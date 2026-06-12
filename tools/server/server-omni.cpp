// Omni streaming HTTP server — standalone omni API endpoints
// Based on the old server.cpp omni handlers, adapted for the new llama.cpp APIs

#include "omni.h"
#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"
#include "session.h"
#include "ws_handler.h"

#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <string>

#include "httplib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static json format_error_response(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}};
}

template<typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key)) {
        try {
            return body.at(key).get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

static void res_ok(httplib::Response & res, const json & data) {
    res.set_content(data.dump(), "application/json");
}

static void res_error(httplib::Response & res, const json & err) {
    res.status = json_value(err, "code", 500);
    res.set_content(err.dump(), "application/json");
}

static bool server_sent_event(httplib::DataSink & sink, const json & ev) {
    std::string str = "data: " + ev.dump() + "\n\n";
    return sink.write(str.data(), str.size());
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static bool ensure_omni_model_paths_from_llm(common_params & params) {
    if (params.model.path.empty()) {
        return false;
    }
    const std::string root = parent_dir(params.model.path);
    if (root.empty()) {
        return false;
    }
    if (params.vpm_model.empty()) {
        params.vpm_model = root + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    }
    if (params.apm_model.empty()) {
        params.apm_model = root + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    }
    if (params.tts_model.empty()) {
        params.tts_model = root + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    }
    if (params.tts_bin_dir.empty()) {
        params.tts_bin_dir = root + "/tts";
    }
    return true;
}

struct omni_server_state {
    omni_context * octx = nullptr;    // WS backend uses this as shared_octx
    std::mutex octx_mutex;            // protects omni_context lifecycle + prefill/decode entry
    SessionManager session_mgr;       // WS backend session management
};

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // omni HTTP server is single-session (1:1 duplex), so 1 sequence is enough.
    // common_params defaults n_parallel to -1 ("auto"); each example resolves it
    // itself (see tools/server/server.cpp). Without this, n_seq_max overflows
    // uint32 and trips LLAMA_MAX_SEQ(256) inside llama_context.
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("Omni HTTP server starting...\n");

    // auto-detect omni model paths
    if (!params.vpm_model.empty() || !params.apm_model.empty() || !params.tts_model.empty()) {
        LOG_INF("Using explicit omni model paths from args\n");
    }

    // HTTP server setup
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif

    omni_server_state state;

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    // POST /v1/stream/omni_init
    svr.Post("/v1/stream/omni_init", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("msg_type") && !data.contains("media_type")) {
            res_error(res, format_error_response("\"msg_type\" or \"media_type\" must be provided"));
            return;
        }

        int media_type = data.value("msg_type", data.value("media_type", 2));
        bool use_tts   = data.value("use_tts", true);
        bool duplex_mode = data.value("duplex_mode", false);
        int tts_gpu_layers = data.value("tts_gpu_layers", 100);
        std::string token2wav_device = data.value("token2wav_device", "gpu:0");
        std::string output_dir = data.value("output_dir", "./tools/omni/output");
        std::string voice_audio = data.value("voice_audio", "");

        // validate key files
        auto check_file = [&](const std::string & role, const std::string & path) -> bool {
            if (path.empty()) return true;
            std::ifstream f(path);
            if (!f.good()) {
                res_error(res, format_error_response(
                    "omni_init missing required model file (" + role + "): " + path));
                return false;
            }
            return true;
        };

        // Keep legacy HTTP aligned with /backend: the LLM path (-m) anchors the
        // fixed MiniCPM-o sub-model layout; request model_dir is ignored.
        if (!ensure_omni_model_paths_from_llm(params)) {
            res_error(res, format_error_response("LLM model path (-m) is required to derive omni model paths"));
            return;
        }

        if (!check_file("LLM",    params.model.path) ||
            !check_file("vision", params.vpm_model)  ||
            !check_file("audio",  params.apm_model)  ||
            (use_tts && !check_file("tts", params.tts_model))) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx) {
                omni_free(state.octx);
                state.octx = nullptr;
            }
        }

        omni_context * octx = omni_init(&params, media_type, use_tts, params.tts_bin_dir, tts_gpu_layers,
                                         token2wav_device, duplex_mode,
                                         /*existing_model=*/nullptr, /*existing_ctx=*/nullptr, output_dir);
        if (!octx) {
            res_error(res, format_error_response("omni_init failed"));
            return;
        }

        // voice clone / assistant prompt
        if (data.contains("voice_clone_prompt")) octx->omni_voice_clone_prompt = data["voice_clone_prompt"];
        if (data.contains("assistant_prompt")) octx->omni_assistant_prompt = data["assistant_prompt"];

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            state.octx = octx;
        }

        res_ok(res, {{"success", true}});
    });

    // POST /v1/stream/prefill
    svr.Post("/v1/stream/prefill", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("audio_path_prefix") || !data.at("audio_path_prefix").is_string()) {
            res_error(res, format_error_response("\"audio_path_prefix\" must be provided as string"));
            return;
        }
        if (!data.contains("cnt") || !data.at("cnt").is_number_integer()) {
            res_error(res, format_error_response("\"cnt\" must be provided as integer"));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string audio_path = data.at("audio_path_prefix");
        std::string img_path   = data.value("img_path_prefix", "");
        std::string text       = data.value("text", "");
        int cnt                = data.at("cnt");
        int max_slice_nums     = data.value("max_slice_nums", -1);

        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            ok = stream_prefill(state.octx, audio_path, img_path, cnt, max_slice_nums, text);
        }

        if (!ok) {
            res_error(res, format_error_response("stream_prefill failed"));
            return;
        }

        res_ok(res, {{"success", true}, {"audio_path_prefix", audio_path}, {"cnt", cnt}});
    });

    // POST /v1/stream/decode (SSE)
    svr.Post("/v1/stream/decode", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string debug_dir = data.value("debug_dir", "./");
        bool stream = data.value("stream", true);
        int round_idx = data.value("round_idx", -1);

        // length_penalty
        if (data.contains("length_penalty") && data.at("length_penalty").is_number()) {
            float lp = data.at("length_penalty").get<float>();
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx != nullptr) {
                state.octx->length_penalty = lp;
            }
        }

        if (!stream) {
            bool ok = false;
            {
                std::lock_guard<std::mutex> lock(state.octx_mutex);
                ok = stream_decode(state.octx, debug_dir, round_idx);
            }
            if (!ok) {
                res_error(res, format_error_response("stream_decode failed"));
                return;
            }
            res_ok(res, {{"success", true}});
            return;
        }

        // SSE streaming
        res.set_chunked_content_provider("text/event-stream",
            [&](size_t, httplib::DataSink & sink) -> bool {
                // reset state
                {
                    std::lock_guard<std::mutex> lock(state.octx->text_mtx);
                    state.octx->text_queue.clear();
                    state.octx->text_done_flag = false;
                    state.octx->text_streaming = true;
                }

                // start decode in background thread
                std::thread worker([&](std::string dd, int ri) {
                    std::lock_guard<std::mutex> lock(state.octx_mutex);
                    (void) stream_decode(state.octx, dd, ri);
                }, debug_dir, round_idx);

                // poll text queue
                while (true) {
                    std::unique_lock<std::mutex> lk(state.octx->text_mtx);
                    state.octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{
                        return !state.octx->text_queue.empty() || state.octx->text_done_flag;
                    });

                    while (!state.octx->text_queue.empty()) {
                        std::string frag = std::move(state.octx->text_queue.front());
                        state.octx->text_queue.pop_front();
                        lk.unlock();

                        json ev;
                        if (frag == "__IS_LISTEN__") {
                            ev = {{"content", ""}, {"stop", false}, {"is_listen", true}, {"end_of_turn", true}};
                        } else if (frag == "__END_OF_TURN__") {
                            ev = {{"content", ""}, {"stop", true}, {"is_listen", false}, {"end_of_turn", true}};
                        } else {
                            ev = {{"content", frag}, {"stop", false}, {"is_listen", false}, {"end_of_turn", false}};
                        }

                        if (!server_sent_event(sink, ev)) {
                            if (worker.joinable()) worker.join();
                            return false;
                        }
                        lk.lock();
                    }

                    if (state.octx->text_done_flag) break;
                }

                if (worker.joinable()) worker.join();

                // send done
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
                return true;
            });
    });

    // POST /v1/stream/update_session_config
    svr.Post("/v1/stream/update_session_config", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        int media_type = data.value("media_type", -1);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized"));
                return;
            }
            if (media_type > 0) {
                state.octx->media_type = media_type;
            }
        }

        res_ok(res, {{"success", true}});
    });

    //
    // Backend Protocol (WebSocket + HTTP unary)
    //
    svr.WebSocket("/backend", [&](const httplib::Request &, httplib::ws::WebSocket & ws) {
        handle_ws_backend(ws, state.session_mgr, params,
                          /*model*/nullptr, /*ctx*/nullptr,
                          state.octx, state.octx_mutex);
    });

    svr.Post("/sessions/:session_id/close", [&](const httplib::Request & req, httplib::Response & res) {
        std::string session_id = req.path_params.at("session_id");
        LOG_INF("Close session requested: %s\n", session_id.c_str());

        auto * session = state.session_mgr.get(session_id);
        if (!session || session->state != SessionState::ACTIVE) {
            res_error(res, format_error_response("session not found", "not_found"));
            res.status = 404;
            return;
        }

        state.session_mgr.request_transport_close(session_id);

        // close is a completion primitive: do not return until inference
        // threads are stopped and the shared omni_context is safe to reuse.
        {
            std::lock_guard<std::mutex> octx_lock(state.octx_mutex);
            auto * closing = state.session_mgr.get(session_id);
            if (closing && closing->octx) {
                closing->octx->break_event = true;
                {
                    std::lock_guard<std::mutex> lk(closing->octx->text_mtx);
                    closing->octx->text_queue.clear();
                    closing->octx->text_done_flag = true;
                }
                closing->octx->text_cv.notify_all();
                omni_prepare_for_reuse(closing->octx);
            }

            state.session_mgr.close(session_id);
        }

        json resp;
        resp["ok"] = true;
        resp["session_id"] = session_id;
        resp["closed"] = true;
        res_ok(res, resp);
    });

    // start server
    svr.listen("0.0.0.0", params.port);

    // cleanup
    {
        std::lock_guard<std::mutex> lock(state.octx_mutex);
        if (state.octx) {
            omni_free(state.octx);
            state.octx = nullptr;
        }
    }
    llama_backend_free();

    return 0;
}
