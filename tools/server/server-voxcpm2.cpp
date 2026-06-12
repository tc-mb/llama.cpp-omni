// VoxCPM2 TTS standalone HTTP server
// Adapted from server-omni.cpp pattern

#include "voxcpm2_runtime.h"
#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"

#include <mutex>
#include <thread>
#include <fstream>
#include <string>
#include <cstring>
#include <atomic>
#include <signal.h>

#include "httplib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ─── base64 decode helpers ────────────────────────────────────────────────

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

using raw_buffer = std::vector<uint8_t>;

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int in_ = 0;
    int in_len = encoded_string.size();
    uint8_t char_array_4[4];
    uint8_t char_array_3[3];
    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }
            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];
            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }
    if (i) {
        for (int j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        for (int j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }
        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];
        for (int j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }
    return ret;
}

// ─── JSON helpers ─────────────────────────────────────────────────────────

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

// ─── WAV encode / decode ──────────────────────────────────────────────────

static std::string voxcpm2_encode_wav(const std::vector<float> & pcm, int sample_rate) {
    const int32_t n_samples = static_cast<int32_t>(pcm.size());
    const int32_t byte_rate = sample_rate * 2;
    const int32_t data_size = n_samples * 2;
    const int32_t chunk_size = 36 + data_size;

    std::string buf;
    buf.resize(44 + data_size);
    auto * p = buf.data();

    auto w32 = [&](int32_t v) { memcpy(p, &v, 4); p += 4; };
    auto w16 = [&](int16_t v) { memcpy(p, &v, 2); p += 2; };

    memcpy(p, "RIFF", 4); p += 4; w32(chunk_size); memcpy(p, "WAVE", 4); p += 4;
    memcpy(p, "fmt ", 4); p += 4; w32(16); w16(1); w16(1); w32(sample_rate); w32(byte_rate); w16(2); w16(16);
    memcpy(p, "data", 4); p += 4; w32(data_size);

    auto * dst = reinterpret_cast<int16_t *>(p);
    for (int32_t i = 0; i < n_samples; ++i) {
        float v = std::max(-1.0f, std::min(1.0f, pcm[i]));
        dst[i] = static_cast<int16_t>(v * 32767.0f);
    }
    return buf;
}

static std::vector<float> voxcpm2_load_wav_from_memory(const std::vector<uint8_t> & data, int & out_sr) {
    if (data.size() < 44) return {};
    const auto * p = data.data();
    if (memcmp(p, "RIFF", 4) != 0 || memcmp(p + 8, "WAVE", 4) != 0) return {};

    int16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    int32_t sample_rate = 0, data_size = 0;
    size_t offset = 12;
    while (offset + 8 <= data.size()) {
        char id[4]; int32_t size;
        memcpy(id, p + offset, 4); offset += 4;
        memcpy(&size, p + offset, 4); offset += 4;
        if (memcmp(id, "fmt ", 4) == 0) {
            if (offset + 16 > data.size()) return {};
            memcpy(&audio_format, p + offset, 2);
            memcpy(&num_channels, p + offset + 2, 2);
            memcpy(&sample_rate, p + offset + 4, 4);
            memcpy(&bits_per_sample, p + offset + 14, 2);
            offset += size;
        } else if (memcmp(id, "data", 4) == 0) {
            data_size = size;
            break;
        } else {
            offset += size;
        }
    }
    if (audio_format != 1 || data_size <= 0 || num_channels < 1) return {};

    out_sr = sample_rate;
    int n_samples = data_size / (bits_per_sample / 8);
    const auto * raw = reinterpret_cast<const int16_t *>(p + offset);
    std::vector<float> pcm(n_samples / num_channels);
    for (int i = 0; i < n_samples / num_channels; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
            sum += static_cast<float>(raw[i * num_channels + ch]) / 32768.0f;
        }
        pcm[i] = sum / static_cast<float>(num_channels);
    }
    return pcm;
}

// ─── Server state ─────────────────────────────────────────────────────────

struct voxcpm2_server_state {
    VoxCPM2Runtime * runtime = nullptr;
    std::mutex mutex;
};

// ─── Static signal handling ───────────────────────────────────────────────

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// ─── main ─────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("VoxCPM2 TTS HTTP server starting...\n");

    // HTTP server setup
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLServer svr(params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
#else
    httplib::Server svr;
#endif

    voxcpm2_server_state state;

    // ── Health ────────────────────────────────────────────────────────────

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "voxcpm2"}};
        res.set_header("X-Engine", "voxcpm2");
        res_ok(res, health);
    });

    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "voxcpm2"}};
        res.set_header("X-Engine", "voxcpm2");
        res_ok(res, health);
    });

    // ── POST /v1/voxcpm2/init ─────────────────────────────────────────────

    svr.Post("/v1/voxcpm2/init", [&](const httplib::Request & req, httplib::Response & res) {
        LOG_INF("handle_voxcpm2_init\n");
        json data = json::parse(req.body);

        std::string base_lm = json_value(data, "base_lm", std::string(""));
        std::string acoustic = json_value(data, "acoustic", std::string(""));
        int n_gpu_layers = json_value(data, "n_gpu_layers", -1);

        if (base_lm.empty() || acoustic.empty()) {
            res_error(res, format_error_response("\"base_lm\" and \"acoustic\" are required"));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (state.runtime) {
                state.runtime->free();
                delete state.runtime;
                state.runtime = nullptr;
            }

            auto * rt = new VoxCPM2Runtime();
            if (!rt->init(base_lm, acoustic, n_gpu_layers, /*use_gpu_backend=*/true)) {
                std::string err = rt->last_error();
                rt->free();
                delete rt;
                res_error(res, format_error_response("VoxCPM2 init failed: " + err, "server_error"));
                return;
            }
            state.runtime = rt;
        }

        json ack = {
            {"success", true},
            {"base_lm", base_lm},
            {"acoustic", acoustic},
            {"sample_rate", state.runtime->sample_rate()}
        };
        res_ok(res, ack);
    });

    // ── POST /v1/audio/speech (OpenAI-compatible) ─────────────────────────

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        LOG_INF("handle_audio_speech\n");
        json data = json::parse(req.body);

        std::string input = json_value(data, "input", std::string(""));
        if (input.empty()) {
            res_error(res, format_error_response("\"input\" is required"));
            return;
        }

        std::string model = json_value(data, "model", std::string(""));
        if (!model.empty() && model != "voxcpm" && model != "voxcpm2") {
            res_error(res, format_error_response("unknown model: \"" + model + "\", expected \"voxcpm\""));
            return;
        }

        std::string response_format = json_value(data, "response_format", std::string("wav"));
        if (response_format != "wav" && response_format != "pcm") {
            res_error(res, format_error_response("supported response_format: wav, pcm"));
            return;
        }

        VoxCPM2GenerateParams gen_params;
        gen_params.seed                = json_value(data, "seed", 42);
        gen_params.cfg_value           = json_value(data, "cfg_value", 2.0f);
        gen_params.inference_timesteps = json_value(data, "inference_timesteps", 10);
        gen_params.max_steps           = json_value(data, "max_steps", 200);
        gen_params.temperature         = json_value(data, "temperature", 1.0f);

        VoxCPM2Runtime * rt = nullptr;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            rt = state.runtime;
        }
        if (!rt || !rt->initialized()) {
            res_error(res, format_error_response("VoxCPM2 not initialized. Call /v1/voxcpm2/init first or provide --voxcpm2-base-lm at startup."));
            return;
        }

        std::vector<float> wav;
        if (data.contains("reference_audio") && data.at("reference_audio").is_string()) {
            std::string ref_b64 = data.at("reference_audio").get<std::string>();
            if (!ref_b64.empty()) {
                std::vector<uint8_t> ref_bytes = base64_decode(ref_b64);
                int ref_sr = 0;
                std::vector<float> ref_pcm = voxcpm2_load_wav_from_memory(ref_bytes, ref_sr);
                if (ref_pcm.empty()) {
                    res_error(res, format_error_response("Invalid reference_audio WAV data"));
                    return;
                }
                gen_params.reference_sample_rate = ref_sr;
                wav = rt->generate_with_clone(input, ref_pcm, gen_params);
            } else {
                wav = rt->generate(input, gen_params);
            }
        } else {
            wav = rt->generate(input, gen_params);
        }

        if (wav.empty()) {
            res_error(res, format_error_response("VoxCPM2 generation failed: " + rt->last_error(), "server_error"));
            return;
        }

        int sr = rt->sample_rate();
        if (response_format == "wav") {
            std::string wav_data = voxcpm2_encode_wav(wav, sr);
            res.set_content(wav_data, "audio/wav");
        } else {
            std::string pcm_data(wav.size() * sizeof(float), '\0');
            memcpy(pcm_data.data(), wav.data(), wav.size() * sizeof(float));
            res.set_content(pcm_data, "audio/pcm");
        }
    });

    // ── POST /v1/audio/speech/stream ──────────────────────────────────────

    svr.Post("/v1/audio/speech/stream", [&](const httplib::Request & req, httplib::Response & res) {
        LOG_INF("handle_audio_speech_stream\n");
        json data = json::parse(req.body);

        std::string input = json_value(data, "input", std::string(""));
        if (input.empty()) {
            res_error(res, format_error_response("\"input\" is required"));
            return;
        }

        std::string model = json_value(data, "model", std::string(""));
        if (!model.empty() && model != "voxcpm" && model != "voxcpm2") {
            res_error(res, format_error_response("unknown model: \"" + model + "\", expected \"voxcpm\""));
            return;
        }

        VoxCPM2GenerateParams gen_params;
        gen_params.seed                = json_value(data, "seed", 42);
        gen_params.cfg_value           = json_value(data, "cfg_value", 2.0f);
        gen_params.inference_timesteps = json_value(data, "inference_timesteps", 10);
        gen_params.max_steps           = json_value(data, "max_steps", 200);
        gen_params.temperature         = json_value(data, "temperature", 1.0f);

        VoxCPM2Runtime * rt = nullptr;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            rt = state.runtime;
        }
        if (!rt || !rt->initialized()) {
            res_error(res, format_error_response("VoxCPM2 not initialized. Call /v1/voxcpm2/init first or provide --voxcpm2-base-lm at startup."));
            return;
        }

        int sr = rt->sample_rate();

        res.set_chunked_content_provider(
            "audio/wav",
            [rt, input, gen_params, sr, data](size_t /*offset*/, httplib::DataSink & sink) -> bool {
                // Write WAV header first (with unknown data size)
                {
                    std::string header;
                    header.resize(44);
                    auto * p = header.data();
                    auto w32 = [&](int32_t v) { memcpy(p, &v, 4); p += 4; };
                    auto w16 = [&](int16_t v) { memcpy(p, &v, 2); p += 2; };
                    memcpy(p, "RIFF", 4); p += 4; w32(0x7FFFFFFF); memcpy(p, "WAVE", 4); p += 4;
                    memcpy(p, "fmt ", 4); p += 4; w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
                    memcpy(p, "data", 4); p += 4; w32(0x7FFFFFFF);
                    sink.write(header.data(), header.size());
                }

                bool has_clone = data.contains("reference_audio") && data.at("reference_audio").is_string()
                                 && !data.at("reference_audio").get<std::string>().empty();

                auto chunk_callback = [&sink](const std::vector<float> & chunk, bool /*is_final*/) {
                    if (chunk.empty()) return;
                    std::string buf(chunk.size() * 2, '\0');
                    auto * dst = reinterpret_cast<int16_t *>(buf.data());
                    for (size_t i = 0; i < chunk.size(); ++i) {
                        float v = std::max(-1.0f, std::min(1.0f, chunk[i]));
                        dst[i] = static_cast<int16_t>(v * 32767.0f);
                    }
                    sink.write(buf.data(), buf.size());
                };

                if (has_clone) {
                    std::string ref_b64 = data.at("reference_audio").get<std::string>();
                    std::vector<uint8_t> ref_bytes = base64_decode(ref_b64);
                    int ref_sr = 0;
                    std::vector<float> ref_pcm = voxcpm2_load_wav_from_memory(ref_bytes, ref_sr);
                    if (ref_pcm.empty()) {
                        return false;
                    }
                    auto p = gen_params;
                    p.reference_sample_rate = ref_sr;
                    VoxCPM2Runtime * mutable_rt = const_cast<VoxCPM2Runtime *>(rt);
                    std::vector<float> wav = mutable_rt->generate_with_clone(input, ref_pcm, p);
                    if (wav.empty()) return false;
                    const size_t chunk_size = sr / 10; // 100ms chunks
                    for (size_t i = 0; i < wav.size(); i += chunk_size) {
                        size_t len = std::min(chunk_size, wav.size() - i);
                        std::vector<float> chunk(wav.begin() + i, wav.begin() + i + len);
                        chunk_callback(chunk, false);
                    }
                } else {
                    VoxCPM2Runtime * mutable_rt = const_cast<VoxCPM2Runtime *>(rt);
                    mutable_rt->generate_streaming(input, chunk_callback, gen_params);
                }

                sink.done();
                return true;
            }
        );
    });

    // ── GET /v1/audio/speech/models ───────────────────────────────────────

    svr.Get("/v1/audio/speech/models", [&](const httplib::Request &, httplib::Response & res) {
        json models = json::array();
        VoxCPM2Runtime * rt = nullptr;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            rt = state.runtime;
        }
        if (rt && rt->initialized()) {
            models.push_back({
                {"id", "voxcpm"},
                {"object", "model"},
                {"sample_rate", rt->sample_rate()},
                {"feat_dim", rt->feat_dim()},
                {"patch_size", rt->patch_size()}
            });
        }
        json resp = {{"object", "list"}, {"data", models}};
        res_ok(res, resp);
    });

    // ── Startup: load VoxCPM2 models ──────────────────────────────────────

    bool has_voxcpm2 = !params.voxcpm2_base_lm.empty() && !params.voxcpm2_acoustic.empty();

    if (!has_voxcpm2) {
        LOG_ERR("No VoxCPM2 models specified. Provide --voxcpm2-base-lm and --voxcpm2-acoustic\n");
        llama_backend_free();
        return 1;
    }

    LOG_INF("Loading VoxCPM2 runtime...\n");
    LOG_INF("  BaseLM:   %s\n", params.voxcpm2_base_lm.c_str());
    LOG_INF("  Acoustic: %s\n", params.voxcpm2_acoustic.c_str());

    {
        auto * rt = new VoxCPM2Runtime();
        if (!rt->init(params.voxcpm2_base_lm, params.voxcpm2_acoustic, params.voxcpm2_n_gpu_layers, /*use_gpu_backend=*/true)) {
            LOG_ERR("VoxCPM2 init failed: %s\n", rt->last_error().c_str());
            rt->free();
            delete rt;
            llama_backend_free();
            return 1;
        }
        state.runtime = rt;
        LOG_INF("VoxCPM2 loaded, sample_rate=%d\n", rt->sample_rate());
    }

    // ── Bind and listen ───────────────────────────────────────────────────

    if (params.n_threads_http < 1) {
        params.n_threads_http = std::max(2, (int) std::thread::hardware_concurrency() - 1);
    }
    svr.new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    bool was_bound = false;
    if (params.hostname.find(".sock") != std::string::npos) {
        svr.set_address_family(AF_UNIX);
        was_bound = svr.bind_to_port(params.hostname, 8080);
    } else {
        if (params.port == 0) {
            int bound_port = svr.bind_to_any_port(params.hostname);
            if ((was_bound = (bound_port >= 0))) {
                params.port = bound_port;
            }
        } else {
            was_bound = svr.bind_to_port(params.hostname, params.port);
        }
    }

    if (!was_bound) {
        LOG_ERR("Couldn't bind HTTP server socket, hostname: %s, port: %d\n", params.hostname.c_str(), params.port);
        if (state.runtime) {
            state.runtime->free();
            delete state.runtime;
        }
        llama_backend_free();
        return 1;
    }

    std::thread http_thread([&]() { svr.listen_after_bind(); });
    svr.wait_until_ready();

    LOG_INF("VoxCPM2 TTS server is listening on http://%s:%d\n", params.hostname.c_str(), params.port);

    // ── Signal handling ───────────────────────────────────────────────────

    std::atomic<bool> running{true};
    shutdown_handler = [&](int) {
        running.store(false);
        svr.stop();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // Block until shutdown
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    LOG_INF("Shutting down...\n");

    if (http_thread.joinable()) {
        http_thread.join();
    }

    if (state.runtime) {
        state.runtime->free();
        delete state.runtime;
    }

    llama_backend_free();
    return 0;
}
