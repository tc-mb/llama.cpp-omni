#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include <cpp-httplib/httplib.h>

#include "ws_handler.h"
#include "session.h"
#include "protocol.h"
#include "omni.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "audition.h"
#include "vision.h"

namespace fs = std::filesystem;

static double elapsed_ms(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static bool utf8_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

// Stream-safe UTF-8: emit only complete code points, hold an incomplete trailing
// byte sequence in `pending` until the next fragment, and replace invalid bytes
// with U+FFFD. Prevents broken multi-byte chars when text is chunked over WS.
static std::string sanitize_utf8_stream(std::string & pending,
                                        const std::string & fragment,
                                        bool flush = false) {
    static const std::string replacement = "\xEF\xBF\xBD";
    std::string input = pending + fragment;
    pending.clear();

    std::string out;
    size_t i = 0;
    while (i < input.size()) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            i++;
            continue;
        }

        int need = 0;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3;
        } else {
            out += replacement;
            i++;
            continue;
        }

        if (i + need >= input.size()) {
            pending = input.substr(i);
            break;
        }

        bool ok = true;
        for (int j = 1; j <= need; ++j) {
            ok = ok && utf8_cont(static_cast<unsigned char>(input[i + j]));
        }
        if (ok && c == 0xE0) {
            ok = static_cast<unsigned char>(input[i + 1]) >= 0xA0;
        } else if (ok && c == 0xED) {
            ok = static_cast<unsigned char>(input[i + 1]) < 0xA0;
        } else if (ok && c == 0xF0) {
            ok = static_cast<unsigned char>(input[i + 1]) >= 0x90;
        } else if (ok && c == 0xF4) {
            ok = static_cast<unsigned char>(input[i + 1]) < 0x90;
        }

        if (!ok) {
            out += replacement;
            i++;
            continue;
        }

        out.append(input, i, need + 1);
        i += need + 1;
    }

    if (flush && !pending.empty()) {
        out += replacement;
        pending.clear();
    }
    return out;
}

// Build the metrics object attached to downlink events from current runtime
// state (KV length, TTS token count, vision tokens) plus the timings passed in.
static ProtocolMetrics make_runtime_metrics(omni_context * octx,
                                            double prefill_ms = 0.0,
                                            double generate_ms = 0.0,
                                            double wall_clock_ms = 0.0,
                                            int n_tokens = 0,
                                            int vision_slices = 0) {
    ProtocolMetrics metrics;
    metrics.backend = "llama.cpp-omni";
    if (octx) {
        metrics.kv_cache_length = std::max(0, octx->n_past);
        metrics.n_tts_tokens = (int)octx->tts_all_generated_tokens.size();
        if (vision_slices > 0 && octx->ctx_vision) {
            metrics.vision_tokens = vision_slices * vision_n_output_tokens(octx->ctx_vision);
        }
    }
    metrics.prefill_ms = prefill_ms;
    metrics.generate_ms = generate_ms;
    metrics.wall_clock_ms = wall_clock_ms;
    metrics.cost_llm_ms = generate_ms;
    metrics.n_tokens = n_tokens;
    metrics.vision_slices = vision_slices;
    return metrics;
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

// Derive vision/audio/TTS sub-model paths from the directory of the main -m
// model (same convention as omni-cli's resolve_model_paths): the llm is the
// only quantized file, the F16 sub-models live in fixed sub-dirs next to it,
// so the -m path is enough to locate them all. Only fills paths left unset.
static void ensure_omni_model_paths(common_params & params) {
    if (params.model.path.empty()) {
        return;
    }
    const std::string root = parent_dir(params.model.path);
    if (root.empty()) {
        return;
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
}

static void clear_text_stream_state(omni_context * octx) {
    std::lock_guard<std::mutex> lk(octx->text_mtx);
    octx->text_queue.clear();
    octx->text_done_flag = false;
    octx->text_streaming = false;
}

static void stop_reusable_octx_threads(omni_context * octx) {
    omni_prepare_for_reuse(octx);
}

// Wipe per-session state on a reused context: stop threads, reset counters/flags,
// clear LLM/TTS/whisper KV caches, so the next session starts clean on the same
// loaded model. Used by the shared-octx reuse path in create_session_octx.
static void reset_octx_for_session(omni_context * octx, const ParsedSessionInit & init,
                                   const std::string & output_dir) {
    stop_reusable_octx_threads(octx);

    const bool duplex_mode = (init.mode == "full_duplex");
    octx->async = true;
    octx->duplex_mode = duplex_mode;
    octx->base_output_dir = output_dir;

    octx->break_event.store(false);
    octx->current_turn_ended = false;
    octx->llm_generation_done = false;
    octx->need_speek = false;
    octx->speek_done = true;

    octx->n_past = 0;
    octx->n_keep = 0;
    octx->system_prompt_initialized = false;
    octx->simplex_round_idx = 0;
    octx->wav_turn_base = 0;
    octx->round_start_positions.clear();
    octx->force_listen_used = 0;

    octx->tts_all_generated_tokens.clear();
    octx->tts_token_buffer.clear();
    octx->tts_n_past_accumulated = 0;
    octx->tts_condition_saved = false;
    octx->tts_condition_embeddings.clear();
    octx->tts_condition_length = 0;
    octx->tts_condition_n_embd = 0;

    clear_text_stream_state(octx);

    if (octx->ctx_llama) {
        llama_memory_t mem = llama_get_memory(octx->ctx_llama);
        if (mem) {
            llama_memory_clear(mem, /*data=*/false);
        }
    }
    if (octx->ctx_tts_llama) {
        llama_memory_t mem = llama_get_memory(octx->ctx_tts_llama);
        if (mem) {
            llama_memory_clear(mem, /*data=*/false);
        }
    }
    if (octx->ctx_audio) {
        audition_whisper_clear_kv_cache(octx->ctx_audio);
    }
    if (octx->ctx_vision) {
        vision_set_max_slice_nums(octx->ctx_vision, -1);
    }

    if (!init.system_prompt.empty()) {
        octx->omni_assistant_prompt = init.system_prompt;
    }
}

// Apply the optional opaque init.config (sampling/decoding knobs, §6) onto the
// context and params. Unknown/missing keys are left at model defaults.
static void apply_session_config(common_params & params, omni_context * octx, const ParsedSessionInit & init) {
    if (!init.config.is_object()) {
        return;
    }
    if (init.config.contains("length_penalty") && init.config.at("length_penalty").is_number()) {
        octx->length_penalty = init.config.at("length_penalty").get<float>();
    }
    if (init.config.contains("listen_prob_scale") && init.config.at("listen_prob_scale").is_number()) {
        octx->listen_prob_scale = init.config.at("listen_prob_scale").get<float>();
    }
    if (init.config.contains("force_listen_count") && init.config.at("force_listen_count").is_number_integer()) {
        octx->force_listen_count = init.config.at("force_listen_count").get<int>();
        octx->force_listen_used = 0;
    }
    if (init.config.contains("max_new_speak_tokens_per_chunk") && init.config.at("max_new_speak_tokens_per_chunk").is_number_integer()) {
        octx->max_new_speak_tokens_per_chunk = init.config.at("max_new_speak_tokens_per_chunk").get<int>();
    }
    if (init.config.contains("tts_temperature") && init.config.at("tts_temperature").is_number()) {
        octx->tts_temperature = init.config.at("tts_temperature").get<float>();
    }
    if (init.config.contains("temperature") && init.config.at("temperature").is_number()) {
        params.sampling.temp = init.config.at("temperature").get<float>();
    }
    if (init.config.contains("top_p") && init.config.at("top_p").is_number()) {
        params.sampling.top_p = init.config.at("top_p").get<float>();
    }
    if (init.config.contains("top_k") && init.config.at("top_k").is_number_integer()) {
        params.sampling.top_k = init.config.at("top_k").get<int>();
    }
}

// ============================================================================
// TempMediaFiles helpers
// ============================================================================

std::string TempMediaFiles::write_temp_file(const std::string & temp_dir, const std::string & prefix,
                                            const std::string & suffix, const void * data, size_t len) {
    fs::path dir(temp_dir);
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    std::string path = (dir / (prefix + suffix)).string();
    std::ofstream out(path, std::ios::binary);
    if (!out) return "";
    out.write(static_cast<const char *>(data), len);
    out.close();
    if (!out) return "";
    return path;
}

std::string TempMediaFiles::write_audio_wav(const std::string & b64, const std::string & temp_dir, int counter) {
    // Decode base64 → float32 PCM samples
    auto pcm = b64_to_float32_pcm(b64);
    if (pcm.empty()) return "";

    // Backend protocol audio payloads are mono float32 PCM in this path.
    int n_samples = static_cast<int>(pcm.size());
    int sample_rate = 16000;
    int n_channels = 1;
    int bits_per_sample = 32;
    int byte_rate = sample_rate * n_channels * bits_per_sample / 8;
    int block_align = n_channels * bits_per_sample / 8;
    int data_size = n_samples * block_align;
    int file_size = 36 + data_size;

    // Build minimal WAV header + PCM data
    std::vector<char> wav(44 + data_size);
    auto wr = [&](int offset, const char * s, int n) { memcpy(&wav[offset], s, n); };
    auto wi = [&](int offset, int32_t val) { memcpy(&wav[offset], &val, 4); };
    auto ws = [&](int offset, int16_t val) { memcpy(&wav[offset], &val, 2); };

    wr(0,  "RIFF", 4); wi(4,  file_size);
    wr(8,  "WAVE", 4);
    wr(12, "fmt ", 4); wi(16, 16);            // subchunk size = 16 for PCM
    ws(20, 3);                                 // audio format = IEEE float
    ws(22, static_cast<int16_t>(n_channels));
    wi(24, sample_rate); wi(28, byte_rate);
    ws(32, static_cast<int16_t>(block_align));
    ws(34, static_cast<int16_t>(bits_per_sample));
    wr(36, "data", 4); wi(40, data_size);
    memcpy(&wav[44], pcm.data(), data_size);

    return write_temp_file(temp_dir, "audio_", "." + std::to_string(counter) + ".wav", wav.data(), wav.size());
}

std::string TempMediaFiles::write_image_jpeg(const std::string & b64, const std::string & temp_dir, int counter) {
    auto raw = b64_decode(b64);
    if (raw.empty()) return "";
    return write_temp_file(temp_dir, "image_", "." + std::to_string(counter) + ".jpg", raw.data(), raw.size());
}

void TempMediaFiles::cleanup() {
    if (!audio_path.empty()) { fs::remove(audio_path); audio_path.clear(); }
    if (!image_path.empty()) { fs::remove(image_path); image_path.clear(); }
}

static std::string shell_quote(const std::string & value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static bool file_nonempty(const std::string & path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::file_size(path, ec) > 0;
}

struct ExtractedVideoMedia {
    std::string video_path;
    std::string audio_path;
    std::vector<std::string> frame_paths;
};

// Decode a base64 MP4 (turn_based `video` content item) to a temp file and use
// ffmpeg to split it into a mono 16k PCM WAV + up to `stack_frames` JPEG frames.
static ExtractedVideoMedia extract_video_mp4_media(const std::string & video_b64,
                                                   const std::string & temp_dir,
                                                   int counter,
                                                   int stack_frames) {
    ExtractedVideoMedia out;
    auto raw = b64_decode(video_b64);
    if (raw.empty()) {
        return out;
    }

    const int n_frames = std::max(1, std::min(stack_frames, 8));
    fs::path dir = fs::path(temp_dir) / ("video_" + std::to_string(counter));
    fs::create_directories(dir);

    out.video_path = (dir / "input.mp4").string();
    {
        std::ofstream f(out.video_path, std::ios::binary);
        if (!f) {
            return out;
        }
        f.write(reinterpret_cast<const char *>(raw.data()), raw.size());
    }
    if (!file_nonempty(out.video_path)) {
        return out;
    }

    std::string audio_path = (dir / "audio.wav").string();
    std::string audio_cmd =
        "ffmpeg -y -hide_banner -loglevel error -i " + shell_quote(out.video_path) +
        " -vn -ac 1 -ar 16000 -c:a pcm_f32le " + shell_quote(audio_path);
    if (std::system(audio_cmd.c_str()) == 0 && file_nonempty(audio_path)) {
        out.audio_path = audio_path;
    }

    std::string frame_pattern = (dir / "frame_%03d.jpg").string();
    std::string frame_cmd =
        "ffmpeg -y -hide_banner -loglevel error -i " + shell_quote(out.video_path) +
        " -an -frames:v " + std::to_string(n_frames) +
        " -q:v 2 " + shell_quote(frame_pattern);
    if (std::system(frame_cmd.c_str()) == 0) {
        for (int i = 1; i <= n_frames; ++i) {
            char name[32];
            snprintf(name, sizeof(name), "frame_%03d.jpg", i);
            std::string frame_path = (dir / name).string();
            if (file_nonempty(frame_path)) {
                out.frame_paths.push_back(frame_path);
            }
        }
    }

    return out;
}

static std::string truncate_for_prompt(const std::string & text, size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars) + "...";
}

static std::string first_system_text(const std::vector<ParsedMessage> & messages) {
    std::string text;
    for (const auto & msg : messages) {
        if (msg.role != "system" || msg.text.empty()) {
            continue;
        }
        if (!text.empty()) {
            text += "\n";
        }
        text += msg.text;
    }
    return text;
}

// Build the ChatML body for a turn: single-turn → just the last user text;
// multi-turn → the last up-to-8 non-system messages with role tags preserved
// (matching the Python backend instead of flattening history into one prompt).
static std::string build_turn_based_chatml_body(const std::vector<ParsedMessage> & messages,
                                                const ParsedMessage * last_user_msg) {
    if (!last_user_msg) {
        return "";
    }
    const std::string last_text = last_user_msg->text;
    if (last_text.empty()) {
        return "";
    }

    int non_system_count = 0;
    for (const auto & msg : messages) {
        if (msg.role != "system" && !msg.text.empty()) {
            non_system_count++;
        }
    }
    if (non_system_count <= 1) {
        return last_text;
    }

    std::vector<const ParsedMessage *> tail;
    tail.reserve(8);
    for (auto it = messages.rbegin(); it != messages.rend() && (int)tail.size() < 8; ++it) {
        if (it->role == "system" || it->text.empty()) {
            continue;
        }
        tail.push_back(&(*it));
    }
    std::reverse(tail.begin(), tail.end());

    std::string prompt;
    bool first_content = true;
    for (const ParsedMessage * msg : tail) {
        const bool is_last = (msg == last_user_msg);
        const std::string text = truncate_for_prompt(msg->text, 1200);

        if (first_content) {
            // stream_prefill(index=0) leaves KV at "<|im_start|>user\n".
            if (msg->role != "user") {
                prompt += "<|im_end|>\n<|im_start|>" + msg->role + "\n";
            }
            first_content = false;
        } else {
            prompt += "<|im_start|>" + msg->role + "\n";
        }

        prompt += text;
        if (!is_last) {
            prompt += "<|im_end|>\n";
        }
    }
    return prompt;
}

// Pick the system/assistant prompt templates for a turn: TTS path uses the
// voice-clone (ref-audio) system prompt; text-only path uses a plain system
// message. Mirrors the Python backend's TTS vs no-TTS chat templates.
static void configure_turn_based_prompt(omni_context * octx,
                                        bool use_tts_template,
                                        const std::string & system_text) {
    if (!octx) {
        return;
    }

    if (use_tts_template) {
        octx->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        octx->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
        octx->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        octx->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
        return;
    }

    const std::string sys = system_text.empty()
        ? "你是一个有用的助手。请准确、清晰地回答用户问题。"
        : system_text;

    // Text-only turn-based chat follows the Python no-TTS path: a normal
    // system message followed by the first user turn. No voice-clone ref audio.
    octx->omni_voice_clone_prompt = "<|im_start|>system\n" + sys;
    octx->omni_assistant_prompt = "<|im_end|>\n<|im_start|>user\n";
    octx->audio_voice_clone_prompt = octx->omni_voice_clone_prompt;
    octx->audio_assistant_prompt = octx->omni_assistant_prompt;
}

// ============================================================================
// Session-level omni init helper
// ============================================================================

static omni_context * create_session_octx(common_params & params, const ParsedSessionInit & init,
                                          llama_model * model, llama_context * ctx,
                                          omni_context *& shared_octx,
                                          const std::string & output_dir) {
    int media_type = 2; // omni
    bool duplex_mode = (init.mode == "full_duplex");
    // Turn-based sessions may request TTS per input.append. Load the TTS-capable
    // context up front, then toggle octx->use_tts per request.
    bool use_tts = true;

    // Build params for omni_init
    auto & p = params;
    p.n_predict = 2048;
    ensure_omni_model_paths(p);

    // Reuse the server-owned context if it matches this session's mode (avoids
    // reloading the model); otherwise tear it down and build a fresh one.
    if (shared_octx && shared_octx->duplex_mode == duplex_mode && shared_octx->use_tts == use_tts) {
        reset_octx_for_session(shared_octx, init, output_dir);
        apply_session_config(p, shared_octx, init);
        LOG_INF("create_session_octx: reused shared octx, duplex=%d, output_dir=%s\n",
                duplex_mode, output_dir.c_str());
        return shared_octx;
    }

    if (shared_octx) {
        omni_free(shared_octx);
        shared_octx = nullptr;
    }

    omni_context * octx = omni_init(&p, media_type, use_tts, p.tts_bin_dir, /*tts_gpu_layers*/99,
                                     /*token2wav_device*/"gpu:0", duplex_mode,
                                     model, ctx, output_dir);
    if (!octx) {
        LOG_ERR("create_session_octx: omni_init failed\n");
        return nullptr;
    }

    octx->async = true;
    octx->duplex_mode = duplex_mode;
    apply_session_config(p, octx, init);

    // Voice clone / system prompt
    if (!init.system_prompt.empty()) {
        octx->omni_assistant_prompt = init.system_prompt;
    }
    shared_octx = octx;

    LOG_INF("create_session_octx: session octx created, duplex=%d, output_dir=%s\n",
            duplex_mode, output_dir.c_str());
    return octx;
}

// ============================================================================
// Main WS handler
// ============================================================================

void handle_ws_backend(httplib::ws::WebSocket & ws,
                        SessionManager & session_mgr,
                        common_params & params_base,
                        llama_model * model,
                        llama_context * ctx,
                        omni_context *& shared_octx,
                        std::mutex & octx_mutex) {
    const std::string temp_dir = fs::temp_directory_path() / "omni_ws";
    fs::create_directories(temp_dir);
    int msg_counter = 0;
    std::vector<std::string> retained_media_files;

    // Helper: fail-fast — send session.closed and close WS
    auto fail_fast = [&](const std::string & session_id, const std::string & reason) {
        if (!session_id.empty()) {
            std::string ev = make_session_closed(session_id, reason).dump();
            ws.send(ev);
        }
        session_mgr.close(session_id);
        for (const auto & path : retained_media_files) {
            fs::remove(path);
        }
        retained_media_files.clear();
        ws.close();
    };

    // Helper: send a JSON event over WS
    auto send_event = [&](const json & ev) -> bool {
        return ws.send(ev.dump());
    };

    // ================================================================
    // Step 1: Read first message — must be session.init
    // ================================================================
    std::string raw_first;
    auto read_result = ws.read(raw_first);
    if (read_result != httplib::ws::ReadResult::Text) {
        LOG_WRN("WS /backend: failed to read init message\n");
        return; // no session yet, just return
    }

    json first_msg;
    try {
        first_msg = json::parse(raw_first);
    } catch (...) {
        LOG_ERR("WS /backend: invalid JSON in init message\n");
        ws.close();
        return;
    }

    auto parsed_init = parse_session_init(first_msg);
    if (!parsed_init.ok) {
        LOG_ERR("WS /backend: session.init parse failed: %s\n", parsed_init.error.c_str());
        // Fail-fast without session — just close
        ws.close();
        return;
    }

    // ================================================================
    // Step 2: Allocate & activate session, create omni_context
    // ================================================================
    std::string session_id = session_mgr.allocate();
    if (session_id.empty()) {
        // Already an active session
        LOG_ERR("WS /backend: session.init rejected — active session exists\n");
        ws.close();
        return;
    }

    std::string session_output_dir = (fs::path(temp_dir) / session_id).string();

    omni_context * octx = nullptr;
    {
        std::lock_guard<std::mutex> lock(octx_mutex);
        octx = create_session_octx(params_base, parsed_init, model, ctx, shared_octx, session_output_dir);
    }
    if (!octx) {
        fail_fast(session_id, "omni_init_failed");
        return;
    }

    // Full-duplex requires index=0 prefill before the first frame. This
    // initializes the system prompt and starts the duplex encoder/LLM pipeline.
    if (parsed_init.mode == "full_duplex" || !parsed_init.ref_audio_b64.empty()) {
        std::string voice_wav;
        if (!parsed_init.ref_audio_b64.empty()) {
            voice_wav = TempMediaFiles::write_audio_wav(parsed_init.ref_audio_b64, temp_dir, msg_counter++);
        }
        if (!parsed_init.ref_audio_b64.empty() && voice_wav.empty()) {
            fail_fast(session_id, "voice_audio_decode_failed");
            return;
        }
        std::lock_guard<std::mutex> lock(octx_mutex);
        if (!stream_prefill(octx, voice_wav, /*img*/"", /*index*/0)) {
            LOG_ERR("WS /backend: voice prefill failed\n");
            if (!voice_wav.empty()) fs::remove(voice_wav);
            fail_fast(session_id, "voice_prefill_failed");
            return;
        }
        if (!voice_wav.empty()) fs::remove(voice_wav);
        if (octx->llm_thread_info) {
            octx->llm_thread_info->start = std::chrono::steady_clock::now();
        }
    }

    // Activate session
    {
        std::lock_guard<std::mutex> lock(octx_mutex);
        if (!session_mgr.activate(session_id, octx, /*owns_octx*/false)) {
            LOG_ERR("WS /backend: session activate failed for %s\n", session_id.c_str());
            fail_fast(session_id, "activate_failed");
            return;
        }
        session_mgr.set_close_callback(session_id, [&ws, session_id]() {
            // Preserve protocol ordering for older runtimes: emit session.closed
            // before the transport close wakes a blocked ws.read().
            ws.send(make_session_closed(session_id, "client_closed").dump());
            ws.close(httplib::ws::CloseStatus::Normal, "client_closed");
        });
    }

    // Send session.created
    send_event(make_session_created(session_id, parsed_init.mode, make_runtime_metrics(octx)));

    LOG_INF("WS /backend: session %s activated, mode=%s\n", session_id.c_str(), parsed_init.mode.c_str());

    // ================================================================
    // Setup audio output callback — sends audio_delta events via WS
    // ================================================================
    struct AudioCbState {
        std::string session_id;
        std::string response_id; // updated per-turn
        httplib::ws::WebSocket * ws = nullptr;
        omni_context * octx = nullptr;
        std::mutex mtx;
        std::chrono::steady_clock::time_point response_start = std::chrono::steady_clock::now();
        // Non-streaming turn-based + TTS: accumulate PCM here and emit it as
        // response.done.audio (full_audio_b64) instead of streaming deltas.
        bool accumulate = false;
        std::vector<float> accum;
        bool emitted_audio = false;
    };
    auto audio_state = std::make_shared<AudioCbState>();
    audio_state->session_id = session_id;
    audio_state->ws = &ws;
    audio_state->octx = octx;

    octx->audio_output_cb = [audio_state](const float * samples, int n_samples, int /*sample_rate*/, bool /*is_final*/) {
        {
            // Non-streaming: buffer the PCM so it can be returned in
            // response.done.audio rather than emitted as audio deltas.
            std::lock_guard<std::mutex> lk(audio_state->mtx);
            if (audio_state->accumulate) {
                if (samples && n_samples > 0) {
                    audio_state->accum.insert(audio_state->accum.end(),
                                              samples, samples + n_samples);
                }
                return;
            }
        }
        if (audio_state->ws && audio_state->ws->is_open()) {
            std::string response_id;
            std::chrono::steady_clock::time_point response_start;
            {
                std::lock_guard<std::mutex> lk(audio_state->mtx);
                response_id = audio_state->response_id;
                response_start = audio_state->response_start;
                audio_state->emitted_audio = true;
            }
            std::string b64 = float32_pcm_to_b64(samples, n_samples);
            ProtocolMetrics metrics = make_runtime_metrics(
                audio_state->octx, 0.0, 0.0, elapsed_ms(response_start));
            json ev = make_audio_delta(audio_state->session_id, response_id, b64, metrics);
            audio_state->ws->send(ev.dump());
        }
    };

    // ================================================================
    // Step 3: Read loop — process input.append messages
    // ================================================================
    std::string raw;
    int response_seq = 0;

    while (true) {
        read_result = ws.read(raw);
        if (read_result != httplib::ws::ReadResult::Text) {
            break; // WS closed or error
        }

        json msg;
        try {
            msg = json::parse(raw);
        } catch (...) {
            fail_fast(session_id, "invalid_json");
            return;
        }

        // Validate message type
        std::string msg_type;
        if (msg.contains("type") && msg.at("type").is_string()) {
            msg_type = msg.at("type").get<std::string>();
        }

        if (msg_type != "input.append") {
            // Unknown or invalid type after init → fail-fast
            LOG_ERR("WS /backend: unexpected message type '%s' from session %s\n",
                    msg_type.c_str(), session_id.c_str());
            fail_fast(session_id, "unexpected_message_type");
            return;
        }

        auto parsed_input = parse_input_append(msg);
        if (!parsed_input.ok) {
            fail_fast(session_id, "invalid_input");
            return;
        }

        const auto t_request_start = std::chrono::steady_clock::now();
        response_seq++;
        std::string response_id = session_id + "_resp_" + std::to_string(response_seq);
        {
            std::lock_guard<std::mutex> lk(audio_state->mtx);
            audio_state->response_id = response_id; // update for audio callback
            audio_state->response_start = t_request_start;
            audio_state->emitted_audio = false;
        }

        // Branch: full_duplex vs turn_based. The input shape MUST match the
        // session mode (schema §7.4 / network §7): turn_based MUST carry
        // `messages`, full_duplex MUST NOT. A mismatch is a fatal protocol
        // violation, not a recoverable branch.
        if (parsed_init.mode == "turn_based") {
            // ================================================================
            // Turn-based input processing
            // ================================================================
            if (parsed_input.messages.is_null()) {
                fail_fast(session_id, "mode_mismatch");
                return;
            }
            auto parsed_msgs = parse_messages_array(parsed_input.messages);
            if (parsed_msgs.empty()) {
                fail_fast(session_id, "empty_messages");
                return;
            }

            TempMediaFiles tmp_files;
            std::vector<std::string> extra_image_paths;
            std::vector<std::string> turn_temp_paths;
            int turn_vision_slices = 0;

            // Images: take all images from the last user message
            const ParsedMessage * last_user_msg = nullptr;
            for (auto it = parsed_msgs.rbegin(); it != parsed_msgs.rend(); ++it) {
                if (it->role == "user") {
                    last_user_msg = &(*it);
                    break;
                }
            }

            if (last_user_msg) {
                int input_index = ++msg_counter;
                for (const auto & img_b64 : last_user_msg->image_b64s) {
                    std::string ipath = TempMediaFiles::write_image_jpeg(img_b64, temp_dir, input_index);
                    if (!ipath.empty()) {
                        tmp_files.image_path = ipath; // last image wins
                    }
                }
                // Audio: use the first audio from the last user message
                if (!last_user_msg->audio_b64s.empty()) {
                    tmp_files.audio_path = TempMediaFiles::write_audio_wav(
                        last_user_msg->audio_b64s[0], temp_dir, input_index);
                }
                for (size_t i = 0; i < last_user_msg->video_b64s.size(); ++i) {
                    const int stack_frames = i < last_user_msg->video_stack_frames.size()
                                           ? last_user_msg->video_stack_frames[i]
                                           : 1;
                    ExtractedVideoMedia video = extract_video_mp4_media(
                        last_user_msg->video_b64s[i], temp_dir, ++msg_counter, stack_frames);
                    if (!video.video_path.empty()) {
                        turn_temp_paths.push_back(video.video_path);
                    }
                    if (!video.audio_path.empty() && tmp_files.audio_path.empty()) {
                        tmp_files.audio_path = video.audio_path;
                    } else if (!video.audio_path.empty()) {
                        turn_temp_paths.push_back(video.audio_path);
                    }
                    for (const auto & frame_path : video.frame_paths) {
                        if (tmp_files.image_path.empty()) {
                            tmp_files.image_path = frame_path;
                        } else {
                            extra_image_paths.push_back(frame_path);
                        }
                    }
                    if (video.frame_paths.empty()) {
                        tmp_files.cleanup();
                        for (const auto & path : turn_temp_paths) fs::remove(path);
                        fail_fast(session_id, "video_decode_failed");
                        return;
                    }
                }
            }

            // The page opens a fresh backend session for every turn. Match the
            // Python backend by preserving message roles in ChatML instead of
            // summarizing history into a single natural-language prompt.
            const std::string system_text = first_system_text(parsed_msgs);
            std::string prompt = build_turn_based_chatml_body(parsed_msgs, last_user_msg);

            turn_vision_slices = (tmp_files.image_path.empty() ? 0 : 1) + (int)extra_image_paths.size();
            double prefill_ms = 0.0;
            double generate_ms = 0.0;
            int n_past_before_decode = 0;

            // Prefill with text + audio + image/video frames
            {
                const auto t_prefill_start = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(octx_mutex);
                if (octx->params) {
                    octx->params->n_predict = parsed_input.max_new_tokens;
                }
                octx->length_penalty = parsed_input.length_penalty;
                const bool prev_use_tts = octx->use_tts;
                octx->use_tts = parsed_input.use_tts_template;
                configure_turn_based_prompt(octx, parsed_input.use_tts_template, system_text);
                if (!octx->system_prompt_initialized) {
                    if (!stream_prefill(octx, /*aud*/"", /*img*/"", /*index*/0)) {
                        octx->use_tts = prev_use_tts;
                        tmp_files.cleanup();
                        for (const auto & path : extra_image_paths) fs::remove(path);
                        for (const auto & path : turn_temp_paths) fs::remove(path);
                        fail_fast(session_id, "system_prefill_failed");
                        return;
                    }
                }
                int input_index = msg_counter > 0 ? msg_counter : ++msg_counter;
                if (!stream_prefill(octx, tmp_files.audio_path, tmp_files.image_path,
                                    input_index, parsed_input.max_slice_nums, prompt)) {
                    octx->use_tts = prev_use_tts;
                    tmp_files.cleanup();
                    for (const auto & path : extra_image_paths) fs::remove(path);
                    for (const auto & path : turn_temp_paths) fs::remove(path);
                    fail_fast(session_id, "prefill_failed");
                    return;
                }
                for (const auto & image_path : extra_image_paths) {
                    if (!stream_prefill(octx, /*aud*/"", image_path,
                                        ++msg_counter, parsed_input.max_slice_nums, /*text*/"")) {
                        octx->use_tts = prev_use_tts;
                        tmp_files.cleanup();
                        for (const auto & path : extra_image_paths) fs::remove(path);
                        for (const auto & path : turn_temp_paths) fs::remove(path);
                        fail_fast(session_id, "video_frame_prefill_failed");
                        return;
                    }
                }
                octx->use_tts = prev_use_tts;
                n_past_before_decode = octx->n_past;
                prefill_ms = elapsed_ms(t_prefill_start);
            }

            if (!tmp_files.audio_path.empty()) retained_media_files.push_back(tmp_files.audio_path);
            if (!tmp_files.image_path.empty()) retained_media_files.push_back(tmp_files.image_path);
            for (const auto & path : extra_image_paths) retained_media_files.push_back(path);
            for (const auto & path : turn_temp_paths) retained_media_files.push_back(path);
            tmp_files.audio_path.clear();
            tmp_files.image_path.clear();

            // Decode
            bool streaming = parsed_input.streaming;
            std::string full_text;
            std::string utf8_pending;
            std::string full_audio_b64;

            {
                std::lock_guard<std::mutex> lk(octx->text_mtx);
                octx->text_queue.clear();
                octx->text_done_flag = false;
                octx->text_streaming = true;
            }

            const bool prev_use_tts = octx->use_tts;
            octx->use_tts = parsed_input.use_tts_template;

            // Non-streaming + TTS: collect PCM during decode/drain and return it
            // in response.done.audio. Streaming keeps emitting audio deltas.
            {
                std::lock_guard<std::mutex> lk(audio_state->mtx);
                audio_state->accumulate = !streaming;
                audio_state->accum.clear();
            }

            const auto t_generate_start = std::chrono::steady_clock::now();
            std::thread decode_thread([octx, debug_dir = session_output_dir]() {
                stream_decode(octx, debug_dir, -1);
            });

            while (true) {
                std::string frag;
                {
                    std::unique_lock<std::mutex> lk(octx->text_mtx);
                    octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{
                        return !octx->text_queue.empty() || octx->text_done_flag;
                    });

                    if (!octx->text_queue.empty()) {
                        frag = std::move(octx->text_queue.front());
                        octx->text_queue.pop_front();
                    }

                    if (octx->text_done_flag && octx->text_queue.empty()) {
                        break;
                    }
                }

                if (!frag.empty()) {
                    if (frag == "__IS_LISTEN__") {
                        // Express listen via the kind=listen delta channel
                        // (schema §5.1 / network §6.3) in both streaming and
                        // non-streaming (non-streaming deltas allowed per
                        // network §4.2). response.done.reason stays turn_end.
                        send_event(make_listen_delta(
                            session_id, response_id,
                            make_runtime_metrics(octx, prefill_ms,
                                                 elapsed_ms(t_generate_start),
                                                 elapsed_ms(t_request_start), 0,
                                                 turn_vision_slices)));
                        break;
                    } else if (frag == "__END_OF_TURN__") {
                        // handled by response.done
                    } else {
                        const std::string text = sanitize_utf8_stream(utf8_pending, frag);
                        full_text += text;
                        if (streaming && !text.empty()) {
                            send_event(make_text_delta(
                                session_id, response_id, text,
                                make_runtime_metrics(octx, prefill_ms,
                                                     elapsed_ms(t_generate_start),
                                                     elapsed_ms(t_request_start), 0,
                                                     turn_vision_slices)));
                        }
                    }
                }

                if (session_mgr.get(session_id) == nullptr) {
                    if (decode_thread.joinable()) decode_thread.join();
                    goto cleanup;
                }
            }

            if (decode_thread.joinable()) decode_thread.join();
            generate_ms = elapsed_ms(t_generate_start);
            {
                const std::string text = sanitize_utf8_stream(utf8_pending, "", true);
                full_text += text;
                if (streaming && !text.empty()) {
                    send_event(make_text_delta(
                        session_id, response_id, text,
                        make_runtime_metrics(octx, prefill_ms, generate_ms,
                                             elapsed_ms(t_request_start), 0,
                                             turn_vision_slices)));
                }
            }
            if (parsed_input.use_tts_template) {
                if (!omni_duplex_drain_tts_audio(octx, /*max_wait_ms*/120000, /*idle_ms*/3000)) {
                    LOG_WRN("WS /backend: timed out waiting for TTS audio drain for session %s\n",
                            session_id.c_str());
                }
            }
            octx->use_tts = prev_use_tts;

            // Non-streaming: hand the buffered PCM to response.done.audio and
            // stop accumulating so later turns / full-duplex stream normally.
            {
                std::lock_guard<std::mutex> lk(audio_state->mtx);
                if (audio_state->accumulate && !audio_state->accum.empty()) {
                    full_audio_b64 = float32_pcm_to_b64(
                        audio_state->accum.data(), audio_state->accum.size());
                }
                audio_state->accumulate = false;
                audio_state->accum.clear();
            }

            // Send response.done (streaming may have sent deltas already, but done is always sent)
            ProtocolMetrics done_metrics = make_runtime_metrics(
                octx, prefill_ms, generate_ms, elapsed_ms(t_request_start),
                std::max(0, octx->n_past - n_past_before_decode), turn_vision_slices);
            if (parsed_input.use_tts_template) {
                done_metrics.cost_tts_ms = std::max(
                    0.0, done_metrics.wall_clock_ms - prefill_ms - generate_ms);
            }
            // reason is a turn-completion reason (network §6.6); the listen
            // decision is conveyed via the kind=listen delta above, not here.
            send_event(make_response_done(session_id, response_id, full_text,
                                          full_audio_b64, "turn_end",
                                          done_metrics));

        } else {
            // ================================================================
            // Full-duplex input processing
            // ================================================================
            // full_duplex MUST NOT carry turn_based `messages` (schema §7.4).
            if (!parsed_input.messages.is_null()) {
                fail_fast(session_id, "mode_mismatch");
                return;
            }
            if (parsed_input.audio_b64.empty()) {
                fail_fast(session_id, "missing_audio");
                return;
            }

            TempMediaFiles tmp_files;
            int input_index = ++msg_counter;
            double prefill_ms = 0.0;
            int turn_vision_slices = parsed_input.video_frames_b64.empty() ? 0 : 1;

            // Write audio to temp WAV
            if (!parsed_input.audio_b64.empty()) {
                tmp_files.audio_path = TempMediaFiles::write_audio_wav(
                    parsed_input.audio_b64, temp_dir, input_index);
            }

            // Write first video frame to temp image
            if (!parsed_input.video_frames_b64.empty()) {
                tmp_files.image_path = TempMediaFiles::write_image_jpeg(
                    parsed_input.video_frames_b64[0], temp_dir, input_index);
            }

            // Prefill
            {
                const auto t_prefill_start = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(octx_mutex);
                if (!stream_prefill(octx, tmp_files.audio_path, tmp_files.image_path,
                                    input_index, parsed_input.max_slice_nums)) {
                    tmp_files.cleanup();
                    fail_fast(session_id, "prefill_failed");
                    return;
                }
                prefill_ms = elapsed_ms(t_prefill_start);
            }

            if (!tmp_files.audio_path.empty()) retained_media_files.push_back(tmp_files.audio_path);
            if (!tmp_files.image_path.empty()) retained_media_files.push_back(tmp_files.image_path);
            tmp_files.audio_path.clear();
            tmp_files.image_path.clear();

            // force_listen: caller forces this step to LISTEN — skip decoding
            // and immediately report a listen delta, then wait for next input.
            if (parsed_input.force_listen) {
                send_event(make_listen_delta(
                    session_id, response_id,
                    make_runtime_metrics(octx, prefill_ms, 0.0,
                                         elapsed_ms(t_request_start), 0,
                                         turn_vision_slices)));
                continue;
            }

            // Decode: start background thread, poll text_queue on this thread
            std::string debug_dir = session_output_dir;

            // Reset text streaming state
            {
                std::lock_guard<std::mutex> lk(octx->text_mtx);
                octx->text_queue.clear();
                octx->text_done_flag = false;
                octx->text_streaming = true;
            }

            const auto t_generate_start = std::chrono::steady_clock::now();
            std::thread decode_thread([octx, debug_dir]() {
                stream_decode(octx, debug_dir, -1);
            });

            // Collect full text for response.done
            std::string full_text;
            std::string utf8_pending;

            // Poll text_queue
            while (true) {
                std::string frag;
                {
                    std::unique_lock<std::mutex> lk(octx->text_mtx);
                    octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{
                        return !octx->text_queue.empty() || octx->text_done_flag;
                    });

                    if (!octx->text_queue.empty()) {
                        frag = std::move(octx->text_queue.front());
                        octx->text_queue.pop_front();
                    }

                    if (octx->text_done_flag && octx->text_queue.empty()) {
                        break;
                    }
                }

                if (!frag.empty()) {
                    if (frag == "__IS_LISTEN__") {
                        // Model switched to listen
                        send_event(make_listen_delta(
                            session_id, response_id,
                            make_runtime_metrics(octx, prefill_ms,
                                                 elapsed_ms(t_generate_start),
                                                 elapsed_ms(t_request_start), 0,
                                                 turn_vision_slices)));
                        break; // Done for this input
                    } else if (frag == "__END_OF_TURN__") {
                        // Turn ended — will be handled by response.done
                    } else {
                        // Text delta
                        const std::string text = sanitize_utf8_stream(utf8_pending, frag);
                        full_text += text;
                        if (!text.empty()) {
                            send_event(make_text_delta(
                                session_id, response_id, text,
                                make_runtime_metrics(octx, prefill_ms,
                                                     elapsed_ms(t_generate_start),
                                                     elapsed_ms(t_request_start), 0,
                                                     turn_vision_slices)));
                        }
                    }
                }

                // Check if session was closed externally
                if (session_mgr.get(session_id) == nullptr) {
                    // Session was closed externally (e.g. HTTP close endpoint)
                    // Join the decode thread and exit both loops
                    if (decode_thread.joinable()) {
                        decode_thread.join();
                    }
                    goto cleanup;
                }
            }

            if (decode_thread.joinable()) {
                decode_thread.join();
            }
            double generate_ms = elapsed_ms(t_generate_start);
            {
                const std::string text = sanitize_utf8_stream(utf8_pending, "", true);
                if (!text.empty()) {
                    full_text += text;
                    send_event(make_text_delta(
                        session_id, response_id, text,
                        make_runtime_metrics(octx, prefill_ms,
                                             elapsed_ms(t_generate_start),
                                             elapsed_ms(t_request_start), 0,
                                             turn_vision_slices)));
                }
            }
            bool emitted_audio = false;
            {
                std::lock_guard<std::mutex> lk(audio_state->mtx);
                emitted_audio = audio_state->emitted_audio;
            }
            if (!full_text.empty() || emitted_audio) {
                // Full-duplex speak responses also need an explicit completion
                // boundary; pure listen steps are represented by listen delta only.
                send_event(make_response_done(
                    session_id, response_id, full_text, /*audio_base64*/"",
                    "turn_end",
                    make_runtime_metrics(octx, prefill_ms, generate_ms,
                                         elapsed_ms(t_request_start), 0,
                                         turn_vision_slices)));
            }
        }
    }

cleanup:

    // ================================================================
    // Step 4: WS disconnect — cleanup
    // ================================================================
    LOG_INF("WS /backend: session %s disconnected\n", session_id.c_str());

    // Send session.closed (best-effort)
    std::string close_ev = make_session_closed(session_id, "client_disconnected").dump();
    ws.send(close_ev);

    // On disconnect, stop inference and recycle the shared context (via
    // omni_prepare_for_reuse) rather than freeing it — the model stays loaded
    // for the next session. Then drop the session and clean up temp media.
    {
        std::lock_guard<std::mutex> lock(octx_mutex);
        OmniSession * session = session_mgr.get(session_id);
        if (session && session->octx) {
            session->octx->break_event = true;
            {
                std::lock_guard<std::mutex> lk(session->octx->text_mtx);
                session->octx->text_queue.clear();
                session->octx->text_done_flag = true;
            }
            session->octx->text_cv.notify_all();
            omni_prepare_for_reuse(session->octx);
        }
        session_mgr.close(session_id);
    }

    for (const auto & path : retained_media_files) {
        fs::remove(path);
    }
    retained_media_files.clear();

    ws.close();
}
