#include "protocol.h"
#include "common/base64.hpp"

#include <algorithm>
#include <chrono>

// ============================================================================
// ProtocolMetrics
// ============================================================================

json ProtocolMetrics::to_json() const {
    json m;
    m["backend"] = backend;
    if (kv_cache_length > 0) {
        m["kv_cache_length"] = kv_cache_length;
    }
    if (prefill_ms > 0.0) {
        m["prefill_ms"] = prefill_ms;
    }
    if (generate_ms > 0.0) {
        m["generate_ms"] = generate_ms;
    }
    if (wall_clock_ms > 0.0) {
        m["wall_clock_ms"] = wall_clock_ms;
    }
    if (cost_llm_ms > 0.0) {
        m["cost_llm_ms"] = cost_llm_ms;
    }
    if (cost_tts_prep_ms > 0.0) {
        m["cost_tts_prep_ms"] = cost_tts_prep_ms;
    }
    if (cost_tts_ms > 0.0) {
        m["cost_tts_ms"] = cost_tts_ms;
    }
    if (cost_token2wav_ms > 0.0) {
        m["cost_token2wav_ms"] = cost_token2wav_ms;
    }
    if (n_tokens > 0) {
        m["n_tokens"] = n_tokens;
    }
    if (n_tts_tokens > 0) {
        m["n_tts_tokens"] = n_tts_tokens;
    }
    if (vision_slices > 0) {
        m["vision_slices"] = vision_slices;
    }
    if (vision_tokens > 0) {
        m["vision_tokens"] = vision_tokens;
    }
    return m;
}

// ============================================================================
// Downstream event builders
// ============================================================================

static double server_timestamp() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

json make_session_created(const std::string & session_id,
                           const std::string & mode,
                           const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "session.created";
    ev["session_id"] = session_id;
    ev["mode"] = mode;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_text_delta(const std::string & session_id,
                      const std::string & response_id,
                      const std::string & text,
                      const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "text";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["text"] = text;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_audio_delta(const std::string & session_id,
                       const std::string & response_id,
                       const std::string & audio_base64,
                       const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "audio";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["audio"] = audio_base64;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_listen_delta(const std::string & session_id,
                        const std::string & response_id,
                        const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "listen";
    ev["session_id"] = session_id;
    if (!response_id.empty()) {
        ev["response_id"] = response_id;
    }
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_response_done(const std::string & session_id,
                         const std::string & response_id,
                         const std::string & full_text,
                         const std::string & audio_base64,
                         const std::string & reason,
                         const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.done";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["text"] = full_text;
    ev["reason"] = reason;
    ev["server_send_ts"] = server_timestamp();

    if (!audio_base64.empty()) {
        ev["audio"] = audio_base64;
    } else {
        ev["audio"] = nullptr;
    }

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_session_closed(const std::string & session_id,
                          const std::string & reason,
                          const std::string & diagnostic_message) {
    json ev;
    ev["type"] = "session.closed";
    ev["session_id"] = session_id;
    ev["reason"] = reason;
    ev["server_send_ts"] = server_timestamp();

    if (!diagnostic_message.empty()) {
        json diag;
        diag["message"] = diagnostic_message;
        ev["diagnostic"] = diag;
    }
    return ev;
}

// ============================================================================
// Upstream message parsers
// ============================================================================

static std::string json_str(const json & j, const std::string & key,
                             const std::string & default_val = "") {
    if (j.contains(key) && j.at(key).is_string()) {
        return j.at(key).get<std::string>();
    }
    return default_val;
}

static bool json_bool(const json & j, const std::string & key, bool default_val = false) {
    if (j.contains(key) && j.at(key).is_boolean()) {
        return j.at(key).get<bool>();
    }
    return default_val;
}

static int json_int(const json & j, const std::string & key, int default_val = 0) {
    if (j.contains(key) && j.at(key).is_number()) {
        return j.at(key).get<int>();
    }
    return default_val;
}

static float json_float(const json & j, const std::string & key, float default_val = 0.0f) {
    if (j.contains(key) && j.at(key).is_number()) {
        return j.at(key).get<float>();
    }
    return default_val;
}

// Return the first non-empty string among several accepted key aliases.
static std::string json_str_any(const json & j, std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        if (j.contains(key) && j.at(key).is_string()) {
            std::string value = j.at(key).get<std::string>();
            if (!value.empty()) {
                return value;
            }
        }
    }
    return "";
}

// Drop a leading "data:...;base64," prefix if present, returning raw base64.
static std::string strip_data_url_prefix(const std::string & value) {
    auto comma = value.find(',');
    if (comma != std::string::npos && value.substr(0, comma).find("base64") != std::string::npos) {
        return value.substr(comma + 1);
    }
    return value;
}

// Accept audio/image as either a bare base64 string or an object carrying it
// under one of several common key names; always return clean base64.
static std::string extract_audio_b64(const json & value) {
    if (value.is_string()) {
        return strip_data_url_prefix(value.get<std::string>());
    }
    if (value.is_object()) {
        return strip_data_url_prefix(json_str_any(value, {"data", "base64", "audio_base64", "audio_data"}));
    }
    return "";
}

static std::string extract_image_b64(const json & value) {
    if (value.is_string()) {
        return strip_data_url_prefix(value.get<std::string>());
    }
    if (value.is_object()) {
        return strip_data_url_prefix(json_str_any(value, {"data", "base64", "image_base64"}));
    }
    return "";
}

ParsedSessionInit parse_session_init(const json & msg) {
    ParsedSessionInit out;

    // Validate type
    if (!msg.contains("type") || msg.at("type") != "session.init") {
        out.error = "expected type=session.init";
        return out;
    }

    if (!msg.contains("payload") || !msg.at("payload").is_object()) {
        out.error = "missing payload";
        return out;
    }

    const json & p = msg.at("payload");

    // mode
    std::string mode = json_str(p, "mode", "full_duplex");
    if (mode != "full_duplex" && mode != "turn_based") {
        out.error = "invalid mode: " + mode + " (expected full_duplex or turn_based)";
        return out;
    }
    out.mode = mode;

    // use_tts — session-level TTS toggle (same name as server init /omni_init)
    out.use_tts = json_bool(p, "use_tts", true);

    // voice (reference audio)
    if (p.contains("voice") && p.at("voice").is_object()) {
        const json & v = p.at("voice");
        out.ref_audio_b64 = json_str_any(v, {"ref_audio", "ref_audio_base64"});
        out.tts_ref_audio_b64 = json_str_any(v, {"tts_ref_audio", "tts_ref_audio_base64"});
        if (out.tts_ref_audio_b64.empty() && !out.ref_audio_b64.empty()) {
            out.tts_ref_audio_b64 = out.ref_audio_b64;
        }
    }

    // system_prompt
    out.system_prompt = json_str(p, "system_prompt");

    // config (opaque pass-through)
    if (p.contains("config") && p.at("config").is_object()) {
        out.config = p.at("config");
    }

    out.ok = true;
    return out;
}

ParsedInput parse_input_append(const json & msg) {
    ParsedInput out;

    if (!msg.contains("type") || msg.at("type") != "input.append") {
        out.error = "expected type=input.append";
        return out;
    }

    if (!msg.contains("input") || !msg.at("input").is_object()) {
        out.error = "missing input";
        return out;
    }

    const json & in = msg.at("input");

    // Full-duplex audio aliases: audio, audio.data/base64, audio_base64, audio_data.
    out.audio_b64 = json_str_any(in, {"audio_base64", "audio_data"});
    if (out.audio_b64.empty() && in.contains("audio")) {
        out.audio_b64 = extract_audio_b64(in.at("audio"));
    }
    out.audio_b64 = strip_data_url_prefix(out.audio_b64);

    auto append_frames = [&](const json & frames) {
        if (!frames.is_array()) {
            return;
        }
        for (const auto & f : frames) {
            std::string frame = extract_image_b64(f);
            if (!frame.empty()) {
                out.video_frames_b64.push_back(frame);
            }
        }
    };

    if (in.contains("video_frames")) {
        append_frames(in.at("video_frames"));
    }
    if (in.contains("frame_base64_list")) {
        append_frames(in.at("frame_base64_list"));
    }
    if (in.contains("frames")) {
        append_frames(in.at("frames"));
    }

    // max_slice_nums: int applies to every frame; int[] must match frame count
    // (schema §4.3). Falls back to hints.max_slice_nums when absent.
    if (in.contains("max_slice_nums")) {
        const json & msn = in.at("max_slice_nums");
        if (msn.is_number()) {
            out.max_slice_nums = msn.get<int>();
        } else if (msn.is_array()) {
            if (!out.video_frames_b64.empty() && msn.size() != out.video_frames_b64.size()) {
                out.error = "max_slice_nums array length must match video frame count";
                return out;
            }
            if (!msn.empty() && msn.at(0).is_number()) {
                out.max_slice_nums = msn.at(0).get<int>();
            }
        } else if (!msn.is_null()) {
            out.error = "max_slice_nums must be int or int[]";
            return out;
        }
    } else if (in.contains("hints") && in.at("hints").is_object()) {
        out.max_slice_nums = json_int(in.at("hints"), "max_slice_nums", -1);
    }

    out.force_listen = json_bool(in, "force_listen", false);
    if (in.contains("hints") && in.at("hints").is_object()) {
        out.force_listen = json_bool(in.at("hints"), "force_listen", out.force_listen);
    }

    // Turn-based fields: messages (required), streaming (required), generation
    if (in.contains("messages") && in.at("messages").is_array()) {
        out.messages = in.at("messages");
    }

    out.streaming = json_bool(in, "streaming", true);

    if (in.contains("generation") && in.at("generation").is_object()) {
        const json & gen = in.at("generation");
        out.max_new_tokens = json_int(gen, "max_new_tokens", 512);
        out.length_penalty = json_float(gen, "length_penalty", 1.1f);
    }

    if (in.contains("image") && in.at("image").is_object()) {
        const json & img = in.at("image");
        out.max_slice_nums = json_int(img, "max_slice_nums", out.max_slice_nums);
    }

    if (in.contains("tts") && in.at("tts").is_object()) {
        const json & tts = in.at("tts");
        out.tts_enabled = json_bool(tts, "enabled", false);
        out.tts_ref_audio_b64 = json_str(tts, "ref_audio_data");
    }

    out.omni_mode = json_bool(in, "omni_mode", false);
    // TTS is on if either use_tts_template or tts.enabled is set (schema §4.2).
    out.use_tts_template = json_bool(in, "use_tts_template", false) || out.tts_enabled;
    out.enable_thinking = json_bool(in, "enable_thinking", false);

    out.ok = true;
    return out;
}

// ============================================================================
// Structured message parsing (turn_based mode)
// ============================================================================

ParsedMessage parse_one_message(const json & msg) {
    ParsedMessage out;

    out.role = json_str(msg, "role", "user");

    if (msg.contains("content")) {
        const json & content = msg.at("content");

        if (content.is_string()) {
            out.text = content.get<std::string>();
        } else if (content.is_array()) {
            std::string text_buf;
            for (const auto & part : content) {
                if (!part.is_object()) continue;

                std::string ptype = json_str(part, "type");

                if (ptype == "text") {
                    std::string t = json_str(part, "text");
                    if (!t.empty()) {
                        if (!text_buf.empty()) text_buf += "\n";
                        text_buf += t;
                    }
                } else if (ptype == "image") {
                    std::string data = extract_image_b64(part);
                    if (!data.empty()) {
                        out.image_b64s.push_back(data);
                    }
                } else if (ptype == "image_url") {
                    if (part.contains("image_url") && part.at("image_url").is_object()) {
                        const json & iu = part.at("image_url");
                        std::string url = json_str(iu, "url");
                        if (!url.empty()) {
                            std::string payload = strip_data_url_prefix(url);
                            if (!payload.empty()) {
                                out.image_b64s.push_back(payload);
                            }
                        }
                    }
                } else if (ptype == "audio") {
                    std::string audio = extract_audio_b64(part);
                    if (!audio.empty()) {
                        out.audio_b64s.push_back(audio);
                    }
                } else if (ptype == "video") {
                    // base64 MP4 container (schema §4.4); decoded into frames +
                    // audio later. stack_frames = frames stacked per sample point.
                    std::string video = json_str_any(part, {"data", "base64"});
                    if (!video.empty()) {
                        out.video_b64s.push_back(strip_data_url_prefix(video));
                        int stack_frames = json_int(part, "stack_frames", 1);
                        out.video_stack_frames.push_back(std::max(1, stack_frames));
                    }
                }
            }
            out.text = text_buf;
        }
    }

    return out;
}

std::vector<ParsedMessage> parse_messages_array(const json & messages) {
    std::vector<ParsedMessage> out;
    if (!messages.is_array()) return out;
    for (const auto & m : messages) {
        out.push_back(parse_one_message(m));
    }
    return out;
}

std::string build_prompt_from_messages(const std::vector<ParsedMessage> & msgs) {
    std::string prompt;
    bool wrote_any = false;
    for (const auto & m : msgs) {
        if (m.role == "system") {
            if (!m.text.empty()) {
                if (wrote_any) prompt += "\n";
                prompt += m.text;
                wrote_any = true;
            }
        } else if (m.role == "user") {
            if (wrote_any) prompt += "<|im_end|>\n";
            prompt += "<|im_start|>user\n" + m.text;
            wrote_any = true;
        } else if (m.role == "assistant") {
            if (wrote_any) prompt += "<|im_end|>\n";
            prompt += "<|im_start|>assistant\n" + m.text;
            wrote_any = true;
        }
    }
    if (wrote_any) prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt;
}

// ============================================================================
// Helpers: base64 ↔ raw bytes
// ============================================================================

std::vector<uint8_t> b64_decode(const std::string & b64) {
    std::string raw = base64::decode(b64);
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

std::vector<float> b64_to_float32_pcm(const std::string & b64) {
    std::string raw = base64::decode(b64);
    const float * ptr = reinterpret_cast<const float *>(raw.data());
    size_t n = raw.size() / sizeof(float);
    return std::vector<float>(ptr, ptr + n);
}

std::string float32_pcm_to_b64(const float * samples, size_t n_samples) {
    const char * ptr = reinterpret_cast<const char *>(samples);
    size_t byte_size = n_samples * sizeof(float);
    return base64::encode(ptr, byte_size);
}
