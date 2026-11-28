#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <filesystem>
#include <mutex>

struct omni_context;
struct common_params;
struct llama_model;
struct llama_context;
class SessionManager;

namespace httplib {
namespace ws { class WebSocket; }
}

inline std::string omni_default_ref_audio_path() {
#ifdef OMNI_ASSET_DIR
    return (std::filesystem::path(OMNI_ASSET_DIR) /
            "default_ref_audio/default_ref_audio.wav").string();
#else
    return "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
#endif
}

inline bool should_remove_prefill_audio(const std::string & prefill_audio_path,
                                        const std::string & tts_ref_audio_path,
                                        bool tts_ref_audio_owned) {
    if (prefill_audio_path.empty()) {
        return false;
    }
    return !tts_ref_audio_owned || prefill_audio_path != tts_ref_audio_path;
}

inline bool should_materialize_tts_ref_audio(bool use_tts,
                                             const std::string & tts_ref_audio_b64) {
    return use_tts && !tts_ref_audio_b64.empty();
}

inline bool release_unactivated_octx_if_owned(
        omni_context * pending_octx,
        omni_context *& shared_octx,
        std::mutex & octx_mutex,
        bool session_activated,
        void (*release_fn)(omni_context *)) {
    if (session_activated || pending_octx == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(octx_mutex);
    if (shared_octx != pending_octx) {
        return false;
    }

    shared_octx = nullptr;
    release_fn(pending_octx);
    return true;
}

inline bool release_failed_shared_octx_if_owned(
        omni_context * failed_octx,
        omni_context *& shared_octx,
        std::mutex & octx_mutex,
        void (*release_fn)(omni_context *)) {
    if (failed_octx == nullptr) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(octx_mutex);
        if (shared_octx != failed_octx) {
            return false;
        }
        shared_octx = nullptr;
    }

    release_fn(failed_octx);
    return true;
}

// ============================================================================
// WS /backend handler — main entry point called from server.cpp
// ============================================================================

void handle_ws_backend(httplib::ws::WebSocket & ws,
                       SessionManager & session_mgr,
                       common_params & params_base,
                       llama_model * model,
                       llama_context * ctx,
                       omni_context *& shared_octx,  // server-owned, reused across sessions
                       std::mutex & octx_mutex);

// ============================================================================
// Helpers: base64 audio/JPEG → temp files
// ============================================================================

struct TempMediaFiles {
    std::string audio_path;      // WAV file path (empty if no audio)
    std::string image_path;      // PNG/JPEG file path (empty if no image)
    
    // Write base64 float32 PCM to a temp WAV file
    // Returns empty string on failure
    static std::string write_audio_wav(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Write base64 JPEG/PNG bytes to a temp file
    // Returns empty string on failure
    static std::string write_image_jpeg(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Create a temp file from raw bytes
    static std::string write_temp_file(const std::string & temp_dir, const std::string & prefix,
                                       const std::string & suffix, const void * data, size_t len);
    
    // Clean up temp files
    void cleanup();
};
