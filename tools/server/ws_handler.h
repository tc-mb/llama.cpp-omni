#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <mutex>

struct omni_context;
struct common_params;
struct llama_model;
struct llama_context;
class SessionManager;

namespace omni {
struct effective_runtime_config;
}

namespace httplib {
namespace ws { class WebSocket; }
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
                       std::mutex & octx_mutex,
                       const omni::effective_runtime_config * runtime_config = nullptr);

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
