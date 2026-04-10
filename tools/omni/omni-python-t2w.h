#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct omni_context;

bool omni_start_python_t2w_service(struct omni_context * ctx_omni);
void omni_stop_python_t2w_service(struct omni_context * ctx_omni);
bool omni_send_python_t2w_command(struct omni_context * ctx_omni, const std::string & cmd_json, std::string & response);
bool omni_init_python_t2w_model(struct omni_context * ctx_omni, const std::string & device);
bool omni_set_python_t2w_ref_audio(struct omni_context * ctx_omni, const std::string & ref_audio_path);
bool omni_process_python_t2w_tokens(
    struct omni_context * ctx_omni,
    const std::vector<int32_t> & tokens,
    bool last_chunk,
    const std::string & output_path,
    double & inference_time_ms,
    double & audio_duration);
bool omni_reset_python_t2w_cache(struct omni_context * ctx_omni);
