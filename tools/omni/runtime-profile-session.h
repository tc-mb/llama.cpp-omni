#pragma once

#include "runtime-profile.h"

#include <cstdint>
#include <string>

struct common_params;

namespace omni {

struct runtime_session_options {
    bool        duplex_mode          = false;
    bool        async_mode           = true;
    int32_t     tts_gpu_layers       = 100;
    std::string token2wav_device     = "gpu:0";
    int32_t     token2wav_threads    = 8;
    bool        strict_runtime_config = false;
};

runtime_session_options resolve_runtime_session_options(const effective_runtime_config * config,
                                                        bool                             requested_duplex_mode,
                                                        int32_t                          requested_tts_gpu_layers,
                                                        const std::string &              requested_token2wav_device,
                                                        int32_t                          requested_token2wav_threads = 8);

bool apply_effective_runtime_config(common_params &                  params,
                                    const effective_runtime_config & config,
                                    std::string &                    error);

}  // namespace omni
