#include "runtime-profile-session.h"

#include "common.h"

namespace omni {

runtime_session_options resolve_runtime_session_options(const effective_runtime_config * config,
                                                        bool                             requested_duplex_mode,
                                                        int32_t                          requested_tts_gpu_layers,
                                                        const std::string &              requested_token2wav_device,
    int32_t                          requested_token2wav_threads) {
    if (config != nullptr) {
        return { config->duplex_mode, config->async_mode, config->tts_gpu_layers, config->token2wav_device,
                 config->token2wav_threads, true };
    }
    return { requested_duplex_mode, true, requested_tts_gpu_layers, requested_token2wav_device,
             requested_token2wav_threads, false };
}

bool apply_effective_runtime_config(common_params &                  params,
                                    const effective_runtime_config & config,
                                    std::string &                    error) {
    error.clear();
    params.model.path       = config.llm_model;
    params.n_gpu_layers     = config.n_gpu_layers;
    params.n_ctx            = config.n_ctx;
    params.vpm_model        = config.vision_model;
    params.apm_model        = config.audio_model;
    params.tts_model        = config.tts_model;
    params.projector_model  = config.projector_model;
    params.vpm_batch_encode = config.vpm_batch_encode;

    const auto separator = config.tts_model.find_last_of("/\\");
    params.tts_bin_dir   = separator == std::string::npos ? "." : config.tts_model.substr(0, separator);

    ggml_backend_dev_t llm_device = config.llm_device == "cpu" ?
                                        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                                        ggml_backend_dev_by_name(config.llm_device.c_str());
    if (llm_device == nullptr) {
        error = "resolved LLM placement device is unavailable: " + config.llm_device;
        return false;
    }
    params.devices    = { llm_device, nullptr };
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu   = 0;
    if (config.llm_device == "cpu") {
        params.n_gpu_layers = 0;
    }
    return true;
}

}  // namespace omni
