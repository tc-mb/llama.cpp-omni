#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace omni {

struct runtime_device {
    std::string name;
    std::string description;
    bool        is_accelerator = false;
    uint64_t    free_memory    = 0;
    uint64_t    total_memory   = 0;
    std::string type;
};

struct hardware_snapshot {
    std::vector<runtime_device> devices;
};

struct model_inventory {
    std::string root;
    std::string llm_f16;
    std::string llm_q8_0;
    std::string llm_q4_k_m;
    std::string vision;
    std::string audio;
    std::string tts;
    std::string projector;
    std::string token2wav_dir;
    std::string token2wav_encoder;
    std::string token2wav_flow_matching;
    std::string token2wav_flow_extra;
    std::string token2wav_hifigan;
    std::string token2wav_prompt_cache;
};

struct module_placement {
    std::string module;
    std::string model;
    std::string precision;
    std::string device;
    std::string backend;
    std::string execution;
    std::string status;
};

struct effective_runtime_config {
    std::string              requested_profile;
    std::string              resolved_profile;
    std::string              reason;
    std::string              primary_device_name;
    std::string              primary_device_description;
    uint64_t                 primary_device_free_memory  = 0;
    uint64_t                 primary_device_total_memory = 0;
    std::string              llm_model;
    std::string              llm_quantization;
    std::string              llm_device;
    std::string              vision_model;
    std::string              vision_device;
    std::string              audio_model;
    std::string              audio_device;
    std::string              tts_model;
    std::string              tts_device;
    std::string              projector_model;
    std::string              token2wav_model_dir;
    int32_t                  n_gpu_layers     = 0;
    int32_t                  tts_gpu_layers   = 0;
    std::string              token2wav_device = "cpu";
    int32_t                  token2wav_threads = 8;
    int32_t                  n_ctx            = 0;
    bool                     duplex_mode      = false;
    bool                     async_mode       = false;
    bool                     vpm_batch_encode = false;
    std::vector<module_placement> placements;
};

struct runtime_profile_overrides {
    std::optional<std::string> llm_model;
    std::optional<std::string> llm_device;
    std::optional<std::string> vision_device;
    std::optional<std::string> audio_device;
    std::optional<std::string> tts_device;
    std::optional<std::string> projector_device;
    std::optional<std::string> token2wav_device;
    std::optional<int32_t>     token2wav_threads;
    std::optional<int32_t>     n_gpu_layers;
    std::optional<int32_t>     n_ctx;
    std::optional<bool>        vpm_batch_encode;
};

struct runtime_profile_result {
    bool                     ok = false;
    effective_runtime_config config;
    std::string              error;
};

model_inventory   discover_model_inventory(const std::string & model_root);
hardware_snapshot detect_hardware_snapshot();

runtime_profile_result resolve_runtime_profile(const std::string &               requested_profile,
                                               const hardware_snapshot &         hardware,
                                               const model_inventory &           inventory,
                                               const runtime_profile_overrides & overrides = {});

runtime_profile_result resolve_runtime_profile_from_config(const std::string &               config_path,
                                                           const hardware_snapshot &         hardware,
                                                           const model_inventory &           inventory,
                                                           const runtime_profile_overrides & overrides = {});

std::string format_effective_runtime_config(const effective_runtime_config & config);

}  // namespace omni
