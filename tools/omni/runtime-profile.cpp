#include "runtime-profile.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace omni {

static std::string existing_file(const std::filesystem::path & root, const std::filesystem::path & relative_path) {
    const auto      path = root / relative_path;
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error)) {
        return path.string();
    }
    return {};
}

model_inventory discover_model_inventory(const std::string & model_root) {
    model_inventory inventory;
    inventory.root = model_root;
    const std::filesystem::path root(model_root);

    inventory.llm_f16    = existing_file(root, "MiniCPM-o-4_5-F16.gguf");
    inventory.llm_q8_0   = existing_file(root, "MiniCPM-o-4_5-Q8_0.gguf");
    inventory.llm_q4_k_m = existing_file(root, "MiniCPM-o-4_5-Q4_K_M.gguf");
    inventory.vision     = existing_file(root, "vision/MiniCPM-o-4_5-vision-F16.gguf");
    inventory.audio      = existing_file(root, "audio/MiniCPM-o-4_5-audio-F16.gguf");
    inventory.tts        = existing_file(root, "tts/MiniCPM-o-4_5-tts-F16.gguf");
    inventory.projector  = existing_file(root, "tts/MiniCPM-o-4_5-projector-F16.gguf");

    const auto      token2wav_dir = root / "token2wav-gguf";
    std::error_code error;
    if (std::filesystem::is_directory(token2wav_dir, error)) {
        inventory.token2wav_dir = token2wav_dir.string();
    }
    inventory.token2wav_encoder       = existing_file(root, "token2wav-gguf/encoder.gguf");
    inventory.token2wav_flow_matching = existing_file(root, "token2wav-gguf/flow_matching.gguf");
    inventory.token2wav_flow_extra    = existing_file(root, "token2wav-gguf/flow_extra.gguf");
    inventory.token2wav_hifigan       = existing_file(root, "token2wav-gguf/hifigan2.gguf");
    inventory.token2wav_prompt_cache  = existing_file(root, "token2wav-gguf/prompt_cache.gguf");
    return inventory;
}

hardware_snapshot detect_hardware_snapshot() {
    hardware_snapshot snapshot;
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }

        size_t free_memory  = 0;
        size_t total_memory = 0;
        ggml_backend_dev_memory(device, &free_memory, &total_memory);
        const auto device_type = ggml_backend_dev_type(device);
        const char * type_name = device_type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" :
                                 device_type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" : "ACCEL";
        snapshot.devices.push_back({
            ggml_backend_dev_name(device),
            ggml_backend_dev_description(device),
            device_type != GGML_BACKEND_DEVICE_TYPE_CPU,
            static_cast<uint64_t>(free_memory),
            static_cast<uint64_t>(total_memory),
            type_name,
        });
    }
    return snapshot;
}

static std::vector<const runtime_device *> accelerator_devices(const hardware_snapshot & hardware) {
    std::vector<const runtime_device *> devices;
    for (const auto & device : hardware.devices) {
        if (device.is_accelerator) {
            devices.push_back(&device);
        }
    }
    std::sort(devices.begin(), devices.end(), [](const runtime_device * lhs, const runtime_device * rhs) {
        if (lhs->free_memory != rhs->free_memory) {
            return lhs->free_memory > rhs->free_memory;
        }
        return lhs->name < rhs->name;
    });
    return devices;
}

static std::string token2wav_device_for(const runtime_device & device) {
    std::string digits;
    for (auto it = device.name.rbegin(); it != device.name.rend() && std::isdigit(static_cast<unsigned char>(*it)); ++it) {
        digits.push_back(*it);
    }
    if (!digits.empty()) {
        std::reverse(digits.begin(), digits.end());
        return "gpu:" + digits;
    }
    return "gpu:0";
}

static const runtime_device * find_device(const hardware_snapshot & hardware, const std::string & name) {
    for (const auto & device : hardware.devices) {
        if (device.name == name) {
            return &device;
        }
    }
    return nullptr;
}

static std::string normalize_device(const std::string & requested,
                                    const hardware_snapshot & hardware,
                                    const std::string & module,
                                    std::string & error) {
    const auto accelerators = accelerator_devices(hardware);
    const runtime_device * primary = accelerators.empty() ? nullptr : accelerators.front();
    if (requested.empty() || requested == "cpu") {
        return requested.empty() ? "cpu" : requested;
    }
    if (requested == "primary" || requested == "secondary") {
        const size_t index = requested == "primary" ? 0 : 1;
        if (index < accelerators.size()) {
            return accelerators[index]->name;
        }
        error = "unsupported " + module + " placement device: " + requested +
                " (the requested accelerator is not available)";
        return {};
    }
    if (requested == "gpu") {
        if (primary != nullptr) {
            return primary->name;
        }
        error = "unsupported " + module + " placement device: gpu (no accelerator is available)";
        return {};
    }
    if (requested.rfind("gpu:", 0) == 0) {
        try {
            const size_t index = static_cast<size_t>(std::stoul(requested.substr(4)));
            const auto accelerators = accelerator_devices(hardware);
            if (index < accelerators.size()) {
                return accelerators[index]->name;
            }
        } catch (...) {
        }
    }
    if (find_device(hardware, requested) != nullptr) {
        return requested;
    }
    error = "unsupported " + module + " placement device: " + requested;
    return {};
}

static void add_module_placement(effective_runtime_config & config,
                                 const std::string & module,
                                 const std::string & model,
                                 const std::string & precision,
                                 const std::string & device,
                                 const std::string & execution = "independent") {
    config.placements.push_back({ module, model, precision, device,
                                  device == "cpu" ? "cpu" : "gpu", execution, "active" });
}

static std::string quantization_from_path(const std::string & path) {
    std::string upper = path;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (upper.find("Q4_K_M") != std::string::npos) {
        return "Q4_K_M";
    }
    if (upper.find("Q8_0") != std::string::npos) {
        return "Q8_0";
    }
    if (upper.find("F16") != std::string::npos) {
        return "F16";
    }
    return "UNKNOWN";
}

static void apply_overrides(effective_runtime_config & config, const runtime_profile_overrides & overrides) {
    if (overrides.llm_model) {
        config.llm_model        = *overrides.llm_model;
        config.llm_quantization = quantization_from_path(*overrides.llm_model);
        config.reason += "; explicit LLM model override applied";
    }
    if (overrides.n_gpu_layers) {
        config.n_gpu_layers = *overrides.n_gpu_layers;
    }
    if (overrides.n_ctx) {
        config.n_ctx = *overrides.n_ctx;
    }
    if (overrides.token2wav_threads) {
        config.token2wav_threads = *overrides.token2wav_threads;
    }
    if (overrides.vpm_batch_encode) {
        config.vpm_batch_encode = *overrides.vpm_batch_encode;
    }
    for (auto & placement : config.placements) {
        if (placement.module == "llm") {
            placement.model = config.llm_model;
            placement.precision = config.llm_quantization;
            placement.device = config.llm_device;
            break;
        }
    }
}

runtime_profile_result resolve_runtime_profile(const std::string &               requested_profile,
                                               const hardware_snapshot &         hardware,
                                               const model_inventory &           inventory,
                                               const runtime_profile_overrides & overrides) {
    runtime_profile_result result;
    (void) hardware;
    (void) inventory;
    (void) overrides;
    result.config.requested_profile = requested_profile;
    result.error = requested_profile == "auto"
                       ? "profile auto requires a static profile config file; use --profile-config or the default "
                         "MODEL_DIR/omni-runtime-profile.json"
                       : "unsupported runtime profile: " + requested_profile;
    return result;
}

using profile_json = nlohmann::ordered_json;

template <typename T>
static bool read_required_profile_value(const profile_json & object,
                                        const char *         section,
                                        const char *         key,
                                        T &                  value,
                                        std::string &        error) {
    if (!object.is_object() || !object.contains(key)) {
        error = std::string("profile config missing required field: ") + section + "." + key;
        return false;
    }
    try {
        value = object.at(key).get<T>();
    } catch (const std::exception & exception) {
        error = std::string("profile config field ") + section + "." + key + " has an invalid type: " +
                exception.what();
        return false;
    }
    return true;
}

static bool read_required_profile_object(const profile_json & root,
                                         const char *         key,
                                         profile_json &       value,
                                         std::string &        error) {
    if (!root.contains(key) || !root.at(key).is_object()) {
        error = std::string("profile config missing required object: ") + key;
        return false;
    }
    value = root.at(key);
    return true;
}

static std::string resolve_profile_model_path(const std::string & model_root, const std::string & configured_path) {
    std::filesystem::path path(configured_path);
    if (path.is_relative()) {
        path = std::filesystem::path(model_root) / path;
    }
    return path.lexically_normal().string();
}

static bool require_profile_file(const std::string & module,
                                 const std::string & path,
                                 std::string &       error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(path, filesystem_error)) {
        error = "profile config " + module + " model file does not exist: " + path;
        return false;
    }
    return true;
}

static bool require_profile_directory(const std::string & module,
                                      const std::string & path,
                                      std::string &       error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(path, filesystem_error)) {
        error = "profile config " + module + " model directory does not exist: " + path;
        return false;
    }
    return true;
}

static bool require_non_empty_profile_value(const char * section,
                                            const char * key,
                                            const std::string & value,
                                            std::string & error) {
    if (value.empty()) {
        error = std::string("profile config field ") + section + "." + key + " must not be empty";
        return false;
    }
    return true;
}

runtime_profile_result resolve_runtime_profile_from_config(const std::string &               config_path,
                                                           const hardware_snapshot &         hardware,
                                                           const model_inventory &           inventory,
                                                           const runtime_profile_overrides & overrides) {
    runtime_profile_result result;
    result.config.requested_profile = "auto";

    if (config_path.empty()) {
        result.error = "profile config file path is empty";
        return result;
    }
    if (inventory.root.empty()) {
        result.error = "profile config requires a model root; pass --model-dir or --model";
        return result;
    }

    std::ifstream file(config_path);
    if (!file.good()) {
        result.error = "profile config file does not exist or cannot be read: " + config_path;
        return result;
    }

    profile_json document;
    try {
        file >> document;
    } catch (const std::exception & exception) {
        result.error = "invalid profile config JSON in " + config_path + ": " + exception.what();
        return result;
    }
    if (!document.is_object()) {
        result.error = "invalid profile config JSON: root must be an object";
        return result;
    }

    int schema_version = 0;
    std::string profile;
    if (!read_required_profile_value(document, "root", "schema_version", schema_version, result.error) ||
        !read_required_profile_value(document, "root", "profile", profile, result.error)) {
        return result;
    }
    if (schema_version != 1) {
        result.error = "unsupported profile config schema_version: " + std::to_string(schema_version);
        return result;
    }
    if (profile != "auto") {
        result.error = "profile config profile must be \"auto\"";
        return result;
    }

    profile_json llm;
    profile_json vision;
    profile_json audio;
    profile_json tts;
    profile_json projector;
    profile_json token2wav;
    profile_json runtime;
    if (!read_required_profile_object(document, "llm", llm, result.error) ||
        !read_required_profile_object(document, "vision", vision, result.error) ||
        !read_required_profile_object(document, "audio", audio, result.error) ||
        !read_required_profile_object(document, "tts", tts, result.error) ||
        !read_required_profile_object(document, "projector", projector, result.error) ||
        !read_required_profile_object(document, "token2wav", token2wav, result.error) ||
        !read_required_profile_object(document, "runtime", runtime, result.error)) {
        return result;
    }

    std::string llm_configured_model;
    std::string llm_quantization;
    std::string llm_configured_device;
    int32_t     n_gpu_layers = 0;
    if (!read_required_profile_value(llm, "llm", "model", llm_configured_model, result.error) ||
        !read_required_profile_value(llm, "llm", "quantization", llm_quantization, result.error) ||
        !read_required_profile_value(llm, "llm", "device", llm_configured_device, result.error) ||
        !read_required_profile_value(llm, "llm", "n_gpu_layers", n_gpu_layers, result.error)) {
        return result;
    }

    std::string vision_configured_model;
    std::string vision_configured_device;
    std::string audio_configured_model;
    std::string audio_configured_device;
    std::string tts_configured_model;
    std::string tts_configured_device;
    std::string projector_configured_model;
    std::string projector_configured_device;
    int32_t     tts_gpu_layers = 0;
    if (!read_required_profile_value(vision, "vision", "model", vision_configured_model, result.error) ||
        !read_required_profile_value(vision, "vision", "device", vision_configured_device, result.error) ||
        !read_required_profile_value(audio, "audio", "model", audio_configured_model, result.error) ||
        !read_required_profile_value(audio, "audio", "device", audio_configured_device, result.error) ||
        !read_required_profile_value(tts, "tts", "model", tts_configured_model, result.error) ||
        !read_required_profile_value(tts, "tts", "device", tts_configured_device, result.error) ||
        !read_required_profile_value(tts, "tts", "gpu_layers", tts_gpu_layers, result.error) ||
        !read_required_profile_value(projector, "projector", "model", projector_configured_model, result.error) ||
        !read_required_profile_value(projector, "projector", "device", projector_configured_device, result.error)) {
        return result;
    }

    std::string token2wav_configured_dir;
    std::string token2wav_configured_device;
    int32_t     token2wav_threads = 0;
    if (!read_required_profile_value(token2wav, "token2wav", "model_dir", token2wav_configured_dir, result.error) ||
        !read_required_profile_value(token2wav, "token2wav", "device", token2wav_configured_device, result.error) ||
        !read_required_profile_value(token2wav, "token2wav", "threads", token2wav_threads, result.error)) {
        return result;
    }

    if (!require_non_empty_profile_value("llm", "model", llm_configured_model, result.error) ||
        !require_non_empty_profile_value("llm", "quantization", llm_quantization, result.error) ||
        !require_non_empty_profile_value("llm", "device", llm_configured_device, result.error) ||
        !require_non_empty_profile_value("vision", "model", vision_configured_model, result.error) ||
        !require_non_empty_profile_value("vision", "device", vision_configured_device, result.error) ||
        !require_non_empty_profile_value("audio", "model", audio_configured_model, result.error) ||
        !require_non_empty_profile_value("audio", "device", audio_configured_device, result.error) ||
        !require_non_empty_profile_value("tts", "model", tts_configured_model, result.error) ||
        !require_non_empty_profile_value("tts", "device", tts_configured_device, result.error) ||
        !require_non_empty_profile_value("projector", "model", projector_configured_model, result.error) ||
        !require_non_empty_profile_value("projector", "device", projector_configured_device, result.error) ||
        !require_non_empty_profile_value("token2wav", "model_dir", token2wav_configured_dir, result.error) ||
        !require_non_empty_profile_value("token2wav", "device", token2wav_configured_device, result.error)) {
        return result;
    }

    if (!read_required_profile_value(runtime, "runtime", "n_ctx", result.config.n_ctx, result.error) ||
        !read_required_profile_value(runtime, "runtime", "duplex", result.config.duplex_mode, result.error) ||
        !read_required_profile_value(runtime, "runtime", "async", result.config.async_mode, result.error) ||
        !read_required_profile_value(runtime, "runtime", "vpm_batch_encode", result.config.vpm_batch_encode,
                                     result.error)) {
        return result;
    }
    if (token2wav_threads <= 0 || result.config.n_ctx <= 0) {
        result.error = "profile config runtime values must be positive";
        return result;
    }

    result.config.llm_model        = resolve_profile_model_path(inventory.root, llm_configured_model);
    result.config.llm_quantization = llm_quantization;
    result.config.vision_model     = resolve_profile_model_path(inventory.root, vision_configured_model);
    result.config.audio_model      = resolve_profile_model_path(inventory.root, audio_configured_model);
    result.config.tts_model        = resolve_profile_model_path(inventory.root, tts_configured_model);
    result.config.projector_model  = resolve_profile_model_path(inventory.root, projector_configured_model);
    result.config.token2wav_model_dir = resolve_profile_model_path(inventory.root, token2wav_configured_dir);
    result.config.n_gpu_layers     = n_gpu_layers;
    result.config.tts_gpu_layers   = tts_gpu_layers;
    result.config.token2wav_threads = token2wav_threads;

    if (!require_profile_file("llm", result.config.llm_model, result.error) ||
        !require_profile_file("vision", result.config.vision_model, result.error) ||
        !require_profile_file("audio", result.config.audio_model, result.error) ||
        !require_profile_file("tts", result.config.tts_model, result.error) ||
        !require_profile_file("projector", result.config.projector_model, result.error) ||
        !require_profile_directory("token2wav", result.config.token2wav_model_dir, result.error)) {
        return result;
    }

    const std::vector<std::pair<const char *, const char *>> token2wav_files = {
        { "encoder", "encoder.gguf" },
        { "flow_matching", "flow_matching.gguf" },
        { "flow_extra", "flow_extra.gguf" },
        { "hifigan2", "hifigan2.gguf" },
        { "prompt_cache", "prompt_cache.gguf" },
    };
    for (const auto & file_entry : token2wav_files) {
        if (!require_profile_file("token2wav " + std::string(file_entry.first),
                                  (std::filesystem::path(result.config.token2wav_model_dir) / file_entry.second).string(),
                                  result.error)) {
            return result;
        }
    }

    const auto accelerators = accelerator_devices(hardware);
    if (!accelerators.empty()) {
        const auto * primary = accelerators.front();
        result.config.primary_device_name         = primary->name;
        result.config.primary_device_description  = primary->description;
        result.config.primary_device_free_memory  = primary->free_memory;
        result.config.primary_device_total_memory = primary->total_memory;
    }

    auto resolve_configured_device = [&](const std::string & requested,
                                         const std::string & module,
                                         std::string &       resolved) {
        resolved = normalize_device(requested, hardware, module, result.error);
        return result.error.empty();
    };
    std::string llm_device;
    std::string vision_device;
    std::string audio_device;
    std::string tts_device;
    std::string projector_device;
    std::string token2wav_backend_device;
    if (!resolve_configured_device(llm_configured_device, "llm", llm_device) ||
        !resolve_configured_device(vision_configured_device, "vision", vision_device) ||
        !resolve_configured_device(audio_configured_device, "audio", audio_device) ||
        !resolve_configured_device(tts_configured_device, "tts", tts_device) ||
        !resolve_configured_device(projector_configured_device, "projector", projector_device) ||
        !resolve_configured_device(token2wav_configured_device, "token2wav", token2wav_backend_device)) {
        return result;
    }

    result.config.llm_device     = llm_device;
    result.config.vision_device  = vision_device;
    result.config.audio_device   = audio_device;
    result.config.tts_device     = tts_device;
    result.config.token2wav_device = token2wav_backend_device == "cpu"
                                         ? "cpu"
                                         : token2wav_device_for(*find_device(hardware, token2wav_backend_device));
    result.config.resolved_profile = "static_config";
    result.config.reason = "loaded static profile config: " + config_path;

    result.config.placements.clear();
    add_module_placement(result.config, "llm", result.config.llm_model, result.config.llm_quantization,
                         result.config.llm_device);
    add_module_placement(result.config, "vision", result.config.vision_model, "F16", result.config.vision_device);
    add_module_placement(result.config, "audio", result.config.audio_model, "F16", result.config.audio_device);
    add_module_placement(result.config, "tts", result.config.tts_model, "F16", result.config.tts_device);
    add_module_placement(result.config, "projector", result.config.projector_model, "F16",
                         projector_device);
    add_module_placement(result.config, "token2wav", result.config.token2wav_model_dir, "F16",
                         token2wav_backend_device);

    apply_overrides(result.config, overrides);
    result.ok = true;
    return result;
}

std::string format_effective_runtime_config(const effective_runtime_config & config) {
    std::ostringstream output;
    output << "profile=" << config.requested_profile << '\n';
    output << "resolved_profile=" << config.resolved_profile << '\n';
    output << "reason=" << config.reason << '\n';
    if (!config.primary_device_name.empty()) {
        output << "primary_device=" << config.primary_device_name;
        if (!config.primary_device_description.empty()) {
            output << " (" << config.primary_device_description << ')';
        }
        output << '\n';
        output << "primary_device_free_memory_mib=" << config.primary_device_free_memory / (1024ULL * 1024ULL) << '\n';
        output << "primary_device_total_memory_mib=" << config.primary_device_total_memory / (1024ULL * 1024ULL)
               << '\n';
    }
    output << "llm_model=" << config.llm_model << '\n';
    output << "llm_quantization=" << config.llm_quantization << '\n';
    output << "llm_device=" << config.llm_device << '\n';
    output << "llm_n_gpu_layers=" << config.n_gpu_layers << '\n';
    output << "vision_model=" << config.vision_model << '\n';
    output << "vision_device=" << config.vision_device << '\n';
    output << "audio_model=" << config.audio_model << '\n';
    output << "audio_device=" << config.audio_device << '\n';
    output << "tts_model=" << config.tts_model << '\n';
    output << "tts_device=" << config.tts_device << '\n';
    output << "tts_gpu_layers=" << config.tts_gpu_layers << '\n';
    output << "projector_model=" << config.projector_model << '\n';
    output << "token2wav_model_dir=" << config.token2wav_model_dir << '\n';
    output << "token2wav_device=" << config.token2wav_device << '\n';
    output << "token2wav_threads=" << config.token2wav_threads << '\n';
    output << "execution=" << (config.async_mode ? "async" : "sync") << ','
           << (config.duplex_mode ? "duplex" : "simplex");
    if (config.vpm_batch_encode) {
        output << ",vpm_batch_encode";
    }
    output << ",ctx_size=" << config.n_ctx << '\n';
    output << "placement_plan_count=" << config.placements.size() << '\n';
    for (const auto & placement : config.placements) {
        output << "placement=" << placement.module
               << ",model=" << placement.model
               << ",precision=" << placement.precision
               << ",device=" << placement.device
               << ",backend=" << placement.backend
               << ",execution=" << placement.execution
               << ",status=" << placement.status << '\n';
    }
    return output.str();
}

}  // namespace omni
