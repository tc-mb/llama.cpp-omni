#include "arg.h"
#include "common.h"
#include "runtime-profile.h"
#include "runtime-profile-session.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;

class temporary_model_tree {
  public:
    temporary_model_tree() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("omni-runtime-profile-test-" + std::to_string(unique));
        std::filesystem::create_directories(root);
    }

    ~temporary_model_tree() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void touch(const std::filesystem::path & relative_path) const {
        const auto path = root / relative_path;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << "test";
    }

    void write(const std::filesystem::path & relative_path, const std::string & contents) const {
        const auto path = root / relative_path;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << contents;
    }

    std::filesystem::path root;
};

static void create_complete_model_tree(const temporary_model_tree & tree) {
    for (const auto & relative_path : {
             std::filesystem::path("MiniCPM-o-4_5-Q4_K_M.gguf"),
             std::filesystem::path("vision/MiniCPM-o-4_5-vision-F16.gguf"),
             std::filesystem::path("audio/MiniCPM-o-4_5-audio-F16.gguf"),
             std::filesystem::path("tts/MiniCPM-o-4_5-tts-F16.gguf"),
             std::filesystem::path("tts/MiniCPM-o-4_5-projector-F16.gguf"),
             std::filesystem::path("token2wav-gguf/encoder.gguf"),
             std::filesystem::path("token2wav-gguf/flow_matching.gguf"),
             std::filesystem::path("token2wav-gguf/flow_extra.gguf"),
             std::filesystem::path("token2wav-gguf/hifigan2.gguf"),
             std::filesystem::path("token2wav-gguf/prompt_cache.gguf") }) {
        tree.touch(relative_path);
    }
}

static std::string complete_profile_config() {
    return R"json({
  "schema_version": 1,
  "profile": "auto",
  "llm": {
    "model": "MiniCPM-o-4_5-Q4_K_M.gguf",
    "quantization": "Q4_K_M",
    "device": "CUDA0",
    "n_gpu_layers": 37
  },
  "vision": {
    "model": "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "device": "CUDA1"
  },
  "audio": {
    "model": "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "device": "CUDA1"
  },
  "tts": {
    "model": "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "device": "CUDA0",
    "gpu_layers": -1
  },
  "projector": {
    "model": "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "device": "CUDA1"
  },
  "token2wav": {
    "model_dir": "token2wav-gguf",
    "device": "CUDA0",
    "threads": 12
  },
  "runtime": {
    "n_ctx": 6144,
    "duplex": true,
    "async": false,
    "vpm_batch_encode": true
  }
})json";
}

static omni::hardware_snapshot two_accelerators() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    hardware.devices.push_back({ "CUDA1", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    return hardware;
}

static omni::runtime_profile_result resolve_from_tree(const temporary_model_tree & tree,
                                                       const omni::hardware_snapshot & hardware) {
    const auto inventory = omni::discover_model_inventory(tree.root.string());
    return omni::resolve_runtime_profile_from_config(
        (tree.root / "omni-runtime-profile.json").string(), hardware, inventory);
}

static void test_auto_requires_static_profile_config() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    omni::model_inventory inventory;
    inventory.root = "/models/MiniCPM-o-4_5-gguf";

    const auto result = omni::resolve_runtime_profile("auto", hardware, inventory);

    assert(!result.ok);
    assert(result.error.find("profile config") != std::string::npos);
}

static void test_static_profile_loads_values_and_maps_logical_devices() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-runtime-profile.json", complete_profile_config());

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(result.ok);
    assert(result.config.requested_profile == "auto");
    assert(result.config.resolved_profile == "static_config");
    assert(result.config.llm_model == (tree.root / "MiniCPM-o-4_5-Q4_K_M.gguf").string());
    assert(result.config.llm_quantization == "Q4_K_M");
    assert(result.config.llm_device == "CUDA0");
    assert(result.config.n_gpu_layers == 37);
    assert(result.config.vision_device == "CUDA1");
    assert(result.config.audio_device == "CUDA1");
    assert(result.config.tts_device == "CUDA0");
    assert(result.config.tts_gpu_layers == -1);
    assert(result.config.token2wav_device == "gpu:0");
    assert(result.config.token2wav_threads == 12);
    assert(result.config.n_ctx == 6144);
    assert(result.config.duplex_mode);
    assert(!result.config.async_mode);
    assert(result.config.vpm_batch_encode);
}

static void test_static_profile_reports_missing_config_file() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("profile config file") != std::string::npos);
    assert(result.error.find("omni-runtime-profile.json") != std::string::npos);
}

static void test_static_profile_reports_invalid_json() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-runtime-profile.json", "{\"schema_version\":");

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("invalid profile config JSON") != std::string::npos);
}

static void test_static_profile_reports_missing_required_field() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_profile_config();
    const auto marker = std::string("  \"audio\": {\n    \"model\": \"audio/MiniCPM-o-4_5-audio-F16.gguf\",\n    \"device\": \"CUDA1\"\n  },\n");
    config.replace(config.find(marker), marker.size(), "");
    tree.write("omni-runtime-profile.json", config);

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("audio") != std::string::npos);
    assert(result.error.find("required") != std::string::npos);
}

static void test_static_profile_reports_missing_model_file() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_profile_config();
    const auto marker = "MiniCPM-o-4_5-Q4_K_M.gguf";
    config.replace(config.find(marker), std::string(marker).size(), "missing.gguf");
    tree.write("omni-runtime-profile.json", config);

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("llm model") != std::string::npos);
    assert(result.error.find("missing.gguf") != std::string::npos);
}

static void test_static_profile_rejects_quantization_filename_mismatch() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_profile_config();
    const auto marker = "\"quantization\": \"Q4_K_M\"";
    config.replace(config.find(marker), std::string(marker).size(), "\"quantization\": \"F16\"");
    tree.write("omni-runtime-profile.json", config);

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("quantization") != std::string::npos);
    assert(result.error.find("Q4_K_M") != std::string::npos);
}

static void test_static_profile_reports_unavailable_logical_device() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-runtime-profile.json", complete_profile_config());

    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    const auto result = resolve_from_tree(tree, hardware);

    assert(!result.ok);
    assert(result.error.find("CUDA1") != std::string::npos);
    assert(result.error.find("vision") != std::string::npos);
}

static void test_static_profile_rejects_empty_device_value() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_profile_config();
    const auto marker = "\"device\": \"CUDA0\"";
    config.replace(config.find(marker), std::string(marker).size(), "\"device\": \"\"");
    tree.write("omni-runtime-profile.json", config);

    const auto result = resolve_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("llm.device") != std::string::npos);
    assert(result.error.find("empty") != std::string::npos);
}

static void test_effective_config_reports_static_source() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-runtime-profile.json", complete_profile_config());

    const auto result = resolve_from_tree(tree, two_accelerators());
    assert(result.ok);
    const auto output = omni::format_effective_runtime_config(result.config);
    assert(output.find("resolved_profile=static_config\n") != std::string::npos);
    assert(output.find("llm_quantization=Q4_K_M\n") != std::string::npos);
}

static void test_runtime_profile_controls_session_options_when_present() {
    omni::effective_runtime_config config;
    config.duplex_mode       = true;
    config.tts_gpu_layers    = -1;
    config.token2wav_device  = "gpu:0";
    config.token2wav_threads = 32;

    const auto profiled = omni::resolve_runtime_session_options(&config, false, 100, "cpu", 4);
    assert(profiled.duplex_mode);
    assert(!profiled.async_mode);
    assert(profiled.tts_gpu_layers == -1);
    assert(profiled.token2wav_device == "gpu:0");
    assert(profiled.token2wav_threads == 32);
    assert(profiled.strict_runtime_config);

    const auto legacy = omni::resolve_runtime_session_options(nullptr, false, 100, "cpu", 4);
    assert(!legacy.duplex_mode);
    assert(legacy.async_mode);
    assert(legacy.tts_gpu_layers == 100);
    assert(legacy.token2wav_device == "cpu");
    assert(legacy.token2wav_threads == 4);
    assert(!legacy.strict_runtime_config);
}

static void test_model_inventory_only_reports_files_that_exist() {
    temporary_model_tree tree;
    tree.touch("MiniCPM-o-4_5-F16.gguf");
    tree.touch("vision/MiniCPM-o-4_5-vision-F16.gguf");

    const auto inventory = omni::discover_model_inventory(tree.root.string());

    assert(inventory.root == tree.root.string());
    assert(inventory.llm_f16 == (tree.root / "MiniCPM-o-4_5-F16.gguf").string());
    assert(inventory.llm_q8_0.empty());
    assert(inventory.llm_q4_k_m.empty());
    assert(inventory.vision == (tree.root / "vision/MiniCPM-o-4_5-vision-F16.gguf").string());
    assert(inventory.token2wav_prompt_cache.empty());
}

static bool parse_omni_server_arguments(std::vector<std::string> arguments) {
    common_params        params;
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }
    return common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);
}

static void test_omni_runtime_profile_arguments_are_parsed() {
    common_params            params;
    std::vector<std::string> arguments = {
        "llama-omni-server", "--profile", "auto", "--model-dir", "/models/MiniCPM-o-4_5-gguf",
        "--profile-config", "/models/profile.json", "--token2wav-threads", "32", "--print-effective-config",
    };
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }

    const bool parsed =
        common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);

    assert(parsed);
    assert(params.omni_runtime_profile.profile == "auto");
    assert(params.omni_runtime_profile.model_dir == "/models/MiniCPM-o-4_5-gguf");
    assert(params.omni_runtime_profile.profile_config == "/models/profile.json");
    assert(params.omni_runtime_profile.token2wav_threads == 32);
    assert(params.omni_runtime_profile.print_effective_config);
}

static void test_omni_runtime_profile_argument_ranges_are_validated() {
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--token2wav-threads", "0" }));
}

static void test_omni_runtime_profile_rejects_autotune_arguments() {
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--profile", "auto", "--autotune", "refresh" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--profile", "auto", "--autotune-cache",
                                          "/tmp/omni-autotune.json" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--profile", "auto", "--autotune-rounds", "3" }));
}

int main() {
    test_auto_requires_static_profile_config();
    test_static_profile_loads_values_and_maps_logical_devices();
    test_static_profile_reports_missing_config_file();
    test_static_profile_reports_invalid_json();
    test_static_profile_reports_missing_required_field();
    test_static_profile_reports_missing_model_file();
    test_static_profile_rejects_quantization_filename_mismatch();
    test_static_profile_reports_unavailable_logical_device();
    test_static_profile_rejects_empty_device_value();
    test_effective_config_reports_static_source();
    test_runtime_profile_controls_session_options_when_present();
    test_model_inventory_only_reports_files_that_exist();
    test_omni_runtime_profile_arguments_are_parsed();
    test_omni_runtime_profile_argument_ranges_are_validated();
    test_omni_runtime_profile_rejects_autotune_arguments();
    return 0;
}
