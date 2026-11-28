#include "token2wav-impl.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class temporary_prompt_bundle {
  public:
    temporary_prompt_bundle() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = fs::temp_directory_path() / ("omni-prompt-bundle-test-" + std::to_string(unique));
        fs::create_directories(root);
    }

    ~temporary_prompt_bundle() {
        std::error_code error;
        fs::remove_all(root, error);
    }

    void write_binary(const std::string & name, const void * data, size_t size) const {
        std::ofstream file(root / name, std::ios::binary);
        file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    }

    void write_manifest(const std::string & contents) const {
        std::ofstream file(root / "manifest.json");
        file << contents;
    }

    fs::path root;
};

static void create_valid_bundle(temporary_prompt_bundle & bundle) {
    const std::vector<int32_t> tokens = { 1, 2, 3, 4, 5, 6, 7 };
    const std::vector<float> mel(8 * 80, 0.0f);
    const std::vector<float> spk(192, 0.0f);

    bundle.write_binary("prompt_tokens_i32.bin", tokens.data(), tokens.size() * sizeof(int32_t));
    bundle.write_binary("prompt_mel_btc_f32.bin", mel.data(), mel.size() * sizeof(float));
    bundle.write_binary("spk_f32.bin", spk.data(), spk.size() * sizeof(float));
    bundle.write_manifest(R"json({
  "schema_version": 1,
  "sample_rate": 16000,
  "channels": 1,
  "prompt_token_count": 7,
  "prompt_mel_frames": 8,
  "mel_channels": 80,
  "speaker_dimensions": 192,
  "prompt_mel_layout": "BTC",
  "dtype": {
    "prompt_tokens": "int32",
    "prompt_mel": "float32",
    "speaker_embedding": "float32"
  }
})json");
}

static void test_loads_valid_prompt_bundle_manifest() {
    temporary_prompt_bundle bundle;
    create_valid_bundle(bundle);

    omni::flow::Token2Mel::PromptBundle prompt;
    assert(omni::flow::Token2Mel::load_prompt_bundle_dir(bundle.root.string(), prompt));
    assert(prompt.B == 1);
    assert(prompt.T_prompt_token == 7);
    assert(prompt.T_prompt_mel == 8);
    assert(prompt.prompt_tokens_bt.size() == 7);
    assert(prompt.prompt_mel_btc.size() == 8 * 80);
    assert(prompt.spk_bc.size() == 192);
}

static void test_rejects_prompt_bundle_with_wrong_schema_contract() {
    temporary_prompt_bundle bundle;
    create_valid_bundle(bundle);
    bundle.write_manifest(R"json({
  "schema_version": 2,
  "sample_rate": 16000,
  "channels": 1,
  "prompt_token_count": 7,
  "prompt_mel_frames": 8,
  "mel_channels": 80,
  "speaker_dimensions": 192,
  "prompt_mel_layout": "BTC",
  "dtype": {
    "prompt_tokens": "int32",
    "prompt_mel": "float32",
    "speaker_embedding": "float32"
  }
})json");

    omni::flow::Token2Mel::PromptBundle prompt;
    assert(!omni::flow::Token2Mel::load_prompt_bundle_dir(bundle.root.string(), prompt));
}

static void test_rejects_prompt_bundle_when_manifest_shape_disagrees_with_binary() {
    temporary_prompt_bundle bundle;
    create_valid_bundle(bundle);
    bundle.write_manifest(R"json({
  "schema_version": 1,
  "sample_rate": 16000,
  "channels": 1,
  "prompt_token_count": 8,
  "prompt_mel_frames": 8,
  "mel_channels": 80,
  "speaker_dimensions": 192,
  "prompt_mel_layout": "BTC",
  "dtype": {
    "prompt_tokens": "int32",
    "prompt_mel": "float32",
    "speaker_embedding": "float32"
  }
})json");

    omni::flow::Token2Mel::PromptBundle prompt;
    assert(!omni::flow::Token2Mel::load_prompt_bundle_dir(bundle.root.string(), prompt));
}

int main() {
    test_loads_valid_prompt_bundle_manifest();
    test_rejects_prompt_bundle_with_wrong_schema_contract();
    test_rejects_prompt_bundle_when_manifest_shape_disagrees_with_binary();
    return 0;
}
