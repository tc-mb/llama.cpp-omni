#pragma once

#include "token2wav-frontend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omni {
namespace flow {

// Run both StepAudio2 frontend models from GGUF weights using ggml.
bool prepare_prompt_bundle_gguf(const AudioFeatures & speech_features,
                                const AudioFeatures & campplus_features,
                                const std::string &  speech_model_path,
                                const std::string &  campplus_model_path,
                                int                    num_threads,
                                std::vector<int32_t> & speech_tokens,
                                std::vector<float> &   speaker_embedding,
                                std::string *           error = nullptr);

}  // namespace flow
}  // namespace omni
