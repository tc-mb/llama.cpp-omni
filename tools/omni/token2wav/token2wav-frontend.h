#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omni {
namespace flow {

struct AudioFeatures {
    int32_t              channels = 0;
    int32_t              frames   = 0;
    std::vector<float>   values;
};

struct AudioBuffer {
    int32_t            sample_rate = 0;
    std::vector<float> samples;
};

struct FrontendPromptBundle {
    std::vector<int32_t> prompt_tokens_bt;
    std::vector<float>   prompt_mel_btc;
    std::vector<float>   spk_bc;

    int64_t B              = 1;
    int64_t T_prompt_token = 0;
    int64_t T_prompt_mel   = 0;
};

class Token2WavFrontend {
  public:
    static constexpr size_t kMaxReferenceWavBytes = 64ULL * 1024ULL * 1024ULL;
    static constexpr uint32_t kMaxReferenceAudioDurationSeconds = 30;

    // Decode an audio file as mono float32 while preserving the source sample rate.
    static bool load_wav_mono(const std::string & path, AudioBuffer & out);

    // Resample mono audio with the StepAudio2-compatible windowed-sinc kernel.
    static bool resample_mono(const AudioBuffer & input, int32_t target_sample_rate, AudioBuffer & out);

    // Compute the 16 kHz speech-tokenizer log-Mel input in [channel, frame] layout.
    static bool compute_s3_log_mel(const std::vector<float> & audio_16k, AudioFeatures & out);

    // Compute the 24 kHz Token2Wav prompt Mel input in [channel, frame] layout.
    static bool compute_token2wav_prompt_mel(const std::vector<float> & audio_24k, AudioFeatures & out);

    // Compute the 16 kHz Kaldi fbank input in [frame, channel] layout.
    static bool compute_campplus_fbank(const std::vector<float> & audio_16k, AudioFeatures & out);

    // Assemble the native model outputs into the PromptBundle contract used by
    // Token2Mel. speech_tokens does not include the three pre-lookahead tokens.
    static bool assemble_prompt_bundle(const std::vector<int32_t> & speech_tokens,
                                       const AudioFeatures &          prompt_mel_ct,
                                       const std::vector<float> &     speaker_embedding,
                                       FrontendPromptBundle &         out,
                                       std::string *                  error = nullptr);

    // Validate the output shapes accepted from the ONNX frontend models.
    // Rank-1 tensors represent an already-squeezed batch; rank-2 tensors
    // must have B=1.
    static bool validate_speech_tokenizer_output_shape(const std::vector<int64_t> & shape,
                                                       std::string *                  error = nullptr);
    static bool validate_campplus_output_shape(const std::vector<int64_t> & shape,
                                               std::string *                  error = nullptr);

    // Run the native WAV frontend. This path is compiled only when
    // OMNI_T2W_ENABLE_NATIVE_FRONTEND is enabled and an ONNX Runtime library
    // is linked into the omni target.
    static bool prepare_prompt_bundle(const std::string & ref_wav_path,
                                      const std::string & speech_tokenizer_onnx,
                                      const std::string & campplus_onnx,
                                      FrontendPromptBundle & out,
                                      std::string *            error = nullptr,
                                      int                      num_threads = 1);
};

}  // namespace flow
}  // namespace omni
