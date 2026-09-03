#include "omni.h"
#include "token2wav-impl.h"

#undef NDEBUG
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool write_wav_mono_i16(const fs::path & path, const std::vector<float> & samples, int sample_rate) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float value = std::isfinite(samples[i]) ? samples[i] : 0.0f;
        value       = std::max(-1.0f, std::min(1.0f, value));
        pcm[i]      = static_cast<int16_t>(value * 32767.0f);
    }

    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t riff_size  = 36u + data_bytes;
    const uint16_t format     = 1;
    const uint16_t channels   = 1;
    const uint16_t bits       = 16;
    const uint16_t block      = channels * sizeof(int16_t);
    const uint32_t byte_rate  = static_cast<uint32_t>(sample_rate) * block;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char *>(&riff_size), sizeof(riff_size));
    file.write("WAVEfmt ", 8);
    const uint32_t fmt_size = 16;
    file.write(reinterpret_cast<const char *>(&fmt_size), sizeof(fmt_size));
    file.write(reinterpret_cast<const char *>(&format), sizeof(format));
    file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    file.write(reinterpret_cast<const char *>(&sample_rate), sizeof(sample_rate));
    file.write(reinterpret_cast<const char *>(&byte_rate), sizeof(byte_rate));
    file.write(reinterpret_cast<const char *>(&block), sizeof(block));
    file.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
    file.write("data", 4);
    file.write(reinterpret_cast<const char *>(&data_bytes), sizeof(data_bytes));
    if (data_bytes > 0) {
        file.write(reinterpret_cast<const char *>(pcm.data()), data_bytes);
    }
    return file.good();
}

static void assert_audio_chunk(const std::vector<float> & samples) {
    assert(!samples.empty());
    for (const float sample : samples) {
        assert(std::isfinite(sample));
    }
}

static std::vector<int32_t> smoke_tokens() {
    return {
        1493, 4299, 4218, 2049, 528,  2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993,
        4769, 2300, 226,  2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763,
    };
}

static std::vector<float> generate_one_window(omni::flow::Token2WavSession & session,
                                               const std::vector<int32_t> & tokens) {
    std::vector<float> samples;
    assert(tokens.size() == 28);
    assert(session.feed_window(tokens, false, samples));
    assert_audio_chunk(samples);
    return samples;
}

static void test_first_nonfinal_stream_output_matches_python_contract() {
    std::vector<float> first = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    omni::flow::token2wav_utils::trim_stream_wave_b1(
        first, /*is_first=*/true, /*is_final=*/false, /*cache_len=*/2);
    assert(first == std::vector<float>({ 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f }));

    std::vector<float> next = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    omni::flow::token2wav_utils::trim_stream_wave_b1(
        next, /*is_first=*/false, /*is_final=*/false, /*cache_len=*/2);
    assert(next == std::vector<float>({ 1.0f, 2.0f, 3.0f, 4.0f }));

    std::vector<float> final = { 1.0f, 2.0f, 3.0f };
    omni::flow::token2wav_utils::trim_stream_wave_b1(
        final, /*is_first=*/true, /*is_final=*/true, /*cache_len=*/2);
    assert(final == std::vector<float>({ 1.0f, 2.0f, 3.0f }));
}

int main(int argc, char ** argv) {
    test_first_nonfinal_stream_output_matches_python_contract();

    std::vector<std::string> configured_args;
    if (argc == 1) {
        const char * const env_names[] = {
            "OMNI_T2W_SESSION_MODEL_DIR",
            "OMNI_T2W_SESSION_SPEECH_TOKENIZER_GGUF",
            "OMNI_T2W_SESSION_CAMPPLUS_GGUF",
            "OMNI_T2W_SESSION_VOICE1_WAV",
            "OMNI_T2W_SESSION_VOICE2_WAV",
            "OMNI_T2W_SESSION_OUTPUT_DIR",
        };
        for (const char * name : env_names) {
            const char * value = std::getenv(name);
            if (!value || value[0] == '\0') {
                std::fprintf(stderr,
                             "SKIP: set all OMNI_T2W_SESSION_* variables to run the Token2Wav voice-switching E2E\n");
                return 77;
            }
            configured_args.emplace_back(value);
        }
    } else if (argc < 7 || argc > 9) {
        std::fprintf(stderr,
                     "usage: %s <model_dir> <speech_tokenizer.gguf> <campplus.gguf> "
                     "<voice1.wav> <voice2.wav> <output_dir> [device] [n_timesteps]\n",
                     argv[0]);
        return 2;
    }

    const fs::path model_dir          = argc == 1 ? configured_args[0] : argv[1];
    const std::string speech_tokenizer = argc == 1 ? configured_args[1] : argv[2];
    const std::string campplus         = argc == 1 ? configured_args[2] : argv[3];
    const fs::path voice1              = argc == 1 ? configured_args[3] : argv[4];
    const fs::path voice2              = argc == 1 ? configured_args[4] : argv[5];
    const fs::path output_dir          = argc == 1 ? configured_args[5] : argv[6];
    const std::string device            = argc >= 8 ? argv[7] : "gpu:0";
    const int n_timesteps              = argc >= 9
                                             ? std::stoi(argv[8])
                                             : omni_tts_python_base_config().n_timesteps;

    fs::create_directories(output_dir);
    const std::string encoder = (model_dir / "encoder.gguf").string();
    const std::string flow_matching = (model_dir / "flow_matching.gguf").string();
    const std::string flow_extra = (model_dir / "flow_extra.gguf").string();
    const std::string prompt_cache = (model_dir / "prompt_cache.gguf").string();
    const std::string vocoder = (model_dir / "hifigan2.gguf").string();

    omni::flow::Token2WavSession session;
    std::fprintf(stderr, "loading Token2Wav models from %s\n", model_dir.c_str());
    if (!session.init_from_prompt_cache_gguf(
            encoder, flow_matching, flow_extra, prompt_cache, vocoder, device, device, -1, 1.0f)) {
        std::fprintf(stderr, "Token2Wav model/cache initialization failed\n");
        return 3;
    }
    std::fprintf(stderr, "Token2Wav static prompt cache initialized\n");

    const auto tokens = smoke_tokens();
    if (!session.set_prompt_wav(voice1.string(), speech_tokenizer, campplus, 1, n_timesteps, 1.0f)) {
        std::fprintf(stderr, "set_prompt_wav failed for voice1: %s\n", voice1.c_str());
        return 4;
    }
    std::fprintf(stderr, "voice1 PromptBundle installed\n");
    const auto voice1_samples = generate_one_window(session, tokens);
    assert(write_wav_mono_i16(output_dir / "voice1.wav", voice1_samples, omni::flow::Token2Wav::kSampleRate));

    if (!session.set_prompt_wav(voice2.string(), speech_tokenizer, campplus, 1, n_timesteps, 1.0f)) {
        std::fprintf(stderr, "set_prompt_wav failed for voice2: %s\n", voice2.c_str());
        return 5;
    }
    std::fprintf(stderr, "voice2 PromptBundle installed\n");
    const auto voice2_samples = generate_one_window(session, tokens);
    assert(write_wav_mono_i16(output_dir / "voice2.wav", voice2_samples, omni::flow::Token2Wav::kSampleRate));

    assert(session.reset_to_prompt_cache_gguf(prompt_cache, -1, 1.0f));
    const auto default_samples = generate_one_window(session, tokens);
    assert(write_wav_mono_i16(output_dir / "default.wav", default_samples, omni::flow::Token2Wav::kSampleRate));

    std::printf("token2wav session: voice1_samples=%zu voice2_samples=%zu sample_rate=%d output_dir=%s\n",
                voice1_samples.size(), voice2_samples.size(), omni::flow::Token2Wav::kSampleRate,
                output_dir.c_str());
    return 0;
}
