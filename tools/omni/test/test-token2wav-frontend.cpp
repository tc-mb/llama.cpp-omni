#include "token2wav-frontend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> sine_wave(int sample_rate, float seconds, float frequency) {
    const size_t n = static_cast<size_t>(sample_rate * seconds);
    std::vector<float> audio(n);
    for (size_t i = 0; i < n; ++i) {
        audio[i] = 0.2f * std::sin(2.0f * static_cast<float>(M_PI) * frequency *
                                   static_cast<float>(i) / static_cast<float>(sample_rate));
    }
    return audio;
}

static void assert_close(float actual, float expected, float tolerance, const char * name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s mismatch: actual=%0.9f expected=%0.9f tolerance=%0.9f\n",
                     name, actual, expected, tolerance);
        assert(false);
    }
}

static void test_resample_matches_stepaudio2_contract() {
    omni::flow::AudioBuffer input;
    input.sample_rate = 24000;
    input.samples.assign(32, 0.0f);
    input.samples[5] = 1.0f;

    omni::flow::AudioBuffer output;
    assert(omni::flow::Token2WavFrontend::resample_mono(input, 16000, output));
    assert(output.sample_rate == 16000);
    assert(output.samples.size() == 22);

    const float expected[] = {
        -0.021723339334f,
        0.050903785974f,
        -0.118959866464f,
        0.543885588646f,
        0.270691812038f,
        -0.093562029302f,
        0.042748011649f,
        -0.017954932526f,
        0.005282599479f,
        -0.000366034976f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    for (size_t i = 0; i < output.samples.size(); ++i) {
        assert_close(output.samples[i], expected[i], 2e-6f, "resample");
    }
}

static void test_s3_tokenizer_mel_contract() {
    const auto audio = sine_wave(16000, 1.0f, 440.0f);
    omni::flow::AudioFeatures features;
    assert(omni::flow::Token2WavFrontend::compute_s3_log_mel(audio, features));
    assert(features.channels == 128);
    assert(features.frames == 100);
    assert(features.values.size() == static_cast<size_t>(128 * 100));
    for (const float value : features.values) {
        assert(std::isfinite(value));
    }
    const float expected[] = {
        0.708549559f,
        0.196541548f,
        -0.713614941f,
        0.214703143f,
        -0.713614941f,
        -0.713614941f,
    };
    const size_t indices[] = {0, 1, 127, 128 * 50, 128 * 99, 128 * 100 - 1};
    for (size_t i = 0; i < 6; ++i) {
        assert_close(features.values[indices[i]], expected[i], 2e-3f, "s3_log_mel");
    }
}

static void test_token2wav_prompt_mel_contract() {
    const auto audio = sine_wave(24000, 1.0f, 440.0f);
    omni::flow::AudioFeatures features;
    assert(omni::flow::Token2WavFrontend::compute_token2wav_prompt_mel(audio, features));
    assert(features.channels == 80);
    assert(features.frames == 50);
    assert(features.values.size() == static_cast<size_t>(80 * 50));
    for (const float value : features.values) {
        assert(std::isfinite(value));
    }
    const float expected[] = {
        -1.431666017f,
        -1.409540534f,
        -6.810078621f,
        -10.043351173f,
        -1.436961055f,
        -6.813612938f,
    };
    const size_t indices[] = {
        0,
        1 * 50,
        79 * 50,
        25,
        49,
        79 * 50 + 49,
    };
    for (size_t i = 0; i < 6; ++i) {
        assert_close(features.values[indices[i]], expected[i], 2e-2f, "token2wav_prompt_mel");
    }
}

static void test_campplus_fbank_contract() {
    const auto audio = sine_wave(16000, 1.0f, 440.0f);
    omni::flow::AudioFeatures features;
    assert(omni::flow::Token2WavFrontend::compute_campplus_fbank(audio, features));
    assert(features.channels == 80);
    assert(features.frames == 98);
    assert(features.values.size() == static_cast<size_t>(80 * 98));
    for (const float value : features.values) {
        assert(std::isfinite(value));
    }
    const float expected[] = {
        -13.403847694f,
        -12.797039986f,
        -15.942384720f,
        -13.475249290f,
        -12.455262184f,
        -14.080187798f,
    };
    const size_t indices[] = {0, 1, 79, 80 * 49, 80 * 97, 80 * 98 - 1};
    for (size_t i = 0; i < 6; ++i) {
        // The C++ float sine fixture and the Python reference use different
        // arithmetic order, so keep this as a regression check rather than
        // an exact cross-language parity test.
        assert_close(features.values[indices[i]], expected[i], 1e-2f, "campplus_fbank");
    }
}

static void test_prompt_bundle_assembly_adds_lookahead_and_converts_mel_layout() {
    const std::vector<int32_t> speech_tokens = { 10, 11, 12, 13 };
    omni::flow::AudioFeatures prompt_mel;
    prompt_mel.channels = 80;
    prompt_mel.frames   = 8;
    prompt_mel.values.resize(static_cast<size_t>(prompt_mel.channels) * prompt_mel.frames);
    for (int c = 0; c < prompt_mel.channels; ++c) {
        for (int t = 0; t < prompt_mel.frames; ++t) {
            prompt_mel.values[static_cast<size_t>(c) * prompt_mel.frames + t] =
                static_cast<float>(c * 1000 + t);
        }
    }
    const std::vector<float> speaker_embedding(192, 0.25f);

    omni::flow::FrontendPromptBundle bundle;
    std::string error;
    assert(omni::flow::Token2WavFrontend::assemble_prompt_bundle(
        speech_tokens, prompt_mel, speaker_embedding, bundle, &error));
    assert(error.empty());
    assert(bundle.B == 1);
    assert(bundle.T_prompt_token == 7);
    assert(bundle.T_prompt_mel == 8);
    assert(bundle.prompt_tokens_bt == std::vector<int32_t>({ 10, 11, 12, 13, 4218, 4218, 4218 }));
    assert(bundle.spk_bc == speaker_embedding);
    for (int t = 0; t < prompt_mel.frames; ++t) {
        for (int c = 0; c < prompt_mel.channels; ++c) {
            const size_t index = static_cast<size_t>(t) * prompt_mel.channels + c;
            assert(bundle.prompt_mel_btc[index] ==
                   static_cast<float>(c * 1000 + t));
        }
    }
}

static void test_prompt_bundle_assembly_rejects_shape_mismatch() {
    const std::vector<int32_t> speech_tokens = { 10, 11, 12, 13 };
    omni::flow::AudioFeatures prompt_mel;
    prompt_mel.channels = 80;
    prompt_mel.frames   = 7;
    prompt_mel.values.assign(static_cast<size_t>(prompt_mel.channels) * prompt_mel.frames, 0.0f);
    const std::vector<float> speaker_embedding(192, 0.25f);

    omni::flow::FrontendPromptBundle bundle;
    std::string error;
    assert(!omni::flow::Token2WavFrontend::assemble_prompt_bundle(
        speech_tokens, prompt_mel, speaker_embedding, bundle, &error));
    assert(error.find("prompt Mel frame count") != std::string::npos);
}

static void test_frontend_output_shape_contracts_require_single_batch() {
    std::string error;
    assert(omni::flow::Token2WavFrontend::validate_speech_tokenizer_output_shape({ 17 }, &error));
    assert(omni::flow::Token2WavFrontend::validate_speech_tokenizer_output_shape({ 1, 17 }, &error));
    assert(!omni::flow::Token2WavFrontend::validate_speech_tokenizer_output_shape({ 2, 9 }, &error));
    assert(error.find("B=1") != std::string::npos);

    error.clear();
    assert(omni::flow::Token2WavFrontend::validate_campplus_output_shape({ 192 }, &error));
    assert(omni::flow::Token2WavFrontend::validate_campplus_output_shape({ 1, 192 }, &error));
    assert(!omni::flow::Token2WavFrontend::validate_campplus_output_shape({ 2, 96 }, &error));
    assert(error.find("B=1") != std::string::npos);
}

static void test_load_wav_rejects_short_multichannel_frame() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "token2wav-short-block-align.wav";
    std::vector<uint8_t> wav(45, 0);
    auto put_u16 = [&](size_t offset, uint16_t value) {
        wav[offset] = static_cast<uint8_t>(value & 0xffU);
        wav[offset + 1] = static_cast<uint8_t>(value >> 8);
    };
    auto put_u32 = [&](size_t offset, uint32_t value) {
        wav[offset] = static_cast<uint8_t>(value & 0xffU);
        wav[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xffU);
        wav[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xffU);
        wav[offset + 3] = static_cast<uint8_t>(value >> 24);
    };
    std::memcpy(wav.data(), "RIFF", 4);
    put_u32(4, static_cast<uint32_t>(wav.size() - 8));
    std::memcpy(wav.data() + 8, "WAVE", 4);
    std::memcpy(wav.data() + 12, "fmt ", 4);
    put_u32(16, 16);
    put_u16(20, 1);
    put_u16(22, 2);
    put_u32(24, 16000);
    put_u32(28, 16000);
    put_u16(32, 1);
    put_u16(34, 16);
    std::memcpy(wav.data() + 36, "data", 4);
    put_u32(40, 1);

    {
        std::ofstream file(path, std::ios::binary);
        assert(file.good());
        file.write(reinterpret_cast<const char *>(wav.data()),
                   static_cast<std::streamsize>(wav.size()));
        assert(file.good());
    }

    omni::flow::AudioBuffer output;
    assert(!omni::flow::Token2WavFrontend::load_wav_mono(path.string(), output));
    std::filesystem::remove(path);
}

static void test_resample_rejects_unbounded_kernel_request() {
    omni::flow::AudioBuffer input;
    input.sample_rate = 1;
    input.samples.assign(8, 0.0f);

    omni::flow::AudioBuffer output;
    assert(!omni::flow::Token2WavFrontend::resample_mono(input, 1000000, output));
}

static void write_pcm16_wav(const std::string & path, int sample_rate, const std::vector<float> & samples) {
    std::ofstream file(path, std::ios::binary);
    assert(file.good());

    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_bytes = 36U + data_bytes;
    auto write_u16 = [&](uint16_t value) {
        file.put(static_cast<char>(value & 0xffU));
        file.put(static_cast<char>((value >> 8) & 0xffU));
    };
    auto write_u32 = [&](uint32_t value) {
        file.put(static_cast<char>(value & 0xffU));
        file.put(static_cast<char>((value >> 8) & 0xffU));
        file.put(static_cast<char>((value >> 16) & 0xffU));
        file.put(static_cast<char>((value >> 24) & 0xffU));
    };

    file.write("RIFF", 4);
    write_u32(riff_bytes);
    file.write("WAVEfmt ", 8);
    write_u32(16);
    write_u16(1);
    write_u16(1);
    write_u32(static_cast<uint32_t>(sample_rate));
    write_u32(static_cast<uint32_t>(sample_rate * sizeof(int16_t)));
    write_u16(sizeof(int16_t));
    write_u16(16);
    file.write("data", 4);
    write_u32(data_bytes);
    for (const float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrintf(clamped * 32767.0f));
        write_u16(static_cast<uint16_t>(pcm));
    }
    assert(file.good());
}

static void test_gguf_frontend_uses_both_model_files() {
    const char * speech_path = std::getenv("OMNI_T2W_TEST_SPEECH_TOKENIZER_GGUF");
    const char * campplus_path = std::getenv("OMNI_T2W_TEST_CAMPPLUS_GGUF");
    if (!speech_path || !campplus_path || *speech_path == '\0' || *campplus_path == '\0') {
        std::fprintf(stderr,
                     "SKIP: set OMNI_T2W_TEST_SPEECH_TOKENIZER_GGUF and "
                     "OMNI_T2W_TEST_CAMPPLUS_GGUF to run the GGUF frontend test\n");
        return;
    }

    const std::filesystem::path wav_path =
        std::filesystem::temp_directory_path() / "token2wav-gguf-frontend.wav";
    const auto audio = sine_wave(16000, 1.0f, 440.0f);
    write_pcm16_wav(wav_path.string(), 16000, audio);

    omni::flow::FrontendPromptBundle bundle;
    std::string error;
    const bool ok = omni::flow::Token2WavFrontend::prepare_prompt_bundle(
        wav_path.string(), speech_path, campplus_path, bundle, &error, 1);
    std::filesystem::remove(wav_path);

    if (!ok) {
        std::fprintf(stderr, "GGUF frontend failed: %s\n", error.c_str());
    }
    assert(ok);
    assert(error.empty());
    assert(bundle.B == 1);
    assert(bundle.T_prompt_token > 3);
    assert(bundle.T_prompt_mel == (bundle.T_prompt_token - 3) * 2);
    assert(bundle.prompt_tokens_bt.size() == static_cast<size_t>(bundle.T_prompt_token));
    assert(bundle.prompt_mel_btc.size() == static_cast<size_t>(bundle.T_prompt_mel * 80));
    assert(bundle.spk_bc.size() == 192);
}

static void write_float_file(const std::string & path, const std::vector<float> & values) {
    std::ofstream file(path, std::ios::binary);
    assert(file.good());
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    assert(file.good());
}

static void dump_native_bundle(const char * output_dir,
                               const std::string & ref_wav,
                               const omni::flow::FrontendPromptBundle & bundle) {
    std::ofstream tokens(std::string(output_dir) + "/prompt_tokens_i32.bin", std::ios::binary);
    std::ofstream mel(std::string(output_dir) + "/prompt_mel_btc_f32.bin", std::ios::binary);
    std::ofstream spk(std::string(output_dir) + "/spk_f32.bin", std::ios::binary);
    assert(tokens.good());
    assert(mel.good());
    assert(spk.good());
    tokens.write(reinterpret_cast<const char *>(bundle.prompt_tokens_bt.data()),
                 static_cast<std::streamsize>(bundle.prompt_tokens_bt.size() * sizeof(int32_t)));
    mel.write(reinterpret_cast<const char *>(bundle.prompt_mel_btc.data()),
              static_cast<std::streamsize>(bundle.prompt_mel_btc.size() * sizeof(float)));
    spk.write(reinterpret_cast<const char *>(bundle.spk_bc.data()),
              static_cast<std::streamsize>(bundle.spk_bc.size() * sizeof(float)));
    assert(tokens.good());
    assert(mel.good());
    assert(spk.good());

    omni::flow::AudioBuffer input;
    omni::flow::AudioBuffer audio_16k;
    omni::flow::AudioFeatures fbank;
    assert(omni::flow::Token2WavFrontend::load_wav_mono(ref_wav, input));
    assert(omni::flow::Token2WavFrontend::resample_mono(input, 16000, audio_16k));
    assert(omni::flow::Token2WavFrontend::compute_campplus_fbank(audio_16k.samples, fbank));
    omni::flow::AudioFeatures speech_features;
    assert(omni::flow::Token2WavFrontend::compute_s3_log_mel(audio_16k.samples, speech_features));
    write_float_file(std::string(output_dir) + "/audio_16k_f32.bin", audio_16k.samples);
    write_float_file(std::string(output_dir) + "/speech_log_mel_ct_f32.bin", speech_features.values);
    write_float_file(std::string(output_dir) + "/campplus_fbank_f32.bin", fbank.values);
}

static void test_native_frontend_with_real_models(const char * ref_wav,
                                                   const char * speech_tokenizer_gguf,
                                                   const char * campplus_gguf,
                                                   const char * dump_dir = nullptr) {
    omni::flow::FrontendPromptBundle bundle;
    std::string error;
    const bool ok = omni::flow::Token2WavFrontend::prepare_prompt_bundle(
        ref_wav, speech_tokenizer_gguf, campplus_gguf, bundle, &error, 1);
    if (!ok) {
        std::fprintf(stderr, "native frontend failed: %s\n", error.c_str());
    }
    assert(ok);
    assert(error.empty());
    assert(bundle.B == 1);
    assert(bundle.T_prompt_token > 3);
    assert(bundle.T_prompt_mel > 0);
    assert(bundle.T_prompt_mel == (bundle.T_prompt_token - 3) * 2);
    assert(bundle.prompt_tokens_bt.size() == static_cast<size_t>(bundle.T_prompt_token));
    assert(bundle.prompt_mel_btc.size() == static_cast<size_t>(bundle.T_prompt_mel * 80));
    assert(bundle.spk_bc.size() == 192);
    for (const float value : bundle.prompt_mel_btc) {
        assert(std::isfinite(value));
    }
    for (const float value : bundle.spk_bc) {
        assert(std::isfinite(value));
    }
    if (dump_dir != nullptr) {
        dump_native_bundle(dump_dir, ref_wav, bundle);
    }
    std::printf("native frontend: tokens=%lld mel_frames=%lld speaker_dim=%zu\n",
                static_cast<long long>(bundle.T_prompt_token),
                static_cast<long long>(bundle.T_prompt_mel),
                bundle.spk_bc.size());
}

int main(int argc, char ** argv) {
    if (argc == 5) {
        test_native_frontend_with_real_models(argv[1], argv[2], argv[3], argv[4]);
        return 0;
    }
    test_resample_matches_stepaudio2_contract();
    test_s3_tokenizer_mel_contract();
    test_token2wav_prompt_mel_contract();
    test_campplus_fbank_contract();
    test_prompt_bundle_assembly_adds_lookahead_and_converts_mel_layout();
    test_prompt_bundle_assembly_rejects_shape_mismatch();
    test_frontend_output_shape_contracts_require_single_batch();
    test_load_wav_rejects_short_multichannel_frame();
    test_resample_rejects_unbounded_kernel_request();
    test_gguf_frontend_uses_both_model_files();
    if (argc == 4) {
        test_native_frontend_with_real_models(argv[1], argv[2], argv[3]);
    } else {
        assert(argc == 1);
    }
    return 0;
}
