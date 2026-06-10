// VoxCPM2 CLI demo — text-to-speech synthesis
//
// Usage:
//   ./voxcpm2-cli [options] <BaseLM.gguf> <Acoustic.gguf>
//
// Examples:
//   # Basic TTS
//   ./voxcpm2-cli -t "Hello, welcome to VoxCPM2." -o output.wav base.gguf acoustic.gguf
//
//   # Voice design
//   ./voxcpm2-cli -t "(A young woman, gentle voice)Hello there!" -o voice.wav base.gguf acoustic.gguf
//
//   # Voice cloning
//   ./voxcpm2-cli -t "This is a cloned voice." -r speaker.wav -o clone.wav base.gguf acoustic.gguf
//
//   # Streaming
//   ./voxcpm2-cli -t "Streaming test." --stream base.gguf acoustic.gguf

#include "voxcpm2_runtime.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct CliConfig {
    std::string base_lm_path;
    std::string acoustic_path;
    std::string text;
    std::string reference_wav_path;
    std::string prompt_wav_path;
    std::string prompt_text;
    std::string output_path = "output.wav";
    bool   use_gpu       = true;
    bool   streaming     = false;
    int    max_steps     = 200;
    int    timesteps     = 10;
    float  cfg_value     = 2.0f;
    float  temperature   = 1.0f;
    int    n_gpu_layers  = -1;
    int    seed          = 42;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "VoxCPM2 CLI — Text-to-Speech Synthesis\n"
        "\n"
        "Usage: %s [options] <BaseLM.gguf> <Acoustic.gguf>\n"
        "\n"
        "Required:\n"
        "  <BaseLM.gguf>        Path to VoxCPM2-BaseLM GGUF file\n"
        "  <Acoustic.gguf>      Path to VoxCPM2-Acoustic GGUF file\n"
        "\n"
        "Options:\n"
        "  -t, --text TEXT      Input text to synthesize\n"
        "  -o, --output PATH    Output WAV file path (default: output.wav)\n"
        "  -r, --reference PATH Reference WAV for voice cloning\n"
        "  --prompt-wav PATH    Prompt WAV for ultimate cloning\n"
        "  --prompt-text TEXT   Transcript of prompt WAV for ultimate cloning\n"
        "  --stream             Streaming mode (output chunks as they arrive)\n"
        "  --steps N            Max decode steps (default: 200)\n"
        "  --timesteps N        CFM inference timesteps (default: 10)\n"
        "  --cfg F              CFG guidance scale (default: 2.0)\n"
        "  --temperature F      Noise temperature (default: 1.0)\n"
        "  --seed N             Random seed (default: 42)\n"
        "  --cpu                Use CPU backend (default: GPU)\n"
        "  --n-gpu-layers N     Number of layers to offload to GPU (default: all)\n"
        "\n"
        "Examples:\n"
        "  # Basic TTS\n"
        "  %s -t \"Hello, welcome to VoxCPM2.\" base.gguf acoustic.gguf\n"
        "\n"
        "  # Voice design (describe voice in parentheses at start of text)\n"
        "  %s -t \"(A young woman, gentle voice)Hello there!\" base.gguf acoustic.gguf\n"
        "\n"
        "  # Voice cloning\n"
        "  %s -t \"Cloned voice text.\" -r speaker.wav base.gguf acoustic.gguf\n"
        "\n",
        prog, prog, prog, prog);
}

static bool parse_args(int argc, char ** argv, CliConfig & cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "-t" || arg == "--text") {
            if (++i < argc) cfg.text = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) cfg.output_path = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i < argc) cfg.reference_wav_path = argv[i];
        } else if (arg == "--prompt-wav") {
            if (++i < argc) cfg.prompt_wav_path = argv[i];
        } else if (arg == "--prompt-text") {
            if (++i < argc) cfg.prompt_text = argv[i];
        } else if (arg == "--stream") {
            cfg.streaming = true;
        } else if (arg == "--cpu") {
            cfg.use_gpu = false;
        } else if (arg == "--steps") {
            if (++i < argc) cfg.max_steps = std::atoi(argv[i]);
        } else if (arg == "--timesteps") {
            if (++i < argc) cfg.timesteps = std::atoi(argv[i]);
        } else if (arg == "--cfg") {
            if (++i < argc) cfg.cfg_value = static_cast<float>(std::atof(argv[i]));
        } else if (arg == "--temperature") {
            if (++i < argc) cfg.temperature = static_cast<float>(std::atof(argv[i]));
        } else if (arg == "--seed") {
            if (++i < argc) cfg.seed = std::atoi(argv[i]);
        } else if (arg == "--n-gpu-layers") {
            if (++i < argc) cfg.n_gpu_layers = std::atoi(argv[i]);
        } else if (cfg.base_lm_path.empty()) {
            cfg.base_lm_path = arg;
        } else if (cfg.acoustic_path.empty()) {
            cfg.acoustic_path = arg;
        }
    }

    if (cfg.base_lm_path.empty() || cfg.acoustic_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    if (cfg.text.empty()) {
        LOG_ERR("No input text specified. Use -t \"your text here\"\n");
        return false;
    }
    return true;
}

static void write_wav(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("Cannot create: %s\n", path.c_str());
        return;
    }
    const int32_t n_samples = static_cast<int32_t>(pcm.size());
    const int32_t byte_rate = sample_rate * 2;  // 16-bit mono
    const int32_t data_size = n_samples * 2;
    const int32_t chunk_size = 36 + data_size;

    auto w32 = [&](int32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto w16 = [&](int16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };

    f.write("RIFF", 4); w32(chunk_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sample_rate); w32(byte_rate); w16(2); w16(16);
    f.write("data", 4); w32(data_size);
    for (float v : pcm) {
        v = std::max(-1.0f, std::min(1.0f, v));
        w16(static_cast<int16_t>(v * 32767.0f));
    }
}

static std::vector<float> load_wav_mono(const std::string & path, int & out_sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("Cannot open: %s\n", path.c_str());
        return {};
    }
    char riff[4]; int32_t chunk_size; char wave[4];
    f.read(riff, 4); f.read(reinterpret_cast<char *>(&chunk_size), 4); f.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        LOG_ERR("Not a valid WAV file: %s\n", path.c_str());
        return {};
    }
    int16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    int32_t sample_rate = 0, data_size = 0;
    while (f) {
        char id[4]; int32_t size;
        f.read(id, 4); f.read(reinterpret_cast<char *>(&size), 4);
        if (!f) break;
        if (std::strncmp(id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char *>(&audio_format), 2);
            f.read(reinterpret_cast<char *>(&num_channels), 2);
            f.read(reinterpret_cast<char *>(&sample_rate), 4);
            int32_t byte_rate, block_align;
            f.read(reinterpret_cast<char *>(&byte_rate), 4);
            f.read(reinterpret_cast<char *>(&block_align), 2);
            f.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            if (size > 16) f.seekg(size - 16, std::ios::cur);
        } else if (std::strncmp(id, "data", 4) == 0) {
            data_size = size;
            break;
        } else {
            f.seekg(size, std::ios::cur);
        }
    }
    if (audio_format != 1 || data_size <= 0) {
        LOG_ERR("Unsupported WAV format: %s (fmt=%d, data=%d)\n", path.c_str(), audio_format, data_size);
        return {};
    }
    out_sample_rate = sample_rate;
    int n_samples = data_size / (bits_per_sample / 8);
    std::vector<int16_t> raw(n_samples);
    f.read(reinterpret_cast<char *>(raw.data()), data_size);

    std::vector<float> pcm(static_cast<size_t>(n_samples) / num_channels);
    for (int i = 0; i < n_samples / num_channels; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
            sum += static_cast<float>(raw[i * num_channels + ch]) / 32768.0f;
        }
        pcm[i] = sum / static_cast<float>(num_channels);
    }
    return pcm;
}

static void report_stats(const std::vector<float> & wav, int sample_rate, double elapsed_s) {
    double dur = static_cast<double>(wav.size()) / static_cast<double>(sample_rate);
    double rms = 0.0, peak = 0.0;
    for (float v : wav) {
        rms += static_cast<double>(v * v);
        peak = std::max(peak, static_cast<double>(std::fabs(v)));
    }
    rms = std::sqrt(rms / static_cast<double>(wav.size()));
    LOG_INF("Audio: %.2fs, %zu samples, %d Hz, RMS=%.3f, peak=%.3f\n",
            dur, wav.size(), sample_rate, rms, peak);
    LOG_INF("Elapsed: %.3fs, RTF=%.3f\n", elapsed_s, elapsed_s / dur);
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_time_init();

    CliConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    LOG_INF("VoxCPM2 CLI — Text-to-Speech Synthesis\n");
    LOG_INF("  BaseLM:   %s\n", cfg.base_lm_path.c_str());
    LOG_INF("  Acoustic: %s\n", cfg.acoustic_path.c_str());
    LOG_INF("  Text:     \"%s\"\n", cfg.text.c_str());
    LOG_INF("  Backend:  %s\n", cfg.use_gpu ? "GPU" : "CPU");
    LOG_INF("  Config:   steps=%d timesteps=%d cfg=%.1f temp=%.1f seed=%d\n",
            cfg.max_steps, cfg.timesteps, cfg.cfg_value, cfg.temperature, cfg.seed);

    VoxCPM2Runtime runtime;
    if (!runtime.init(cfg.base_lm_path, cfg.acoustic_path, cfg.n_gpu_layers, cfg.use_gpu)) {
        LOG_ERR("Failed to initialize VoxCPM2 runtime: %s\n", runtime.last_error().c_str());
        return 1;
    }

    VoxCPM2GenerateParams params;
    params.max_steps           = cfg.max_steps;
    params.inference_timesteps = cfg.timesteps;
    params.cfg_value           = cfg.cfg_value;
    params.temperature         = cfg.temperature;
    params.seed                = static_cast<uint32_t>(cfg.seed);

    std::vector<float> wav;
    auto t_start = std::chrono::steady_clock::now();

    if (!cfg.reference_wav_path.empty()) {
        // Voice cloning mode
        LOG_INF("Loading reference WAV: %s\n", cfg.reference_wav_path.c_str());
        int ref_sr = 0;
        std::vector<float> ref_wav = load_wav_mono(cfg.reference_wav_path, ref_sr);
        if (ref_wav.empty()) {
            LOG_ERR("Failed to load reference WAV\n");
            return 1;
        }
        LOG_INF("  Reference: %.2fs, %d Hz\n",
                static_cast<double>(ref_wav.size()) / ref_sr, ref_sr);

        params.reference_sample_rate = ref_sr;
        wav = runtime.generate_with_clone(cfg.text, ref_wav, params);
        if (wav.empty()) {
            LOG_ERR("Voice cloning failed: %s\n", runtime.last_error().c_str());
            return 1;
        }
    } else if (cfg.streaming) {
        // Streaming mode
        LOG_INF("Generating (streaming)...\n");
        std::vector<float> all_wav;
        runtime.generate_streaming(cfg.text,
            [&all_wav](const std::vector<float> & chunk, bool is_final) {
                all_wav.insert(all_wav.end(), chunk.begin(), chunk.end());
                LOG_INF("  Chunk: %zu samples%s\n", chunk.size(), is_final ? " (final)" : "");
            },
            params);
        wav = std::move(all_wav);
    } else {
        // Standard TTS / voice design
        LOG_INF("Generating...\n");
        wav = runtime.generate(cfg.text, params);
        if (wav.empty()) {
            LOG_ERR("Generation failed: %s\n", runtime.last_error().c_str());
            return 1;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    int sample_rate = runtime.sample_rate();
    report_stats(wav, sample_rate, elapsed);
    write_wav(cfg.output_path, wav, sample_rate);
    LOG_INF("WAV saved: %s\n", cfg.output_path.c_str());

    return 0;
}
