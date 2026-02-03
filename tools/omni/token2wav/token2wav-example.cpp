#include "token2wav-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {


bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return (bool) f;
}

// pcm转wav
bool write_wav_mono_i16(const std::string & path, const std::vector<float> & wave_bt, int sample_rate) {
    const int16_t num_channels    = 1;
    const int16_t bits_per_sample = 16;
    const int16_t block_align     = num_channels * (bits_per_sample / 8);
    const int32_t byte_rate       = sample_rate * block_align;

    std::vector<int16_t> pcm((size_t) wave_bt.size());
    for (size_t i = 0; i < wave_bt.size(); ++i) {
        float x = wave_bt[i];
        if (!std::isfinite(x)) {
            x = 0.0f;
        }
        x = std::max(-1.0f, std::min(1.0f, x));
        const float y = x * 32767.0f;
        if (y >= 32767.0f) {
            pcm[i] = (int16_t) 32767;
        } else if (y <= -32768.0f) {
            pcm[i] = (int16_t) -32768;
        } else {
            pcm[i] = (int16_t) (y);
        }
    }

    const uint32_t data_bytes = (uint32_t) (pcm.size() * sizeof(int16_t));
    const uint32_t riff_size  = 36u + data_bytes;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char *>(&riff_size), 4);
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    const uint32_t fmt_size     = 16;
    const uint16_t audio_format = 1;  // PCM 格式
    out.write(reinterpret_cast<const char *>(&fmt_size), 4);
    out.write(reinterpret_cast<const char *>(&audio_format), 2);
    out.write(reinterpret_cast<const char *>(&num_channels), 2);
    out.write(reinterpret_cast<const char *>(&sample_rate), 4);
    out.write(reinterpret_cast<const char *>(&byte_rate), 4);
    out.write(reinterpret_cast<const char *>(&block_align), 2);
    out.write(reinterpret_cast<const char *>(&bits_per_sample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char *>(&data_bytes), 4);
    out.write(reinterpret_cast<const char *>(pcm.data()), (std::streamsize) data_bytes);
    return true;
}

bool ensure_dir(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(dir), ec);
    return !ec;
}

std::string parent_dir_of(const std::string & path) {
    const auto p = std::filesystem::path(path).parent_path();
    return p.empty() ? std::string() : p.string();
}

}  // namespace

int main() {
    using clock = std::chrono::steady_clock;
    const auto t_program0 = clock::now();

    // token2wav-example：主要关注初始化和两种送入输出方式即可
    // 目前是使用读取prompt_cache.gguf的方式初始化然后以call的方式来流式输出pcm并转换为wav

    // 默认路径，根据5个gguf和两个输出位置改动
    std::string model_dir = "./tools/omni/convert/gguf/token2wav-gguf";
    std::string encoder_gguf = model_dir + "/encoder.gguf";
    std::string flow_matching_gguf = model_dir + "/flow_matching.gguf";
    std::string flow_extra_gguf = model_dir + "/flow_extra.gguf";
    std::string vocoder_gguf = model_dir + "/hifigan2.gguf";
    std::string prompt_cache_gguf = model_dir + "/prompt_cache.gguf";

    std::string out_wav = "/tmp/token2wav_example_stream.wav";
    std::string out_chunk_wav_dir = "/tmp/token2wav_example_chunks";

    std::string device_token2mel = "gpu";
    std::string device_vocoder   = "gpu";

    int       n_timesteps = 10;
    float     temperature = 1.0f;
    const int sr          = omni::flow::Token2Wav::kSampleRate;

    {
        const std::string out_parent = parent_dir_of(out_wav);
        if (!out_parent.empty() && !ensure_dir(out_parent)) {
            std::fprintf(stderr, "failed to create out_wav parent dir: %s\n", out_parent.c_str());
            return 2;
        }
        if (!out_chunk_wav_dir.empty() && !ensure_dir(out_chunk_wav_dir)) {
            std::fprintf(stderr, "failed to create out_chunk_wav_dir: %s\n", out_chunk_wav_dir.c_str());
            return 2;
        }
    }

    if (!file_exists(prompt_cache_gguf)) {
        std::fprintf(stderr, "prompt_cache.gguf not found: %s\n", prompt_cache_gguf.c_str());
        return 2;
    }

    // 例子 token
    const std::vector<int32_t> tokens = {
        1493, 4299, 4218, 2049, 528,  2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300,
        226,  2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725,
        6048, 3816, 6058, 958,  63,   4460, 5914, 2379, 735,  5319, 4593, 2328, 890,  35,   751,  1483,
        1484, 1483, 2112, 303,  4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860,
        2031, 414,  4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831,
        5084, 5027, 4946, 4946, 2678, 575,  575,  521,  518,  638,  1367, 2804, 3402, 4299,
    };

    omni::flow::Token2WavSession sess;
    // 初始化：加载 encoder/flow/vocoder 模型，导入 prompt_cache用于初始化
    const auto t_init0 = clock::now();
    if (!sess.init_from_prompt_cache_gguf(encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                                          vocoder_gguf, device_token2mel, device_vocoder, n_timesteps, temperature)) {
        std::fprintf(stderr, "init_from_prompt_cache_gguf failed\n");
        return 3;
    }
    const auto t_init1 = clock::now();

    constexpr int32_t step_valid  = 25;
    constexpr int32_t chunk_total = 28;
    int64_t           pos         = 0;
    const int64_t     n           = (int64_t) tokens.size();

    std::vector<float> wave_all;

    int call_id = 0;
    const auto t_infer0 = clock::now();
    while (pos + chunk_total <= n) {
        // 滑窗规则：每次取 28 个 token（25 主要内容 + 3 lookahead），下一次 pos += 25
        std::vector<int32_t> win(tokens.begin() + pos, tokens.begin() + pos + chunk_total);
        // callback 推流：调用 Token2WavSession::feed_window(callback形式)，一窗推理完成后立刻把音频分块回调
        // 需要注意pcm 指针只在回调执行期间有效
        if (!sess.feed_window(win, false, [&](const float * pcm, int64_t n_samples) {
                wave_all.insert(wave_all.end(), pcm, pcm + n_samples);
                if (!out_chunk_wav_dir.empty()) {
                    const std::string  chunk_path = out_chunk_wav_dir + "/call" + std::to_string(call_id) + ".wav";
                    std::vector<float> tmp(pcm, pcm + n_samples);
                    if (!write_wav_mono_i16(chunk_path, tmp, sr)) {
                        std::fprintf(stderr, "failed to write chunk wav: %s\n", chunk_path.c_str());
                    }
                }
            })) {
            std::fprintf(stderr, "feed_window failed\n");
            return 4;
        }
        pos += step_valid;
        call_id++;
    }

    {
        // final flush：最后一段不足 28 时也要调用一次，然后传is_final=true 把剩下的缓存吐干净
        std::vector<int32_t> tail;
        if (pos < n) {
            tail.assign(tokens.begin() + pos, tokens.end());
        }
        if (!sess.feed_window(tail, true, [&](const float * pcm, int64_t n_samples) {
                wave_all.insert(wave_all.end(), pcm, pcm + n_samples);
                if (!out_chunk_wav_dir.empty()) {
                    const std::string  chunk_path = out_chunk_wav_dir + "/call" + std::to_string(call_id) + ".wav";
                    std::vector<float> tmp(pcm, pcm + n_samples);
                    if (!write_wav_mono_i16(chunk_path, tmp, sr)) {
                        std::fprintf(stderr, "failed to write chunk wav: %s\n", chunk_path.c_str());
                    }
                }
            })) {
            std::fprintf(stderr, "feed_window(final) failed\n");
            return 4;
        }
        call_id++;
    }
    const auto t_infer1 = clock::now();

    const auto t_write0 = clock::now();
    if (!write_wav_mono_i16(out_wav, wave_all, sr)) {
        std::fprintf(stderr, "failed to write wav: %s\n", out_wav.c_str());
        return 5;
    }
    const auto t_write1 = clock::now();

    {
        const double init_ms  = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
        const double infer_ms = std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count();
        const double write_ms = std::chrono::duration<double, std::milli>(t_write1 - t_write0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_program0).count();
        std::fprintf(stderr,
                     "[timing-total] init=%.3fms infer=%.3fms write=%.3fms total=%.3fms\n",
                     init_ms, infer_ms, write_ms, total_ms);
    }

    std::printf("[done] out_wav=%s sr=%d total_samples=%zu n_calls=%d\n", out_wav.c_str(), sr, wave_all.size(),
                call_id);
    if (!out_chunk_wav_dir.empty()) {
        std::printf("[done] out_chunk_wav_dir=%s\n", out_chunk_wav_dir.c_str());
    }
    return 0;
}


