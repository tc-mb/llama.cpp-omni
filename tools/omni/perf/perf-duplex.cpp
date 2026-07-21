/**
 * 双工 (Duplex) 性能 / 可行性 Profiling
 *
 * 目标：在不侵入 omni.cpp 推理流程的前提下，复刻 test-duplex.cpp 的
 *       push_frame / wait_next_frame 流程，额外通过 *已存在* 的
 *       ctx_omni->audio_output_cb 钩子采集「每个 wav chunk 落盘时刻 + 时长」，
 *       最终把每帧 LLM 判定延迟 + 音频生成时间线写成 JSON，交给
 *       analyze_perf.py 产出「该机器能否支撑双工」的报告。
 *
 * 与 test-duplex.cpp 的唯一区别：
 *   1. 注册 audio_output_cb 采集音频时间线（零修改 omni.cpp）。
 *   2. 用统一的 session 时钟给所有事件打相对时间戳。
 *   3. 把 frame 结果 + audio chunk 时间线序列化成 JSON。
 *
 * 判据参考（详见 DUPLEX_PROFILING.md）：
 *   - 每帧 push→判定 (ms_total) 的 P95 < 进帧间隔（默认 1000ms），否则跟不上实时进帧。
 *   - SPEAK 轮次首个 wav 的 push→落盘（首响）尽量 < 1000ms。
 *   - 音频 RTF < 1.0（生成 1s 音频耗时 < 1s），否则播放会饿死。
 */

#include "omni-impl.h"
#include "omni.h"

#include "arg.h"
#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_interrupted = false;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_interrupted) _exit(1);
        g_is_interrupted = true;
    }
}
#endif

// ==================== session 时钟 ====================
// 所有事件（push / frame done / audio chunk）都用相对于 g_t0 的毫秒数，
// 方便 Python 端在同一条时间线上对齐。

static std::chrono::high_resolution_clock::time_point g_t0;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - g_t0).count();
}

// ==================== 采集到的数据 ====================

struct AudioChunkRecord {
    double t_complete_ms = 0;  // 相对 session 时钟：该 wav chunk 落盘/回调时刻
    int    n_samples     = 0;
    int    sample_rate   = 0;
    double duration_s    = 0;  // n_samples / sample_rate
    bool   is_final      = false;
};

struct FrameRecord {
    int64_t user_seq      = 0;
    int64_t frame_id      = -1;
    double  t_push_ms     = 0;  // 相对 session 时钟：push_frame 时刻
    double  t_done_ms     = 0;  // 相对 session 时钟：result 出队时刻
    bool    ok            = false;
    bool    is_speak      = false;
    double  ms_decode     = 0;
    double  ms_total      = 0;
    int     n_past_after  = 0;
    int     text_len      = 0;
    std::string text;           // 仅取前 80 字节，JSON 转义
};

struct PerfCollector {
    std::mutex                    mtx;
    std::vector<AudioChunkRecord> audio;          // audio_output_cb 采集
    std::unordered_map<int64_t, double> push_ms;  // frame_id -> push 时刻
    std::vector<FrameRecord>      frames;
};

static PerfCollector g_perf;

// ==================== 模型路径解析（复用 test-duplex 逻辑） ====================

struct TestModelPaths {
    std::string llm, vision, audio, tts, projector, vision_coreml, base_dir;
};

static std::string get_parent_dir(const std::string & path) {
    size_t last_slash = path.find_last_of("/\\");
    return (last_slash != std::string::npos) ? path.substr(0, last_slash) : ".";
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

static TestModelPaths resolve_model_paths(const std::string & llm_path) {
    TestModelPaths p;
    p.llm = llm_path;
    p.base_dir = get_parent_dir(llm_path);
    p.vision        = p.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    p.audio         = p.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    p.tts           = p.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    p.projector     = p.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    p.vision_coreml = p.base_dir + "/vision/coreml_minicpmo45_vit_all_f16.mlmodelc";
    return p;
}

// ==================== JSON 序列化（手写，无外部依赖） ====================

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

static bool write_json_report(const std::string & path,
                              int    interval_ms,
                              int    n_threads,
                              int    sample_rate_hint,
                              const std::string & llm_path,
                              const std::string & vision_backend,
                              bool   use_tts,
                              int    media_type) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[perf] 无法写 JSON: %s\n", path.c_str());
        return false;
    }

    std::lock_guard<std::mutex> lk(g_perf.mtx);

    fprintf(f, "{\n");
    fprintf(f, "  \"meta\": {\n");
    fprintf(f, "    \"stream_interval_ms\": %d,\n", interval_ms);
    fprintf(f, "    \"n_threads\": %d,\n", n_threads);
    fprintf(f, "    \"sample_rate_hint\": %d,\n", sample_rate_hint);
    fprintf(f, "    \"llm_path\": \"%s\",\n", json_escape(llm_path).c_str());
    fprintf(f, "    \"vision_backend\": \"%s\",\n", json_escape(vision_backend).c_str());
    fprintf(f, "    \"use_tts\": %s,\n", use_tts ? "true" : "false");
    fprintf(f, "    \"media_type\": %d\n", media_type);
    fprintf(f, "  },\n");

    // frames
    fprintf(f, "  \"frames\": [\n");
    for (size_t i = 0; i < g_perf.frames.size(); ++i) {
        const auto & fr = g_perf.frames[i];
        fprintf(f,
            "    {\"user_seq\": %lld, \"frame_id\": %lld, \"t_push_ms\": %.3f, "
            "\"t_done_ms\": %.3f, \"ok\": %s, \"is_speak\": %s, \"ms_decode\": %.3f, "
            "\"ms_total\": %.3f, \"n_past_after\": %d, \"text_len\": %d, \"text\": \"%s\"}%s\n",
            (long long)fr.user_seq, (long long)fr.frame_id, fr.t_push_ms,
            fr.t_done_ms, fr.ok ? "true" : "false", fr.is_speak ? "true" : "false",
            fr.ms_decode, fr.ms_total, fr.n_past_after, fr.text_len,
            json_escape(fr.text).c_str(),
            (i + 1 < g_perf.frames.size()) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // audio chunks
    fprintf(f, "  \"audio_chunks\": [\n");
    for (size_t i = 0; i < g_perf.audio.size(); ++i) {
        const auto & a = g_perf.audio[i];
        fprintf(f,
            "    {\"t_complete_ms\": %.3f, \"n_samples\": %d, \"sample_rate\": %d, "
            "\"duration_s\": %.4f, \"is_final\": %s}%s\n",
            a.t_complete_ms, a.n_samples, a.sample_rate, a.duration_s,
            a.is_final ? "true" : "false",
            (i + 1 < g_perf.audio.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return true;
}

// ==================== 双工 profiling 核心 ====================

static void duplex_perf_run(struct omni_context * ctx_omni,
                            const std::string & data_path_prefix,
                            int cnt,
                            int stream_interval_ms) {
    printf("\n=== Duplex perf: %d chunks, interval=%dms ===\n", cnt, stream_interval_ms);

    std::vector<OmniDuplexFrame> frames(cnt);
    for (int il = 0; il < cnt; ++il) {
        char idx[16]; snprintf(idx, sizeof(idx), "%04d", il);
        frames[il].aud_fname = data_path_prefix + idx + ".wav";
        std::string img = data_path_prefix + idx + ".jpg";
        if (file_exists(img)) frames[il].img_fname = img;
        frames[il].user_seq = il + 1;
        if (!file_exists(frames[il].aud_fname)) {
            fprintf(stderr, "[错误] 音频不存在: %s\n", frames[il].aud_fname.c_str());
            return;
        }
    }

    if (!omni_duplex_session_begin(ctx_omni, /*voice_audio=*/"", /*debug_dir=*/"./")) {
        fprintf(stderr, "[错误] omni_duplex_session_begin failed\n");
        return;
    }

    // session 时钟从这里起算
    g_t0 = std::chrono::high_resolution_clock::now();

    std::thread producer([&]() {
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int il = 0; il < cnt && !g_is_interrupted; ++il) {
            if (stream_interval_ms > 0) {
                std::this_thread::sleep_until(
                    t_start + std::chrono::milliseconds((int64_t)il * stream_interval_ms));
            }
            double t_push = now_ms();
            int64_t fid = omni_duplex_push_frame(ctx_omni, frames[il]);
            if (fid < 0) {
                fprintf(stderr, "[push] frame %d 提交失败\n", il + 1);
                break;
            }
            std::lock_guard<std::mutex> lk(g_perf.mtx);
            g_perf.push_ms[fid] = t_push;
        }
    });

    int speak = 0, listen = 0, completed = 0;
    double sum_decode = 0, sum_e2e = 0;

    for (int il = 0; il < cnt && !g_is_interrupted; ++il) {
        OmniDuplexFrameResult r;
        if (!omni_duplex_wait_next_frame(ctx_omni, &r, /*timeout_ms=*/30000) || !r.ok) {
            fprintf(stderr, "[错误] frame %d 处理失败/超时\n", il + 1);
            break;
        }
        (r.is_speak ? speak : listen)++;
        sum_decode += r.ms_decode;
        sum_e2e    += r.ms_total;
        completed++;

        FrameRecord fr;
        fr.user_seq     = r.user_seq;
        fr.frame_id     = r.frame_id;
        fr.t_done_ms    = now_ms();
        fr.ok           = r.ok;
        fr.is_speak     = r.is_speak;
        fr.ms_decode    = r.ms_decode;
        fr.ms_total     = r.ms_total;
        fr.n_past_after = r.n_past_after;
        fr.text_len     = (int)r.text.size();
        fr.text         = r.text.size() > 80 ? r.text.substr(0, 80) : r.text;
        {
            std::lock_guard<std::mutex> lk(g_perf.mtx);
            auto it = g_perf.push_ms.find(r.frame_id);
            fr.t_push_ms = (it != g_perf.push_ms.end()) ? it->second : 0.0;
            g_perf.frames.push_back(std::move(fr));
        }

        printf("--- Chunk %lld/%d --- decode %.1fms | e2e %.1fms | n_past %d | %s\n",
               (long long)r.user_seq, cnt, r.ms_decode, r.ms_total, r.n_past_after,
               r.is_speak ? "<|speak|>" : "<|listen|>");
    }

    if (producer.joinable()) producer.join();
    omni_duplex_session_end(ctx_omni);

    printf("\n=== Summary: %d/%d chunks | avg decode %.1fms | avg e2e %.1fms | speak %d listen %d ===\n",
           completed, cnt,
           completed ? sum_decode / completed : 0,
           completed ? sum_e2e    / completed : 0,
           speak, listen);
}

// ==================== 帮助信息 ====================

static void show_usage(const char * prog_name) {
    printf(
        "MiniCPM-o Duplex Perf / Feasibility Profiler\n\n"
        "Usage: %s -m <llm_model_path> [options]\n\n"
        "Required:\n"
        "  -m <path>           LLM 模型路径\n\n"
        "Options:\n"
        "  --vision <path>     覆盖 vision 模型路径\n"
        "  --audio <path>      覆盖 audio 模型路径\n"
        "  --tts <path>        覆盖 TTS 模型路径\n"
        "  --projector <path>  覆盖 projector 模型路径\n"
        "  --ref-audio <path>  参考音频路径\n"
        "  -c, --ctx-size <n>  上下文大小 (默认: 4096)\n"
        "  -ngl <n>            GPU 层数 (默认: 99)\n"
        "  --no-tts            禁用 TTS\n"
        "  --omni              启用 omni 模式 (audio+vision)；默认已开\n"
        "  --vision-backend <m>  'metal'(默认) 或 'coreml'(ANE)\n"
        "  --vision-coreml <p>   CoreML/ANE 模型路径 (.mlmodelc)\n"
        "  --test <prefix> <n> 指定测试数据前缀和 chunk 数量\n"
        "                      (默认: duplex_omni_test_case 共 36 帧)\n"
        "  --stream-interval <ms>  push frame 的最小间隔 (默认 1000=模拟真实流式)\n"
        "  -o <dir>            输出目录 (默认: ./tools/omni/output)\n"
        "  --out-json <path>   性能报告 JSON 路径 (默认: <out>/perf_report.json)\n"
        "  -h, --help          显示帮助\n\n"
        "Example (全部用默认硬编码样本/参考音频):\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf\n",
        prog_name, prog_name
    );
}

// ==================== Main ====================

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string llm_path;
    std::string vision_path_override, audio_path_override, tts_path_override, projector_path_override;
    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    std::string output_dir = "./tools/omni/output";
    std::string out_json;
    std::string vision_backend = "metal";
    std::string vision_coreml_model_path;
    int n_ctx = 4096;
    int n_gpu_layers = 99;
    int media_type = 2;     // 1=audio only, 2=omni；默认 omni（硬编码样本含图像）
    bool use_tts = true;
    bool run_test = false;
    int  stream_interval_ms = 1000;  // 默认模拟真实 1s 流式输入
    std::string test_prefix;
    int test_count = 0;
    std::string token2wav_device = "gpu";
    if (const char * v = std::getenv("OMNI_T2W_DEVICE")) {
        if (*v) token2wav_device = v;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { show_usage(argv[0]); return 0; }
        else if (arg == "-m" && i + 1 < argc) { llm_path = argv[++i]; }
        else if (arg == "--vision" && i + 1 < argc) { vision_path_override = argv[++i]; }
        else if (arg == "--audio" && i + 1 < argc) { audio_path_override = argv[++i]; }
        else if (arg == "--tts" && i + 1 < argc) { tts_path_override = argv[++i]; }
        else if (arg == "--projector" && i + 1 < argc) { projector_path_override = argv[++i]; }
        else if (arg == "--ref-audio" && i + 1 < argc) { ref_audio_path = argv[++i]; }
        else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) { n_ctx = std::atoi(argv[++i]); }
        else if (arg == "-ngl" && i + 1 < argc) { n_gpu_layers = std::atoi(argv[++i]); }
        else if (arg == "--no-tts") { use_tts = false; }
        else if (arg == "--omni") { media_type = 2; }
        else if (arg == "--vision-backend" && i + 1 < argc) {
            vision_backend = argv[++i];
            if (vision_backend != "metal" && vision_backend != "coreml") {
                fprintf(stderr, "Error: --vision-backend must be 'metal' or 'coreml'\n");
                return 1;
            }
        }
        else if (arg == "--vision-coreml" && i + 1 < argc) {
            vision_coreml_model_path = argv[++i];
            vision_backend = "coreml";
        }
        else if (arg == "-o" && i + 1 < argc) { output_dir = argv[++i]; }
        else if (arg == "--out-json" && i + 1 < argc) { out_json = argv[++i]; }
        else if (arg == "--test" && i + 2 < argc) {
            run_test = true;
            test_prefix = argv[++i];
            test_count = std::atoi(argv[++i]);
        }
        else if (arg == "--stream-interval" && i + 1 < argc) {
            stream_interval_ms = std::atoi(argv[++i]);
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            show_usage(argv[0]);
            return 1;
        }
    }

    if (llm_path.empty()) {
        fprintf(stderr, "Error: -m <llm_model_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#endif

    TestModelPaths paths = resolve_model_paths(llm_path);
    if (!vision_path_override.empty()) paths.vision = vision_path_override;
    if (!audio_path_override.empty()) paths.audio = audio_path_override;
    if (!tts_path_override.empty()) paths.tts = tts_path_override;
    if (!projector_path_override.empty()) paths.projector = projector_path_override;
    if (!vision_coreml_model_path.empty()) paths.vision_coreml = vision_coreml_model_path;

    if (out_json.empty()) out_json = output_dir + "/perf_report.json";

    printf("=== Duplex Perf Profiler ===\n");
    printf("  LLM:    %s\n", paths.llm.c_str());
    printf("  Vision: %s (backend=%s)\n", paths.vision.c_str(), vision_backend.c_str());
    printf("  Audio:  %s\n", paths.audio.c_str());
    printf("  TTS:    %s (use_tts=%d)\n", paths.tts.c_str(), use_tts ? 1 : 0);
    printf("  Interval: %dms | out-json: %s\n", stream_interval_ms, out_json.c_str());

    if (!file_exists(paths.llm))   { fprintf(stderr, "Error: LLM not found\n");   return 1; }
    if (!file_exists(paths.audio)) { fprintf(stderr, "Error: Audio not found\n"); return 1; }
    if (use_tts && !file_exists(paths.tts)) {
        fprintf(stderr, "Warning: TTS not found, disabling TTS\n");
        use_tts = false;
    }

    common_params params;
    params.model.path   = paths.llm;
    params.vpm_model    = paths.vision;
    params.apm_model    = paths.audio;
    params.tts_model    = paths.tts;
    params.n_ctx        = n_ctx;
    params.n_gpu_layers = n_gpu_layers;
    if (vision_backend == "coreml") {
        params.vision_coreml_model_path = paths.vision_coreml;
    }

    {
        const char * seed_env = std::getenv("OMNI_SAMPLER_SEED");
        uint32_t     seed     = seed_env ? (uint32_t) std::strtoul(seed_env, nullptr, 10) : 42u;
        params.sampling.seed  = seed;
    }

    std::string tts_bin_dir = get_parent_dir(paths.tts);

    common_init();

    auto ctx_omni = omni_init(&params, media_type, use_tts, tts_bin_dir,
                              /*tts_gpu_layers=*/-1, /*token2wav_device=*/token2wav_device,
                              /*duplex_mode=*/true,
                              /*existing_model=*/nullptr, /*existing_ctx=*/nullptr,
                              /*base_output_dir=*/output_dir);
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: Failed to initialize omni context\n");
        return 1;
    }
    ctx_omni->async = true;
    ctx_omni->ref_audio_path = ref_audio_path;

    // 🔑 [低侵入核心] 注册已存在的 audio_output_cb 钩子采集 wav chunk 时间线。
    //    T2W 线程每写完一个 wav chunk 就会调用它（cpp / python 两条 T2W 路径都会调）。
    ctx_omni->audio_output_cb =
        [](const float * /*samples*/, int n_samples, int sample_rate, bool is_final) {
            AudioChunkRecord rec;
            rec.t_complete_ms = now_ms();
            rec.n_samples     = n_samples;
            rec.sample_rate   = sample_rate;
            rec.duration_s    = (sample_rate > 0) ? (double)n_samples / sample_rate : 0.0;
            rec.is_final      = is_final;
            std::lock_guard<std::mutex> lk(g_perf.mtx);
            g_perf.audio.push_back(rec);
        };

    // 硬编码默认样本：duplex omni 测试集（audio + image，共 36 帧 0000~0035）
    const std::string default_prefix =
        "tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_";
    const int default_count = 36;
    duplex_perf_run(ctx_omni,
                    run_test ? test_prefix : default_prefix,
                    run_test ? test_count  : default_count,
                    stream_interval_ms);

    // 等所有 speak 帧的 audio 落盘（audio_output_cb 也会在这期间继续触发）。
    omni_duplex_drain_tts_audio(ctx_omni);

    // 写报告（在 omni_free 之前，确保 audio_output_cb 已不会再被调用）。
    int sr_hint = g_perf.audio.empty() ? 24000 : g_perf.audio.front().sample_rate;
    write_json_report(out_json, stream_interval_ms, params.cpuparams.n_threads,
                      sr_hint, paths.llm, vision_backend, use_tts, media_type);
    printf("\n[perf] JSON 报告已写入: %s\n", out_json.c_str());
    printf("[perf] 运行 analyze_perf.py 生成可读报告与可行性判定。\n");

    llama_perf_context_print(ctx_omni->ctx_llama);
    omni_free(ctx_omni);

    printf("\n=== Duplex perf finished ===\n");
    return 0;
}
