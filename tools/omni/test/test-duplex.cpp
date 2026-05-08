/**
 * 双工 (Duplex) Omni 模式测试
 *
 * 这份测试脚本的职责仅有：
 *   1. 解析命令行 / 模型路径，调 omni_init
 *   2. 调用 omni_duplex_session_begin / push_frame / wait_next_frame / session_end
 *   3. 打印每帧结果与汇总统计
 *
 * 所有线程调度（VPM/APM 编码、LLM prefill/decode 流水线、TTS/T2W 落盘）
 * 都封装在 omni 内部。同一组 API 可被 server / cli 复用。
 */

#include "omni-impl.h"
#include "omni.h"

#include "arg.h"
#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <thread>

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
        if (g_is_interrupted) {
            _exit(1);
        }
        g_is_interrupted = true;
    }
}
#endif

// ==================== 模型路径解析（复用 cli 逻辑） ====================

struct TestModelPaths {
    std::string llm;
    std::string vision;
    std::string audio;
    std::string tts;
    std::string projector;
    std::string vision_coreml;  // CoreML/ANE mlmodelc 路径，可选
    std::string base_dir;
};

static std::string get_parent_dir(const std::string & path) {
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return path.substr(0, last_slash);
    }
    return ".";
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

static TestModelPaths resolve_model_paths(const std::string & llm_path) {
    TestModelPaths paths;
    paths.llm = llm_path;
    paths.base_dir = get_parent_dir(llm_path);
    paths.vision = paths.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    paths.audio = paths.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    paths.tts = paths.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    paths.projector = paths.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    paths.vision_coreml = paths.base_dir + "/vision/coreml_minicpmo45_vit_all_f16.mlmodelc";
    return paths;
}

// ==================== 双工测试核心 ====================

static void duplex_test_case(struct omni_context * ctx_omni,
                             const std::string & data_path_prefix,
                             int cnt,
                             int stream_interval_ms) {
    printf("\n=== Duplex test: %d chunks, interval=%dms ===\n", cnt, stream_interval_ms);

    // 准备 N 个 chunk 的 (audio, image) 文件路径
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

    auto total_t0 = std::chrono::high_resolution_clock::now();

    // push 节奏 = stream_interval_ms。push 本身非阻塞（除非内部队列满），
    // 因此即便 LLM 处理慢于 push 间隔，pipeline 仍能把后续帧排队等候。
    std::thread producer([&]() {
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int il = 0; il < cnt && !g_is_interrupted; ++il) {
            if (stream_interval_ms > 0) {
                std::this_thread::sleep_until(
                    t_start + std::chrono::milliseconds((int64_t)il * stream_interval_ms));
            }
            if (omni_duplex_push_frame(ctx_omni, frames[il]) < 0) {
                fprintf(stderr, "[push] frame %d 提交失败\n", il + 1);
                break;
            }
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

        printf("--- Chunk %lld/%d --- decode %.1fms | e2e %.1fms | n_past %d | %s\n",
               (long long)r.user_seq, cnt, r.ms_decode, r.ms_total, r.n_past_after,
               r.is_speak ? ("<|speak|> \"" + (r.text.size() > 60 ? r.text.substr(0,60)+"..." : r.text) + "\"").c_str()
                          : "<|listen|>");
    }

    if (producer.joinable()) producer.join();
    omni_duplex_session_end(ctx_omni);

    double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - total_t0).count();
    printf("\n=== Summary: %d/%d chunks, %.3fs | avg decode %.1fms | avg e2e %.1fms | speak %d listen %d ===\n",
           completed, cnt, total_s,
           completed ? sum_decode / completed : 0,
           completed ? sum_e2e    / completed : 0,
           speak, listen);
}

// ==================== 帮助信息 ====================

static void show_usage(const char * prog_name) {
    printf(
        "MiniCPM-o Duplex Mode Test\n\n"
        "Usage: %s -m <llm_model_path> [options]\n\n"
        "Required:\n"
        "  -m <path>           LLM 模型路径\n\n"
        "Options:\n"
        "  --vision <path>     覆盖 vision 模型路径\n"
        "  --audio <path>      覆盖 audio 模型路径\n"
        "  --tts <path>        覆盖 TTS 模型路径\n"
        "  --projector <path>  覆盖 projector 模型路径\n"
        "  --ref-audio <path>  参考音频路径 (默认: tools/omni/assets/default_ref_audio/default_ref_audio.wav)\n"
        "  -c, --ctx-size <n>  上下文大小 (默认: 4096)\n"
        "  -ngl <n>            GPU 层数 (默认: 99)\n"
        "  --no-tts            禁用 TTS\n"
        "  --omni              启用 omni 模式 (audio+vision, media_type=2)\n"
        "  --vision-backend <m>  Vision compute backend: 'metal'(默认) 或 'coreml'(ANE)\n"
        "  --vision-coreml <p>   CoreML/ANE 模型路径 (.mlmodelc)；--vision-backend=coreml 时\n"
        "                        若未指定则默认 <llm 同级目录>/vision/coreml_minicpmo45_vit_all_f16.mlmodelc\n"
        "  --test <prefix> <n> 指定测试数据前缀和 chunk 数量\n"
        "  --stream-interval <ms>  push frame 的最小间隔 (默认 0=背靠背压测；\n"
        "                          设为 1000 模拟真实 MiniCPM-o 流式输入)\n"
        "  -o <dir>            输出目录 (默认: ./tools/omni/output)\n"
        "  -h, --help          显示帮助\n\n"
        "Example:\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \\\n"
        "     --omni --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9\n",
        prog_name, prog_name
    );
}

// ==================== Main ====================

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string llm_path;
    std::string vision_path_override;
    std::string audio_path_override;
    std::string tts_path_override;
    std::string projector_path_override;
    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    std::string output_dir = "./tools/omni/output";
    std::string vision_backend = "metal";  // 'metal' or 'coreml'
    std::string vision_coreml_model_path;  // 可选；若为空且 backend=coreml 则用默认推断路径
    int n_ctx = 4096;
    int n_gpu_layers = 99;
    int media_type = 1;     // 1=audio only, 2=omni (audio+vision)
    bool use_tts = true;
    bool run_test = false;
    int  stream_interval_ms = 0;  // 0 = 背靠背（压测）；真实流式建议 1000
    std::string test_prefix;
    int test_count = 0;
    std::string token2wav_device = "gpu";
    if (const char * v = std::getenv("OMNI_T2W_DEVICE")) {
        if (*v) token2wav_device = v;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        }
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
                fprintf(stderr, "Error: --vision-backend must be 'metal' or 'coreml', got '%s'\n", vision_backend.c_str());
                return 1;
            }
        }
        else if (arg == "--vision-coreml" && i + 1 < argc) {
            vision_coreml_model_path = argv[++i];
            // 显式指定路径时自动开 coreml 后端
            vision_backend = "coreml";
        }
        else if (arg == "-o" && i + 1 < argc) { output_dir = argv[++i]; }
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

    // 解析模型路径
    TestModelPaths paths = resolve_model_paths(llm_path);
    if (!vision_path_override.empty()) paths.vision = vision_path_override;
    if (!audio_path_override.empty()) paths.audio = audio_path_override;
    if (!tts_path_override.empty()) paths.tts = tts_path_override;
    if (!projector_path_override.empty()) paths.projector = projector_path_override;
    // CoreML 路径：用户显式指定 > resolve_model_paths 默认推断
    if (!vision_coreml_model_path.empty()) paths.vision_coreml = vision_coreml_model_path;

    printf("=== Duplex Test ===\n");
    printf("  LLM:    %s\n", paths.llm.c_str());
    printf("  Vision: %s (backend=%s)\n", paths.vision.c_str(), vision_backend.c_str());
    printf("  Audio:  %s\n", paths.audio.c_str());
    printf("  TTS:    %s (use_tts=%d)\n", paths.tts.c_str(), use_tts ? 1 : 0);
    if (vision_backend == "coreml") {
        struct stat st;
        bool ok = (stat(paths.vision_coreml.c_str(), &st) == 0);
        printf("  CoreML: %s %s\n", paths.vision_coreml.c_str(), ok ? "[OK]" : "[NOT FOUND]");
    }

    if (!file_exists(paths.llm))   { fprintf(stderr, "Error: LLM not found\n");   return 1; }
    if (!file_exists(paths.audio)) { fprintf(stderr, "Error: Audio not found\n"); return 1; }
    if (use_tts && !file_exists(paths.tts)) {
        fprintf(stderr, "Warning: TTS not found, disabling TTS\n");
        use_tts = false;
    }

    // 设置参数
    common_params params;
    params.model.path = paths.llm;
    params.vpm_model = paths.vision;
    params.apm_model = paths.audio;
    params.tts_model = paths.tts;
    params.n_ctx = n_ctx;
    params.n_gpu_layers = n_gpu_layers;
    if (vision_backend == "coreml") {
        params.vision_coreml_model_path = paths.vision_coreml;
    }

    // 🔧 [bit-exact A/B] 固定 LLM / TTS 采样种子，配合 token2wav 中已有的 mt19937(42)
    // 让两次独立进程产生完全相同的输出（A/B 对比可用 md5 验证）
    {
        const char * seed_env = std::getenv("OMNI_SAMPLER_SEED");
        uint32_t     seed     = seed_env ? (uint32_t) std::strtoul(seed_env, nullptr, 10) : 42u;
        params.sampling.seed  = seed;
        printf("  Sampler seed: %u (env OMNI_SAMPLER_SEED to override)\n", seed);
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

    const std::string default_prefix = "tools/omni/assets/test_case/audio_test_case/audio_test_case_";
    duplex_test_case(ctx_omni,
                     run_test ? test_prefix : default_prefix,
                     run_test ? test_count  : 2,
                     stream_interval_ms);

    // 等所有 speak 帧的 audio 文件落盘后再销毁；omni_free 会处理后续所有线程 join。
    omni_duplex_drain_tts_audio(ctx_omni);
    llama_perf_context_print(ctx_omni->ctx_llama);
    omni_free(ctx_omni);

    printf("\n=== Duplex test finished ===\n");
    return 0;
}
