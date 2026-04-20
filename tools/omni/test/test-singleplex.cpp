/**
 * 单工 (Singleplex) Omni 模式批量测试
 *
 * 单工模式下，所有音频（+可选图片）一次性 prefill 完成，最后统一 decode 一次生成完整回复。
 *
 * 与双工模式的区别：
 *   - 单工：所有 chunk 一次性 prefill，然后统一 decode 一次
 *   - 双工：每个 chunk prefill 后立即 decode，模型实时决策
 *
 * 用法:
 *   llama-omni-test-singleplex -m <llm_model_path> [options]
 *     --test <prefix> <n>   指定测试数据前缀和 chunk 数量 (必需)
 *     --ref-audio <path>    参考音频路径
 *     --no-tts              禁用 TTS
 *     --omni                启用 omni 模式 (audio+vision)
 *     --vision-backend <mode>  Vision 后端: 'metal'(默认) 或 'coreml'
 *     --vision-coreml <path>   CoreML 模型路径 (backend=coreml 时需要)
 *     -c <n>                上下文大小 (默认 4096)
 *     -ngl <n>              GPU 层数 (默认 99)
 *
 * 示例:
 *   llama-omni-test-singleplex \
 *       -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
 *       --omni --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9
 */

#include "omni-impl.h"
#include "omni.h"

#include "arg.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
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

// ==================== 模型路径解析 ====================

struct OmniModelPaths {
    std::string llm;
    std::string vision;
    std::string audio;
    std::string tts;
    std::string projector;
    std::string vision_coreml;
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

static OmniModelPaths resolve_model_paths(const std::string & llm_path) {
    OmniModelPaths paths;
    paths.llm = llm_path;
    paths.base_dir = get_parent_dir(llm_path);

    paths.vision = paths.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    paths.audio = paths.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    paths.tts = paths.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    paths.projector = paths.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    paths.vision_coreml = paths.base_dir + "/vision/coreml_minicpmo45_vit_all_f16.mlmodelc";

    return paths;
}

static void print_model_paths(const OmniModelPaths & paths) {
    printf("=== Model Paths ===\n");
    printf("  Base dir:   %s\n", paths.base_dir.c_str());
    printf("  LLM:        %s %s\n", paths.llm.c_str(), file_exists(paths.llm) ? "[OK]" : "[NOT FOUND]");
    printf("  Vision:     %s %s\n", paths.vision.c_str(), file_exists(paths.vision) ? "[OK]" : "[NOT FOUND]");
    printf("  Audio:      %s %s\n", paths.audio.c_str(), file_exists(paths.audio) ? "[OK]" : "[NOT FOUND]");
    printf("  TTS:        %s %s\n", paths.tts.c_str(), file_exists(paths.tts) ? "[OK]" : "[NOT FOUND]");
    printf("  Projector:  %s %s\n", paths.projector.c_str(), file_exists(paths.projector) ? "[OK]" : "[NOT FOUND]");
    printf("===================\n");
}

// ==================== 批量测试主流程 ====================

// 单工批量测试：所有 chunk 同步 prefill，最后统一 decode 一次
static void run_singleplex_test(struct omni_context * ctx_omni,
                                const std::string & data_path_prefix,
                                int cnt) {
    // 单工模式：先 prefill 所有输入，然后 decode 一次生成完整回复
    // 使用同步模式 prefill，避免 async 模式下的竞态条件
    ctx_omni->system_prompt_initialized = false;
    bool orig_async = ctx_omni->async;
    ctx_omni->async = false;  // 同步模式 prefill，确保所有数据被处理

    // index=0 仅用于系统 prompt + ref_audio 初始化，不会处理用户音频。
    // 用户音频必须从 index>=1 开始喂，否则第一个 chunk 会被丢掉（见 omni.cpp stream_prefill）。
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        stream_prefill(ctx_omni, std::string(), std::string(), 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "prefill init (system prompt) : "
                  << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
    }

    for (int il = 0; il < cnt; ++il) {
        char idx_str[16];
        snprintf(idx_str, sizeof(idx_str), "%04d", il);
        std::string aud_fname = data_path_prefix + idx_str + ".wav";

        // omni 模式：自动检测同名 .jpg 图片
        std::string img_fname;
        std::string img_candidate = data_path_prefix + idx_str + ".jpg";
        if (file_exists(img_candidate)) {
            img_fname = img_candidate;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        // 用户音频 chunk 从 index=1 开始喂入（index=0 保留给 system prompt 初始化）
        stream_prefill(ctx_omni, aud_fname, img_fname, il + 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = t1 - t0;
        double dt = elapsed_seconds.count();
        if (img_fname.empty()) {
            std::cout << "prefill " << il << " (audio) : " << dt << " s" << std::endl;
        } else {
            std::cout << "prefill " << il << " (audio+vision) : " << dt << " s" << std::endl;
        }
    }

    // 所有数据同步 prefill 完成后恢复 async 模式并调用 decode
    // stream_decode 内部会根据 async 标志启动 TTS/T2W 线程
    ctx_omni->async = orig_async;
    stream_decode(ctx_omni, "./");
}

static void show_usage(const char * prog_name) {
    printf(
        "MiniCPM-o Singleplex Test - Batch prefill + single decode\n\n"
        "Usage: %s -m <llm_model_path> --test <prefix> <n> [options]\n\n"
        "Required:\n"
        "  -m <path>                LLM GGUF 模型路径\n"
        "  --test <prefix> <n>      测试数据前缀和 chunk 数量\n\n"
        "Options:\n"
        "  --vision <path>          覆盖 vision 模型路径\n"
        "  --audio <path>           覆盖 audio 模型路径\n"
        "  --tts <path>             覆盖 TTS 模型路径\n"
        "  --projector <path>       覆盖 projector 模型路径\n"
        "  --ref-audio <path>       参考音频路径\n"
        "  -c, --ctx-size <n>       上下文大小 (默认 4096)\n"
        "  -ngl <n>                 GPU 层数 (默认 99)\n"
        "  --no-tts                 禁用 TTS\n"
        "  --omni                   启用 omni 模式 (audio+vision, media_type=2)\n"
        "  --vision-backend <mode>  Vision 后端: 'metal'(默认) 或 'coreml'\n"
        "  --vision-coreml <path>   CoreML 模型路径 (backend=coreml 时需要)\n"
        "  -h, --help               显示帮助\n\n"
        "Example:\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \\\n"
        "     --omni --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9\n",
        prog_name, prog_name
    );
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string llm_path;
    std::string vision_path_override;
    std::string audio_path_override;
    std::string tts_path_override;
    std::string projector_path_override;
    std::string vision_backend = "metal";
    std::string vision_coreml_model_path;
    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    int n_ctx = 4096;
    int n_gpu_layers = 99;
    int media_type = 1;  // 1=audio only, 2=omni(audio+vision)
    bool use_tts = true;
    std::string test_audio_prefix;
    int test_count = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            llm_path = argv[++i];
        } else if (arg == "--vision" && i + 1 < argc) {
            vision_path_override = argv[++i];
        } else if (arg == "--audio" && i + 1 < argc) {
            audio_path_override = argv[++i];
        } else if (arg == "--tts" && i + 1 < argc) {
            tts_path_override = argv[++i];
        } else if (arg == "--projector" && i + 1 < argc) {
            projector_path_override = argv[++i];
        } else if (arg == "--ref-audio" && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            n_ctx = std::atoi(argv[++i]);
        } else if (arg == "-ngl" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--no-tts") {
            use_tts = false;
        } else if (arg == "--omni") {
            media_type = 2;
        } else if (arg == "--vision-backend" && i + 1 < argc) {
            vision_backend = argv[++i];
            if (vision_backend != "metal" && vision_backend != "coreml") {
                fprintf(stderr, "Error: --vision-backend must be 'metal' or 'coreml', got '%s'\n", vision_backend.c_str());
                return 1;
            }
        } else if (arg == "--vision-coreml" && i + 1 < argc) {
            vision_coreml_model_path = argv[++i];
        } else if (arg == "--test" && i + 2 < argc) {
            test_audio_prefix = argv[++i];
            test_count = std::atoi(argv[++i]);
        } else {
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
    if (test_audio_prefix.empty() || test_count <= 0) {
        fprintf(stderr, "Error: --test <prefix> <n> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }

    OmniModelPaths paths = resolve_model_paths(llm_path);
    if (!vision_path_override.empty())    paths.vision    = vision_path_override;
    if (!audio_path_override.empty())     paths.audio     = audio_path_override;
    if (!tts_path_override.empty())       paths.tts       = tts_path_override;
    if (!projector_path_override.empty()) paths.projector = projector_path_override;
    print_model_paths(paths);

    if (!file_exists(paths.llm)) {
        fprintf(stderr, "Error: LLM model not found: %s\n", paths.llm.c_str());
        return 1;
    }
    if (!file_exists(paths.audio)) {
        fprintf(stderr, "Error: Audio model not found: %s\n", paths.audio.c_str());
        return 1;
    }
    if (use_tts && !file_exists(paths.tts)) {
        fprintf(stderr, "Warning: TTS model not found: %s, disabling TTS\n", paths.tts.c_str());
        use_tts = false;
    }

    common_params params;
    params.model.path = paths.llm;
    params.vpm_model  = paths.vision;
    params.apm_model  = paths.audio;
    params.tts_model  = paths.tts;
    if (vision_backend == "coreml") {
        if (vision_coreml_model_path.empty()) {
            vision_coreml_model_path = paths.vision_coreml;
        }
        params.vision_coreml_model_path = vision_coreml_model_path;
    }
    params.n_ctx        = n_ctx;
    params.n_gpu_layers = n_gpu_layers;

    std::string tts_bin_dir = get_parent_dir(paths.tts);

    common_init();

    printf("=== Initializing Omni Context ===\n");
    printf("  Media type: %d (%s)\n", media_type, media_type == 2 ? "omni: audio+vision" : "audio only");
    printf("  TTS enabled: %s\n", use_tts ? "yes" : "no");
    printf("  Context size: %d\n", n_ctx);
    printf("  GPU layers: %d\n", n_gpu_layers);
    printf("  Vision backend: %s\n", vision_backend.c_str());
    if (vision_backend == "coreml") {
        printf("  Vision CoreML: %s\n", vision_coreml_model_path.c_str());
    }
    printf("  TTS bin dir: %s\n", tts_bin_dir.c_str());
    printf("  Ref audio: %s\n", ref_audio_path.c_str());

    // Token2Wav 使用 GPU（Metal），已用 ggml_add+ggml_repeat 替代不支持的 ggml_add1
    auto ctx_omni = omni_init(&params, media_type, use_tts, tts_bin_dir, -1, "gpu:0");
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: Failed to initialize omni context\n");
        return 1;
    }
    ctx_omni->async = true;
    ctx_omni->ref_audio_path = ref_audio_path;

    printf("=== Running singleplex test ===\n");
    printf("  Audio prefix: %s\n", test_audio_prefix.c_str());
    printf("  Count: %d\n", test_count);
    run_singleplex_test(ctx_omni, test_audio_prefix, test_count);

    // 等待 T2W 完成所有音频生成后再停止线程
    if (ctx_omni->async && ctx_omni->use_tts) {
        std::string done_flag = std::string(ctx_omni->base_output_dir) + "/round_000/tts_wav/generation_done.flag";
        fprintf(stderr, "Waiting for audio generation to complete...\n");
        for (int i = 0; i < 1200; ++i) {  // 最多等 120 秒
            FILE * f = fopen(done_flag.c_str(), "r");
            if (f) { fclose(f); fprintf(stderr, "Audio generation completed.\n"); break; }
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
            usleep(100000);  // 100ms
#elif defined(_WIN32)
            Sleep(100);
#endif
        }
    }

    if (ctx_omni->async) {
        omni_stop_threads(ctx_omni);
        if (ctx_omni->llm_thread.joinable()) { ctx_omni->llm_thread.join(); printf("llm thread end\n"); }
        if (ctx_omni->use_tts && ctx_omni->tts_thread.joinable()) { ctx_omni->tts_thread.join(); printf("tts thread end\n"); }
        if (ctx_omni->use_tts && ctx_omni->t2w_thread.joinable()) { ctx_omni->t2w_thread.join(); printf("t2w thread end\n"); }
    }

    llama_perf_context_print(ctx_omni->ctx_llama);

    omni_free(ctx_omni);
    return 0;
}
