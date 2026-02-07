#include "omni-impl.h"
#include "omni.h"

#include "arg.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <limits.h>
#include <cinttypes>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_usage(const char * prog_name) {
    printf(
        "MiniCPM-o Omni CLI - Multimodal inference tool\n\n"
        "Usage: %s -m <llm_model_path> [options]\n\n"
        "Required:\n"
        "  -m <path>           Path to LLM GGUF model (e.g., MiniCPM-o-4_5-Q4_K_M.gguf)\n"
        "                      Other model paths will be auto-detected from directory structure:\n"
        "                        {dir}/vision/MiniCPM-o-4_5-vision-F16.gguf\n"
        "                        {dir}/audio/MiniCPM-o-4_5-audio-F16.gguf\n"
        "                        {dir}/tts/MiniCPM-o-4_5-tts-F16.gguf\n"
        "                        {dir}/tts/MiniCPM-o-4_5-projector-F16.gguf\n\n"
        "Options:\n"
        "  --vision <path>     Override vision model path\n"
        "  --audio <path>      Override audio model path\n"
        "  --tts <path>        Override TTS model path\n"
        "  --projector <path>  Override projector model path\n"
        "  --ref-audio <path>  Reference audio for voice cloning (default: tools/omni/assets/default_ref_audio/default_ref_audio.wav)\n"
        "  -c, --ctx-size <n>  Context size (default: 4096)\n"
        "  -ngl <n>            Number of GPU layers (default: 99)\n"
        "  --no-tts            Disable TTS output\n"
        "  --test <prefix> <n> Run test case with audio prefix and count\n"
        "  -h, --help          Show this help message\n\n"
        "Example:\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf --no-tts\n",
        prog_name, prog_name, prog_name
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// 从 LLM 模型路径推断其他模型路径
// 目录结构:
// MiniCPM-o-4_5-gguf/
// ├── MiniCPM-o-4_5-{量化}.gguf          (LLM)
// ├── audio/
// │   └── MiniCPM-o-4_5-audio-F16.gguf
// ├── tts/
// │   ├── MiniCPM-o-4_5-projector-F16.gguf
// │   └── MiniCPM-o-4_5-tts-F16.gguf
// └── vision/
//     └── MiniCPM-o-4_5-vision-F16.gguf
struct OmniModelPaths {
    std::string llm;         // LLM 模型路径
    std::string vision;      // 视觉模型路径
    std::string audio;       // 音频模型路径
    std::string tts;         // TTS 模型路径
    std::string projector;   // Projector 模型路径
    std::string base_dir;    // 模型根目录
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
    
    // 自动推断其他模型路径
    paths.vision = paths.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    paths.audio = paths.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    paths.tts = paths.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    paths.projector = paths.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    
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

void test_case(struct omni_context *ctx_omni, common_params& params, std::string audio_path_prefix, int cnt){
    // 单工模式：
    //   1. stream_prefill("", "", 0) — 初始化 system prompt (ref_audio 在内部自动处理)
    //   2. stream_prefill(user_audio, "", 1) — 用户音频输入
    //   3. stream_decode — 生成回复
    ctx_omni->system_prompt_initialized = false;
    bool orig_async = ctx_omni->async;
    ctx_omni->async = false;

    // Step 1: 初始化 system prompt (index=0, 不传用户音频)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        stream_prefill(ctx_omni, "", "", 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        printf("prefill 0 (system prompt): %.3f s\n", dt);
    }

    // Step 2: prefill 所有用户音频 (index >= 1)
    for (int il = 0; il < cnt; ++il) {
        char idx_str[16];
        snprintf(idx_str, sizeof(idx_str), "%04d", il);
        std::string aud_fname = audio_path_prefix + idx_str + ".wav";

        auto t0 = std::chrono::high_resolution_clock::now();
        stream_prefill(ctx_omni, aud_fname, "", il + 1);  // index 从 1 开始
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        printf("prefill %d (%s): %.3f s\n", il + 1, aud_fname.c_str(), dt);
    }

    ctx_omni->async = orig_async;
    stream_decode(ctx_omni, "./");
}

int main(int argc, char ** argv) {
    ggml_time_init();

    // 命令行参数
    std::string llm_path;
    std::string vision_path_override;
    std::string audio_path_override;
    std::string tts_path_override;
    std::string projector_path_override;
    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    int n_ctx = 4096;
    int n_gpu_layers = 99;  // GPU 层数，默认 99
    bool use_tts = true;
    bool run_test = false;
    std::string test_audio_prefix;
    int test_count = 0;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        }
        else if (arg == "-m" && i + 1 < argc) {
            llm_path = argv[++i];
        }
        else if (arg == "--vision" && i + 1 < argc) {
            vision_path_override = argv[++i];
        }
        else if (arg == "--audio" && i + 1 < argc) {
            audio_path_override = argv[++i];
        }
        else if (arg == "--tts" && i + 1 < argc) {
            tts_path_override = argv[++i];
        }
        else if (arg == "--projector" && i + 1 < argc) {
            projector_path_override = argv[++i];
        }
        else if (arg == "--ref-audio" && i + 1 < argc) {
            ref_audio_path = argv[++i];
        }
        else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            n_ctx = std::atoi(argv[++i]);
        }
        else if (arg == "-ngl" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        }
        else if (arg == "--no-tts") {
            use_tts = false;
        }
        else if (arg == "--test" && i + 2 < argc) {
            run_test = true;
            test_audio_prefix = argv[++i];
            test_count = std::atoi(argv[++i]);
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            show_usage(argv[0]);
            return 1;
        }
    }
    
    // 检查必需参数
    if (llm_path.empty()) {
        fprintf(stderr, "Error: -m <llm_model_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }
    
    // 解析模型路径
    OmniModelPaths paths = resolve_model_paths(llm_path);
    
    // 应用覆盖路径
    if (!vision_path_override.empty()) paths.vision = vision_path_override;
    if (!audio_path_override.empty()) paths.audio = audio_path_override;
    if (!tts_path_override.empty()) paths.tts = tts_path_override;
    if (!projector_path_override.empty()) paths.projector = projector_path_override;
    
    // 打印模型路径
    print_model_paths(paths);
    
    // 检查必需文件
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
    
    // 设置参数
    common_params params;
    params.model.path = paths.llm;
    params.vpm_model = paths.vision;
    params.apm_model = paths.audio;
    params.tts_model = paths.tts;
    params.n_ctx = n_ctx;
    params.n_gpu_layers = n_gpu_layers;
    
    // Projector 路径需要通过 tts_bin_dir 传递
    // omni.cpp 中 projector 路径计算: gguf_root_dir + "/projector.gguf"
    // 其中 gguf_root_dir = tts_bin_dir 的父目录
    // 但我们的结构是 projector 在 tts/ 目录下
    // 所以需要修改 omni.cpp 或者创建符号链接
    // 这里暂时使用 tts 目录作为 tts_bin_dir
    std::string tts_bin_dir = get_parent_dir(paths.tts);
    
    common_init();
    
    printf("=== Initializing Omni Context ===\n");
    printf("  TTS enabled: %s\n", use_tts ? "yes" : "no");
    printf("  Context size: %d\n", n_ctx);
    printf("  GPU layers: %d\n", n_gpu_layers);
    printf("  TTS bin dir: %s\n", tts_bin_dir.c_str());
    printf("  Ref audio: %s\n", ref_audio_path.c_str());
    
    // 🔧 Token2Wav 使用 GPU（Metal），已用 ggml_add+ggml_repeat 替代不支持的 ggml_add1
    auto ctx_omni = omni_init(&params, 1, use_tts, tts_bin_dir, -1, "gpu:0");
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: Failed to initialize omni context\n");
        return 1;
    }
    ctx_omni->async = true;
    ctx_omni->ref_audio_path = ref_audio_path;  // 设置参考音频路径

    if (run_test) {
        printf("=== Running test case ===\n");
        printf("  Audio prefix: %s\n", test_audio_prefix.c_str());
        printf("  Count: %d\n", test_count);
        test_case(ctx_omni, params, test_audio_prefix, test_count);
    } else {
        // 默认测试用例
        test_case(ctx_omni, params, std::string("tools/omni/assets/test_case/audio_test_case/audio_test_case_"), 2);
    }

    // 等待 T2W 完成所有音频生成后再停止线程
    if(ctx_omni->async && ctx_omni->use_tts) {
        std::string done_flag = std::string(ctx_omni->base_output_dir) + "/round_000/tts_wav/generation_done.flag";
        fprintf(stderr, "Waiting for audio generation to complete...\n");
        for (int i = 0; i < 1200; ++i) {  // 最多等 120 秒
            FILE * f = fopen(done_flag.c_str(), "r");
            if (f) { fclose(f); fprintf(stderr, "Audio generation completed.\n"); break; }
            usleep(100000);  // 100ms
        }
    }

    if(ctx_omni->async) {
        omni_stop_threads(ctx_omni);
        if(ctx_omni->llm_thread.joinable()) { ctx_omni->llm_thread.join(); printf("llm thread end\n"); }
        if(ctx_omni->use_tts && ctx_omni->tts_thread.joinable()) { ctx_omni->tts_thread.join(); printf("tts thread end\n"); }
        if(ctx_omni->use_tts && ctx_omni->t2w_thread.joinable()) { ctx_omni->t2w_thread.join(); printf("t2w thread end\n"); }
    }

    llama_perf_context_print(ctx_omni->ctx_llama);

    omni_free(ctx_omni);
    return 0;
}
