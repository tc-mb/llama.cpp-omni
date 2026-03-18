/**
 * 双工 (Duplex) Omni 模式测试
 *
 * 双工模式下，每个音频 chunk 独立执行 prefill → decode 循环，
 * 模型可以在任意时刻决定 <|speak|>（说话）或 <|listen|>（继续听）。
 *
 * 与单工模式的区别：
 *   - 单工：所有 chunk 一次性 prefill，然后统一 decode 一次
 *   - 双工：每个 chunk prefill 后立即 decode，模型实时决策
 *
 * 用法:
 *   llama-omni-test-duplex -m <llm_model_path> [options]
 *     --test <prefix> <n>   指定测试数据前缀和 chunk 数量
 *     --ref-audio <path>    参考音频路径
 *     --no-tts              禁用 TTS
 *     --omni                启用 omni 模式 (audio+vision)
 *     -c <n>                上下文大小 (默认 4096)
 *     -ngl <n>              GPU 层数 (默认 99)
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

// ==================== 模型路径解析（复用 cli 逻辑） ====================

struct TestModelPaths {
    std::string llm;
    std::string vision;
    std::string audio;
    std::string tts;
    std::string projector;
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
    return paths;
}

// ==================== 双工测试核心 ====================

static void duplex_test_case(struct omni_context * ctx_omni,
                             common_params & /*params*/,
                             const std::string & data_path_prefix,
                             int cnt) {
    printf("\n========================================\n");
    printf("  双工模式测试: %d chunks\n", cnt);
    printf("  数据前缀: %s\n", data_path_prefix.c_str());
    printf("========================================\n\n");

    // async 模式：TTS 线程需要 async=true 才能正常工作
    // 已修复 stream_decode 中 need_speek 重复设置导致的 prefill_done 竞态条件

    auto total_t0 = std::chrono::high_resolution_clock::now();
    int speak_count = 0;
    int listen_count = 0;
    double total_prefill_s = 0;
    double total_decode_s = 0;
    int chunks_completed = 0;

    for (int il = 0; il < cnt; ++il) {
        if (g_is_interrupted) {
            printf("\n[中断] 用户中断测试\n");
            break;
        }

        char idx_str[16];
        snprintf(idx_str, sizeof(idx_str), "%04d", il);
        std::string aud_fname = data_path_prefix + idx_str + ".wav";

        std::string img_fname;
        std::string img_candidate = data_path_prefix + idx_str + ".jpg";
        if (file_exists(img_candidate)) {
            img_fname = img_candidate;
        }

        if (!file_exists(aud_fname)) {
            fprintf(stderr, "[错误] 音频文件不存在: %s\n", aud_fname.c_str());
            break;
        }

        printf("\n--- Chunk %d/%d ---\n", il + 1, cnt);
        if (!img_fname.empty()) {
            printf("  音频+图片: %s\n", aud_fname.c_str());
        } else {
            printf("  音频: %s\n", aud_fname.c_str());
        }

        // Step 1: Prefill
        auto t0 = std::chrono::high_resolution_clock::now();
        bool prefill_ok = stream_prefill(ctx_omni, aud_fname, img_fname, il);
        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_dt = std::chrono::duration<double>(t1 - t0).count();

        if (!prefill_ok) {
            fprintf(stderr, "[错误] Chunk %d prefill 失败\n", il);
            break;
        }

        // Step 2: Decode
        auto t2 = std::chrono::high_resolution_clock::now();
        bool decode_ok = stream_decode(ctx_omni, "./");
        auto t3 = std::chrono::high_resolution_clock::now();
        double decode_dt = std::chrono::duration<double>(t3 - t2).count();

        if (!decode_ok) {
            fprintf(stderr, "[错误] Chunk %d decode 失败\n", il);
            break;
        }

        bool ended_listen = ctx_omni->ended_with_listen.load();

        // stream_decode 返回后从 text_queue 读取生成的文本
        std::string generated_text;
        {
            std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
            while (!ctx_omni->text_queue.empty()) {
                std::string piece = ctx_omni->text_queue.front();
                ctx_omni->text_queue.pop_front();
                if (piece == "__IS_LISTEN__" || piece == "__END_OF_TURN__") {
                    continue;
                }
                generated_text += piece;
            }
        }

        if (ended_listen) {
            listen_count++;
        } else {
            speak_count++;
        }

        total_prefill_s += prefill_dt;
        total_decode_s += decode_dt;
        chunks_completed++;

        printf("  prefill: %.3f s | decode: %.3f s | total: %.3f s | n_past: %d\n",
               prefill_dt, decode_dt, prefill_dt + decode_dt, ctx_omni->n_past);
        if (ended_listen) {
            printf("  决策: <|listen|>\n");
        } else {
            printf("  决策: <|speak|> → \"%s\"\n",
                   generated_text.length() > 60
                       ? (generated_text.substr(0, 60) + "...").c_str()
                       : generated_text.c_str());
        }
    }

    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_dt = std::chrono::duration<double>(total_t1 - total_t0).count();

    printf("\n========================================\n");
    printf("  双工测试完成\n");
    printf("========================================\n");
    printf("  Chunks:      %d / %d\n", chunks_completed, cnt);
    printf("  总耗时:      %.3f s\n", total_dt);
    printf("  avg prefill: %.1f ms\n", chunks_completed > 0 ? total_prefill_s / chunks_completed * 1000.0 : 0);
    printf("  avg decode:  %.1f ms\n", chunks_completed > 0 ? total_decode_s / chunks_completed * 1000.0 : 0);
    printf("  avg/chunk:   %.1f ms\n", chunks_completed > 0 ? total_dt / chunks_completed * 1000.0 : 0);
    printf("  speak: %d | listen: %d\n", speak_count, listen_count);
    printf("  最终 n_past: %d\n", ctx_omni->n_past);
    printf("========================================\n");

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
        "  --test <prefix> <n> 指定测试数据前缀和 chunk 数量\n"
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
    int n_ctx = 4096;
    int n_gpu_layers = 99;
    int media_type = 1;     // 1=audio only, 2=omni (audio+vision)
    bool use_tts = true;
    bool run_test = false;
    std::string test_prefix;
    int test_count = 0;

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
        else if (arg == "-o" && i + 1 < argc) { output_dir = argv[++i]; }
        else if (arg == "--test" && i + 2 < argc) {
            run_test = true;
            test_prefix = argv[++i];
            test_count = std::atoi(argv[++i]);
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

    printf("=== Duplex Test - Model Paths ===\n");
    printf("  LLM:        %s %s\n", paths.llm.c_str(), file_exists(paths.llm) ? "[OK]" : "[NOT FOUND]");
    printf("  Vision:     %s %s\n", paths.vision.c_str(), file_exists(paths.vision) ? "[OK]" : "[NOT FOUND]");
    printf("  Audio:      %s %s\n", paths.audio.c_str(), file_exists(paths.audio) ? "[OK]" : "[NOT FOUND]");
    printf("  TTS:        %s %s\n", paths.tts.c_str(), file_exists(paths.tts) ? "[OK]" : "[NOT FOUND]");
    printf("  Projector:  %s %s\n", paths.projector.c_str(), file_exists(paths.projector) ? "[OK]" : "[NOT FOUND]");
    printf("================================\n");

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

    std::string tts_bin_dir = get_parent_dir(paths.tts);

    common_init();

    printf("=== Initializing Duplex Omni Context ===\n");
    printf("  Media type: %d (%s)\n", media_type, media_type == 2 ? "omni: audio+vision" : "audio only");
    printf("  TTS enabled: %s\n", use_tts ? "yes" : "no");
    printf("  Context size: %d\n", n_ctx);
    printf("  GPU layers: %d\n", n_gpu_layers);
    printf("  Output dir: %s\n", output_dir.c_str());
    printf("  Ref audio: %s\n", ref_audio_path.c_str());
    printf("  Mode: DUPLEX\n");

    // 关键: duplex_mode=true
    auto ctx_omni = omni_init(&params, media_type, use_tts, tts_bin_dir,
                              /*tts_gpu_layers=*/-1, /*token2wav_device=*/"cpu",
                              /*duplex_mode=*/true,
                              /*existing_model=*/nullptr, /*existing_ctx=*/nullptr,
                              /*base_output_dir=*/output_dir);
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: Failed to initialize omni context\n");
        return 1;
    }
    ctx_omni->async = true;
    ctx_omni->ref_audio_path = ref_audio_path;

    // 执行测试
    if (run_test) {
        printf("=== Running duplex test case ===\n");
        printf("  Prefix: %s\n", test_prefix.c_str());
        printf("  Count: %d\n", test_count);
        duplex_test_case(ctx_omni, params, test_prefix, test_count);
    } else {
        printf("=== Running default duplex test (audio_test_case, 2 chunks) ===\n");
        duplex_test_case(ctx_omni, params,
                         "tools/omni/assets/test_case/audio_test_case/audio_test_case_", 2);
    }

    // 等待 TTS 和 T2W 队列清空
    if (ctx_omni->async && ctx_omni->use_tts) {
        fprintf(stderr, "Waiting for TTS/T2W processing to complete...\n");
        int idle_count = 0;
        for (int i = 0; i < 1200; ++i) {
            if (omni_tts_queues_empty(ctx_omni)) {
                idle_count++;
                if (idle_count >= 30) {
                    fprintf(stderr, "TTS/T2W queues idle for 3s, proceeding.\n");
                    break;
                }
            } else {
                idle_count = 0;
            }
            usleep(100000);
        }
    }

    // 停止线程并等待
    if (ctx_omni->async) {
        omni_stop_threads(ctx_omni);
        if (ctx_omni->llm_thread.joinable()) { ctx_omni->llm_thread.join(); printf("llm thread end\n"); }
        if (ctx_omni->use_tts && ctx_omni->tts_thread.joinable()) { ctx_omni->tts_thread.join(); printf("tts thread end\n"); }
        if (ctx_omni->use_tts && ctx_omni->t2w_thread.joinable()) { ctx_omni->t2w_thread.join(); printf("t2w thread end\n"); }
    }

    llama_perf_context_print(ctx_omni->ctx_llama);
    omni_free(ctx_omni);

    printf("\n=== Duplex test finished ===\n");
    return 0;
}
