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

/**
 * MiniCPM-o Omni 单轮 CLI（同步）
 *
 * 设计：
 *   - 同步单轮：prefill 与 decode 串行，阻塞直到全部产出完成后退出。
 *   - 三类输入（audio / image / text）均可选，至少给一个。
 *   - --omni 控制是否加载视觉模型；没开就拒绝 -i。
 *   - 启用 TTS 时把生成的 wav 写入 -o 指定（或默认）的输出目录；
 *     禁用 TTS 时不起 TTS 线程，仅把文字回复打到 stdout。
 */

static void show_usage(const char * prog_name) {
    printf(
        "MiniCPM-o Omni CLI - 单轮同步多模态推理\n\n"
        "Usage: %s -m <llm_model_path> [-a <audio>] [-i <image>] [-p <text>] [options]\n\n"
        "必需:\n"
        "  -m <path>                LLM GGUF 模型路径 (e.g., MiniCPM-o-4_5-Q4_K_M.gguf)\n"
        "                           其他模型路径从目录结构自动推断:\n"
        "                             {dir}/vision/MiniCPM-o-4_5-vision-F16.gguf (仅 --omni)\n"
        "                             {dir}/audio/MiniCPM-o-4_5-audio-F16.gguf\n"
        "                             {dir}/tts/MiniCPM-o-4_5-tts-F16.gguf       (默认启用 TTS)\n"
        "                             {dir}/tts/MiniCPM-o-4_5-projector-F16.gguf\n\n"
        "输入 (至少提供一项):\n"
        "  -a, --audio-input <path> 输入音频 (.wav)\n"
        "  -i, --image <path>       输入图片 (.jpg/.png) —— 需同时开启 --omni\n"
        "  -p, --prompt <text>      输入文字\n\n"
        "选项:\n"
        "      --omni               加载视觉模型，允许 -i；不设则不加载视觉\n"
        "      --no-tts             关闭 TTS，只输出文字回复\n"
        "  -o, --output-dir <path>  TTS 音频输出根目录 (默认: ./tools/omni/output)\n"
        "      --ref-audio <path>   参考音频（声音克隆）\n"
        "                           默认: tools/omni/assets/default_ref_audio/default_ref_audio.wav\n"
        "  -c, --ctx-size <n>       上下文长度 (默认 4096)\n"
        "  -ngl <n>                 GPU 层数 (默认 99)\n"
        "      --vision <path>      覆盖 vision 模型路径\n"
        "      --audio <path>       覆盖 audio 模型路径\n"
        "      --tts <path>         覆盖 TTS 模型路径\n"
        "      --projector <path>   覆盖 projector 模型路径\n"
        "      --vision-backend <mode>  metal (默认) 或 coreml\n"
        "      --vision-coreml <path>   CoreML 模型路径 (backend=coreml 时需要)\n"
        "  -h, --help               显示帮助\n\n"
        "示例:\n"
        "  纯文字 QA:\n"
        "    %s -m ./models/MiniCPM-o-4_5-Q4_K_M.gguf -p \"介绍一下你自己\" --no-tts\n"
        "  音频 + 文字 + 视觉 + TTS 输出:\n"
        "    %s -m ./models/MiniCPM-o-4_5-Q4_K_M.gguf --omni \\\n"
        "       -a ./input.wav -i ./input.jpg -p \"描述一下图片里的场景\" \\\n"
        "       -o ./out\n",
        prog_name, prog_name, prog_name
    );
}

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

    paths.vision        = paths.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    paths.audio         = paths.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    paths.tts           = paths.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    paths.projector     = paths.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    paths.vision_coreml = paths.base_dir + "/vision/coreml_minicpmo45_vit_all_f16.mlmodelc";

    return paths;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string llm_path;
    std::string audio_input;
    std::string image_input;
    std::string text_input;
    std::string vision_path_override;
    std::string audio_path_override;
    std::string tts_path_override;
    std::string projector_path_override;
    std::string vision_backend = "metal";
    std::string vision_coreml_model_path;
    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    std::string output_dir = "./tools/omni/output";
    int  n_ctx        = 4096;
    int  n_gpu_layers = 99;
    bool use_omni     = false;   // 是否加载视觉模型
    bool use_tts      = true;    // 是否生成 TTS 音频

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        }
        else if (arg == "-m" && i + 1 < argc)                               { llm_path = argv[++i]; }
        else if ((arg == "-a" || arg == "--audio-input") && i + 1 < argc)   { audio_input = argv[++i]; }
        else if ((arg == "-i" || arg == "--image") && i + 1 < argc)         { image_input = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc)        { text_input = argv[++i]; }
        else if (arg == "--omni")                                           { use_omni = true; }
        else if (arg == "--no-tts")                                         { use_tts = false; }
        else if ((arg == "-o" || arg == "--output-dir") && i + 1 < argc)    { output_dir = argv[++i]; }
        else if (arg == "--ref-audio" && i + 1 < argc)                      { ref_audio_path = argv[++i]; }
        else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc)      { n_ctx = std::atoi(argv[++i]); }
        else if (arg == "-ngl" && i + 1 < argc)                             { n_gpu_layers = std::atoi(argv[++i]); }
        else if (arg == "--vision" && i + 1 < argc)                         { vision_path_override = argv[++i]; }
        else if (arg == "--audio" && i + 1 < argc)                          { audio_path_override = argv[++i]; }
        else if (arg == "--tts" && i + 1 < argc)                            { tts_path_override = argv[++i]; }
        else if (arg == "--projector" && i + 1 < argc)                      { projector_path_override = argv[++i]; }
        else if (arg == "--vision-backend" && i + 1 < argc) {
            vision_backend = argv[++i];
            if (vision_backend != "metal" && vision_backend != "coreml") {
                fprintf(stderr, "Error: --vision-backend must be 'metal' or 'coreml', got '%s'\n", vision_backend.c_str());
                return 1;
            }
        }
        else if (arg == "--vision-coreml" && i + 1 < argc)                  { vision_coreml_model_path = argv[++i]; }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            show_usage(argv[0]);
            return 1;
        }
    }

    // ===== 参数校验 =====
    if (llm_path.empty()) {
        fprintf(stderr, "Error: -m <llm_model_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }

    const bool has_audio = !audio_input.empty();
    const bool has_image = !image_input.empty();
    const bool has_text  = !text_input.empty();
    if (!has_audio && !has_image && !has_text) {
        fprintf(stderr, "Error: 至少提供 -a / -i / -p 中的一项输入\n\n");
        show_usage(argv[0]);
        return 1;
    }
    if (has_image && !use_omni) {
        fprintf(stderr, "Error: 使用 -i 必须同时加上 --omni 来加载视觉模型\n");
        return 1;
    }
    if (has_audio && !file_exists(audio_input)) {
        fprintf(stderr, "Error: 音频文件不存在: %s\n", audio_input.c_str()); return 1;
    }
    if (has_image && !file_exists(image_input)) {
        fprintf(stderr, "Error: 图片文件不存在: %s\n", image_input.c_str()); return 1;
    }

    // ===== 解析模型路径 =====
    OmniModelPaths paths = resolve_model_paths(llm_path);
    if (!vision_path_override.empty())    paths.vision    = vision_path_override;
    if (!audio_path_override.empty())     paths.audio     = audio_path_override;
    if (!tts_path_override.empty())       paths.tts       = tts_path_override;
    if (!projector_path_override.empty()) paths.projector = projector_path_override;

    if (!file_exists(paths.llm)) {
        fprintf(stderr, "Error: LLM model not found: %s\n", paths.llm.c_str()); return 1;
    }
    // audio encoder 一直需要，因为系统 prompt 里会塞参考音频做声音克隆
    if (!file_exists(paths.audio)) {
        fprintf(stderr, "Error: Audio model not found: %s\n", paths.audio.c_str()); return 1;
    }
    if (use_omni && !file_exists(paths.vision)) {
        fprintf(stderr, "Error: --omni 开启但 vision 模型不存在: %s\n", paths.vision.c_str()); return 1;
    }
    if (use_tts && !file_exists(paths.tts)) {
        fprintf(stderr, "Warning: TTS 模型不存在，自动禁用 TTS: %s\n", paths.tts.c_str());
        use_tts = false;
    }
    if (!file_exists(ref_audio_path)) {
        fprintf(stderr, "Warning: ref_audio 不存在，系统 prompt 的声音克隆步骤可能失败: %s\n", ref_audio_path.c_str());
    }

    const int media_type = use_omni ? 2 : 1;

    // ===== 组装 params =====
    common_params params;
    params.model.path = paths.llm;
    params.vpm_model  = use_omni ? paths.vision : std::string();
    params.apm_model  = paths.audio;
    params.tts_model  = use_tts ? paths.tts : std::string();
    if (use_omni && vision_backend == "coreml") {
        if (vision_coreml_model_path.empty()) {
            vision_coreml_model_path = paths.vision_coreml;
        }
        params.vision_coreml_model_path = vision_coreml_model_path;
    }
    params.n_ctx        = n_ctx;
    params.n_gpu_layers = n_gpu_layers;

    const std::string tts_bin_dir = get_parent_dir(paths.tts);

    common_init();

    printf("=== Config ===\n");
    printf("  Omni (vision) : %s\n", use_omni ? "on" : "off");
    printf("  TTS           : %s\n", use_tts  ? "on" : "off");
    printf("  Context size  : %d\n", n_ctx);
    printf("  GPU layers    : %d\n", n_gpu_layers);
    if (use_omni) {
        printf("  Vision backend: %s\n", vision_backend.c_str());
        if (vision_backend == "coreml") {
            printf("  Vision CoreML : %s\n", vision_coreml_model_path.c_str());
        }
    }
    if (use_tts) {
        printf("  Output dir    : %s\n", output_dir.c_str());
        printf("  Ref audio     : %s\n", ref_audio_path.c_str());
    }
    printf("  Inputs        : audio=%s  image=%s  text=%s\n",
           has_audio ? audio_input.c_str() : "(none)",
           has_image ? image_input.c_str() : "(none)",
           has_text  ? "(provided)"        : "(none)");

    // ===== 初始化 omni ctx =====
    auto * ctx_omni = omni_init(&params, media_type, use_tts, tts_bin_dir,
                                /*tts_gpu_layers=*/ -1,
                                /*token2wav_device=*/ "gpu:0",
                                /*duplex_mode=*/ false,
                                /*existing_model=*/ nullptr,
                                /*existing_ctx=*/ nullptr,
                                /*base_output_dir=*/ output_dir);
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: Failed to initialize omni context\n");
        return 1;
    }
    ctx_omni->ref_audio_path = ref_audio_path;
    ctx_omni->system_prompt_initialized = false;

    // ===== Prefill 阶段：始终同步 =====
    // 说明：stream_prefill(index=0) 在 async=true 时会 spawn LLM/TTS/T2W 线程，
    // 会与同步 prefill 产生竞态；所以 prefill 阶段固定 async=false。
    ctx_omni->async = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) 系统 prompt + ref_audio 初始化（无论输入组合如何都要走这一步）
    if (!stream_prefill(ctx_omni, /*aud=*/ "", /*img=*/ "", /*index=*/ 0)) {
        fprintf(stderr, "Error: stream_prefill(index=0) failed\n");
        omni_free(ctx_omni);
        return 1;
    }

    // 2) 用户多模态输入（index=1）：有音频或图片才调用
    if (has_audio || has_image) {
        if (!stream_prefill(ctx_omni, audio_input, image_input, /*index=*/ 1)) {
            fprintf(stderr, "Error: stream_prefill(index=1) failed\n");
            omni_free(ctx_omni);
            return 1;
        }
    }

    // 3) 用户文字输入：eval_string 到当前 user turn 里
    if (has_text) {
        if (!omni_prefill_text(ctx_omni, text_input)) {
            fprintf(stderr, "Error: omni_prefill_text failed\n");
            omni_free(ctx_omni);
            return 1;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prefill_elapsed = t1 - t0;
    printf("\nprefill 耗时: %.3f s\n", prefill_elapsed.count());

    // ===== Decode 阶段 =====
    // 开 TTS 时 async=true：stream_decode 会起 tts/t2w 线程把 wav 写到 base_output_dir；
    // 不开 TTS 时保持 async=false：stream_decode 走纯同步路径生成文字。
    ctx_omni->async = use_tts;

    auto decode_t0 = std::chrono::high_resolution_clock::now();
    if (!stream_decode(ctx_omni, output_dir)) {
        fprintf(stderr, "Error: stream_decode failed\n");
        omni_free(ctx_omni);
        return 1;
    }
    auto decode_t1 = std::chrono::high_resolution_clock::now();

    // 读取 text_queue 打印完整文本回复
    std::string full_text;
    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        for (const auto & frag : ctx_omni->text_queue) full_text += frag;
    }

    printf("\n=== Assistant ===\n%s\n", full_text.c_str());
    printf("decode 耗时: %.3f s\n",
           std::chrono::duration<double>(decode_t1 - decode_t0).count());

    // ===== TTS 收尾 =====
    if (use_tts) {
        // tts/t2w 在独立线程里跑，等 flag 文件表示音频全部写完再收摊
        std::string done_flag = std::string(ctx_omni->base_output_dir) +
                                "/round_000/tts_wav/generation_done.flag";
        fprintf(stderr, "Waiting for audio generation to complete...\n");
        bool tts_done = false;
        for (int i = 0; i < 1200; ++i) {  // 最多等 120s
            FILE * f = fopen(done_flag.c_str(), "r");
            if (f) {
                fclose(f);
                tts_done = true;
                break;
            }
            usleep(100000);
        }
        if (tts_done) {
            fprintf(stderr, "Audio generation completed.\n");
            printf("音频输出目录: %s/round_000/tts_wav/\n", ctx_omni->base_output_dir.c_str());
        } else {
            fprintf(stderr, "Warning: TTS 超时 (120s) 未完成，可能有音频未落盘\n");
        }

        omni_stop_threads(ctx_omni);
        if (ctx_omni->llm_thread.joinable()) { ctx_omni->llm_thread.join(); }
        if (ctx_omni->tts_thread.joinable()) { ctx_omni->tts_thread.join(); }
        if (ctx_omni->t2w_thread.joinable()) { ctx_omni->t2w_thread.join(); }
    }

    llama_perf_context_print(ctx_omni->ctx_llama);

    omni_free(ctx_omni);
    return 0;
}
