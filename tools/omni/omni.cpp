#include "omni-impl.h"
#include "vision.h"
#include "audition.h"
#include "omni.h"
#include "token2wav/token2wav-impl.h"

#include "llama.h"
#include "common/common.h"
#include "common/sampling.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <set>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>
#include <random>
#include <cstdarg>
#include <signal.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #include <io.h>
    #include <process.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    // Windows compatibility macros
    #define popen  _popen
    #define pclose _pclose
    #define unlink _unlink
    #define stat   _stat
    #define S_IFDIR _S_IFDIR
#else
    #include <sys/time.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <dirent.h>
#endif

// ============================================================
// Cross-platform helper: recursive directory creation
// Replaces "mkdir -p" shell command for Windows compatibility
// ============================================================
static bool cross_platform_mkdir_p(const std::string& path) {
    if (path.empty()) return false;
    
    std::string normalized = path;
#ifdef _WIN32
    for (char& c : normalized) {
        if (c == '/') c = '\\';
    }
    size_t pos = 0;
    if (normalized.size() >= 2 && normalized[1] == ':') {
        pos = 2;
        if (normalized.size() > 2 && normalized[2] == '\\') pos = 3;
    }
    while (pos < normalized.size() && normalized[pos] == '\\') pos++;
    
    while (pos < normalized.size()) {
        pos = normalized.find('\\', pos);
        if (pos == std::string::npos) pos = normalized.size();
        std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct _stat info;
            if (_stat(sub.c_str(), &info) != 0) {
                if (_mkdir(sub.c_str()) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
        if (pos < normalized.size()) pos++;
    }
    return true;
#else
    for (char& c : normalized) {
        if (c == '\\') c = '/';
    }
    size_t pos = 0;
    if (!normalized.empty() && normalized[0] == '/') pos = 1;
    while (pos < normalized.size()) {
        pos = normalized.find('/', pos);
        if (pos == std::string::npos) pos = normalized.size();
        std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct stat info;
            if (::stat(sub.c_str(), &info) != 0) {
                if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
        if (pos < normalized.size()) pos++;
    }
    return true;
#endif
}

// 前向声明：Python Token2Wav 服务函数（定义在文件后面）
static bool start_python_t2w_service(struct omni_context * ctx_omni);
static void stop_python_t2w_service(struct omni_context * ctx_omni);
static bool send_python_t2w_command(struct omni_context * ctx_omni, const std::string& cmd_json, std::string& response);
static bool init_python_t2w_model(struct omni_context * ctx_omni, const std::string& device);
static bool set_python_t2w_ref_audio(struct omni_context * ctx_omni, const std::string& ref_audio_path);
static bool process_python_t2w_tokens(struct omni_context * ctx_omni, const std::vector<int32_t>& tokens, bool last_chunk, const std::string& output_path, double& inference_time_ms, double& audio_duration);
static bool reset_python_t2w_cache(struct omni_context * ctx_omni);


//
// Forward declarations
//
void print_with_timestamp(const char* format, ...);

// ==================== 特殊 Token 分类 ====================
// 
// 双工模式下，模型生成的特殊 token 分为以下几类：
// 
// 1. 状态切换 Token:
//    - <|speak|>: 模型决定开始说话
//    - <|listen|>: 模型决定开始听（双工）
// 
// 2. Chunk 终止 Token:
//    - <|chunk_eos|>: 语义 chunk 结束，仍在同一轮内
//    - <|chunk_tts_eos|>: TTS chunk 结束，触发 TTS 处理
// 
// 3. 轮次/序列终止 Token:
//    - <|turn_eos|>: 当前轮次结束，可切换回 listen
//    - <|tts_eos|>: 旧版 TTS 结束（单工模式）
//    - </s>: 序列完全结束
//
// Python 参考 (MiniCPMODuplex):
//   CHUNK_EOS_TOKEN_ID = 128261
//   CHUNK_TTS_EOS_TOKEN_ID = 128268
//   TURN_EOS_TOKEN_ID = 128260
//   LISTEN_TOKEN_ID = 128267
//   SPEAK_TOKEN_ID = 128266

enum class OmniTokenType {
    NORMAL,           // 普通 token (文本、audio code 等)
    SPEAK,            // <|speak|> - 开始说话
    LISTEN,           // <|listen|> - 开始听 (双工)
    CHUNK_EOS,        // <|chunk_eos|> - 语义 chunk 结束
    CHUNK_TTS_EOS,    // <|chunk_tts_eos|> - TTS chunk 结束
    TURN_EOS,         // <|turn_eos|> - 轮次结束
    TTS_EOS,          // <|tts_eos|> - 旧版 TTS 结束 (单工)
    EOS               // </s> - 序列结束
};

// 获取 token 类型
static OmniTokenType get_token_type(struct omni_context * ctx, llama_token token) {
    if (token == ctx->special_token_speak) {
        return OmniTokenType::SPEAK;
    } else if (token == ctx->special_token_listen) {
        return OmniTokenType::LISTEN;
    } else if (token == ctx->special_token_chunk_eos) {
        return OmniTokenType::CHUNK_EOS;
    } else if (token == ctx->special_token_chunk_tts_eos) {
        return OmniTokenType::CHUNK_TTS_EOS;
    } else if (token == ctx->special_token_turn_eos) {
        return OmniTokenType::TURN_EOS;
    } else if (token == ctx->special_token_tts_eos) {
        return OmniTokenType::TTS_EOS;
    } else if (token == ctx->special_token_eos) {
        return OmniTokenType::EOS;
    }
    return OmniTokenType::NORMAL;
}

// 检查是否是会话/轮次结束 token
static bool is_end_token(struct omni_context * ctx, llama_token token) {
    OmniTokenType type = get_token_type(ctx, token);
    
    if (ctx->duplex_mode) {
        // 双工模式:
        // - chunk_eos/chunk_tts_eos: 结束当前 stream_decode 调用，Python server 管理多轮
        // - listen: 结束当前发言段
        // - turn_eos/tts_eos/eos: 在 stream_decode 内层循环中单独处理（设 llm_finish + is_end_of_turn）
        return 
            type == OmniTokenType::LISTEN ||      // 双工模式下 <|listen|> 结束当前发言
            type == OmniTokenType::CHUNK_EOS ||   // <|chunk_eos|> 结束当前 chunk
            type == OmniTokenType::CHUNK_TTS_EOS; // <|chunk_tts_eos|> 结束当前 TTS chunk
    } else {
        // 单工流式 TTS 模式: 
        // Python (ChunkPrefillChunkGenerate): terminators=["<|tts_eos|>", "<|im_end|>", "</s>"]
        // 🔧 [与旧版本对齐] 单工模式下只检查主要终止 token，避免误判
        return type == OmniTokenType::TTS_EOS ||      // <|tts_eos|>
               type == OmniTokenType::EOS;            // </s> 或 <|endoftext|>
    }
}

// 检查是否是 chunk 结束 token (但不是会话结束)
static bool is_chunk_end_token(struct omni_context * ctx, llama_token token) {
    OmniTokenType type = get_token_type(ctx, token);
    return type == OmniTokenType::CHUNK_EOS || 
           type == OmniTokenType::CHUNK_TTS_EOS;
}

// 获取 token 类型名称（用于日志）
static const char * get_token_type_name(OmniTokenType type) {
    switch (type) {
        case OmniTokenType::NORMAL:        return "NORMAL";
        case OmniTokenType::SPEAK:         return "SPEAK";
        case OmniTokenType::LISTEN:        return "LISTEN";
        case OmniTokenType::CHUNK_EOS:     return "CHUNK_EOS";
        case OmniTokenType::CHUNK_TTS_EOS: return "CHUNK_TTS_EOS";
        case OmniTokenType::TURN_EOS:      return "TURN_EOS";
        case OmniTokenType::TTS_EOS:       return "TTS_EOS";
        case OmniTokenType::EOS:           return "EOS";
        default:                           return "UNKNOWN";
    }
}

//
// omni structure
//
struct unit_buffer {
    std::vector<float> buffer;
    std::string text;
    bool completed = false;
    int unit_n_past = 0;
    float duration;
};

struct omni_output {
    std::vector<unit_buffer *> output;
    int idx;
};

struct LLMOut {
    std::string text;
    int n_past;
    bool llm_finish = false;
    std::string debug_dir;
    // 添加token IDs和hidden states用于TTS条件生成
    std::vector<llama_token> token_ids;  // LLM生成的token IDs
    std::vector<float> hidden_states;   // LLM的hidden states (n_tokens * n_embd)
    int n_embd = 0;  // hidden states的维度
    
    // 🔧 [修复双工缺字问题] 该 chunk 是否是 turn 的最后一个 chunk
    // 此状态随数据一起传递，避免全局状态 current_turn_ended 的时序问题
    // 只有当 LLM 检测到 TURN_EOS/TTS_EOS/EOS 时才设置为 true
    bool is_end_of_turn = false;
};

 struct TTSThreadInfo{
    const int MAX_QUEUE_SIZE;
    std::queue<LLMOut*> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    int n_past = 0;

    TTSThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};

// 前向声明
static void kv_cache_slide_window(struct omni_context* ctx_omni, common_params* params, int chunk_size);

//
// omni mtmd embed
//
bool omni_eval_embed(llama_context * ctx_llama, const struct omni_embed * omni_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < omni_embed->n_pos; i += n_batch) {
        int n_eval = omni_embed->n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (omni_embed->embed + i*n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();
        
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool prefill_with_emb(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int*n_past) {
    kv_cache_slide_window(ctx_omni, params, n_pos);
    
    int n_embd  = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i*n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();
        
        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 prefill_with_emb 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 n_pos * n_embd * sizeof(float)，调用者负责释放
bool prefill_emb_with_hidden(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int* n_past, float *& hidden_states) {
    if (n_pos == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, n_pos);

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(n_pos * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i * n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_ERR("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_ERR("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

void omni_embed_free(struct omni_embed * embed) {
    free(embed->embed);
    free(embed);
}

// 🔧 [高清模式] 返回分离的 chunk embeds（二维 vector）
// 返回值: vision_chunks[0] = overview, vision_chunks[1..n] = slices
static bool encode_image_with_vision_chunks(vision_ctx * ctx_vision, int n_threads, const vision_image_u8 * img, 
                                            std::vector<std::vector<float>> & vision_chunks) {
    const int64_t t_img_enc_start_us = ggml_time_us();
    vision_image_f32_batch img_res_v;
    img_res_v.entries.resize(0);
    img_res_v.entries.clear();
    if (!vision_image_preprocess(ctx_vision, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        img_res_v.entries.clear();
        return false;
    }
    
    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    
    vision_chunks.clear();
    vision_chunks.resize(img_res_v.entries.size());
    
    for (size_t i = 0; i < img_res_v.entries.size(); i++) {
        const int64_t t_img_enc_step_start_us = ggml_time_us();
        
        // 为每个 chunk 分配空间
        vision_chunks[i].resize(n_embd * n_tokens);
        
        bool encoded = vision_image_encode(ctx_vision, n_threads, img_res_v.entries[i].get(), vision_chunks[i].data());
        if (!encoded) {
            LOG_ERR("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.entries.size());
            return false;
        }
        const int64_t t_img_enc_steop_batch_us = ggml_time_us();
        LOG_INF("%s: step %d of %d encoded in %8.2f ms\n", __func__, (int)i+1, (int)img_res_v.entries.size(), (t_img_enc_steop_batch_us - t_img_enc_step_start_us) / 1000.0);
    }
    const int64_t t_img_enc_batch_us = ggml_time_us();
    LOG_INF("%s: all %d chunks encoded in %8.2f ms (grid: %dx%d)\n", __func__, 
            (int)img_res_v.entries.size(), (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0,
            img_res_v.grid_x, img_res_v.grid_y);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    int total_tokens = (int)vision_chunks.size() * n_tokens;
    LOG_INF("\n%s: image encoded in %8.2f ms by vision (%8.2f ms per chunk, %d total tokens)\n", 
            __func__, t_img_enc_ms, t_img_enc_ms / vision_chunks.size(), total_tokens);

    return true;
}

// 保留原有函数用于兼容（将所有 chunk 拼成一个 flat buffer）
static bool encode_image_with_vision(vision_ctx * ctx_vision, int n_threads, const vision_image_u8 * img, float * image_embd, int * n_img_pos) {
    std::vector<std::vector<float>> vision_chunks;
    if (!encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks)) {
        return false;
    }
    
    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    int n_img_pos_out = 0;
    
    for (size_t i = 0; i < vision_chunks.size(); i++) {
        std::memcpy(image_embd + n_img_pos_out * n_embd, vision_chunks[i].data(), n_embd * n_tokens * sizeof(float));
        n_img_pos_out += n_tokens;
    }
    *n_img_pos = n_img_pos_out;
    LOG_INF("%s: image embedding created: %d tokens from %d chunks\n", __func__, *n_img_pos, (int)vision_chunks.size());
    
    return true;
}

static void build_vision_image_from_data(const stbi_uc * data, int nx, int ny, vision_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    std::memcpy(img->buf.data(), data, img->buf.size());
}

bool vision_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct vision_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_vision_image_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

struct omni_embed * omni_image_embed_make_with_bytes(struct vision_ctx * ctx_vision, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }
    int num_max_patches = 10;
    float * image_embed = (float *)malloc(vision_n_mmproj_embd(ctx_vision) * vision_n_output_tokens(ctx_vision) * num_max_patches * sizeof(float));
    if (!image_embed) {
        vision_image_u8_free(img);
        LOG_ERR("Unable to allocate memory for image embeddings\n");
        return NULL;
    }

    LOG_INF("%s: omni_image_embed_make_with_filename s1\n", __func__);
    int n_img_pos = 0;
    if (!encode_image_with_vision(ctx_vision, n_threads, img, image_embed, &n_img_pos)) {
        vision_image_u8_free(img);
        free(image_embed);
        LOG_ERR("%s: cannot encode image, aborting\n", __func__);
        return NULL;
    }
    LOG_INF("%s: omni_image_embed_make_with_filename s2\n", __func__);

    vision_image_u8_free(img);
    auto result = (omni_embed*)malloc(sizeof(omni_embed));
    result->embed = image_embed;
    result->n_pos = n_img_pos;
    return result;
}

struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision, int n_threads, std::string image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path.c_str());
        return NULL;
    }
    LOG_INF("%s: omni_image_embed_make_with_filename: %s\n", __func__, image_path.c_str());
    omni_embed *embed = omni_image_embed_make_with_bytes(ctx_vision, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

// 🔧 [高清模式] 创建带 chunks 的 vision embed（用于 V2.6 slice schema）
// 返回的 vector: [0] = overview, [1..n] = slices
bool omni_image_embed_make_chunks_with_filename(struct vision_ctx * ctx_vision, int n_threads, 
                                                 std::string image_path, 
                                                 std::vector<std::vector<float>> & vision_chunks) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path.c_str());
        return false;
    }
    
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        free(image_bytes);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return false;
    }
    free(image_bytes);
    
    bool success = encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks);
    vision_image_u8_free(img);
    
    if (success) {
        LOG_INF("%s: created %d vision chunks from %s\n", __func__, (int)vision_chunks.size(), image_path.c_str());
    }
    return success;
}

bool audition_read_binary_file(const char * fname, std::vector<uint8_t> * buf_res) {
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return false;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf_res->resize(file_size);

    size_t n_read = fread(buf_res->data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s\n", fname);
        return false;
    }

    return true;
}

struct omni_embed * omni_audio_embed_make_with_bytes(audition_ctx * ctx_audio, int n_threads, audition_audio_u8 * audio) {
    audition_audio_f32 * res_auds = audition_audio_f32_init();
    // printf("omni_audio_embed_make_with_bytes 1 :\n");
    if (!audition_audio_preprocess(ctx_audio, audio, &res_auds)) {
        LOG_ERR("%s: failed to preprocess audio file\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 2 :\n");
    // 分配空间
    int n_embd = audition_n_mmproj_embd(ctx_audio);
    int n_tokens = audition_n_output_tokens(ctx_audio, res_auds);
    std::vector<float> output_buffer(n_embd * n_tokens);
    
    if (!audition_audio_encode(ctx_audio, n_threads, res_auds, output_buffer.data())) {
        LOG_ERR("%s: cannot encode audio, aborting\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 4 :\n");
    auto result = (omni_embed*)malloc(sizeof(omni_embed));
    result->embed = (float *)malloc(output_buffer.size() * sizeof(float));
    if (!result->embed) {
        free(result);
        audition_audio_f32_free(res_auds);
        LOG_ERR("%s: failed to allocate memory for audio embeddings\n", __func__);
        return NULL;
    }
    std::memcpy(result->embed, output_buffer.data(), output_buffer.size() * sizeof(float));
    result->n_pos = n_tokens;
    
    audition_audio_f32_free(res_auds);
    // printf("===audio embed tokens: %d %d\n", result->n_pos, ret.buf.size() / ret.n_len);
    return result;
}

struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audio, int n_threads, std::string audio_path) {
    audition_audio_u8 * audio = audition_audio_u8_init();
    // printf("omni_audio_embed_make_with_filename 1 :%s\n", audio_path.c_str());
    if (!audition_read_binary_file(audio_path.c_str(), &audio->buf)) {
        LOG_ERR("%s: failed to read audio file %s\n", __func__,  audio_path.c_str());
        return NULL;
    }
    // printf("omni_audio_embed_make_with_filename 2 :%s\n", audio_path.c_str());
    omni_embed *embed = omni_audio_embed_make_with_bytes(ctx_audio, n_threads, audio);
    if (embed == NULL) {
        LOG_ERR("%s: failed to preprocess audio file, %s\n", __func__, audio_path.c_str());
    }

    audition_audio_u8_free(audio);
    // printf("omni_audio_embed_make_with_filename 3 :%s\n", audio_path.c_str());
    return embed;
}

//
// omni llm eval
//
static void kv_cache_slide_window(struct omni_context* ctx_omni, common_params* params, int chunk_size) {
    const int n_ctx = params->n_ctx;
    
    // 检查是否需要滑动窗口
    if (ctx_omni->n_past + chunk_size < n_ctx) {
        return; // 还有足够空间，无需滑动
    }
    
    // 🔧 [诊断] 打印滑动窗口触发信息
    print_with_timestamp("⚠️ KV Cache 滑动窗口触发: n_past=%d, chunk_size=%d, n_ctx=%d, n_keep=%d, 轮次数=%zu\n",
                         ctx_omni->n_past, chunk_size, n_ctx, ctx_omni->n_keep, 
                         ctx_omni->round_start_positions.size());
    
    int n_discard = 0;
    int delete_end_pos = ctx_omni->n_keep;  // 删除范围的结束位置
    
    // ==================== 按轮次边界删除（优先使用） ====================
    // 🔧 [重要] round_start_positions 记录的是轮次**结束**位置（也是下一轮开始位置）
    // 轮次布局：
    //   轮次 0: [n_keep, round_start_positions[0])
    //   轮次 1: [round_start_positions[0], round_start_positions[1])
    //   轮次 i: [round_start_positions[i-1], round_start_positions[i])
    //   当前轮（未结束）: [round_start_positions[size-1], n_past)
    if (ctx_omni->max_preserved_context > 0 && ctx_omni->round_start_positions.size() >= 1) {
        // 策略：保留尽可能多的最近轮次，但总长度不超过 max_preserved_context
        // 从最新轮次往前数，累计长度直到超过 max_preserved_context
        
        const auto& rounds = ctx_omni->round_start_positions;
        int cumulative_length = 0;
        int keep_from_round = rounds.size();  // 从哪个轮次开始保留（0-indexed）
        int total_rounds = rounds.size();  // 总轮次数（已完成的轮次）
        
        // 从最后一个已完成轮次往前遍历
        for (int i = total_rounds - 1; i >= 0; --i) {
            // 计算第 i 轮的长度
            // 轮次 i 的范围是 [round_start(i), round_end(i))
            int round_start = (i == 0) ? ctx_omni->n_keep : rounds[i - 1];
            int round_end = rounds[i];
            int round_length = round_end - round_start;
            
            if (cumulative_length + round_length > ctx_omni->max_preserved_context) {
                // 加上这一轮会超过限制，停止
                break;
            }
            
            cumulative_length += round_length;
            keep_from_round = i;
        }
        
        // 至少保留最近一轮
        if (keep_from_round >= total_rounds) {
            keep_from_round = total_rounds - 1;
        }
        
        // 计算要删除的范围
        // 保留轮次 [keep_from_round, total_rounds)
        // 删除轮次 [0, keep_from_round)
        // 删除范围：[n_keep, 轮次 keep_from_round 的开始位置)
        int delete_start = ctx_omni->n_keep;
        delete_end_pos = (keep_from_round == 0) ? ctx_omni->n_keep : rounds[keep_from_round - 1];
        
        if (delete_end_pos > delete_start) {
            n_discard = delete_end_pos - delete_start;
            
            print_with_timestamp("⚠️ 按轮次删除: 删除轮次 0-%d，保留轮次 %d-%d，保留长度=%d\n",
                                 keep_from_round - 1, keep_from_round, total_rounds - 1, cumulative_length);
            
            // 更新 round_start_positions：删除早期轮次，调整剩余轮次的位置
            std::vector<int> new_rounds;
            for (int i = keep_from_round; i < total_rounds; ++i) {
                new_rounds.push_back(rounds[i] - n_discard);
            }
            ctx_omni->round_start_positions = new_rounds;
            
            print_with_timestamp("⚠️ 更新轮次边界: 新边界数=%zu，首轮结束位置=%d\n",
                                 new_rounds.size(), new_rounds.empty() ? -1 : new_rounds[0]);
        } else {
            // 没有可删除的完整轮次，回退到按比例删除
            print_with_timestamp("⚠️ 没有可删除的完整轮次（keep_from_round=%d），回退到按比例删除\n", keep_from_round);
            n_discard = 0;
        }
    }
    
    // ==================== 回退策略：按比例删除（旧逻辑） ====================
    if (n_discard == 0) {
        const int n_left = ctx_omni->n_past - ctx_omni->n_keep;
        n_discard = n_left / 2;
        delete_end_pos = ctx_omni->n_keep + n_discard;
        
        // 边界检查
        if (n_left <= 0 || n_discard <= 0) {
            print_with_timestamp("⚠️ KV Cache 滑动窗口: 边界检查失败 n_left=%d, n_discard=%d，跳过滑动\n",
                                 n_left, n_discard);
            return;
        }
        
        // 按比例删除时，更新 round_start_positions
        // round_start_positions[i] 表示第 i 轮的结束位置
        // 删除范围 [n_keep, delete_end_pos) 会影响轮次边界
        std::vector<int> new_rounds;
        for (int pos : ctx_omni->round_start_positions) {
            if (pos > delete_end_pos) {
                // 这个轮次结束位置在删除范围之后，需要前移
                new_rounds.push_back(pos - n_discard);
            }
            // 如果 pos <= delete_end_pos，说明这个轮次被删除或部分删除，丢弃
        }
        ctx_omni->round_start_positions = new_rounds;
        
        print_with_timestamp("⚠️ 按比例删除后轮次边界: 剩余 %zu 个轮次\n", new_rounds.size());
        
        print_with_timestamp("⚠️ 按比例删除: n_left=%d, n_discard=%d, 删除范围=[%d, %d)\n",
                             n_left, n_discard, ctx_omni->n_keep, delete_end_pos);
    }
    
    // ==================== 执行 KV Cache 操作 ====================
    print_with_timestamp("⚠️ KV Cache 滑动窗口执行: 删除范围=[%d, %d), n_discard=%d\n",
                         ctx_omni->n_keep, delete_end_pos, n_discard);
    
    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (mem) {
        // 1. 删除 [n_keep, delete_end_pos) 范围的 token
        bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, delete_end_pos);
        (void)rm_ok;
        
        // 2. 将 [delete_end_pos, n_past) 范围的 token 位置前移 n_discard
        llama_memory_seq_add(mem, 0, delete_end_pos, ctx_omni->n_past, -n_discard);
    }
    
    // 3. 更新 n_past
    int old_n_past = ctx_omni->n_past;
    ctx_omni->n_past -= n_discard;
    print_with_timestamp("⚠️ KV Cache 滑动窗口完成: n_past 从 %d 减少到 %d\n", old_n_past, ctx_omni->n_past);
}

static bool eval_tokens(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, bool get_emb = false) {
    int N = (int) tokens.size();
    kv_cache_slide_window(ctx_omni, params, N);

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, true);
        }
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }
        
        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, false);
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 eval_tokens 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 N * n_embd * sizeof(float)，调用者负责释放
static bool eval_tokens_with_hidden(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, float *& hidden_states) {
    int N = (int) tokens.size();
    if (N == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, N);

    const int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(N * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            // 将当前 batch 的 embeddings 复制到 hidden_states 的对应位置
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

static bool eval_id(struct omni_context * ctx_omni, common_params* params, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_omni, params, tokens, 1, n_past);
}

static bool eval_id_with_hidden(struct omni_context * ctx_omni, common_params* params, int id, int * n_past, float *& hidden_states) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens_with_hidden(ctx_omni, params, tokens, 1, n_past, hidden_states);
}

static bool eval_string(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, bool get_emb = false) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_omni, params, embd_inp, n_batch, n_past, get_emb);
}

static bool eval_string_with_hidden(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, float *& hidden_states) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens_with_hidden(ctx_omni, params, embd_inp, n_batch, n_past, hidden_states);
}

static const char * sample(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id(ctx_omni, params, id, n_past);
    return ret.c_str();
}

static const char * sample_with_hidden(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

static const char * llama_loop(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past) {
    const char * tmp = sample(smpl, ctx_omni, params, &n_past);
    return tmp;
}

// 修改sample_with_hidden来返回token ID（通过引用参数）
// 🔧 [双工模式] 支持 listen_prob_scale 参数，增加 <|listen|> 的采样概率
// 🔧 [双工模式] 支持 forbidden_token_ids，禁止采样 <|tts_pad|> 等 token
static const char * sample_with_hidden_and_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states, llama_token & token_id) {
    float * logits = llama_get_logits_ith(ctx_omni->ctx_llama, -1);
    
    // 🔧 [双工模式] 在采样前调整 logits
    if (ctx_omni->duplex_mode) {
        if (logits != nullptr) {
            // 1. 调整 <|listen|> 的 logit（listen_prob_scale）
            // listen_prob_scale > 1.0 会增加 <|listen|> 的概率，让模型更倾向于先听
            if (ctx_omni->special_token_listen >= 0) {
                // 使用 listen_prob_scale 调整 <|listen|> 的 logit
                // 默认值 1.0 不改变，> 1.0 增加 listen 概率
                // 这里我们使用加法而不是乘法，因为 logit 可能是负数
                // 添加一个偏置值来增加 listen 的概率
                // listen_prob_bias = log(listen_prob_scale) ≈ (listen_prob_scale - 1.0) for small values
                float listen_bias = (ctx_omni->listen_prob_scale - 1.0f) * 2.0f;  // 放大效果
                logits[ctx_omni->special_token_listen] += listen_bias;
            }
            
            // 2. 🔧 [与 Python 对齐] 禁止采样 <|tts_pad|> token
            // Python: self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
            //         logits[:, self.forbidden_token_ids] = float("-inf")
            // <|tts_pad|> 是填充 token，模型不应该主动生成它
            // 如果不禁止，模型可能生成 <|speak|> → <|tts_pad|> → <|chunk_eos|>，导致无有效输出
            if (ctx_omni->special_token_tts_pad >= 0) {
                logits[ctx_omni->special_token_tts_pad] = -INFINITY;
            }
        }
    }
    
    // 🔧 [Length Penalty] 调整 EOS token 的 logit 值（单工模式）
    // length_penalty > 1.0 会降低 EOS 概率，让模型生成更长的输出
    if (!ctx_omni->duplex_mode && ctx_omni->length_penalty != 1.0f && ctx_omni->special_token_tts_eos >= 0) {
        if (logits != nullptr) {
            float eos_logit = logits[ctx_omni->special_token_tts_eos];
            if (eos_logit > 0) {
                // logit > 0 时，除以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit / ctx_omni->length_penalty;
            } else {
                // logit <= 0 时，乘以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit * ctx_omni->length_penalty;
            }
        }
    }
    
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    token_id = id;  // 保存token ID
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

static const char * llama_loop_with_hidden(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states) {
    llama_token dummy_token;
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, dummy_token);
    return tmp;
}

// 新增：返回token ID的版本
static const char * llama_loop_with_hidden_and_token(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states, llama_token & token_id) {
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, token_id);
    return tmp;
}

//
// TTS specific helper functions
//

// Helper function to get RNG from sampler chain for multinomial sampling
// Uses common_sampler_get_rng() from common/sampling.cpp
// This ensures audio_bos sampling uses the same RNG as non-audio_bos sampling
static std::mt19937* get_sampler_rng(struct common_sampler * smpl) {
    void* rng_ptr = common_sampler_get_rng(smpl);
    return static_cast<std::mt19937*>(rng_ptr);
}

// ==============================================================================
// Projector Semantic 实现 (精度验证版本)
// 使用 ggml 后端进行计算，支持 CUDA 加速
// forward(x): relu(linear1(x)) -> linear2
// ==============================================================================
bool projector_init(projector_model & model, const std::string & fname, bool use_cuda) {
    
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("Projector: failed to open '%s'\n", fname.c_str());
        return false;
    }
    
#ifdef GGML_USE_CUDA
    if (use_cuda) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
        if (!model.backend) {
            model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
    } else {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
#else
    (void)use_cuda;
    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
#endif

    if (!model.backend) {
        LOG_ERR("Projector: failed to init backend\n");
        gguf_free(ctx_gguf);
        return false;
    }
    
    model.buf_type = ggml_backend_get_default_buffer_type(model.backend);
    
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    model.ctx_w = ggml_init(ctx_params);
    
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        enum ggml_type type = gguf_get_tensor_type(ctx_gguf, i);
        struct ggml_tensor * tensor = nullptr;
        
        if (strcmp(name, "linear1.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 4096, 768);
            model.layer.linear1_weight = tensor;
            model.hparams.in_dim = 4096;
            model.hparams.out_dim = 768;
        } else if (strcmp(name, "linear1.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear1_bias = tensor;
        } else if (strcmp(name, "linear2.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 768, 768);
            model.layer.linear2_weight = tensor;
        } else if (strcmp(name, "linear2.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear2_bias = tensor;
        } else {
            continue;
        }
        
        if (tensor) {
            ggml_set_name(tensor, name);
        }
    }
    
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        LOG_ERR("Projector: failed to open file for reading\n");
        return false;
    }
    
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        fseek(f, offset, SEEK_SET);
        
        size_t tensor_size = ggml_nbytes(tensor);
        void * data = malloc(tensor_size);
        if (fread(data, 1, tensor_size, f) != tensor_size) {
            LOG_ERR("Projector: failed to read tensor %s\n", name);
            free(data);
            fclose(f);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, data, 0, tensor_size);
        free(data);
    }
    
    fclose(f);
    gguf_free(ctx_gguf);
    
    model.initialized = true;
    return true;
}

void projector_free(projector_model & model) {
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.buf_w) ggml_backend_buffer_free(model.buf_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.ctx_w = nullptr;
    model.buf_w = nullptr;
    model.backend = nullptr;
    model.initialized = false;
}

static struct ggml_cgraph * projector_build_graph(projector_model & model, struct ggml_context * ctx, struct ggml_tensor * input) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // linear1 + relu
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, model.layer.linear1_weight, input);
    hidden = ggml_add(ctx, hidden, model.layer.linear1_bias);
    hidden = ggml_relu(ctx, hidden);
    
    // linear2
    struct ggml_tensor * output = ggml_mul_mat(ctx, model.layer.linear2_weight, hidden);
    output = ggml_add(ctx, output, model.layer.linear2_bias);
    
    ggml_build_forward_expand(gf, output);
    return gf;
}

std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens) {
    const int in_dim = model.hparams.in_dim;
    const int out_dim = model.hparams.out_dim;
    
    // 🔧 [安全检查] 验证参数
    if (n_tokens <= 0 || n_tokens > 10000) {
        LOG_ERR("projector_forward: invalid n_tokens=%d\n", n_tokens);
        return {};
    }
    if (in_dim <= 0 || in_dim > 10000 || out_dim <= 0 || out_dim > 10000) {
        LOG_ERR("projector_forward: invalid dimensions in_dim=%d, out_dim=%d\n", in_dim, out_dim);
        return {};
    }
    
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    struct ggml_cgraph * gf = projector_build_graph(model, ctx, input);
    
    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buf_compute) {
        LOG_ERR("Projector: failed to allocate compute buffer\n");
        ggml_free(ctx);
        return {};
    }
    
    ggml_backend_tensor_set(input, input_data, 0, n_tokens * in_dim * sizeof(float));
    
    enum ggml_status status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("Projector: graph compute failed with status %d\n", (int)status);
        ggml_backend_buffer_free(buf_compute);
        ggml_free(ctx);
        return {};
    }
    
    struct ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float> result(n_tokens * out_dim);
    ggml_backend_tensor_get(output, result.data(), 0, n_tokens * out_dim * sizeof(float));
    
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx);
    
    return result;
}
// ==============================================================================

// Load TTS weights from GGUF file
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path) {
    
    // Initialize GGUF context
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(tts_model_path, params);
    if (!ctx_gguf) {
        LOG_ERR("TTS: Failed to load GGUF file: %s\n", tts_model_path);
        return false;
    }
    
    // Load emb_code.0.weight: (num_audio_tokens=6562, hidden_size=768)
    // This is used to convert audio token IDs to embeddings during decode phase
    const char * emb_code_name = "emb_code.0.weight";
    int64_t emb_code_idx = gguf_find_tensor(ctx_gguf, emb_code_name);
    if (emb_code_idx >= 0) {
        struct ggml_tensor * emb_code_tensor = ggml_get_tensor(ctx_meta, emb_code_name);
        if (emb_code_tensor) {
            // emb_code is Embedding(num_audio_tokens, hidden_size)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (transposed)
            int64_t dim0 = emb_code_tensor->ne[0];
            int64_t dim1 = emb_code_tensor->ne[1];
            
            // Determine which dimension is which based on expected values
            int64_t num_audio_tokens = 6562;
            int64_t hidden_size = 768;
            
            // GGUF stores as (hidden_size, num_audio_tokens) = [768, 6562]
            if (dim0 == hidden_size && dim1 == num_audio_tokens) {
                // Correct: stored as (hidden_size, num_audio_tokens)
            } else if (dim0 == num_audio_tokens && dim1 == hidden_size) {
                // Stored as (num_audio_tokens, hidden_size) - need to transpose
                num_audio_tokens = dim0;
                hidden_size = dim1;
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unexpected shape [%ld, %ld], expected [768, 6562] or [6562, 768]\n", dim0, dim1);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory and copy data
            size_t emb_code_size = dim0 * dim1 * sizeof(float);
            ctx_omni->emb_code_weight = (float *)malloc(emb_code_size);
            if (!ctx_omni->emb_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_code.0.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_code_type = emb_code_tensor->type;
            int64_t emb_code_elements = dim0 * dim1;
            
            if (emb_code_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(ctx_omni->emb_code_weight, emb_code_tensor->data, emb_code_size);
            } else if (emb_code_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)emb_code_tensor->data;
                for (int64_t i = 0; i < emb_code_elements; ++i) {
                    ctx_omni->emb_code_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unsupported type: %d\n", emb_code_type);
                free(ctx_omni->emb_code_weight);
                ctx_omni->emb_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->emb_code_vocab_size = num_audio_tokens;  // 6562
            ctx_omni->emb_code_hidden_size = hidden_size;     // 768
            // NOTE: GGUF data is stored in row-major order, matching NumPy/PyTorch
            // Even when metadata shape is [768, 6562], the actual data layout is (6562, 768) row-major
            // So we should access as: weight[token_idx * hidden_size + j]
            // This means: emb_code_stored_as_transposed should be FALSE
            ctx_omni->emb_code_stored_as_transposed = false; // Data is always (vocab_size, hidden_size) in memory
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_code_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file (this is OK if using token IDs)\n", emb_code_name);
        // Note: emb_code is optional if we use token IDs, but we prefer embeddings
    }
    
    // Load emb_text.weight: (vocab_size=152064, hidden_size=768)
    // PyTorch: nn.Embedding(vocab_size, hidden_size) -> weight shape is [vocab_size, hidden_size] = [152064, 768]
    // GGUF: may be stored as [hidden_size, vocab_size] = [768, 152064] (transposed)
    const char * emb_text_name = "emb_text.weight";
    int64_t emb_text_idx = gguf_find_tensor(ctx_gguf, emb_text_name);
    if (emb_text_idx >= 0) {
        struct ggml_tensor * emb_text_tensor = ggml_get_tensor(ctx_meta, emb_text_name);
        if (emb_text_tensor) {
            int64_t dim0 = emb_text_tensor->ne[0];
            int64_t dim1 = emb_text_tensor->ne[1];
            
            // Expected values
            int64_t expected_vocab_size = 152064;
            int64_t expected_hidden_size = 768;
            
            // GGML tensor 维度理解：
            // ne[0] = 最内层维度 (stride=1) = hidden_size = 768
            // ne[1] = 外层维度 = vocab_size = 152064
            // 内存布局是 row-major，即 [vocab_size][hidden_size]
            // 所以 ne[0]=768, ne[1]=152064 意味着数据已经是 [vocab_size, hidden_size] 格式
            // 不需要转置！
            
            int64_t vocab_size, hidden_size;
            
            if (dim0 == expected_hidden_size && dim1 == expected_vocab_size) {
                // GGML shape: ne[0]=768, ne[1]=152064
                // 这意味着内存布局是 [vocab_size=152064][hidden_size=768]
                // 不需要转置
                vocab_size = dim1;   // 152064
                hidden_size = dim0;  // 768
            } else if (dim0 == expected_vocab_size && dim1 == expected_hidden_size) {
                // GGML shape: ne[0]=152064, ne[1]=768 (unusual)
                // 这意味着内存布局是 [hidden_size=768][vocab_size=152064]
                // 这种情况需要转置
                vocab_size = dim0;   // 152064
                hidden_size = dim1;  // 768
                LOG_ERR("TTS: emb_text.weight has unusual GGML shape, not handled\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            } else {
                LOG_ERR("TTS: emb_text.weight has unexpected shape [%ld, %ld], expected ne=[%ld, %ld]\n", 
                       dim0, dim1, expected_hidden_size, expected_vocab_size);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory for the weight
            size_t emb_text_size = vocab_size * hidden_size * sizeof(float);
            ctx_omni->emb_text_weight = (float *)malloc(emb_text_size);
            if (!ctx_omni->emb_text_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_text.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_text_type = emb_text_tensor->type;
            int64_t emb_text_elements = vocab_size * hidden_size;
            
            if (emb_text_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(ctx_omni->emb_text_weight, emb_text_tensor->data, emb_text_size);
            } else if (emb_text_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)emb_text_tensor->data;
                for (int64_t i = 0; i < emb_text_elements; ++i) {
                    ctx_omni->emb_text_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_text.weight has unsupported type: %d\n", emb_text_type);
                free(ctx_omni->emb_text_weight);
                ctx_omni->emb_text_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->emb_text_vocab_size = vocab_size;   // 152064
            ctx_omni->emb_text_hidden_size = hidden_size;  // 768
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_text_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", emb_text_name);
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }
    
    // Load projector_semantic weights
    const char * projector_names[] = {
        "projector_semantic.linear1.weight",
        "projector_semantic.linear1.bias",
        "projector_semantic.linear2.weight",
        "projector_semantic.linear2.bias"
    };
    
    float ** projector_ptrs[] = {
        &ctx_omni->projector_semantic_linear1_weight,
        &ctx_omni->projector_semantic_linear1_bias,
        &ctx_omni->projector_semantic_linear2_weight,
        &ctx_omni->projector_semantic_linear2_bias
    };
    
    // PyTorch nn.Linear(in_features, out_features) weight shape is (out_features, in_features)
    // GGUF may store as (out_features, in_features) or (in_features, out_features)
    // We need to detect and handle both cases
    int64_t expected_shapes_pytorch[][2] = {
        {768, 4096},  // linear1.weight: PyTorch shape (out_features, in_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: PyTorch shape (out_features, in_features)
        {768, 0}      // linear2.bias (1D)
    };
    
    int64_t expected_shapes_transposed[][2] = {
        {4096, 768},  // linear1.weight: transposed shape (in_features, out_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: same for square matrix
        {768, 0}      // linear2.bias (1D)
    };
    
    // Track whether weights need transposition
    bool need_transpose[2] = {false, false};  // [linear1, linear2]
    
    for (int i = 0; i < 4; i++) {
        int64_t tensor_idx = gguf_find_tensor(ctx_gguf, projector_names[i]);
        if (tensor_idx >= 0) {
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, projector_names[i]);
            if (tensor) {
                int64_t dim0 = tensor->ne[0];
                int64_t dim1 = (ggml_n_dims(tensor) > 1) ? tensor->ne[1] : 0;
                
                if (i % 2 == 0) {  // weight (2D)
                    // Check if stored as PyTorch shape (out_features, in_features) or transposed
                    bool is_pytorch_shape = (dim0 == expected_shapes_pytorch[i][0] && 
                                           dim1 == expected_shapes_pytorch[i][1]);
                    bool is_transposed_shape = (dim0 == expected_shapes_transposed[i][0] && 
                                               dim1 == expected_shapes_transposed[i][1]);
                    
                    if (is_pytorch_shape) {
                        // Stored as PyTorch shape (out_features, in_features), need to transpose
                        need_transpose[i / 2] = true;
                    } else if (is_transposed_shape) {
                        // Already transposed, use directly
                        need_transpose[i / 2] = false;
                    } else {
                        LOG_ERR("TTS: %s has unexpected shape: [%ld, %ld], expected [%ld, %ld] or [%ld, %ld]\n",
                               projector_names[i], dim0, dim1, 
                               expected_shapes_pytorch[i][0], expected_shapes_pytorch[i][1],
                               expected_shapes_transposed[i][0], expected_shapes_transposed[i][1]);
                        // Try to continue, assume PyTorch shape
                        need_transpose[i / 2] = true;
                    }
                } else {  // bias (1D)
                    if (dim0 != expected_shapes_pytorch[i][0] || dim1 != 0) {
                        LOG_ERR("TTS: %s has wrong shape: [%ld, %ld], expected [%ld, 0]\n",
                               projector_names[i], dim0, dim1, expected_shapes_pytorch[i][0]);
                    }
                }
                
                // Check tensor type for F16 conversion
                enum ggml_type proj_tensor_type = tensor->type;
                
                if (i % 2 == 0 && need_transpose[i / 2]) {
                    // Weight needs transposition: allocate transposed size (always F32 output)
                    int64_t in_dim = expected_shapes_pytorch[i][1];  // PyTorch in_features
                    int64_t out_dim = expected_shapes_pytorch[i][0];  // PyTorch out_features
                    size_t transposed_size = in_dim * out_dim * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(transposed_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for transposed %s\n", projector_names[i]);
                        // Clean up
                        for (int j = 0; j < i; j++) {
                            if (*projector_ptrs[j]) {
                                free(*projector_ptrs[j]);
                                *projector_ptrs[j] = nullptr;
                            }
                        }
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                    
                    // Transpose: src[out_dim][in_dim] -> dst[in_dim][out_dim], handling F16 if needed
                    float * dst_data = *projector_ptrs[i];
                    if (proj_tensor_type == GGML_TYPE_F32) {
                        const float * src_data = (const float *)tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = src_data[out * in_dim + in];
                            }
                        }
                    } else if (proj_tensor_type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src_data = (const ggml_fp16_t *)tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = ggml_fp16_to_fp32(src_data[out * in_dim + in]);
                            }
                        }
                    } else {
                        LOG_ERR("TTS: %s has unsupported type: %d\n", projector_names[i], proj_tensor_type);
                        free(*projector_ptrs[i]);
                        *projector_ptrs[i] = nullptr;
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                } else {
                    // Direct copy (bias or already transposed weight), handle F16
                    int64_t num_elements = (dim1 > 0) ? dim0 * dim1 : dim0;
                    size_t output_size = num_elements * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(output_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for %s\n", projector_names[i]);
                        // Clean up
                        for (int j = 0; j < i; j++) {
                            if (*projector_ptrs[j]) {
                                free(*projector_ptrs[j]);
                                *projector_ptrs[j] = nullptr;
                            }
                        }
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                    
                    if (proj_tensor_type == GGML_TYPE_F32) {
                        memcpy(*projector_ptrs[i], tensor->data, output_size);
                    } else if (proj_tensor_type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)tensor->data;
                        for (int64_t k = 0; k < num_elements; ++k) {
                            (*projector_ptrs[i])[k] = ggml_fp16_to_fp32(src_f16[k]);
                        }
                    } else {
                        LOG_ERR("TTS: %s has unsupported type: %d\n", projector_names[i], proj_tensor_type);
                        free(*projector_ptrs[i]);
                        *projector_ptrs[i] = nullptr;
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                    if (dim1 > 0) {
                    }
                }
            } else {
                LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", projector_names[i]);
                // Clean up
                for (int j = 0; j < i; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
        } else {
            LOG_ERR("TTS: Tensor %s not found in GGUF file\n", projector_names[i]);
            // Clean up
            for (int j = 0; j < i; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    }
    
    // Set projector dimensions
    ctx_omni->projector_semantic_input_dim = 4096;
    ctx_omni->projector_semantic_output_dim = 768;
    
    // Load head_code.weight: (hidden_size=768, num_audio_tokens=6562)
    // Note: num_vq=1, so we only load head_code.0.weight
    const char * head_code_name = "head_code.0.weight";
    int64_t head_code_idx = gguf_find_tensor(ctx_gguf, head_code_name);
    if (head_code_idx >= 0) {
        struct ggml_tensor * head_code_tensor = ggml_get_tensor(ctx_meta, head_code_name);
        if (head_code_tensor) {
            // head_code is Linear(hidden_size, num_audio_tokens, bias=False)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (already transposed)
            int64_t dim0 = head_code_tensor->ne[0];
            int64_t dim1 = (ggml_n_dims(head_code_tensor) > 1) ? head_code_tensor->ne[1] : 0;
            
            // Expected shape in GGUF: (hidden_size=768, num_audio_tokens=6562)
            int64_t expected_hidden_size = 768;
            int64_t expected_num_audio_tokens = 6562;
            
            // Allocate memory for weight: (hidden_size, num_audio_tokens) = [768, 6562]
            size_t head_code_size = expected_hidden_size * expected_num_audio_tokens * sizeof(float);
            ctx_omni->head_code_weight = (float *)malloc(head_code_size);
            if (!ctx_omni->head_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for head_code.0.weight\n");
                // Clean up already loaded weights
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // CRITICAL FIX: The conversion script transposes head_code weight to [768, 6562] before saving,
            // but the GGUF metadata shape may still be [6562, 768] due to how add_tensor works.
            // We need to detect the actual data format by testing both layouts.
            // 
            // Strategy: Try both formats and see which one produces correct logits.
            // But a simpler approach: Since the conversion script always transposes to [768, 6562],
            // and the actual data is stored in that format, we should always use the data as-is
            // (treating it as [768, 6562]) regardless of metadata shape.
            //
            // However, to be safe, we'll check: if metadata says [6562, 768], the data might be
            // in that format (old conversion) or already transposed (new conversion).
            // We'll use a heuristic: check if the first few values match what we expect.
            
            const float * src_data = (const float *)head_code_tensor->data;
            bool need_transpose = false;
            
            // CRITICAL FIX: Based on conversion script analysis, the script always transposes
            // head_code to [768, 6562] before saving. So the actual data is always [768, 6562],
            // regardless of metadata shape. We should NOT transpose based on metadata.
            //
            // However, if metadata says [6562, 768], it might be an old conversion that didn't transpose.
            // We'll use a simple heuristic: if metadata says [6562, 768], assume data needs transpose.
            // If metadata says [768, 6562], assume data is already correct.
            
            if (dim0 == expected_hidden_size && dim1 == expected_num_audio_tokens) {
                // Metadata says (768, 6562) - data should already be in correct format
                need_transpose = false;
            } else if (dim0 == expected_num_audio_tokens && dim1 == expected_hidden_size) {
                // Metadata says (6562, 768) - but conversion script may have already transposed the data
                // We need to check: if conversion script transposed, data is actually [768, 6562] and we should NOT transpose
                // If conversion script didn't transpose, data is [6562, 768] and we SHOULD transpose
                //
                // Since the conversion script ALWAYS transposes (line 351: W_transposed = W.T),
                // the data is always [768, 6562] regardless of metadata.
                // So we should NOT transpose.
                need_transpose = false;  // CRITICAL FIX: Don't transpose, data is already [768, 6562]
            } else {
                LOG_ERR("TTS: head_code.0.weight has unexpected shape [%ld, %ld], expected [%d, %d] or [%d, %d]\n",
                       dim0, dim1, expected_hidden_size, expected_num_audio_tokens, expected_num_audio_tokens, expected_hidden_size);
                // Clean up
                free(ctx_omni->head_code_weight);
                ctx_omni->head_code_weight = nullptr;
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Check tensor type and copy/convert accordingly
            // ⚡ 优化：转置存储为 [6562, 768]，使每个output token的权重连续存储
            // 这样在计算logits时可以用高效的向量点积
            // 原始: weight[j * 6562 + i] = W[j, i]  (j=hidden, i=output)
            // 转置后: weight[i * 768 + j] = W[i, j] (i=output, j=hidden)
            enum ggml_type tensor_type = head_code_tensor->type;
            int64_t total_elements = expected_hidden_size * expected_num_audio_tokens;
            
            print_with_timestamp("TTS: head_code shape: dim0=%ld, dim1=%ld, will transpose to [%ld, %ld]\n", 
                                dim0, dim1, expected_num_audio_tokens, expected_hidden_size);
            
            if (tensor_type == GGML_TYPE_F32) {
                // F32: copy with transpose from [768, 6562] to [6562, 768]
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        // src: [j * 6562 + i], dst: [i * 768 + j]
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = src_data[j * expected_num_audio_tokens + i];
                    }
                }
            } else if (tensor_type == GGML_TYPE_F16) {
                // F16: convert to F32 with transpose
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)src_data;
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = 
                            ggml_fp16_to_fp32(src_f16[j * expected_num_audio_tokens + i]);
                    }
                }
            } else {
                LOG_ERR("TTS: head_code.0.weight has unsupported type: %d\n", tensor_type);
                free(ctx_omni->head_code_weight);
                ctx_omni->head_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->head_code_hidden_size = expected_hidden_size;
            ctx_omni->head_code_num_audio_tokens = expected_num_audio_tokens;
            
            // 🔍 调试：验证加载的数据
            // Python weight[0, 0:5] = [0.01385498, -0.01647949, 0.0111084, -0.01367188, -0.01141357]
            // C++ head_code_weight[0..4] 应该和 Python 一致
            print_with_timestamp("TTS: head_code loaded, verifying first few values:\n");
            print_with_timestamp("  head_code_weight[0] = %.8f (expect ~0.01385498)\n", ctx_omni->head_code_weight[0]);
            print_with_timestamp("  head_code_weight[1] = %.8f (expect ~-0.01647949)\n", ctx_omni->head_code_weight[1]);
            print_with_timestamp("  head_code_weight[768] = %.8f (this is weight[1, 0], expect ~0.04150391)\n", ctx_omni->head_code_weight[768]);
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", head_code_name);
            // Clean up
            if (ctx_omni->emb_text_weight) {
                free(ctx_omni->emb_text_weight);
                ctx_omni->emb_text_weight = nullptr;
            }
            for (int j = 0; j < 4; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", head_code_name);
        // Clean up
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
        }
        for (int j = 0; j < 4; j++) {
            if (*projector_ptrs[j]) {
                free(*projector_ptrs[j]);
                *projector_ptrs[j] = nullptr;
            }
        }
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }
    
    ggml_free(ctx_meta);
    gguf_free(ctx_gguf);
    return true;
}

// TODO: 实现TTS的emb_text层调用
// 功能：将token ID转换为embedding
// 输入：token_id (llama_token), TTS模型上下文
// 输出：embedding向量 (float*, 大小为tts_n_embd)
// 实现方式：
//   1. 从TTS模型文件中加载emb_text权重（152064 x 768）
//      - 在omni_init中加载emb_text权重到ctx_omni中
//      - 权重名称：根据GGUF格式，可能是"tts.emb_text.weight"或类似
//   2. 查找：embedding = emb_text_weight[token_id]
//   3. 返回embedding向量
// 注意：需要确保token_id在[0, 152064)范围内
// 
// 实现步骤：
// 1. 在omni_context中添加emb_text_weight字段（float*, 152064 * 768）
// 2. 在omni_init中从TTS模型文件加载emb_text权重
// 3. 在这里实现查找逻辑
static bool tts_emb_text(struct omni_context * ctx_omni, llama_token token_id, float * embedding_out, int tts_n_embd) {
    // Check if weights are loaded
    if (!ctx_omni->emb_text_weight) {
        LOG_ERR("TTS: emb_text_weight not loaded\n");
        return false;
    }
    
    // Check token_id range
    if (token_id < 0 || token_id >= ctx_omni->emb_text_vocab_size) {
        LOG_ERR("TTS: token_id %d out of range [0, %d)\n", token_id, ctx_omni->emb_text_vocab_size);
        return false;
    }
    
    // Check embedding dimension
    if (tts_n_embd != ctx_omni->emb_text_hidden_size) {
        LOG_ERR("TTS: tts_n_embd (%d) != emb_text_hidden_size (%d)\n", tts_n_embd, ctx_omni->emb_text_hidden_size);
        return false;
    }
    
    // Copy embedding: embedding_out = emb_text_weight[token_id]
    const float * src = ctx_omni->emb_text_weight + token_id * tts_n_embd;
    memcpy(embedding_out, src, tts_n_embd * sizeof(float));
    
    return true;
}

// TTS的projector_semantic层实现
// 功能：将LLM hidden states（4096维）投影到TTS hidden_dim（768维）
// 输入：llm_hidden_states (float*, n_tokens * 4096), n_tokens, llm_n_embd (4096)
// 输出：projected_hidden_states (float*, n_tokens * 768), tts_n_embd (768)
// 
// 权重存储格式：
//   - PyTorch nn.Linear(in_features, out_features)的权重形状是(out_features, in_features)
//   - GGUF可能存储为(out_features, in_features)或(in_features, out_features)
//   - 加载时会自动检测并转置为(in_features, out_features)格式以便矩阵乘法
//   - linear1_weight: 转置后存储为(4096, 768) - 存储在ctx_omni->projector_semantic_linear1_weight
//   - linear1_bias: (768,) - 存储在ctx_omni->projector_semantic_linear1_bias
//   - linear2_weight: (768, 768) - 存储在ctx_omni->projector_semantic_linear2_weight
//   - linear2_bias: (768,) - 存储在ctx_omni->projector_semantic_linear2_bias
// 
// 矩阵乘法实现：
//   对每个token的hidden state：
//     hidden_proj = ReLU(linear1(hidden) + bias1)  // (1, 4096) @ (4096, 768) = (1, 768)
//     hidden_proj = linear2(hidden_proj) + bias2    // (1, 768) @ (768, 768) = (1, 768)
//   归一化在调用者中完成（使用normalize_l2_per_token）
static bool tts_projector_semantic(struct omni_context * ctx_omni, 
                                    const float * llm_hidden_states, int n_tokens, int llm_n_embd,
                                    float * projected_hidden_states, int tts_n_embd) {
    // 优先使用新的 ggml 实现 (精度验证版本)
    if (ctx_omni->projector.initialized) {
        // 检查维度
        if (llm_n_embd != ctx_omni->projector.hparams.in_dim) {
            LOG_ERR("TTS: llm_n_embd (%d) != projector in_dim (%d)\n", 
                    llm_n_embd, ctx_omni->projector.hparams.in_dim);
            return false;
        }
        if (tts_n_embd != ctx_omni->projector.hparams.out_dim) {
            LOG_ERR("TTS: tts_n_embd (%d) != projector out_dim (%d)\n", 
                    tts_n_embd, ctx_omni->projector.hparams.out_dim);
            return false;
        }
        
        // 使用 ggml 后端计算
        std::vector<float> result = projector_forward(ctx_omni->projector, llm_hidden_states, n_tokens);
        if (result.empty()) {
            LOG_ERR("TTS: projector_forward failed\n");
            return false;
        }
        
        // 复制结果
        memcpy(projected_hidden_states, result.data(), n_tokens * tts_n_embd * sizeof(float));
        return true;
    }
    
    // Fallback: 使用旧的 float* 权重实现
    // Check if weights are loaded
    if (!ctx_omni->projector_semantic_linear1_weight || 
        !ctx_omni->projector_semantic_linear1_bias ||
        !ctx_omni->projector_semantic_linear2_weight ||
        !ctx_omni->projector_semantic_linear2_bias) {
        LOG_ERR("TTS: projector_semantic weights not loaded (both ggml and legacy)\n");
        return false;
    }
    
    // Check dimensions
    if (llm_n_embd != ctx_omni->projector_semantic_input_dim) {
        LOG_ERR("TTS: llm_n_embd (%d) != projector_semantic_input_dim (%d)\n", 
                llm_n_embd, ctx_omni->projector_semantic_input_dim);
        return false;
    }
    
    if (tts_n_embd != ctx_omni->projector_semantic_output_dim) {
        LOG_ERR("TTS: tts_n_embd (%d) != projector_semantic_output_dim (%d)\n", 
                tts_n_embd, ctx_omni->projector_semantic_output_dim);
        return false;
    }
    
    const int input_dim = ctx_omni->projector_semantic_input_dim;   // 4096
    const int output_dim = ctx_omni->projector_semantic_output_dim; // 768
    
    // Process each token (legacy CPU implementation)
    for (int t = 0; t < n_tokens; t++) {
        const float * hidden = llm_hidden_states + t * input_dim;
        float * output = projected_hidden_states + t * output_dim;
        
        // Temporary buffer for intermediate results
        std::vector<float> temp(output_dim);
        
        // Step 1: linear1: temp = hidden @ linear1_weight + linear1_bias
        for (int j = 0; j < output_dim; j++) {
            float sum = ctx_omni->projector_semantic_linear1_bias[j];
            for (int i = 0; i < input_dim; i++) {
                sum += hidden[i] * ctx_omni->projector_semantic_linear1_weight[i * output_dim + j];
            }
            temp[j] = sum;
        }
        
        // Step 2: ReLU activation
        for (int j = 0; j < output_dim; j++) {
            temp[j] = (temp[j] > 0.0f) ? temp[j] : 0.0f;
        }
        
        // Step 3: linear2: output = temp @ linear2_weight + linear2_bias
        for (int j = 0; j < output_dim; j++) {
            float sum = ctx_omni->projector_semantic_linear2_bias[j];
            for (int i = 0; i < output_dim; i++) {
                sum += temp[i] * ctx_omni->projector_semantic_linear2_weight[i * output_dim + j];
            }
            output[j] = sum;
        }
    }
    
    return true;
}

// 辅助函数：L2归一化（对每个token的embedding分别归一化）
// 匹配Python的 F.normalize(hidden_embeds, p=2, dim=-1)
// 注意：PyTorch的F.normalize使用sqrt(sum(x^2) + eps)，然后除以norm
static void normalize_l2_per_token(float * embeddings, int n_tokens, int n_embd, float eps = 1e-8f) {
    for (int t = 0; t < n_tokens; t++) {
        float * vec = embeddings + t * n_embd;
        
        // Calculate L2 norm (matching PyTorch: sqrt(sum(x^2) + eps))
        float norm_sq = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float val = vec[i];
            norm_sq += val * val;
        }
        // PyTorch F.normalize: norm = sqrt(sum(x^2) + eps), then x = x / norm
        float norm = std::sqrt(norm_sq + eps);
        
        // Normalize: divide by norm (matching PyTorch F.normalize with p=2, dim=-1)
        // CRITICAL: Always normalize, even if norm is very small (eps ensures norm > 0)
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < n_embd; i++) {
                vec[i] *= inv_norm;
            }
        } else {
            // If norm is zero (shouldn't happen), set to unit vector
            LOG_WRN("TTS: WARNING - zero norm detected for token %d, setting to unit vector\n", t);
            float inv_sqrt_n = 1.0f / std::sqrt((float)n_embd);
            for (int i = 0; i < n_embd; i++) {
                vec[i] = inv_sqrt_n;
            }
        }
        
        // Verify normalization (always check to catch bugs)
        float verify_norm_sq = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float val = vec[i];
            verify_norm_sq += val * val;
        }
        float verify_norm = std::sqrt(verify_norm_sq);
        if (std::abs(verify_norm - 1.0f) > 0.01f) {
            LOG_ERR("TTS: ERROR - normalization verification failed for token %d: norm=%.6f (expected ~1.0), norm_sq=%.6f\n", 
                    t, verify_norm, verify_norm_sq);
        }
    }
}
static bool eval_tokens_tts(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past_tts) {
    fflush(stdout);
    int N = (int) tokens.size();
    fflush(stdout);
    // Note: TTS model might need different KV cache management
    // For now, we'll use a simple approach similar to LLM
    fflush(stdout);
    fflush(stdout);
    for (int i = 0; i < N; i += n_batch) {
        fflush(stdout);
        int n_eval = (int) tokens.size() - i;
        fflush(stdout);
        if (n_eval > n_batch) {
            n_eval = n_batch;
            fflush(stdout);
        }
        if (n_eval == 0) {
            fflush(stdout);
            break;
        }
        fflush(stdout);
        
        // Use llama_batch_get_one and manually set pos
        // Note: llama_batch_get_one may return batch with nullptr pos, so we need to handle it
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        fflush(stdout);
        
        // If batch.pos is nullptr, we need to allocate it
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            fflush(stdout);
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        
        // Set pos values to ensure correct KV cache position
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past_tts + j;
        }
        fflush(stdout);
        
        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        fflush(stdout);
        int decode_ret = llama_decode(ctx_omni->ctx_tts_llama, batch);
        
        // Keep embeddings enabled for sample_tts_token to use
        
        if (decode_ret != 0) {
            LOG_ERR("%s : failed to eval TTS tokens. token %d/%d (batch size %d, n_past %d), decode_ret=%d\n", 
                    __func__, i, N, n_batch, *n_past_tts, decode_ret);
            return false;
        }
        *n_past_tts += n_eval;
    }
    return true;
}

static bool eval_string_tts(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past_tts, bool add_bos) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_tts_llama, str2, add_bos, true);
    return eval_tokens_tts(ctx_omni, params, embd_inp, n_batch, n_past_tts);
}

// 使用embedding作为TTS输入的prefill函数（类似prefill_with_emb，但针对TTS模型）
bool prefill_with_emb_tts(struct omni_context* ctx_omni, common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts) {
    // 🔧 [安全检查] 验证输入参数
    if (n_pos <= 0) {
        LOG_ERR("%s: invalid n_pos=%d, skipping\n", __func__, n_pos);
        return false;
    }
    if (n_pos > 10000) {
        LOG_ERR("%s: n_pos=%d seems too large, likely data corruption\n", __func__, n_pos);
        return false;
    }
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("%s: TTS model not loaded\n", __func__);
        return false;
    }
    
    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
    
    // 🔧 [安全检查] 验证 n_embd 是合理值
    if (n_embd <= 0 || n_embd > 10000) {
        LOG_ERR("%s: invalid n_embd=%d from TTS model, likely model corruption\n", __func__, n_embd);
        return false;
    }
    
    // 🔧 [安全检查] 检查乘法溢出
    if (n_pos > (INT_MAX / n_embd)) {
        LOG_ERR("%s: n_pos=%d * n_embd=%d would overflow\n", __func__, n_pos, n_embd);
        return false;
    }
    
    // Save condition embeddings for first audio token re-forward (if not already saved)
    // This is needed to match Python's behavior: first audio token re-forwards the condition
    if (!ctx_omni->tts_condition_saved && n_pos > 0) {
        ctx_omni->tts_condition_embeddings.resize(n_pos * n_embd);
        std::memcpy(ctx_omni->tts_condition_embeddings.data(), embed, n_pos * n_embd * sizeof(float));
        ctx_omni->tts_condition_length = n_pos;
        ctx_omni->tts_condition_n_embd = n_embd;
        ctx_omni->tts_condition_saved = true;
    }
    
    // Save the starting position before the loop
    int text_start_pos = *n_past_tts;
    
    // Check if we need to save all hidden states (for alignment testing)
    const char* save_hidden_states_dir = getenv("TTS_SAVE_HIDDEN_STATES_DIR");
    bool save_all_hidden_states = (save_hidden_states_dir != nullptr);
    
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd = (embed + i * n_embd);  // 使用embedding作为输入
        
        // 设置pos值以确保正确的KV cache位置
        // Python: pos_ids = torch.arange(text_start_pos, text_start_pos + condition_length)
        // C++: batch.pos[j] = text_start_pos + i + j (where i is the offset within the current batch)
        std::vector<llama_pos> pos_vec(n_eval);
        batch.pos = pos_vec.data();
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = text_start_pos + i + j;  // Fix: use text_start_pos + i + j instead of *n_past_tts + j
        }
        
        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        
        if (llama_decode(ctx_omni->ctx_tts_llama, batch)) {
            LOG_ERR("%s : failed to eval TTS embeddings. pos %d/%d (batch size %d, n_past %d)\n", 
                    __func__, i, n_pos, n_batch, *n_past_tts);
            llama_set_embeddings(ctx_omni->ctx_tts_llama, false);
            return false;
        }
        
        // Save hidden states for each token in the batch (for alignment testing)
        // Note: llama_get_embeddings_ith uses negative indices relative to the end of the batch
        // For a batch of n_eval tokens: -1 is last, -2 is second-to-last, ..., -n_eval is first
        if (save_all_hidden_states) {
            for (int j = 0; j < n_eval; j++) {
                int token_idx = text_start_pos + i + j;
                // Get j-th token in current batch: j=0 -> -n_eval (first), j=n_eval-1 -> -1 (last)
                int llama_idx = j - n_eval;
                const float* hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, llama_idx);
                if (hidden_state) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/hidden_states_%03d.bin", save_hidden_states_dir, token_idx);
                    FILE* f = fopen(filepath, "wb");
                    if (f) {
                        fwrite(&token_idx, sizeof(int32_t), 1, f);
                        fwrite(&n_embd, sizeof(int32_t), 1, f);
                        fwrite(hidden_state, sizeof(float), n_embd, f);
                        fclose(f);
                    }
                } else {
                    LOG_WRN("TTS: Failed to get hidden state for token %d (llama_idx=%d)\n", token_idx, llama_idx);
                }
            }
        }
        
        // Keep embeddings enabled for sample_tts_token to use
    }
    
    // Update n_past_tts after all tokens are processed
    *n_past_tts = text_start_pos + n_pos;
    
    return true;
}

// Save logits to file for Python comparison
static void save_logits_to_file(const char* filepath, const float* logits, int num_tokens, int token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/logits_%03d.bin", filepath, token_index);
    
    FILE* f = fopen(full_path, "wb");
    if (!f) {
        LOG_ERR("Failed to open logits file for writing: %s\n", full_path);
        return;
    }
    
    // Write: token_index (int32), num_tokens (int32), logits (float32 array)
    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&num_tokens, sizeof(int32_t), 1, f);
    fwrite(logits, sizeof(float), num_tokens, f);
    fclose(f);
}

// Save hidden states to file for Python comparison
static void save_hidden_states_to_file(const char* filepath, const float* hidden_states, int hidden_size, int token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/hidden_states_%03d.bin", filepath, token_index);
    
    FILE* f = fopen(full_path, "wb");
    if (!f) {
        LOG_ERR("Failed to open hidden states file for writing: %s\n", full_path);
        return;
    }
    
    // Write: token_index (int32), hidden_size (int32), hidden_states (float32 array)
    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&hidden_size, sizeof(int32_t), 1, f);
    fwrite(hidden_states, sizeof(float), hidden_size, f);
    fclose(f);
}

// ========== RAS (Repetition Aware Sampling) Implementation ==========
// Ported from Python: tts_streaming_generate.py
// Key idea: Use nucleus sampling (top_p + top_k), but if too many recent tokens repeat,
// fall back to random sampling to break the pattern.

// Nucleus sampling (top-p + top-k)
// Returns a sampled token index from the logits
static int nucleus_sampling_tts(const float* logits, int num_tokens, float top_p, int top_k, std::mt19937& rng) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Sort by probability descending
    std::vector<std::pair<float, int>> sorted_probs;
    sorted_probs.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        sorted_probs.push_back({probs[i], i});
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 3. Collect tokens within top_p and top_k
    std::vector<float> filtered_probs;
    std::vector<int> filtered_indices;
    float cum_prob = 0.0f;
    
    for (const auto& p : sorted_probs) {
        if (cum_prob < top_p && (int)filtered_probs.size() < top_k) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else {
            break;
        }
    }
    
    // 4. Renormalize filtered probs
    float filtered_sum = 0.0f;
    for (float p : filtered_probs) {
        filtered_sum += p;
    }
    for (float& p : filtered_probs) {
        p /= filtered_sum;
    }
    
    // 5. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cum += filtered_probs[i];
        if (r <= cum) {
            return filtered_indices[i];
        }
    }
    return filtered_indices.back();  // Fallback
}

// Random sampling (uniform multinomial from all tokens)
static int random_sampling_tts(const float* logits, int num_tokens, std::mt19937& rng) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        cum += probs[i];
        if (r <= cum) {
            return i;
        }
    }
    return num_tokens - 1;  // Fallback
}

// RAS: Repetition Aware Sampling
// If the sampled token repeats too often in recent history, use random sampling instead
// 🔧 [与 Python TTSSamplingParams 对齐] (modeling_minicpmo.py line 69-76)
static int ras_sampling_tts(
    const float* logits, 
    int num_tokens, 
    const std::vector<int>& decoded_tokens,  // Recent tokens (relative indices)
    std::mt19937& rng,
    float top_p = 0.85f,   // TTSSamplingParams.top_p = 0.85
    int top_k = 25,        // TTSSamplingParams.top_k = 25
    int win_size = 16,     // TTSSamplingParams.win_size = 16
    float tau_r = 0.1f     // TTSSamplingParams.tau_r = 0.1
) {
    // 1. First, do nucleus sampling to get a candidate
    int top_id = nucleus_sampling_tts(logits, num_tokens, top_p, top_k, rng);
    
    // 2. Count how many times this token appears in the recent window
    int start_idx = std::max(0, (int)decoded_tokens.size() - win_size);
    int rep_num = 0;
    for (int i = start_idx; i < (int)decoded_tokens.size(); ++i) {
        if (decoded_tokens[i] == top_id) {
            rep_num++;
        }
    }
    
    // 3. If repetition exceeds threshold, switch to random sampling
    // Python: if rep_num >= win_size * tau_r
    if (rep_num >= (int)(win_size * tau_r)) {
        return random_sampling_tts(logits, num_tokens, rng);
    }
    
    return top_id;
}

// Apply repetition penalty to logits - matching Python's CustomRepetitionPenaltyLogitsProcessorRepeat
// Python implementation:
//   freq = F.one_hot(input_ids, num_tokens).sum(1)  # count frequency
//   alpha = torch.pow(penalty, freq)
//   inp = scores.multiply(alpha)  # for negative scores
//   oth = scores.divide(alpha)    # for positive scores
//   out = torch.where(scores < 0, inp, oth)
static void apply_repetition_penalty_tts(
    float* logits, 
    int num_tokens,
    const std::vector<int>& decoded_tokens,  // Recent tokens (relative indices)
    float penalty,
    int past_window = 16
) {
    if (decoded_tokens.empty() || penalty == 1.0f) {
        return;
    }
    
    // Get the window of recent tokens
    int start_idx = std::max(0, (int)decoded_tokens.size() - past_window);
    
    // Count frequency of each token in the window
    std::vector<int> freq(num_tokens, 0);
    for (int i = start_idx; i < (int)decoded_tokens.size(); ++i) {
        int tok = decoded_tokens[i];
        if (tok >= 0 && tok < num_tokens) {
            freq[tok]++;
        }
    }
    
    // Apply penalty: alpha = penalty ^ freq
    // For positive logits: divide by alpha (makes them smaller, lower probability)
    // For negative logits: multiply by alpha (makes them more negative, lower probability)
    for (int i = 0; i < num_tokens; ++i) {
        if (freq[i] > 0) {
            float alpha = powf(penalty, (float)freq[i]);
            if (logits[i] < 0) {
                logits[i] *= alpha;  // More negative
            } else {
                logits[i] /= alpha;  // Smaller positive
            }
        }
    }
}

// Nucleus sampling with min_tokens_to_keep (matching Python's TopPLogitsWarper and TopKLogitsWarper)
static int nucleus_sampling_with_min_keep_tts(
    const float* logits, 
    int num_tokens, 
    float top_p, 
    int top_k, 
    int min_tokens_to_keep,  // Python default: 3
    std::mt19937& rng
) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Sort by probability descending
    std::vector<std::pair<float, int>> sorted_probs;
    sorted_probs.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        sorted_probs.push_back({probs[i], i});
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 3. Collect tokens within top_p and top_k, but keep at least min_tokens_to_keep
    std::vector<float> filtered_probs;
    std::vector<int> filtered_indices;
    float cum_prob = 0.0f;
    
    for (const auto& p : sorted_probs) {
        // Always keep min_tokens_to_keep tokens
        if ((int)filtered_probs.size() < min_tokens_to_keep) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        }
        // After min_tokens_to_keep, apply top_p and top_k
        else if (cum_prob < top_p && (int)filtered_probs.size() < top_k) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else {
            break;
        }
    }
    
    // 4. Renormalize filtered probs
    float filtered_sum = 0.0f;
    for (float p : filtered_probs) {
        filtered_sum += p;
    }
    for (float& p : filtered_probs) {
        p /= filtered_sum;
    }
    
    // 5. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cum += filtered_probs[i];
        if (r <= cum) {
            return filtered_indices[i];
        }
    }
    return filtered_indices.back();  // Fallback
}

// ==================== 单工版本的 sample_tts_token ====================
// 直接从 omni_sinplex.cpp 复制，保证单工模式行为完全一致
// 🔧 [与 Python 对齐] 添加 is_final_text_chunk 参数：
//    - 非 final chunk：采样到 EOS 时不 prefill，避免污染 KV cache
//    - final chunk：采样到 EOS 时正常 prefill
static llama_token sample_tts_token_simplex(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past_tts, const std::vector<llama_token> * all_generated_tokens, int token_index_in_chunk, bool force_no_eos = false, bool is_final_text_chunk = false) {
    const char* logits_debug_dir = getenv("TTS_LOGITS_DEBUG_DIR");
    
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    const int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
    
    // 单工版本：is_audio_bos 只有在整个生成过程的第一个 token 时才为 true
    bool is_audio_bos = (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    if (is_audio_bos) {
        print_with_timestamp("TTS simplex: is_audio_bos=true (first audio token)\n");
    }
    
    // Re-forward condition for first audio token
    if (is_audio_bos && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        }
        int condition_n_past = 0;
        if (!prefill_with_emb_tts(ctx_omni, params, 
                                 ctx_omni->tts_condition_embeddings.data(), 
                                 ctx_omni->tts_condition_length, 
                                 params->n_batch, 
                                 &condition_n_past)) {
            LOG_ERR("TTS simplex: Failed to re-forward condition\n");
            return 0;
        }
        *n_past_tts = condition_n_past;
    }
    
    // 使用 head_code 层计算 audio logits
    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);
    if (hidden_state == nullptr) {
        LOG_ERR("TTS simplex: failed to get hidden state\n");
        return 0;
    }
    
    if (ctx_omni->head_code_weight == nullptr) {
        LOG_ERR("TTS simplex: head_code weight not loaded\n");
        return 0;
    }
    
    if (ctx_omni->head_code_hidden_size != 768 || ctx_omni->head_code_num_audio_tokens != num_audio_tokens) {
        LOG_ERR("TTS simplex: head_code dimensions mismatch\n");
        return 0;
    }
    
    std::vector<float> audio_logits(num_audio_tokens, 0.0f);
    const float * head_code_w = ctx_omni->head_code_weight;
    const int hidden_size = ctx_omni->head_code_hidden_size;
    
    for (int i = 0; i < num_audio_tokens; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }
    
    std::mt19937* rng = get_sampler_rng(smpl);
    std::mt19937 local_rng;
    if (rng == nullptr) {
        local_rng = std::mt19937(std::random_device{}());
        rng = &local_rng;
    }
    
    // 单工版本：使用 all_generated_tokens 做 repetition penalty
    std::vector<int> decoded_tokens_relative;
    if (all_generated_tokens != nullptr) {
        for (llama_token tid : *all_generated_tokens) {
            int relative_idx = tid - audio_bos_token_id;
            if (relative_idx >= 0 && relative_idx < num_audio_tokens) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }
    
    // 🔧 [与 Python streaming 对齐] TTS 采样参数
    // Python tts_streaming_generate.py 使用：
    // - temperature (从 TTSSamplingParams 传入，默认 0.8)
    // - repetition_penalty = 1.05
    // - window = 8 (recent_ids = logits_token[:, -8:])
    // - multinomial 采样 (无 top-p/top-k)
    float temperature = 0.8f;
    float repetition_penalty = 1.05f;
    int win_size = 8;  // 🔧 [与 Python 对齐] Python: recent_ids = logits_token[:, -8:]
    
    bool use_argmax = (params->sampling.temp <= 0.0f);
    
    int selected_relative_idx;
    if (use_argmax) {
        float max_logit = audio_logits[0];
        selected_relative_idx = 0;
        for (int i = 1; i < num_audio_tokens; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        // Step 1: 应用 temperature
        // Python: logits /= self.temperature
        for (int i = 0; i < num_audio_tokens; ++i) {
            audio_logits[i] /= temperature;
        }
        
        // Step 2: 只有 t > 0 时才应用 repetition penalty
        // Python: if t > 0: ... recent_ids = logits_token[:, -8:]
        if (!is_audio_bos && !decoded_tokens_relative.empty()) {
            // 🔧 [与 Python 对齐] 获取最近 win_size 个 tokens
            int start_idx = std::max(0, (int)decoded_tokens_relative.size() - win_size);
            
            // 🔧 [与 Python 对齐] 使用 bool mask 而不是频率计数
            // Python: occurred = F.one_hot(recent_ids, ...).sum(dim=1).bool()
            std::vector<bool> occurred(num_audio_tokens, false);
            for (int i = start_idx; i < (int)decoded_tokens_relative.size(); ++i) {
                int tok = decoded_tokens_relative[i];
                if (tok >= 0 && tok < num_audio_tokens) {
                    occurred[tok] = true;
                }
            }
            
            // 🔧 [与 Python 对齐] 应用 repetition penalty
            // Python: logits = torch.where(occurred & (logits >= 0), logits / penalty, logits)
            //         logits = torch.where(occurred & (logits < 0), logits * penalty, logits)
            for (int i = 0; i < num_audio_tokens; ++i) {
                if (occurred[i]) {
                    if (audio_logits[i] >= 0) {
                        audio_logits[i] /= repetition_penalty;
                    } else {
                        audio_logits[i] *= repetition_penalty;
                    }
                }
            }
        }
        
        // Step 3: 如果 force_no_eos=true，将 EOS logit 设为 -inf
        if (force_no_eos) {
            audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();
        }
        
        // Step 4: 🔧 [与 Python 对齐] 使用 multinomial 采样 (无 top-p/top-k)
        // Python: scores = F.softmax(logits, dim=-1)
        //         next_token = torch.multinomial(scores, num_samples=1)
        selected_relative_idx = random_sampling_tts(audio_logits.data(), num_audio_tokens, *rng);
    }
    
    if (selected_relative_idx < 0 || selected_relative_idx >= num_audio_tokens) {
        selected_relative_idx = 0;
    }
    
    const llama_token id = audio_bos_token_id + selected_relative_idx;
    int relative_idx = selected_relative_idx;
    
    common_sampler_accept(smpl, id, true);
    
    // 🔧 [与 Python 对齐] 检测是否采样到 EOS
    bool is_eos = (relative_idx == eos_relative_idx);
    
    // 🔧 [与 Python 对齐] EOS prefill 逻辑：
    // Python 中 EOS token 不会被加入 all_generated_tokens，所以下一轮不会 forward 进 TTS
    // C++ 中需要显式控制：
    //   - 非 final chunk：采样到 EOS 时不 prefill，直接返回（让调用方知道已结束）
    //   - final chunk：整个生成结束，prefill EOS（如果需要的话，保持状态一致性）
    if (is_eos && !is_final_text_chunk) {
        // 非 final chunk 采样到 EOS，不 prefill，直接返回
        // 这样 TTS 模型的 KV cache 不会包含 EOS，下一个 chunk 可以继续使用
        return id;
    }
    
    // 使用 emb_code 获取 embedding，然后通过 prefill_with_emb_tts 输入模型
    if (ctx_omni->emb_code_weight != nullptr && relative_idx >= 0 && relative_idx < ctx_omni->emb_code_vocab_size) {
        const float * emb_code_w = ctx_omni->emb_code_weight;
        const int emb_code_hidden_size = ctx_omni->emb_code_hidden_size;
        const int emb_code_vocab_size = ctx_omni->emb_code_vocab_size;
        
        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        
        if (ctx_omni->emb_code_stored_as_transposed) {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + relative_idx];
            }
        } else {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[relative_idx * emb_code_hidden_size + j];
            }
        }
        
        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS simplex: failed to decode audio token embedding\n");
            return 0;
        }
    } else {
        LOG_ERR("TTS simplex: emb_code not available\n");
        return 0;
    }
    
    return id;
}

llama_token sample_tts_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past_tts, const std::vector<llama_token> * all_generated_tokens, const std::vector<llama_token> * chunk_generated_tokens, int token_index_in_chunk, bool force_no_eos, bool is_final_text_chunk = false) {
    // Debug: Save logits directory (set via environment variable)
    const char* logits_debug_dir = getenv("TTS_LOGITS_DEBUG_DIR");
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;  // head_code output size
    
    // 🔧 [差异2修复] 分离两个概念，与 Python generate_chunk 对齐：
    // 1. is_first_token_overall: 整个生成过程的第一个 token（用于 re-forward condition）
    //    Python TTSStreamingGenerator: audio_bos = len(self.all_generated_tokens) == 0 and t == 0
    // 2. is_chunk_first_token: 当前 chunk 的第一个 token（用于跳过 sampling processors）
    //    Python generate_chunk: if t == 0: audio_bos = True
    //    Python generate_chunk 中，每个 chunk 的第一个 token 不应用 repetition penalty 和 warpers
    
    // is_first_token_overall: 用于控制是否 re-forward condition（清空 KV cache）
    bool is_first_token_overall = (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    
    // 🔧 [单双工适配] skip_processors: 控制是否跳过 sampling processors
    // - 双工模式：每个 chunk 的第一个 token 都跳过（与 Python generate_chunk 对齐）
    //   Python generate_chunk: if t == 0: audio_bos = True
    // - 单工模式：只有整个生成过程的第一个 token 跳过（与 Python TTSStreamingGenerator 对齐）
    //   Python: audio_bos = len(self.all_generated_tokens) == 0 and t == 0
    bool skip_processors;
    if (ctx_omni->duplex_mode) {
        // 双工模式：每个 chunk 的第一个 token 都跳过
        skip_processors = (token_index_in_chunk == 0);
    } else {
        // 单工模式：只有整个生成过程的第一个 token 跳过
        skip_processors = is_first_token_overall;
    }
    
    // 只在第一个token时打印
    if (is_first_token_overall) {
        print_with_timestamp("TTS sample: is_first_token_overall=true, duplex_mode=%d\n", ctx_omni->duplex_mode);
    }
    
    // CRITICAL FIX: For the first audio token of the ENTIRE generation, we need to re-forward the entire condition
    // This matches Python's behavior where past_key_values=None on the first forward
    // Python: outputs = self.tts.model(position_ids=pos_ids, past_key_values=None, inputs_embeds=inputs_embeds, use_cache=True)
    // 🔧 [差异2修复] 只在整个生成过程的第一个 token 时 re-forward，不是每个 chunk 的第一个
    if (is_first_token_overall && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        // 🔧 [安全检查] 验证 tts_condition_* 值是否合理
        int cond_len = ctx_omni->tts_condition_length;
        int cond_n_embd = ctx_omni->tts_condition_n_embd;
        size_t cond_emb_size = ctx_omni->tts_condition_embeddings.size();
        size_t expected_size = (size_t)cond_len * cond_n_embd;
        
        if (cond_len <= 0 || cond_len > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_length=%d\n", cond_len);
            return 0;
        }
        if (cond_n_embd <= 0 || cond_n_embd > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_n_embd=%d\n", cond_n_embd);
            return 0;
        }
        if (cond_emb_size != expected_size) {
            LOG_ERR("TTS sample: tts_condition_embeddings size mismatch: %zu != %zu (len=%d * n_embd=%d)\n",
                    cond_emb_size, expected_size, cond_len, cond_n_embd);
            return 0;
        }
        // Clear KV cache to match Python's past_key_values=None behavior
        // Use llama_memory_seq_rm to remove all tokens from sequence 0
        // seq_id=0, p0=0 (from beginning), p1=-1 (to end) removes all tokens
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        } else {
            LOG_ERR("TTS: Failed to get memory for KV cache clear\n");
        }
        
        // Reset n_past_tts to 0 (like Python's text_start_pos=0)
        int condition_n_past = 0;
        
        // Re-forward the entire condition
        if (!prefill_with_emb_tts(ctx_omni, params, 
                                 ctx_omni->tts_condition_embeddings.data(), 
                                 ctx_omni->tts_condition_length, 
                                 params->n_batch, 
                                 &condition_n_past)) {
            LOG_ERR("TTS: Failed to re-forward condition for first audio token\n");
            return 0;
        }
        
        // Update n_past_tts to match the condition length
        *n_past_tts = condition_n_past;
    }
    
    // 1. 获取TTS模型的最后一个位置的hidden state
    // hidden_state shape: (hidden_size=768,)
    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);
    if (hidden_state == nullptr) {
        LOG_ERR("TTS: failed to get hidden state from TTS model\n");
        return 0;
    }
    
    // Debug: Save hidden states to file for Python comparison
    if (logits_debug_dir != nullptr) {
        int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        save_hidden_states_to_file(logits_debug_dir, hidden_state, hidden_size, token_index_in_chunk);
    }
    
    // 2. 检查head_code权重是否已加载
    if (ctx_omni->head_code_weight == nullptr) {
        LOG_ERR("TTS: head_code weight not loaded\n");
        return 0;
    }
    
    if (ctx_omni->head_code_hidden_size != 768 || ctx_omni->head_code_num_audio_tokens != num_audio_tokens) {
        LOG_ERR("TTS: head_code dimensions mismatch: expected (768, 6562), got (%d, %d)\n",
                ctx_omni->head_code_hidden_size, ctx_omni->head_code_num_audio_tokens);
        return 0;
    }
    
    // 3. 使用head_code层计算audio token的logits
    // logits = hidden_state @ head_code_weight
    // hidden_state: (768,), head_code_weight: [6562, 768] (转置后存储) -> logits: (6562,)
    std::vector<float> audio_logits(num_audio_tokens, 0.0f);
    const float * head_code_w = ctx_omni->head_code_weight;
    const int hidden_size = ctx_omni->head_code_hidden_size;
    
    // ⚡ 优化：head_code_weight已转置为[6562, 768]，每行连续存储
    // 连续内存访问，cache友好
    for (int i = 0; i < num_audio_tokens; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }
    
    int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
    
    // Debug: Save raw logits to file for Python comparison
    if (logits_debug_dir != nullptr) {
        save_logits_to_file(logits_debug_dir, audio_logits.data(), num_audio_tokens, token_index_in_chunk);
    }
    
    // Save hidden state and logits for comparison (before evaluation)
    const char* output_dir = getenv("TTS_OUTPUT_DIR");
    if (output_dir != nullptr) {
        // Save hidden state (for first token only)
        if (token_index_in_chunk == 0) {
            char hidden_state_path[512];
            snprintf(hidden_state_path, sizeof(hidden_state_path), "%s/cpp_first_hidden_state.bin", output_dir);
            FILE* f_hidden = fopen(hidden_state_path, "wb");
            if (f_hidden) {
                fwrite(hidden_state, sizeof(float), hidden_size, f_hidden);
                fclose(f_hidden);
            }
        }
        
        // Save logits for all tokens (for debugging)
        char logits_path[512];
        snprintf(logits_path, sizeof(logits_path), "%s/cpp_logits_token_%d.bin", output_dir, token_index_in_chunk);
        FILE* f_logits = fopen(logits_path, "wb");
        if (f_logits) {
            fwrite(audio_logits.data(), sizeof(float), num_audio_tokens, f_logits);
            fclose(f_logits);
        }
    }
    
    // 4. 采样流程 - 与 Python MiniCPMTTS.generate() 完全对齐
    // Python 采样流程:
    //   1. logits /= temperature (默认 0.8)
    //   2. if not audio_bos: apply repetition_penalty (penalty=1.05, past_window=16)
    //   3. if not audio_bos: apply TopP (0.85) + TopK (25) warper (min_tokens_to_keep=3)
    //   4. softmax + multinomial
    
    // Get RNG from sampler
    std::mt19937* rng = get_sampler_rng(smpl);
    std::mt19937 local_rng;
    if (rng == nullptr) {
        // Fallback to local RNG
        LOG_WRN("TTS: sampler RNG not available, using local RNG\n");
        local_rng = std::mt19937(std::random_device{}());
        rng = &local_rng;
    }
    
    // 🔧 [单双工适配] Collect decoded tokens (relative indices) for repetition penalty
    // - 双工模式：使用 chunk_generated_tokens（当前 chunk 内的 tokens）
    //   Python generate_chunk: input_ids_sliced = new_tokens[:, 0:t]
    // - 单工模式：使用 all_generated_tokens（所有生成的 tokens）
    //   Python TTSStreamingGenerator: self.all_generated_tokens
    std::vector<int> decoded_tokens_relative;
    const std::vector<llama_token> * tokens_for_penalty;
    if (ctx_omni->duplex_mode) {
        // 双工模式：优先使用当前 chunk 的 tokens
        tokens_for_penalty = chunk_generated_tokens ? chunk_generated_tokens : all_generated_tokens;
    } else {
        // 单工模式：使用所有生成的 tokens
        tokens_for_penalty = all_generated_tokens;
    }
    if (tokens_for_penalty != nullptr) {
        for (llama_token tid : *tokens_for_penalty) {
            // Convert absolute token ID to relative index
            int relative_idx = tid - audio_bos_token_id;
            if (relative_idx >= 0 && relative_idx < num_audio_tokens) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }
    
    // 🔧 [与 Python TTSSamplingParams 对齐] (modeling_minicpmo.py line 69-76)
    float temperature = 0.8f;       // TTSSamplingParams.temperature = 0.8
    float top_p = 0.85f;            // TTSSamplingParams.top_p = 0.85
    int top_k = 25;                 // TTSSamplingParams.top_k = 25
    float repetition_penalty = 1.05f; // TTSSamplingParams.repetition_penalty = 1.05
    int win_size = 16;              // TTSSamplingParams.win_size = 16
    float tau_r = 0.1f;             // TTSSamplingParams.tau_r = 0.1
    int min_tokens_to_keep = 3;     // Python: TopPLogitsWarper/TopKLogitsWarper default
    
    // Get temperature from params if specified (for argmax mode)
    bool use_argmax = (params->sampling.temp <= 0.0f);
    
    int selected_relative_idx;
    if (use_argmax) {
        // Deterministic sampling: select argmax (highest logit)
        float max_logit = audio_logits[0];
        selected_relative_idx = 0;
        for (int i = 1; i < num_audio_tokens; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        // Step 1: Apply temperature
        for (int i = 0; i < num_audio_tokens; ++i) {
            audio_logits[i] /= temperature;
        }
        
        // 🔧 [单双工适配] Step 2 & 3: Apply repetition penalty and TopP/TopK
        // - 双工模式：每个 chunk 的第一个 token 跳过
        // - 单工模式：只有整个生成过程的第一个 token 跳过
        if (!skip_processors && !decoded_tokens_relative.empty()) {
            // Apply repetition penalty (matching Python's CustomRepetitionPenaltyLogitsProcessorRepeat)
            apply_repetition_penalty_tts(audio_logits.data(), num_audio_tokens, 
                                        decoded_tokens_relative, repetition_penalty, win_size);
        }
        
        // 🔧 [差异1修复] 在采样前阻止 EOS token 被采样
        // Python generate_chunk: if force_no_stop or t < min_new_tokens: logits[:, eos_token] = -torch.inf
        // 这样可以确保在达到 min_new_tokens 之前不会生成 EOS
        if (ctx_omni->duplex_mode && force_no_eos) {
            int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
            audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();
        }
        
        // Step 4: Nucleus sampling with min_tokens_to_keep (matching Python's warpers)
        selected_relative_idx = nucleus_sampling_with_min_keep_tts(
            audio_logits.data(), 
            num_audio_tokens, 
            top_p,
            top_k,
            min_tokens_to_keep,
            *rng
        );
    }
    
    // 5. 验证采样结果在有效范围内
    if (selected_relative_idx < 0 || selected_relative_idx >= num_audio_tokens) {
        LOG_ERR("TTS: invalid selected index %d, should be in [0, %d)\n", 
                selected_relative_idx, num_audio_tokens);
        // Fallback: use first token
        selected_relative_idx = 0;
    }
    
    // Convert relative index to absolute token ID
    const llama_token id = audio_bos_token_id + selected_relative_idx;
    int relative_idx = selected_relative_idx;
    
    common_sampler_accept(smpl, id, true);
    
    // 🔧 [与 Python 对齐] 检测是否采样到 EOS
    int eos_relative_idx_check = num_audio_tokens - 1;  // EOS token relative index: 6561
    bool is_eos = (relative_idx == eos_relative_idx_check);
    
    // 🔧 [与 Python 对齐] EOS prefill 逻辑：
    // Python 中 EOS token 不会被加入 all_generated_tokens，所以下一轮不会 forward 进 TTS
    // C++ 中需要显式控制：
    //   - 非 final chunk：采样到 EOS 时不 prefill，直接返回（让调用方知道已结束）
    //   - final chunk：整个生成结束，prefill EOS（如果需要的话，保持状态一致性）
    if (ctx_omni->duplex_mode && is_eos && !is_final_text_chunk) {
        // 非 final chunk 采样到 EOS，不 prefill，直接返回
        // 这样 TTS 模型的 KV cache 不会包含 EOS，下一个 chunk 可以继续使用
        return id;
    }
    
    // 9. 使用emb_code将audio token ID转换为embedding，然后使用prefill_with_emb_tts
    // 这样可以避免token ID超出词汇表范围的问题
    if (ctx_omni->emb_code_weight != nullptr && relative_idx >= 0 && relative_idx < ctx_omni->emb_code_vocab_size) {
        // Get embedding from emb_code
        const float * emb_code_w = ctx_omni->emb_code_weight;
        const int emb_code_hidden_size = ctx_omni->emb_code_hidden_size;
        const int emb_code_vocab_size = ctx_omni->emb_code_vocab_size;
        
        // Allocate embedding buffer
        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        
        if (ctx_omni->emb_code_stored_as_transposed) {
            // Stored as (hidden_size, num_audio_tokens) = [768, 6562]
            // Access: embedding[j] = emb_code_weight[j * num_audio_tokens + relative_idx]
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + relative_idx];
            }
        } else {
            // Stored as (num_audio_tokens, hidden_size) = [6562, 768]
            // Access: embedding[j] = emb_code_weight[relative_idx * hidden_size + j]
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[relative_idx * emb_code_hidden_size + j];
            }
        }
        
        // Use prefill_with_emb_tts to decode with embedding instead of token ID
        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token embedding\n");
            return 0;
        }
    } else {
        // Fallback: use token IDs (may fail if token ID exceeds vocab size)
        LOG_ERR("TTS: emb_code not available, falling back to token IDs (may fail if token exceeds vocab)\n");
        std::vector<llama_token> tokens;
        tokens.push_back(id);
        if (!eval_tokens_tts(ctx_omni, params, tokens, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token ID (token may exceed vocab size)\n");
            return 0;
        }
    }
    
    return id;
}

// Check if a token is an audio token (based on config: num_audio_tokens = 6562)
// Audio tokens are typically in a specific range
static bool is_audio_token(llama_token token, int audio_bos_token_id = 151687, int num_audio_tokens = 6562) {
    // Audio tokens are typically in range [audio_bos_token_id, audio_bos_token_id + num_audio_tokens)
    // Check if token is in the audio token range
    return (token >= audio_bos_token_id && token < audio_bos_token_id + num_audio_tokens);
}

// Simple UTF-8 incomplete byte checker
// Returns the number of incomplete bytes at the end of the string
// Returns 0 if the string ends with a complete UTF-8 sequence
static size_t findIncompleteUtf8(const std::string& str) {
    if (str.empty()) return 0;
    
    size_t len = str.length();
    
    // Check from the end backwards to find incomplete UTF-8 sequences
    size_t pos = len;
    int expected_continuation_bytes = 0;
    
    while (pos > 0) {
        unsigned char c = (unsigned char)str[pos - 1];
        
        if ((c & 0x80) == 0) {
            // ASCII character (0xxxxxxx), complete sequence
            break;
        } else if ((c & 0xC0) == 0x80) {
            // Continuation byte (10xxxxxx), part of a multi-byte sequence
            expected_continuation_bytes++;
            pos--;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence start (110xxxxx)
            // Should have exactly 1 continuation byte after it
            if (expected_continuation_bytes == 1) {
                // Complete 2-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + 1;  // Return incomplete bytes including the start byte
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence start (1110xxxx)
            // Should have exactly 2 continuation bytes after it
            if (expected_continuation_bytes == 2) {
                // Complete 3-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (3 - expected_continuation_bytes);
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence start (11110xxx)
            // Should have exactly 3 continuation bytes after it
            if (expected_continuation_bytes == 3) {
                // Complete 4-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (4 - expected_continuation_bytes);
            }
        } else {
            // Invalid UTF-8 byte pattern (should not happen in valid UTF-8)
            // Treat as complete to avoid breaking valid text
            break;
        }
    }
    
    // If we've checked all bytes and still have expected continuation bytes,
    // the sequence is incomplete
    if (pos == 0 && expected_continuation_bytes > 0) {
        return len;
    }
    
    return 0;  // String ends with complete UTF-8 sequence
}

// ==================== 滑动窗口实现 (#39) ====================
// 基于 Python sliding_utils.py 和 stream_decoder.py

/**
 * 重置滑动窗口状态
 */
void sliding_window_reset(struct omni_context * ctx_omni) {
    if (!ctx_omni) return;
    
    int old_unit_count = ctx_omni->unit_history.size();
    
    ctx_omni->unit_history.clear();
    ctx_omni->next_unit_id = 0;
    ctx_omni->pending_unit_id = -1;
    ctx_omni->pending_unit_start_cache_len = 0;
    ctx_omni->system_preserve_length = 0;
    ctx_omni->position_offset = 0;
    
    // 统计信息
    ctx_omni->sliding_event_count = 0;
    ctx_omni->total_dropped_tokens = 0;
    ctx_omni->total_dropped_units = 0;
    
    if (old_unit_count > 0) {
        print_with_timestamp("[SW] reset: cleared %d units, all sliding window state reset\n", old_unit_count);
    }
}

/**
 * 获取当前 KV cache 长度
 */
static int get_cache_length(struct omni_context * ctx_omni) {
    if (!ctx_omni || !ctx_omni->ctx_llama) return 0;
    return ctx_omni->n_past;
}

/**
 * 注册 unit 开始
 * 在每个 unit（音频/视频/omni chunk）处理开始时调用
 * @return 返回分配的 unit_id
 */
int sliding_window_register_unit_start(struct omni_context * ctx_omni) {
    if (!ctx_omni) return -1;
    
    ctx_omni->pending_unit_id = ctx_omni->next_unit_id;
    ctx_omni->pending_unit_start_cache_len = get_cache_length(ctx_omni);
    
    print_with_timestamp("[SW] unit_start: pending_unit_id=%d, cache_len=%d, preserve=%d, units=%zu\n",
                        ctx_omni->pending_unit_id,
                        ctx_omni->pending_unit_start_cache_len,
                        ctx_omni->system_preserve_length,
                        ctx_omni->unit_history.size());
    
    return ctx_omni->pending_unit_id;
}

/**
 * 注册 unit 结束
 * 在每个 unit 处理完成后调用，记录该 unit 的信息
 */
void sliding_window_register_unit_end(struct omni_context * ctx_omni, 
                                      const std::string & input_type,
                                      const std::vector<llama_token> & generated_tokens,
                                      bool is_listen) {
    if (!ctx_omni) return;
    
    if (ctx_omni->pending_unit_id < 0) {
        print_with_timestamp("[SW] WARNING: register_unit_end called without register_unit_start\n");
        return;
    }
    
    int current_cache_len = get_cache_length(ctx_omni);
    int unit_len = current_cache_len - ctx_omni->pending_unit_start_cache_len;
    
    if (unit_len > 0) {
        UnitEntry entry;
        entry.unit_id = ctx_omni->pending_unit_id;
        entry.length = unit_len;
        entry.type = input_type;
        entry.generated_tokens = generated_tokens;
        entry.is_listen = is_listen;
        
        ctx_omni->unit_history.push_back(entry);
        
        print_with_timestamp("[SW] unit_end: unit_id=%d type=%s len=%d gen_tokens=%zu is_listen=%d | cache=%d preserve=%d total_units=%zu\n",
                            entry.unit_id, entry.type.c_str(), entry.length,
                            entry.generated_tokens.size(), entry.is_listen,
                            current_cache_len, ctx_omni->system_preserve_length,
                            ctx_omni->unit_history.size());
    } else {
        print_with_timestamp("[SW] WARNING: unit_end: unit_id=%d has zero length (start=%d, current=%d), not recorded\n",
                            ctx_omni->pending_unit_id, ctx_omni->pending_unit_start_cache_len, current_cache_len);
    }
    
    ctx_omni->pending_unit_id = -1;
    ctx_omni->pending_unit_start_cache_len = 0;
    ctx_omni->next_unit_id++;
}

/**
 * 注册 system prompt
 * 在 system prompt prefill 完成后调用，记录保护长度
 */
void sliding_window_register_system_prompt(struct omni_context * ctx_omni) {
    if (!ctx_omni) return;
    
    ctx_omni->system_preserve_length = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] system_prompt registered: preserve_length=%d (will be protected from sliding)\n",
                        ctx_omni->system_preserve_length);
}

/**
 * 从 KV cache 中删除指定数量的 tokens
 * 删除位于 [preserve, preserve + length) 区间的 tokens
 * 
 * 注意：llama.cpp 的 llama_memory_seq_rm 函数可以直接删除指定范围的 tokens
 * 但删除后需要进行 RoPE 位置重对齐（暂时不支持，需要更深入的修改）
 * 
 * 当前实现：使用简化的策略，删除 tokens 后更新 position_offset
 * 后续生成时使用 position_offset 来调整 position_ids
 */
bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length) {
    if (!ctx_omni || !ctx_omni->ctx_llama || length <= 0) {
        print_with_timestamp("[SW] drop_tokens: invalid params (length=%d)\n", length);
        return false;
    }
    
    int cache_len_before = get_cache_length(ctx_omni);
    int preserve = ctx_omni->system_preserve_length;
    
    if (cache_len_before <= preserve) {
        print_with_timestamp("[SW] drop_tokens: cache_len=%d <= preserve=%d, nothing to drop\n",
                            cache_len_before, preserve);
        return false;
    }
    
    int available = cache_len_before - preserve;
    if (available < length) {
        print_with_timestamp("[SW] drop_tokens: cannot drop %d tokens, only %d available (cache=%d, preserve=%d)\n",
                            length, available, cache_len_before, preserve);
        return false;
    }
    
    // 使用 llama_memory_seq_rm 删除 [preserve, preserve + length) 区间的 tokens
    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (!mem) {
        print_with_timestamp("[SW] drop_tokens: failed to get llama memory\n");
        return false;
    }
    
    // 删除指定范围的 tokens
    // llama_memory_seq_rm(mem, seq_id, p0, p1) 删除 [p0, p1) 范围的 tokens
    bool success = llama_memory_seq_rm(mem, 0, preserve, preserve + length);
    
    if (success) {
        // 更新 n_past
        ctx_omni->n_past = cache_len_before - length;
        
        // 更新 position_offset（用于后续 RoPE 计算）
        ctx_omni->position_offset += length;
        
        print_with_timestamp("[SW] drop_tokens: SUCCESS, dropped %d tokens from [%d, %d), cache %d -> %d, offset=%d\n",
                            length, preserve, preserve + length, cache_len_before, ctx_omni->n_past, ctx_omni->position_offset);
    } else {
        print_with_timestamp("[SW] drop_tokens: FAILED to drop %d tokens\n", length);
    }
    
    return success;
}

/**
 * 删除指定的 unit
 */
static bool sliding_window_drop_unit(struct omni_context * ctx_omni, int unit_id) {
    if (!ctx_omni) return false;
    
    // 查找 unit
    auto it = std::find_if(ctx_omni->unit_history.begin(), ctx_omni->unit_history.end(),
                          [unit_id](const UnitEntry& e) { return e.unit_id == unit_id; });
    
    if (it == ctx_omni->unit_history.end()) {
        print_with_timestamp("[SW] drop_unit: unit_id=%d not found\n", unit_id);
        return false;
    }
    
    int total_len = it->length;
    if (total_len <= 0) {
        print_with_timestamp("[SW] drop_unit: unit_id=%d has zero length, removing from history\n", unit_id);
        ctx_omni->unit_history.erase(it);
        return false;
    }
    
    int cache_before = get_cache_length(ctx_omni);
    if (!sliding_window_drop_tokens_from_cache(ctx_omni, total_len)) {
        print_with_timestamp("[SW] drop_unit: failed to drop %d tokens for unit_id=%d\n", total_len, unit_id);
        return false;
    }
    
    int cache_after = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] 🗑️ DROPPED unit_id=%d type=%s len=%d gen_tokens=%zu | cache %d -> %d, offset=%d\n",
                        it->unit_id, it->type.c_str(), it->length, it->generated_tokens.size(),
                        cache_before, cache_after, ctx_omni->position_offset);
    
    ctx_omni->unit_history.erase(it);
    return true;
}

/**
 * 删除最早的一个非 system unit
 */
static bool sliding_window_drop_next_unit(struct omni_context * ctx_omni) {
    if (!ctx_omni) return false;
    
    for (const auto& entry : ctx_omni->unit_history) {
        // 跳过 system 类型
        if (entry.type == "system") {
            print_with_timestamp("[SW] drop_next_unit: skipping system unit_id=%d\n", entry.unit_id);
            continue;
        }
        
        print_with_timestamp("[SW] drop_next_unit: attempting to drop unit_id=%d\n", entry.unit_id);
        if (sliding_window_drop_unit(ctx_omni, entry.unit_id)) {
            return true;
        }
    }
    
    print_with_timestamp("[SW] drop_next_unit: no droppable unit found in %zu units\n", ctx_omni->unit_history.size());
    return false;
}

/**
 * 执行滑动窗口策略
 * 当 cache 长度超过高水位线时，循环移除最早的 unit，直到降到低水位线以下
 * 
 * @return true 如果执行了滑窗操作，false 如果没有需要滑窗
 */
bool sliding_window_enforce(struct omni_context * ctx_omni) {
    if (!ctx_omni) return false;
    
    const auto& cfg = ctx_omni->sliding_window_config;
    
    // 检查是否启用滑窗
    if (cfg.mode == "off") {
        return false;
    }
    
    int cache_len_before = get_cache_length(ctx_omni);
    
    // 检查是否超过高水位线
    if (cache_len_before <= cfg.high_water_tokens) {
        return false;  // 未超过高水位线，不触发
    }
    
    // 超过高水位线，开始滑窗
    print_with_timestamp("[SW] ⚡ SLIDING TRIGGERED: cache=%d > high_water=%d, target=low_water=%d\n",
                        cache_len_before, cfg.high_water_tokens, cfg.low_water_tokens);
    
    int dropped_count = 0;
    int cache_len = cache_len_before;
    
    while (cache_len > cfg.low_water_tokens) {
        if (!sliding_window_drop_next_unit(ctx_omni)) {
            print_with_timestamp("[SW] enforce_window: no more units to drop, stopping\n");
            break;
        }
        dropped_count++;
        cache_len = get_cache_length(ctx_omni);
    }
    
    if (dropped_count > 0) {
        // 更新统计
        ctx_omni->sliding_event_count++;
        ctx_omni->total_dropped_tokens += cache_len_before - cache_len;
        ctx_omni->total_dropped_units += dropped_count;
        
        // 一致性检查
        int expected = ctx_omni->system_preserve_length;
        for (const auto& u : ctx_omni->unit_history) {
            expected += u.length;
        }
        bool is_consistent = (expected == cache_len);
        
        print_with_timestamp("[SW] ✅ SLIDING DONE: cache %d -> %d, dropped %d units, remaining %zu units | consistency: expected=%d actual=%d %s\n",
                            cache_len_before, cache_len, dropped_count, ctx_omni->unit_history.size(),
                            expected, cache_len, is_consistent ? "✓" : "✗ MISMATCH!");
        
        if (!is_consistent) {
            print_with_timestamp("[SW] ❌ CONSISTENCY ERROR! preserve=%d + sum(units)=%d != cache=%d, offset=%d\n",
                                ctx_omni->system_preserve_length,
                                expected - ctx_omni->system_preserve_length,
                                cache_len, ctx_omni->position_offset);
        }
    }
    
    return dropped_count > 0;
}

//
// omni main
//
std::condition_variable g_decode_cv;
bool prefill_done = true;
std::mutex speek_mtx;
std::condition_variable speek_cv;
bool last_speek_done_flag = false;

// 让 thread 可以结束
std::atomic<bool> llm_thread_running(true);
std::atomic<bool> tts_thread_running(true);
std::atomic<bool> t2w_thread_running(true);

// 读取 omni_output 互斥
std::mutex buffer_mutex;

void print_with_timestamp(const char* format, ...)
{
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    // 格式化时间戳
    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    std::cout << std::put_time(&buf, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count() << " ";
    
    // 打印格式化字符串
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

static struct llama_model * llama_init(common_params * params, std::string model_path) {
    llama_backend_init();
    llama_numa_init(params->numa);
    
    llama_model_params model_params = common_model_params_to_llama(*params);
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

// TTS专用模型加载 - 支持独立的GPU层数设置
// 通过环境变量 TTS_GPU_LAYERS 控制，-1 表示使用与LLM相同的设置
static struct llama_model * llama_init_tts(common_params * params, std::string model_path, int n_gpu_layers_override = -1) {
    llama_backend_init();
    llama_numa_init(params->numa);
    
    llama_model_params model_params = common_model_params_to_llama(*params);
    
    // 如果指定了override值(>=0)，使用它；否则保持与LLM相同的设置
    if (n_gpu_layers_override >= 0) {
        model_params.n_gpu_layers = n_gpu_layers_override;
    }
    
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load TTS model\n" , __func__);
        return NULL;
    }
    return model;
}

struct omni_context * omni_init(struct common_params * params, int media_type, bool use_tts, std::string tts_bin_dir,
                                int tts_gpu_layers, const std::string & token2wav_device, bool duplex_mode,
                                llama_model * existing_model, llama_context * existing_ctx,
                                const std::string & base_output_dir) {
    // process the prompt
    print_with_timestamp("=== omni_init start\n");
    // if (params->prompt.empty() && params->interactive == false) {
    //     LOG_INF("prompt should be given or interactive mode should be on");
    //     return NULL;
    // }
    // auto ctx_omni = (struct omni_context *)malloc(sizeof(omni_context));
    auto ctx_omni = new omni_context();

    ctx_omni->params = params;
    ctx_omni->media_type = media_type;
    ctx_omni->use_tts = use_tts;
    ctx_omni->duplex_mode = duplex_mode;
    ctx_omni->base_output_dir = base_output_dir;  // 🔧 [多实例支持] 设置可配置的输出目录
    print_with_timestamp("media_type = %d, duplex_mode = %d, base_output_dir = %s\n", media_type, duplex_mode, base_output_dir.c_str());
    // 🔧 [对齐 Python MiniCPM-o-4_5-latest] prompt 格式
    // Python default_tts_chat_template:
    //   {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}
    //   {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' + think_str + '<|tts_bos|>' }}{% endif %}
    // 
    // Python audio_assistant 模式的 system prompt:
    //   vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
    //   vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
    //   sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
    // 
    // 完整格式 (audio_assistant 模式):
    //   <|im_start|>system
    //   模仿音频样本的音色并生成新的内容。
    //   <|audio_start|>[ref_audio_embed]<|audio_end|>你的任务是用这种声音模式来当一个助手。...
    //   <|im_end|>
    //   <|im_start|>user
    //   <|audio_start|>[user_audio_embed]<|audio_end|>
    //   <|im_end|>
    //   <|im_start|>assistant
    //   <think>
    //   
    //   </think>
    //   
    //   <|tts_bos|>
    // 
    // 注意: voice_clone_prompt 是 system prompt 的 prefix，assistant_prompt 是 system prompt 的 suffix
    //       stream_decode 会添加实际的 assistant generation prompt
    if (duplex_mode) {
        // 🔧 [与 Python 对齐] Audio 双工模式：嵌入参考音频
        // 双工模式不需要 <|im_start|>user\n，用 <unit> 标记用户输入
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        
        // 🔧 [修复] Omni 双工模式：也需要嵌入参考音频，格式与 Audio 双工相同
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 🔧 [与 Python 对齐] 非双工模式 Audio 格式 (audio_assistant 模式)
        // 格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n
        // 🔧 [整合] 在 sys prompt 末尾直接添加 <|im_start|>user\n，不再在 stream_prefill 里动态添加
        // 这样更稳妥，不依赖 Python 端的 counter 重置
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
        
        // Omni 模式（非双工）：与 Audio 模式类似，末尾也添加 <|im_start|>user\n
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
    }

    llama_model * model = nullptr;
    llama_context * ctx_llama = nullptr;
    
    // 支持模型复用（单工模式常用）
    if (existing_model != nullptr && existing_ctx != nullptr) {
        print_with_timestamp("=== omni_init: reusing existing LLM model and context\n");
        model = existing_model;
        ctx_llama = existing_ctx;
        ctx_omni->owns_model = false;  // 不拥有模型，omni_free 时不释放
        
        // 🔧 [模式切换修复] 清理 LLM 的 KV cache，避免位置冲突
        // 当从一个模式切换到另一个模式时，需要清理旧的 KV cache
        llama_memory_t mem = llama_get_memory(ctx_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);  // 清除 sequence 0 的所有 KV cache
            print_with_timestamp("=== omni_init: cleared LLM KV cache for mode switch\n");
        }
    } else {
        // 加载新模型
        print_with_timestamp("=== omni_init: loading new LLM model\n");
        model = llama_init(params, params->model.path);
        if (model == NULL) {
            return NULL;
        }
        llama_context_params ctx_params = common_context_params_to_llama(*params);
        ctx_params.n_ctx                = params->n_ctx;
        
        ctx_llama = llama_new_context_with_model(model, ctx_params);
        if (ctx_llama == NULL) {
            LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
            return NULL;
        }
        ctx_omni->owns_model = true;  // 拥有模型，omni_free 时需要释放
    }
    
    struct common_sampler * sampler = common_sampler_init(model, params->sampling);
    ctx_omni->ctx_llama = ctx_llama;
    ctx_omni->model = model;
    ctx_omni->ctx_sampler = sampler;

    if (use_tts && !params->tts_model.empty()) {
        print_with_timestamp("=== omni_init: loading TTS model\n");
        // 使用TTS专用的模型加载函数，支持独立的GPU层数设置
        // tts_gpu_layers 从 omni_init 参数传入，-1 表示使用与LLM相同的设置
        print_with_timestamp("TTS model: loading with n_gpu_layers=%d\n", tts_gpu_layers);
        llama_model * tts_model = llama_init_tts(params, params->tts_model, tts_gpu_layers);
        if (tts_model == NULL) {
            LOG_ERR("%s: error: failed to init TTS model from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // TTS 模型使用独立的上下文参数
        // 注意：TTS 模型可能需要不同的上下文大小和批处理大小
        llama_context_params tts_ctx_params = common_context_params_to_llama(*params);
        // 如果 TTS 模型需要更小的上下文窗口，可以在这里调整
        // 例如：tts_ctx_params.n_ctx = std::min(params->n_ctx, 2048); // 限制 TTS 上下文大小
        tts_ctx_params.n_ctx = params->n_ctx;  // 暂时使用相同的 n_ctx，后续可以根据需要调整
        
        llama_context * ctx_tts_llama = llama_new_context_with_model(tts_model, tts_ctx_params);
        if (ctx_tts_llama == NULL) {
            LOG_ERR("%s: error: failed to create the TTS llama_context\n", __func__);
            llama_free_model(tts_model);
            // 清理已分配的资源
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // 创建 TTS 的采样器
        // 🔧 TTS流式采样参数 - 与 Python ras_sampling 对齐：
        // Python TTSSamplingParams 默认 temperature=0.8 (modeling_minicpmo.py line 75)
        common_params_sampling tts_sampling = params->sampling;
        tts_sampling.temp = 0.8f;              // 🔧 [与 Python 对齐] TTSSamplingParams.temperature=0.8
        tts_sampling.top_p = 0.85f;  // 🔧 [与 Python 对齐] TTSSamplingParams.top_p=0.85             // 🔧 [与 Python streaming 对齐] top_p=0.8
        tts_sampling.top_k = 25;               // top_k = 25 (ras_sampling 参数)
        tts_sampling.penalty_repeat = 1.05f;   // repetition_penalty = 1.05
        tts_sampling.min_p = 0.01f;            // min_p = 0.01
        // Python: CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16)
        tts_sampling.penalty_last_n = 16;      // past_window = 16 (与Python对齐)
        struct common_sampler * tts_sampler = common_sampler_init(tts_model, tts_sampling);
        print_with_timestamp("TTS sampler: temp=%.2f, top_p=%.2f, top_k=%d, rep_penalty=%.2f\n",
                            tts_sampling.temp, tts_sampling.top_p, tts_sampling.top_k, tts_sampling.penalty_repeat);
        
        ctx_omni->model_tts = tts_model;
        ctx_omni->ctx_tts_llama = ctx_tts_llama;
        ctx_omni->ctx_tts_sampler = tts_sampler;
        
        // Load TTS weights from GGUF file
        print_with_timestamp("TTS: loading weights from GGUF (emb_code, emb_text, projector_semantic, head_code)...\n");
        if (!load_tts_weights_from_gguf(ctx_omni, params->tts_model.c_str())) {
            LOG_ERR("%s: error: failed to load TTS weights from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_tts_llama);
            llama_free_model(tts_model);
            common_sampler_free(tts_sampler);
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        print_with_timestamp("TTS: weights loaded successfully\n");
        
        // Load Projector Semantic from GGUF file
        // 路径: {tts_bin_dir}/MiniCPM-o-4_5-projector-F16.gguf
        std::string projector_path = tts_bin_dir + "/MiniCPM-o-4_5-projector-F16.gguf";
        print_with_timestamp("Projector: loading from %s\n", projector_path.c_str());
        if (projector_init(ctx_omni->projector, projector_path, true)) {
            print_with_timestamp("Projector: loaded successfully\n");
        } else {
            print_with_timestamp("Projector: failed to load, will use fallback implementation\n");
        }
    }

    ctx_omni->omni_emb.resize((64 + 10 + 1) * 4096); // temp fix for omni embed
    ctx_omni->audio_emb.resize((10 + 1) * 4096); // temp fix for audio embed
    print_with_timestamp("=== omni_init: loading APM model\n");
    if (params->apm_model.empty()) {
        LOG_ERR("%s: error: apm_model path is empty\n", __func__);
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }
    ctx_omni->ctx_audio = audition_init(params->apm_model.c_str(), audition_context_params{true, GGML_LOG_LEVEL_INFO});
    print_with_timestamp("APM: init from %s\n", params->apm_model.c_str());
    if (ctx_omni->ctx_audio == nullptr) {
        LOG_ERR("%s: error: failed to init audition model from %s\n", __func__, params->apm_model.c_str());
        // 清理 TTS 模型资源（如果已加载）
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }

    ctx_omni->n_past = 0;
    
    if (media_type == 2) {
        LOG_INF("init vision....");
        const char * vision_path = ctx_omni->params->vpm_model.c_str();
        auto * ctx_vision = vision_init(vision_path, vision_context_params{true, GGML_LOG_LEVEL_INFO, nullptr});
        ctx_omni->ctx_vision = ctx_vision;

        // Set CoreML model path if available (for vision ANE acceleration)
        // Note: .mlmodelc is a directory, not a file, so use stat instead of ifstream
        if (ctx_vision && !ctx_omni->params->vision_coreml_model_path.empty()) {
            struct stat coreml_stat;
            if (stat(ctx_omni->params->vision_coreml_model_path.c_str(), &coreml_stat) == 0) {
                vision_set_coreml_model_path(ctx_vision, ctx_omni->params->vision_coreml_model_path.c_str());
                LOG_INF("Vision CoreML model path set to: %s\n", ctx_omni->params->vision_coreml_model_path.c_str());
            } else {
                LOG_WRN("Vision CoreML model path does not exist: %s, skipping ANE\n", ctx_omni->params->vision_coreml_model_path.c_str());
            }
        }
    }
    
    ctx_omni->llm_thread_info = new LLMThreadInfo(1000);
    if (ctx_omni->use_tts) {
        LOG_INF("init tts....");
        ctx_omni->tts_thread_info = new TTSThreadInfo(1);
        ctx_omni->omni_output = new omni_output();
        ctx_omni->tts_bin_dir = tts_bin_dir;
        
        // Initialize T2W thread info
        LOG_INF("init t2w....");
        ctx_omni->t2w_thread_info = new T2WThreadInfo(25);  // Queue size of 10 chunks
        
        // Initialize C++ Token2Wav session
        // Try to load token2wav GGUF models from {model_dir}/token2wav-gguf/
        // Fallback to tools/omni/token2wav-gguf if not found
        ctx_omni->token2wav_initialized = false;
        
        // 🔧 如果使用 Python Token2Wav，跳过 C++ 的初始化以节省显存
        bool skip_cpp_token2wav = ctx_omni->use_python_token2wav;
        
        // Check if token2wav model files exist
        // 优先检查 HF 模型目录下的 token2wav-gguf (tts_bin_dir 的父目录)
        // 目录结构: {model_dir}/token2wav-gguf/
        std::string gguf_root_dir = tts_bin_dir;
        size_t last_slash = gguf_root_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            gguf_root_dir = gguf_root_dir.substr(0, last_slash);  // 获取 tts 的父目录
        }
        ctx_omni->token2wav_model_dir = gguf_root_dir + "/token2wav-gguf";
        
        std::string encoder_test = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        {
            std::ifstream f(encoder_test);
            if (!f.good()) {
                // 尝试备用路径 (本地开发用)
                ctx_omni->token2wav_model_dir = "tools/omni/token2wav-gguf";
                print_with_timestamp("Token2Wav: trying fallback path %s\n", ctx_omni->token2wav_model_dir.c_str());
            } else {
                print_with_timestamp("Token2Wav: found models in %s\n", ctx_omni->token2wav_model_dir.c_str());
            }
        }
        std::string encoder_gguf = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        std::string flow_matching_gguf = ctx_omni->token2wav_model_dir + "/flow_matching.gguf";
        std::string flow_extra_gguf = ctx_omni->token2wav_model_dir + "/flow_extra.gguf";
        std::string vocoder_gguf = ctx_omni->token2wav_model_dir + "/hifigan2.gguf";
        std::string prompt_cache_gguf = ctx_omni->token2wav_model_dir + "/prompt_cache.gguf";
        
        // Check if all files exist
        bool all_files_exist = true;
        std::vector<std::string> gguf_files = {encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf};
        for (const auto& file : gguf_files) {
            std::ifstream f(file);
            if (!f.good()) {
                all_files_exist = false;
                break;
            } else {
            }
        }
        
        if (all_files_exist && !skip_cpp_token2wav) {
            print_with_timestamp("Token2Wav: all model files found, initializing session...\n");
            ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
            
            // Device configuration - 使用 omni_init 传入的 token2wav_device 参数
            // 格式: "gpu", "gpu:0", "gpu:1", "cpu"
            std::string device_token2mel = token2wav_device;

            // Vocoder 设备策略：
            //   CUDA: vocoder 跟随 token2wav_device（GPU），因为 CUDA kernel launch 开销低
            //   Metal (macOS): vocoder 强制用 CPU，因为 vocoder 有大量小操作，
            //     Metal kernel dispatch 开销累积后远慢于 CPU 直接计算
            //   可通过 OMNI_VOC_DEVICE 环境变量覆盖
            const char * voc_dev_env = getenv("OMNI_VOC_DEVICE");
            std::string device_vocoder;
            if (voc_dev_env) {
                device_vocoder = voc_dev_env;
                print_with_timestamp("Token2Wav: vocoder device overridden by OMNI_VOC_DEVICE=%s\n", voc_dev_env);
            } else {
#ifdef GGML_USE_CUDA
                device_vocoder = token2wav_device;
                print_with_timestamp("Token2Wav: CUDA detected, vocoder using GPU (%s)\n", device_vocoder.c_str());
#else
                device_vocoder = "cpu";
                print_with_timestamp("Token2Wav: non-CUDA backend, vocoder using CPU for better performance\n");
#endif
            }
            
            // 🔧 优先使用 prompt_bundle (setup_cache 路径)，否则 fallback 到 prompt_cache.gguf
            std::string prompt_bundle_dir = "tools/omni/assets/default_ref_audio";
            std::string spk_file = prompt_bundle_dir + "/spk_f32.bin";
            std::string tokens_file = prompt_bundle_dir + "/prompt_tokens_i32.bin";
            std::string mel_file = prompt_bundle_dir + "/prompt_mel_btc_f32.bin";
            
            bool use_prompt_bundle = false;
            {
                std::ifstream f1(spk_file), f2(tokens_file), f3(mel_file);
                use_prompt_bundle = f1.good() && f2.good() && f3.good();
            }
            
            bool init_ok = false;
            // 优先级: prompt_cache.gguf > prompt_bundle (实时计算 fallback)
            print_with_timestamp("Token2Wav: using prompt_cache from %s\n", prompt_cache_gguf.c_str());
            init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                    encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                    vocoder_gguf, device_token2mel, device_vocoder, 5, 1.0f);
            if (!init_ok && use_prompt_bundle) {
                print_with_timestamp("Token2Wav: prompt_cache failed, fallback to prompt_bundle from %s\n", prompt_bundle_dir.c_str());
                init_ok = ctx_omni->token2wav_session->init_from_prompt_bundle(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_bundle_dir,
                        vocoder_gguf, device_token2mel, device_vocoder, 5, 1.0f);
            }
            // Fallback to CPU
            if (!init_ok) {
                print_with_timestamp("Token2Wav: GPU init failed, trying CPU mode...\n");
                ctx_omni->token2wav_session.reset();
                ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
                init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                        vocoder_gguf, "cpu", "cpu", 5, 1.0f);
            }
            
            if (init_ok) {
                ctx_omni->token2wav_initialized = true;
                // Initialize token2wav buffer with 3 silence tokens (4218) as Python does
                // Python: buffer = [4218] * 3  # 预先放入3个前缀静音token
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
                print_with_timestamp("Token2Wav: initialized successfully\n");
            } else {
                ctx_omni->token2wav_session.reset();
                print_with_timestamp("Token2Wav: initialization failed\n");
            }
        } else {
            print_with_timestamp("Token2Wav: model files not found in %s\n", ctx_omni->token2wav_model_dir.c_str());
        }
        
        // ==================== 初始化 Python Token2Wav ====================
        // 🔧 默认使用 Python Token2Wav（精度更高）
        // 设置 Python T2W 脚本目录和模型目录
        // Python T2W 脚本目录：tools/omni/pyt2w/
        // Python T2W 模型目录：dependencies/token2wav/
        
        // 计算 Python T2W 脚本目录（相对于 tts_bin_dir）
        // tts_bin_dir 通常是 /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 我们需要 /xxx/tools/omni/pyt2w
        std::string t2w_script_dir = tts_bin_dir;  // /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 回退到 tools/omni/
        size_t convert_pos = t2w_script_dir.find("/convert/gguf/tts");
        if (convert_pos != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else if ((convert_pos = t2w_script_dir.find("/convert/gguf")) != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else {
            // 尝试从当前工作目录构建
            t2w_script_dir = "./tools/omni/pyt2w";
        }
        ctx_omni->python_t2w_script_dir = t2w_script_dir;
        
        // Python T2W 模型目录（stepaudio2 模型）
        // 默认路径：相对于 script_dir 的 token2wav 子目录
        ctx_omni->python_t2w_model_dir = t2w_script_dir + "/token2wav";
        
        // 参考音频路径
        std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
        
        print_with_timestamp("Python T2W: script_dir=%s, model_dir=%s\n", 
                             ctx_omni->python_t2w_script_dir.c_str(),
                             ctx_omni->python_t2w_model_dir.c_str());
        
        if (ctx_omni->use_python_token2wav) {
            print_with_timestamp("Python T2W: 使用 Python Token2Wav 实现\n");
            
            // 🔧 Python T2W GPU 配置
            // C++ LLM+TTS 占用约 22GB，Python T2W 占用约 3.3GB
            // 单卡 24GB 放不下，需要配置独立 GPU
            // 
            // 通过环境变量 PYTHON_T2W_GPU 配置独立 GPU
            // 例如: export PYTHON_T2W_GPU=1  # Python T2W 使用 GPU 1
            // 
            // 优先级：PYTHON_T2W_GPU 环境变量 > 外部 CUDA_VISIBLE_DEVICES > token2wav_device
            const char* env_python_t2w_gpu = getenv("PYTHON_T2W_GPU");
            if (env_python_t2w_gpu && strlen(env_python_t2w_gpu) > 0) {
                ctx_omni->python_t2w_dedicated_gpu = env_python_t2w_gpu;
            }
            
            ctx_omni->python_t2w_gpu_id = "";
            
            if (!ctx_omni->python_t2w_dedicated_gpu.empty()) {
                // 使用配置的独立 GPU
                ctx_omni->python_t2w_gpu_id = ctx_omni->python_t2w_dedicated_gpu;
                print_with_timestamp("Python T2W: 使用独立 GPU %s (C++ 和 Python 分开)\n", ctx_omni->python_t2w_gpu_id.c_str());
            } else {
                const char* env_cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
                if (env_cuda_visible && strlen(env_cuda_visible) > 0) {
                    // 外部已设置，Python 子进程会继承，不需要额外设置
                    print_with_timestamp("Python T2W: 继承外部 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", env_cuda_visible);
                } else if (token2wav_device.find("gpu") != std::string::npos) {
                    // 外部未设置，从 token2wav_device 提取
                    size_t colon_pos = token2wav_device.find(':');
                    if (colon_pos != std::string::npos) {
                        ctx_omni->python_t2w_gpu_id = token2wav_device.substr(colon_pos + 1);
                    } else {
                        ctx_omni->python_t2w_gpu_id = "0";
                    }
                    print_with_timestamp("Python T2W: 设置 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", ctx_omni->python_t2w_gpu_id.c_str());
                } else {
                    print_with_timestamp("Python T2W: CPU 模式\n");
                }
            }
            
            // 启动 Python 服务
            if (start_python_t2w_service(ctx_omni)) {
                // 初始化模型
                if (init_python_t2w_model(ctx_omni, token2wav_device)) {
                    // 设置参考音频
                    if (set_python_t2w_ref_audio(ctx_omni, ref_audio_path)) {
                        print_with_timestamp("Python T2W: 初始化成功\n");
                    } else {
                        print_with_timestamp("Python T2W: 设置参考音频失败\n");
                        ctx_omni->use_python_token2wav = false;
                    }
                } else {
                    print_with_timestamp("Python T2W: 初始化模型失败\n");
                    ctx_omni->use_python_token2wav = false;
                }
            } else {
                print_with_timestamp("Python T2W: 启动服务失败\n");
                ctx_omni->use_python_token2wav = false;
            }
            
            // 如果 Python 初始化失败，回退到 C++ 实现
            if (!ctx_omni->use_python_token2wav) {
                print_with_timestamp("Python T2W: 回退到 C++ 实现\n");
            }
        } else {
            print_with_timestamp("Token2Wav: 使用 C++ 实现\n");
        }
    }
    ctx_omni->async = true;
    
    // ==================== 初始化特殊 Token ID ====================
    // 从 LLM 词表中查找并缓存特殊 token ID
    // 这些 token 用于控制双工模式下的状态切换
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab) {
        // 使用 llama_tokenize 直接将字符串转换为 token ID
        // parse_special=true 确保特殊 token 被正确解析
        auto find_token = [&](const char * token_str) -> llama_token {
            llama_token tokens[4];  // 预留空间
            int n_tokens = llama_tokenize(vocab, token_str, strlen(token_str), tokens, 4, false, true);
            if (n_tokens == 1) {
                return tokens[0];
            }
            // 如果 tokenize 失败，尝试遍历词表查找（使用 special=true）
            int n_vocab = llama_vocab_n_tokens(vocab);
            for (int i = 0; i < n_vocab; i++) {
                char buf[128];
                int len = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);  // special=true
                if (len > 0 && len < (int)sizeof(buf)) {
                    buf[len] = '\0';
                    if (strcmp(buf, token_str) == 0) {
                        return i;
                    }
                }
            }
            return -1;
        };
        
        ctx_omni->special_token_speak = find_token("<|speak|>");
        ctx_omni->special_token_listen = find_token("<|listen|>");
        ctx_omni->special_token_chunk_eos = find_token("<|chunk_eos|>");
        ctx_omni->special_token_chunk_tts_eos = find_token("<|chunk_tts_eos|>");
        ctx_omni->special_token_turn_eos = find_token("<|turn_eos|>");
        ctx_omni->special_token_tts_eos = find_token("<|tts_eos|>");
        ctx_omni->special_token_eos = llama_vocab_eos(vocab);
        
        // 同时初始化 tts_bos_token_id（用于双工模式强制继续说话）
        llama_token tts_bos = find_token("<|tts_bos|>");
        if (tts_bos >= 0) {
            ctx_omni->tts_bos_token_id = tts_bos;
        }
        // 初始化 </unit> token（用于双工模式 chunk 边界标记）
        ctx_omni->special_token_unit_end = find_token("</unit>");
        
        // 🔧 [双工模式] 初始化 <|tts_pad|> token（双工模式下禁止采样此 token）
        // Python: self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
        ctx_omni->special_token_tts_pad = find_token("<|tts_pad|>");
    }
        
    // ANE/CoreML warmup: pre-load models into NPU to avoid first-inference latency
    omni_warmup_ane(ctx_omni);

    print_with_timestamp("=== omni_init success: ctx_llama = %p\n", (void*)ctx_omni->ctx_llama);
    return ctx_omni;
}

//
// ANE/CoreML warmup — pre-load models into NPU to avoid first-inference latency
//
void omni_warmup_ane(struct omni_context * ctx_omni) {
#if defined(__APPLE__)
    if (!ctx_omni) return;

    LOG_INF("%s: starting ANE/CoreML warmup...\n", __func__);

    // 1. Vision ANE warmup
    if (ctx_omni->ctx_vision) {
        vision_coreml_warmup(ctx_omni->ctx_vision);
    }

    // 2. Future: audio ANE warmup
    // if (ctx_omni->ctx_audio) {
    //     audition_coreml_warmup(ctx_omni->ctx_audio);
    // }

    // 3. Future: other module ANE warmup
    // ...

    LOG_INF("%s: ANE/CoreML warmup finished\n", __func__);
#else
    (void)ctx_omni;
#endif
}

// 停止所有线程（发送信号，不等待）
void omni_stop_threads(struct omni_context * ctx_omni) {
    // 发送停止信号
    llm_thread_running = false;
    tts_thread_running = false;
    t2w_thread_running = false;
    
    // 唤醒所有等待的线程
    if (ctx_omni->llm_thread_info) {
        ctx_omni->llm_thread_info->cv.notify_all();
    }
    if (ctx_omni->tts_thread_info) {
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    if (ctx_omni->t2w_thread_info) {
        ctx_omni->t2w_thread_info->cv.notify_all();
    }
    
    print_with_timestamp("omni_stop_threads: stop signals sent\n");
}

void omni_free(struct omni_context * ctx_omni) {
    
    // 等待 llm 和 tts thread 停止
    llm_thread_running = false; // Signal the thread to stop
    if (ctx_omni->llm_thread.joinable()) {
        ctx_omni->llm_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
        ctx_omni->llm_thread.join(); // Wait for the thread to finish
    }
 
    if (ctx_omni->use_tts) {
        tts_thread_running = false; // Signal the thread to stop
        if (ctx_omni->tts_thread.joinable()) {
            ctx_omni->tts_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->tts_thread.join(); // Wait for the thread to finish
        }
        
        // Stop T2W thread
        t2w_thread_running = false; // Signal the thread to stop
        if (ctx_omni->t2w_thread.joinable()) {
            ctx_omni->t2w_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->t2w_thread.join(); // Wait for the thread to finish
        }
    }
    
    delete ctx_omni->ctx_vision;
    audition_free(ctx_omni->ctx_audio);
    
    if (ctx_omni->use_tts) {
        llama_free(ctx_omni->ctx_tts_llama);
        llama_free_model(ctx_omni->model_tts);
        common_sampler_free(ctx_omni->ctx_tts_sampler);
        
        // Free TTS weights
        if (ctx_omni->emb_code_weight) {
            free(ctx_omni->emb_code_weight);
            ctx_omni->emb_code_weight = nullptr;
        }
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_weight) {
            free(ctx_omni->projector_semantic_linear1_weight);
            ctx_omni->projector_semantic_linear1_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_bias) {
            free(ctx_omni->projector_semantic_linear1_bias);
            ctx_omni->projector_semantic_linear1_bias = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_weight) {
            free(ctx_omni->projector_semantic_linear2_weight);
            ctx_omni->projector_semantic_linear2_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_bias) {
            free(ctx_omni->projector_semantic_linear2_bias);
            ctx_omni->projector_semantic_linear2_bias = nullptr;
        }
        if (ctx_omni->head_code_weight) {
            free(ctx_omni->head_code_weight);
            ctx_omni->head_code_weight = nullptr;
        }
        
        // Free C++ Token2Wav session
        if (ctx_omni->token2wav_session) {
            ctx_omni->token2wav_session.reset();
            ctx_omni->token2wav_initialized = false;
            LOG_INF("Token2Wav (C++): session released\n");
        }
        
        // 🔧 停止 Python Token2Wav 服务
        if (ctx_omni->python_t2w_initialized) {
            stop_python_t2w_service(ctx_omni);
        }
        
        // Free ggml-based projector model
        if (ctx_omni->projector.initialized) {
            projector_free(ctx_omni->projector);
        }
    }
    
    // 🔧 [单双工适配] 只有在拥有模型时才释放 LLM model 和 context
    // 如果是外部传入的模型（模型复用），则不释放
    if (ctx_omni->owns_model) {
        llama_free(ctx_omni->ctx_llama);
        llama_free_model(ctx_omni->model);
    }
    common_sampler_free(ctx_omni->ctx_sampler);
    // delete ctx_omni->ctx_tts;
    delete ctx_omni->llm_thread_info;
    delete ctx_omni->audio_input_manager;
    
    // omni_output 还要把里面 output 的每个元素也 delete 下
    if (ctx_omni->use_tts) {
        delete ctx_omni->tts_thread_info;
        delete ctx_omni->t2w_thread_info;
        
        // omni_output 还要把里面 output 的每个元素也 delete 下
        if (ctx_omni->omni_output) {
            for (auto &buffer : ctx_omni->omni_output->output) {
                delete buffer;
            }
            ctx_omni->omni_output->output.clear(); // Clear the vector
            delete ctx_omni->omni_output;
        }
    }

    llama_backend_free();

    delete ctx_omni;
}

// ==================== 语言设置函数 ====================
// 设置语言并更新 system prompt（zh=中文，en=英文）
// 基于 Python MiniCPM-o-4_5 modeling_minicpmo.py 中的 audio_assistant 模式 prompt
void omni_set_language(struct omni_context * ctx_omni, const std::string & lang) {
    if (ctx_omni == nullptr) {
        print_with_timestamp("omni_set_language: ctx_omni is null\n");
        return;
    }
    
    ctx_omni->language = lang;
    print_with_timestamp("omni_set_language: setting language to '%s'\n", lang.c_str());
    
    if (ctx_omni->duplex_mode) {
        // 双工模式：prompt 固定使用英文（与 Python 对齐）
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 非双工模式（audio_assistant 模式）：根据语言设置 prompt
        if (lang == "en") {
            // 英文 prompt（来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni.<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style.<|im_end|>\n<|im_start|>user\n";
        } else {
            // 中文 prompt（默认，来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
        }
    }
    
    // 🔧 [关键] 重置 system_prompt_initialized，让下次 stream_prefill(index=0) 重新 prefill system prompt
    ctx_omni->system_prompt_initialized = false;
    
    print_with_timestamp("omni_set_language: prompts updated for language '%s', system_prompt_initialized reset to false\n", lang.c_str());
}

static void process_audio(struct omni_context * ctx_omni, struct omni_embed * embeds, common_params * params, bool save_spk_emb=false) {
    LOG_INF("%s: audio token past: %d\n", __func__, ctx_omni->n_past);
    omni_eval_embed(ctx_omni->ctx_llama, embeds, params->n_batch, &ctx_omni->n_past);
    LOG_INF("%s: audio token past after eval: %d\n", __func__, ctx_omni->n_past);
}

void eval_prefix(struct omni_context* ctx_omni, common_params* params){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false);
}

void eval_prefix_with_hidden(struct omni_context* ctx_omni, common_params* params, float *& hidden_states){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string_with_hidden(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false, hidden_states);
}

/**
 * LLM线程函数：负责处理多模态（视觉+音频）嵌入的前缀填充（prefill）
 * 
 * 这个函数在一个独立线程中运行，主要职责是：
 * 1. 从队列中获取视觉和音频嵌入数据
 * 2. 将嵌入数据组合成LLM可以处理的格式
 * 3. 执行前缀填充，为后续的文本生成做准备
 * 4. 协调与解码线程的同步
 * 
 * 运行逻辑：
 * - 主循环持续运行，直到 llm_thread_running 为 false
 * - 等待条件：队列不为空 OR need_speek 为 true OR 线程需要停止
 * - 两个主要分支：
 *   分支1：队列不为空 -> 处理嵌入数据的前缀填充
 *   分支2：队列为空且 need_speek 为 true -> 通知解码线程可以开始生成
 */
void llm_thread_func(omni_context* ctx_omni, common_params* params){
    print_with_timestamp("LLM thread started\n");
    // 获取模型的隐藏层维度，用于计算token数量
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    
    // ========== 主循环：持续处理嵌入数据 ==========
    while(llm_thread_running){
        // 获取队列的互斥锁，保护共享资源
        std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        auto& queue = ctx_omni->llm_thread_info->queue;

        // 打印当前状态（调试用）
        
        // ========== 等待条件满足 ==========
        // 等待以下任一条件满足：
        // 1. 队列不为空（有新嵌入数据需要处理）
        // 2. need_speek 为 true（需要开始生成文本）
        // 3. llm_thread_running 为 false（线程需要停止）
        ctx_omni->llm_thread_info->cv.wait(lock, [&] { 
            return !queue.empty() || ctx_omni->need_speek || !llm_thread_running; 
        });
        
        // 检查是否需要停止线程
        if (!llm_thread_running) {
            break;
        }

        // ========== 分支1：处理队列中的嵌入数据（前缀填充） ==========
        if (!queue.empty()){
            // 🔧 [诊断] 打印 prefill 开始时的 n_past
            print_with_timestamp("LLM thread: start prefill, n_past=%d, n_keep=%d, n_ctx=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep, params->n_ctx);
            
            // 🔧 [修复] prefill 阶段不清除 KV cache，保持累积
            // 无论单工还是双工模式，prefill 都是累积用户输入
            // KV cache 只在以下情况清除：
            //   1. 新会话开始（通过 reset API）
            //   2. 滑动窗口触发（context 满了）
            print_with_timestamp("LLM thread: prefill continuing, n_past=%d (no KV cache clear)\n", ctx_omni->n_past);
            
            // 标记前缀填充未完成，防止解码线程过早开始
            prefill_done = false;
            
            // 步骤1：批量取出队列中的所有嵌入数据
            // 这样可以一次性处理多个嵌入，提高效率
            std::vector<omni_embeds*> llm_embeds;
            llm_embeds.clear();
            while (!queue.empty()) {
                llm_embeds.push_back(queue.front());
                queue.pop();
            }
            
            // 释放锁，允许其他线程继续向队列添加数据
            lock.unlock();
            
            // 如果批量处理多个嵌入，打印日志
            print_with_timestamp("Batch processing %zu llm prefill\n", llm_embeds.size());
            if (llm_embeds.size() > 1)
            
            // 通知等待的生产者线程，队列有空间了
            ctx_omni->llm_thread_info->cv.notify_all();

            // 🔧 [与 Python 对齐] 只有非双工模式才添加 <|im_start|>user\n
            // 双工模式: 直接用 <unit> 标记用户输入开始，不需要 <|im_start|>user\n
            // 非双工模式: <|im_start|>system....<|im_end|>\n<|im_start|>user\n<|audio_start|>audio<|audio_end|><|im_end|>\n<|im_start|>assistant...
            // 🔧 [整合] <|im_start|>user\n 已在 sys prompt 末尾添加（第一轮），
            // 后续轮次在 stream_decode 结束时添加
            // 不再需要在这里动态添加

            // 🔧 [重构] 逐个处理嵌入数据，正确添加特殊标记
            // 遍历所有嵌入数据
            for (int il = 0; il < (int)llm_embeds.size(); ++il) {
                auto embeds = llm_embeds[il];
                
                // 🔧 [#39 滑动窗口] 注册 unit 开始
                if (ctx_omni->sliding_window_config.mode != "off") {
                    sliding_window_register_unit_start(ctx_omni);
                }
                
                // ========== 子分支1：处理包含视觉嵌入的数据 ==========
                // 🔧 [高清模式] vision_embed 现在是二维 vector: [0]=overview, [1..n]=slices
                if (embeds->vision_embed.size() > 0){
                    int n_chunks = (int)embeds->vision_embed.size();
                    int tokens_per_chunk = (int)embeds->vision_embed[0].size() / hidden_size;
                    int n_audio_tokens = embeds->audio_embed.size() / hidden_size;
                    bool has_audio = (n_audio_tokens > 0);
                    bool has_slices = (n_chunks > 1);
                    
                    // 🔧 [与 Python 对齐] 根据模式决定是否添加 <unit>
                    if (ctx_omni->duplex_mode) {
                        eval_string(ctx_omni, params, "<unit><image>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        eval_string(ctx_omni, params, "<image>", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    // Prefill overview embedding (第一个 chunk)
                    prefill_with_emb(ctx_omni, params, embeds->vision_embed[0].data(), tokens_per_chunk, 
                                    params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, params, "</image>", params->n_batch, &ctx_omni->n_past, false);
                    
                    // 🔧 [高清模式 V2.6 schema] 如果有 slices，添加 <slice> 标记
                    // 格式: <image>(overview)</image><slice>(slice1)</slice><slice>(slice2)</slice>\n
                    if (has_slices) {
                        for (int i = 1; i < n_chunks; i++) {
                            eval_string(ctx_omni, params, "<slice>", params->n_batch, &ctx_omni->n_past, false);
                            prefill_with_emb(ctx_omni, params, embeds->vision_embed[i].data(), tokens_per_chunk,
                                            params->n_batch, &ctx_omni->n_past);
                            eval_string(ctx_omni, params, "</slice>", params->n_batch, &ctx_omni->n_past, false);
                        }
                        // V2.6 格式在 slices 后添加换行
                        eval_string(ctx_omni, params, "\n", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    print_with_timestamp("Omni模式: %d vision chunks (%d tokens each), %d audio tokens, has_slices=%d\n", 
                                        n_chunks, tokens_per_chunk, n_audio_tokens, has_slices);
                    
                    // 音频部分
                    if (has_audio) {
                        if (!ctx_omni->duplex_mode) {
                            // 单工格式：<|audio_start|> + audio + <|audio_end|>
                            eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                        }
                        prefill_with_emb(ctx_omni, params, embeds->audio_embed.data(), n_audio_tokens,
                                        params->n_batch, &ctx_omni->n_past);
                        if (!ctx_omni->duplex_mode) {
                            eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
                        }
                    }
                }
                // ========== 子分支2：处理只有音频嵌入的数据（纯音频模式） ==========
                else {
                    int n_audio_tokens = embeds->audio_embed.size() / hidden_size;
                    print_with_timestamp("用户语音: %d audio tokens\n", n_audio_tokens);
                    
                    // 🔧 [根据模式选择格式]
                    if (ctx_omni->duplex_mode) {
                        // 双工格式：<unit> + audio_embedding（无 audio_start/end）
                        eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        // 单工格式：<|audio_start|> + audio + <|audio_end|>
                        eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    // Prefill 音频 embedding
                    prefill_with_emb(ctx_omni, params, embeds->audio_embed.data(), n_audio_tokens,
                                    params->n_batch, &ctx_omni->n_past);
                    
                    // 单工格式需要 <|audio_end|>
                    if (!ctx_omni->duplex_mode) {
                        eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
                    }
                }
                
                // 🔧 [#39 滑动窗口] 注册 unit 结束
                if (ctx_omni->sliding_window_config.mode != "off") {
                    std::string input_type = embeds->vision_embed.size() > 0 ? "omni" : "audio";
                    sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
                }
                
                // 释放嵌入数据的内存（由生产者线程分配）
                delete embeds;
            }
            
            // 🔧 [诊断] 打印 prefill 结束后的 n_past
            print_with_timestamp("LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep, 
                                 ctx_omni->n_past - ctx_omni->n_keep,
                                 ctx_omni->duplex_mode);
            
            // 🔧 [#39 滑动窗口] prefill 完成后检查是否需要滑窗
            if (ctx_omni->sliding_window_config.mode != "off") {
                sliding_window_enforce(ctx_omni);
            }
        }
        
        // ========== 分支2：队列为空且需要开始生成文本 ==========
        // 这个分支在以下情况触发：
        // 1. 所有嵌入数据都已处理完成（队列为空）
        // 2. 解码线程设置了 need_speek = true，表示需要开始生成文本

        if (queue.empty() && ctx_omni->need_speek){
            // 标记前缀填充完成
            prefill_done = true;
            
            // 如果使用TTS，重置speek_done标志，允许TTS线程开始工作
            if (ctx_omni->use_tts && !ctx_omni->duplex_mode) {
                ctx_omni->speek_done = false;
            }
            
            // 重置need_speek标志
            ctx_omni->need_speek = false;
            
            // 通知等待的解码线程：前缀填充已完成，可以开始生成文本了
            g_decode_cv.notify_all();
        }
    }
}

// Special token IDs to filter (from file_loader.py)
// 🔧 [与 Python 对齐] Python MiniCPMODuplex.streaming_generate 中的过滤逻辑：
//   1. chunk_speak_token_ids = [speak_token_id] 中的 token 不会被添加到 TTS 条件
//   2. j != 0 条件跳过循环中的第一个 token
// 注意：这些 token 大部分也会被 tid >= 150000 规则过滤，但显式列出便于理解和调试
static const std::vector<llama_token> g_special_token_ids = {
    151667,  // <think>
    151668,  // </think>
    151704,  // <|tts_eos|> - TTS 结束标记
    151706,  // <|speak|> - 🔧 Python 的 chunk_speak_token_ids 会过滤此 token
    151705,  // <|listen|> - 双工模式切换到听的标记
    151718,  // <|chunk_eos|> - chunk 结束标记
    151721,  // <|chunk_tts_eos|> - TTS chunk 结束标记
    151717,  // <|turn_eos|> - 轮次结束标记
    271,     // <reserved_182> (newline, appears near think tags)
};

// Known empty token IDs (from file_loader.py)
// WARNING: These tokens were incorrectly marked as "empty" but they actually decode to real text!
// 99692 = "好的", 104314 = "给你", 99526 = "讲"
// DO NOT add tokens here without verifying they actually decode to empty string!
static const std::set<llama_token> g_known_empty_token_ids = {
    // Intentionally empty - no tokens confirmed to decode to truly empty strings
};

// Helper function to check if a token should be filtered out
// Returns true if the token is valid for TTS, false if it should be skipped
static bool is_valid_tts_token(llama_token tid) {
    // Skip special tokens
    for (llama_token sid : g_special_token_ids) {
        if (tid == sid) {
            return false;
        }
    }
    
    // Skip tokens >= 150000 (usually special tokens like <|im_end|>, <|tts_bos|>, etc.)
    if (tid >= 150000) {
        return false;
    }
    
    // Skip known empty tokens
    if (g_known_empty_token_ids.find(tid) != g_known_empty_token_ids.end()) {
        return false;
    }
    
    return true;
}

// Helper function to filter special tokens (matching Python file_loader.py logic)
static void filter_special_tokens(
    std::vector<llama_token>& token_ids,
    std::vector<float>& hidden_states,
    int n_embd
) {
    // Validate input: hidden_states should have size = token_ids.size() * n_embd
    if (hidden_states.size() != token_ids.size() * n_embd) {
        LOG_ERR("filter_special_tokens: hidden_states size (%zu) != token_ids.size() * n_embd (%zu * %d)\n",
                hidden_states.size(), token_ids.size(), n_embd);
        return;  // Don't filter if sizes don't match
    }
    
    size_t original_size = token_ids.size();
    
    // Filter out special tokens and tokens >= 150000
    // IMPORTANT: Keep token_ids and hidden_states aligned by filtering them together
    std::vector<llama_token> filtered_token_ids;
    std::vector<float> filtered_hidden_states;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        llama_token tid = token_ids[i];
        
        // Use the shared is_valid_tts_token function
        if (!is_valid_tts_token(tid)) continue;
        
        // Keep this token and its corresponding hidden state
        filtered_token_ids.push_back(tid);
        // Copy corresponding hidden state (n_embd floats per token)
        for (int j = 0; j < n_embd; ++j) {
            filtered_hidden_states.push_back(hidden_states[i * n_embd + j]);
        }
    }
    
    // Filter out leading empty tokens (already handled by is_valid_tts_token for known_empty_token_ids)
    size_t start_idx = 0;
    for (size_t i = 0; i < filtered_token_ids.size(); ++i) {
        if (g_known_empty_token_ids.find(filtered_token_ids[i]) == g_known_empty_token_ids.end()) {
            start_idx = i;
            break;
        }
    }
    
    // Apply start_idx filter to both token_ids and hidden_states
    if (start_idx > 0) {
        filtered_token_ids.erase(filtered_token_ids.begin(), filtered_token_ids.begin() + start_idx);
        filtered_hidden_states.erase(filtered_hidden_states.begin(), 
                                     filtered_hidden_states.begin() + start_idx * n_embd);
    }
    
    // Validate alignment after filtering
    if (filtered_hidden_states.size() != filtered_token_ids.size() * n_embd) {
        LOG_ERR("filter_special_tokens: alignment error after filtering! token_ids.size()=%zu, hidden_states.size()=%zu, n_embd=%d\n",
                filtered_token_ids.size(), filtered_hidden_states.size(), n_embd);
        return;  // Don't update if alignment is broken
    }
    
    // Update input vectors
    token_ids = filtered_token_ids;
    hidden_states = filtered_hidden_states;
    
    if (original_size != token_ids.size()) {
    }
}

// ==============================================================================
// 单工版本的 TTS Audio Token Generation
// 直接从 omni_sinplex.cpp 复制，保证单工模式行为完全一致
// ==============================================================================
static bool generate_audio_tokens_local_simplex(
    struct omni_context * ctx_omni,
    common_params * params,
    const std::vector<float>& merged_embeddings,
    int n_tokens,
    int tts_n_embd,
    int chunk_idx,
    std::vector<int32_t>& output_audio_tokens,
    const std::string& output_dir = "",
    bool is_final_text_chunk = false  // 🔧 [与 Python 对齐] 是否是最后一个 text chunk
) {
    print_with_timestamp("TTS Simplex: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d)\n",
                        chunk_idx, n_tokens, tts_n_embd);
    
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    // 🔧 [与 Python 对齐] Python: max_new_token=500，每个 LLM condition 生成直到 EOS
    // 然后每 25 个 tokens yield 一次
    const int max_audio_tokens = 500;
    
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Simplex: TTS model not loaded\n");
        return false;
    }
    
    if (!ctx_omni->head_code_weight || !ctx_omni->emb_code_weight) {
        LOG_ERR("TTS Simplex: TTS weights not loaded\n");
        return false;
    }
    
    if (merged_embeddings.size() != (size_t)(n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Simplex: merged_embeddings size mismatch: %zu != %d * %d\n",
                merged_embeddings.size(), n_tokens, tts_n_embd);
        return false;
    }
    
    // 🔧 [修复] 在 prefill 之前动态添加 audio_bos embedding
    // Python 中 audio_bos 是在 TTS 类内部（TTSStreamingGenerator.generate_with_buffer）添加的
    // 每个 chunk 都会加 audio_bos: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> condition_with_bos = merged_embeddings;  // 复制一份
    int extra_tokens = 0;  // 额外添加的 tokens 数量
    
    // 🔧 [修复 text_eos_embed 时机] text_eos_embed 不再在 condition 构建阶段添加
    // 原因：一个 text chunk 会生成多个 audio chunk，text_eos_embed 放在 condition 中
    // 会导致所有 audio chunk 都受到影响。正确做法是在第一轮 audio 生成结束（EOS）后，
    // 再注入 text_eos_embed + audio_bos 做第二轮生成，这样只影响最后一批 audio tokens。
    const int text_eos_token_id = 151692;  // TTS 的 text_eos_token_id
    
    // 🔧 [与 Python 对齐] 只添加 audio_bos（总是添加）
    // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), 
                                  audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Simplex: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                            chunk_idx, condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Simplex: failed to get audio_bos embedding\n");
    }
    int n_tokens_with_bos = n_tokens + extra_tokens;  // 包含 audio_bos
    
    // Save condition embeddings (包含 audio_bos)
    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd = tts_n_embd;
    ctx_omni->tts_condition_saved = true;
    
    int n_past_tts = 0;
    if (chunk_idx == 0) {
        // 第一个 chunk：清空 KV cache
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Simplex: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;
        ctx_omni->tts_all_generated_tokens.clear();
        ctx_omni->tts_token_buffer.clear();  // 🔧 [与 Python 对齐] 新轮对话清空 buffer
        print_with_timestamp("TTS Simplex: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Simplex: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }
    
    // 使用包含 audio_bos 的 condition 进行 prefill
    if (!prefill_with_emb_tts(ctx_omni, params, 
                              condition_with_bos.data(),
                              n_tokens_with_bos, params->n_batch, &n_past_tts)) {
        LOG_ERR("TTS Simplex: prefill_with_emb_tts failed\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: prefill completed, n_past_tts=%d\n", n_past_tts);
    
    // Create sampler - matching Python TTSStreamingGenerator
    // Create sampler - matching Python TTSStreamingGenerator
    // Python TTSSamplingParams 默认 temperature=0.8 (modeling_minicpmo.py line 75)
    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp = 0.8f;    // 🔧 [与 Python 对齐] TTSSamplingParams.temperature=0.8
    tts_sampling.top_p = 0.85f;  // 🔧 [与 Python 对齐] TTSSamplingParams.top_p=0.85   // 🔧 [与 Python streaming 对齐] top_p=0.8
    tts_sampling.top_k = 25;
    tts_sampling.penalty_repeat = 1.05f;
    tts_sampling.min_p = 0.01f;
    tts_sampling.penalty_last_n = 16;
    
    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (!tts_sampler) {
        LOG_ERR("TTS Simplex: failed to create sampler\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: sampler created\n");
    
    output_audio_tokens.clear();
    
    // 🔧 [与 Python 对齐] 统一使用 tts_token_buffer 管理，和 Python _token_buffer 一致
    // Python chunk_size=25，每凑够 25 个才 yield
    const int CHUNK_SIZE = 25;  // 与 Python TTSStreamingGenerator.chunk_size=25 对齐
    
    // 🔧 [与 Python 对齐] 不强制最小生成数量，让 TTS 自然决定何时结束
    const int min_new_tokens = 0;
    
    // 🔧 [修复 text_eos_embed 时机] 两阶段生成：
    // Phase 1: 不带 text_eos_embed，正常生成 audio tokens 直到 EOS
    // Phase 2 (仅 is_final_text_chunk): 注入 text_eos_embed + audio_bos，生成最后一批 audio tokens
    // 这样 text_eos_embed 只影响最后一批 audio tokens，而非整个 text chunk 的所有 audio
    bool need_phase2 = false;  // 是否需要第二阶段生成
    
    // ===== Phase 1: 正常生成（不带 text_eos_embed） =====
    for (int t = 0; t < max_audio_tokens; ++t) {
        // 🔧 [P0-立即打断] 检测 break_event，立即停止 TTS 生成
        if (ctx_omni->break_event.load()) {
            print_with_timestamp("TTS Simplex: break_event detected at step %d, stopping immediately\n", t);
            break;
        }
        
        // 🔧 [修复过早EOS] 如果还没达到 min_new_tokens，阻止 EOS 被采样
        bool force_no_eos = (t < min_new_tokens);
        
        // Phase 1 始终使用 is_final_text_chunk=false：EOS 不 prefill，保持 KV cache 干净
        llama_token sampled_token_abs = sample_tts_token_simplex(
            tts_sampler,
            ctx_omni,
            params,
            &n_past_tts,
            &ctx_omni->tts_all_generated_tokens,
            t,
            force_no_eos,
            false  // Phase 1: 不 prefill EOS，留给 Phase 2 处理
        );
        
        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Simplex: sample_tts_token failed at step %d\n", t);
            break;
        }
        
        int relative_idx = sampled_token_abs - audio_bos_token_id;
        if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
            LOG_ERR("TTS Simplex: invalid token ID %d at step %d\n", sampled_token_abs, t);
            break;
        }
        
        output_audio_tokens.push_back(relative_idx);
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
        
        // 🔧 [与 Python 对齐] EOS token 检测
        bool is_eos = (relative_idx == num_audio_tokens - 1);
        if (is_eos) {
            print_with_timestamp("TTS Simplex Phase1: EOS token at step %d\n", t + 1);
            output_audio_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
            // 如果是最后一个 text chunk，需要进入 Phase 2
            if (is_final_text_chunk) {
                need_phase2 = true;
                print_with_timestamp("TTS Simplex: is_final_text_chunk=true, will enter Phase 2 for text_eos_embed\n");
            }
        } else {
            // 🔧 [与 Python 对齐] 非 EOS token 加入 tts_token_buffer
            ctx_omni->tts_token_buffer.push_back(relative_idx);
        }
        
        // 🔧 [与 Python 对齐] 当 buffer >= chunk_size 时，yield 出 chunk_size 个
        while ((int)ctx_omni->tts_token_buffer.size() >= CHUNK_SIZE && ctx_omni->t2w_thread_info) {
            T2WOut *t2w_out = new T2WOut();
            t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                         ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
            t2w_out->is_final = false;
            t2w_out->round_idx = ctx_omni->simplex_round_idx;
            
            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();
            
            print_with_timestamp("TTS Simplex Phase1: yield %d tokens 到 T2W (step %d, buffer=%zu)\n", 
                                CHUNK_SIZE, t + 1, ctx_omni->tts_token_buffer.size());
            ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(), 
                                              ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
        }
        
        if (t < 5 || (t + 1) % 25 == 0) {
            print_with_timestamp("TTS Simplex Phase1: token %d/%d: rel_id=%d\n", t + 1, max_audio_tokens, relative_idx);
        }
        
        // 如果是 EOS，退出 Phase 1 循环
        if (is_eos) {
            break;
        }
    }
    
    // ===== Phase 2: 注入 text_eos_embed，生成最后一批 audio tokens =====
    // 仅在 is_final_text_chunk 且 Phase 1 正常结束（EOS）时执行
    if (need_phase2 && !ctx_omni->break_event.load()) {
        print_with_timestamp("TTS Simplex Phase2: injecting text_eos_embed + audio_bos at n_past=%d\n", n_past_tts);
        
        // 注入 text_eos_embed 到 TTS KV cache
        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        bool inject_ok = false;
        if (tts_emb_text(ctx_omni, text_eos_token_id, text_eos_embed.data(), tts_n_embd)) {
            if (prefill_with_emb_tts(ctx_omni, params, text_eos_embed.data(), 1, 1, &n_past_tts)) {
                print_with_timestamp("TTS Simplex Phase2: text_eos_embed injected, n_past=%d\n", n_past_tts);
                
                // 再注入 audio_bos，开始新一轮 audio 生成
                std::vector<float> audio_bos_embed2(tts_n_embd, 0.0f);
                if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed2.data(), tts_n_embd)) {
                    if (prefill_with_emb_tts(ctx_omni, params, audio_bos_embed2.data(), 1, 1, &n_past_tts)) {
                        print_with_timestamp("TTS Simplex Phase2: audio_bos injected, n_past=%d, starting final generation\n", n_past_tts);
                        inject_ok = true;
                    }
                }
            }
        }
        
        if (inject_ok) {
            // Phase 2 生成循环：text_eos_embed 已注入，生成最后的 audio tokens
            for (int t2 = 0; t2 < max_audio_tokens; ++t2) {
                if (ctx_omni->break_event.load()) {
                    print_with_timestamp("TTS Simplex Phase2: break_event at step %d\n", t2);
                    break;
                }
                
                // Phase 2 使用 is_final_text_chunk=true：EOS 会被 prefill
                llama_token sampled_token_abs = sample_tts_token_simplex(
                    tts_sampler,
                    ctx_omni,
                    params,
                    &n_past_tts,
                    &ctx_omni->tts_all_generated_tokens,
                    t2,
                    false,  // 不强制阻止 EOS
                    true    // Phase 2: is_final，EOS 会被 prefill
                );
                
                if (sampled_token_abs == 0) {
                    LOG_ERR("TTS Simplex Phase2: sample failed at step %d\n", t2);
                    break;
                }
                
                int relative_idx = sampled_token_abs - audio_bos_token_id;
                if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
                    LOG_ERR("TTS Simplex Phase2: invalid token %d at step %d\n", sampled_token_abs, t2);
                    break;
                }
                
                output_audio_tokens.push_back(relative_idx);
                ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
                
                bool is_eos = (relative_idx == num_audio_tokens - 1);
                if (is_eos) {
                    print_with_timestamp("TTS Simplex Phase2: EOS at step %d (final end)\n", t2 + 1);
                    output_audio_tokens.pop_back();
                    ctx_omni->tts_all_generated_tokens.pop_back();
                } else {
                    ctx_omni->tts_token_buffer.push_back(relative_idx);
                }
                
                // yield audio chunks
                while ((int)ctx_omni->tts_token_buffer.size() >= CHUNK_SIZE && ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                                 ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
                    t2w_out->is_final = false;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS Simplex Phase2: yield %d tokens 到 T2W\n", CHUNK_SIZE);
                    ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(), 
                                                      ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
                }
                
                if (is_eos) {
                    break;
                }
            }
        } else {
            LOG_ERR("TTS Simplex Phase2: failed to inject text_eos_embed/audio_bos\n");
        }
    }
    
    // 🔧 [与 Python 对齐] chunk 结束时，tts_token_buffer 中可能有剩余的 tokens (< CHUNK_SIZE)
    // 这些 tokens 保留在 buffer 中，等下一个 text chunk 继续累积
    // 只有 is_final_text_chunk 时才 flush 所有剩余
    print_with_timestamp("TTS Simplex: chunk 结束，tts_token_buffer 剩余 %zu tokens (保留等下一个 chunk)\n", 
                        ctx_omni->tts_token_buffer.size());
    
    // 🔧 [与 Python 对齐] 只有最后一个 text chunk 时才 flush 剩余的 buffer
    // Python: if finished.all() and text_finished: yield all remaining; buffer = []
    if (is_final_text_chunk && !ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                     ctx_omni->tts_token_buffer.end());
        t2w_out->is_final = false;  // 注意：这里不设 is_final=true，is_final 由 tts_thread_func 在 llm_finish 时发送
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
        
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
        
        print_with_timestamp("TTS Simplex: is_final_text_chunk=true, flush 剩余 %zu tokens 从 tts_token_buffer\n", 
                            ctx_omni->tts_token_buffer.size());
        ctx_omni->tts_token_buffer.clear();
    }
    
    // 🔧 [与 Python 对齐] 只有 duplex_mode 时才发送 is_chunk_end 信号
    // 单工模式下，T2W 依赖滑动窗口机制，中间 chunk 不需要 flush
    if (ctx_omni->duplex_mode && ctx_omni->t2w_thread_info) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.clear();  // 空的 token 列表
        t2w_out->is_final = false;  // Not final (turn not ended)
        t2w_out->is_chunk_end = true;  // 🔧 标记 chunk 结束，T2W 需要 flush buffer
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
        
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
        print_with_timestamp("TTS Simplex: 发送 is_chunk_end=true 信号\n");
    }
    
    common_sampler_free(tts_sampler);
    
    print_with_timestamp("TTS Simplex: generated %zu audio tokens for chunk %d\n",
                        output_audio_tokens.size(), chunk_idx);
    
    // 🔧 [诊断] 输出前几个和后几个 audio tokens，帮助定位重复问题
    if (output_audio_tokens.size() > 0) {
        std::string first_tokens = "";
        std::string last_tokens = "";
        int show_count = std::min((int)output_audio_tokens.size(), 5);
        for (int i = 0; i < show_count; i++) {
            first_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        for (int i = std::max(0, (int)output_audio_tokens.size() - 5); i < (int)output_audio_tokens.size(); i++) {
            last_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        print_with_timestamp("TTS Simplex: chunk %d first tokens: [%s], last tokens: [%s]\n",
                            chunk_idx, first_tokens.c_str(), last_tokens.c_str());
    }
    
    ctx_omni->tts_n_past_accumulated = n_past_tts;
    print_with_timestamp("TTS Simplex: updated tts_n_past_accumulated=%d, total_generated_tokens=%zu\n", 
                        ctx_omni->tts_n_past_accumulated, ctx_omni->tts_all_generated_tokens.size());
    
    // Save tokens to file if output_dir is specified
    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE* f = fopen(tokens_file.c_str(), "wb");
        if (f) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }
    
    return !output_audio_tokens.empty();
}

// ==============================================================================
// Local TTS Audio Token Generation
// Uses the verified TTS model to generate audio tokens from merged_embeddings
// ==============================================================================
static bool generate_audio_tokens_local(
    struct omni_context * ctx_omni,
    common_params * params,
    const std::vector<float>& merged_embeddings,
    int n_tokens,
    int tts_n_embd,
    int chunk_idx,
    std::vector<int32_t>& output_audio_tokens,
    bool is_end_of_turn = false,  // 🔧 [与 Python 对齐] 是否是轮次结束
    const std::string& output_dir = ""
) {
    print_with_timestamp("TTS Local: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d, emb_size=%zu)\n", 
                         chunk_idx, n_tokens, tts_n_embd, merged_embeddings.size());
    
    // 🔧 [安全检查] 验证输入参数
    // 🔧 [修复尾音问题] 当 is_end_of_turn=true 时，允许 n_tokens=0
    // 因为我们会添加 text_eos_embed 和 audio_bos，让 TTS 生成最后的 audio tokens
    if (n_tokens < 0) {
        LOG_ERR("TTS Local: invalid n_tokens=%d\n", n_tokens);
        return false;
    }
    if (n_tokens == 0 && !is_end_of_turn) {
        LOG_ERR("TTS Local: n_tokens=0 but is_end_of_turn=false, nothing to generate\n");
        return false;
    }
    if (n_tokens > 10000) {
        LOG_ERR("TTS Local: n_tokens=%d seems too large, likely data corruption\n", n_tokens);
        return false;
    }
    if (tts_n_embd <= 0 || tts_n_embd > 10000) {
        LOG_ERR("TTS Local: invalid tts_n_embd=%d\n", tts_n_embd);
        return false;
    }
    
    // 🔧 [DEBUG] 记录特殊情况
    if (n_tokens == 0 && is_end_of_turn) {
        print_with_timestamp("TTS Local: n_tokens=0 but is_end_of_turn=true, will add text_eos_embed and generate final tokens\n");
    }
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    // 🔧 [单双工适配] max_audio_tokens:
    // - 双工模式: 26 (与 Python 对齐，max_token_per_chunk = 25 + 1)
    // - 单工模式: 500 (允许更长的生成，靠 EOS 结束)
    const int max_audio_tokens = ctx_omni->duplex_mode ? 26 : 500;
    print_with_timestamp("TTS Local: mode=%s, max_audio_tokens=%d\n", 
                         ctx_omni->duplex_mode ? "duplex" : "simplex", max_audio_tokens);
    
    // Verify TTS model is loaded
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Local: TTS model not loaded\n");
        return false;
    }
    
    // Verify TTS weights are loaded
    if (!ctx_omni->head_code_weight || !ctx_omni->emb_code_weight) {
        LOG_ERR("TTS Local: TTS weights not loaded (head_code or emb_code)\n");
        return false;
    }
    
    // Verify merged_embeddings size
    if (merged_embeddings.size() != (size_t)(n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Local: merged_embeddings size mismatch: %zu != %d * %d\n",
                merged_embeddings.size(), n_tokens, tts_n_embd);
        return false;
    }
    
    // 🔧 [修复] 在 prefill 之前动态添加 text_eos_embed（如果是轮次结束）和 audio_bos embedding
    // Python TTSStreamingGenerator.generate_with_buffer 逻辑：
    //   if text_finished: condition = torch.cat([condition, self.text_eos_embed], dim=1)
    //   condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> condition_with_bos = merged_embeddings;  // 复制一份
    int extra_tokens = 0;
    
    // 🔧 [与 Python 对齐] 如果是轮次结束，先添加 text_eos_embed
    // Python: if text_finished: condition = torch.cat([condition, self.text_eos_embed], dim=1)
    const int text_eos_token_id = 151692;  // TTS 的 text_eos_token_id
    if (is_end_of_turn) {
        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        if (tts_emb_text(ctx_omni, text_eos_token_id, text_eos_embed.data(), tts_n_embd)) {
            condition_with_bos.insert(condition_with_bos.end(), 
                                      text_eos_embed.begin(), text_eos_embed.end());
            extra_tokens++;
            print_with_timestamp("TTS Local: is_end_of_turn=true, 添加 text_eos_embed (chunk_idx=%d, new_size=%zu)\n", 
                                chunk_idx, condition_with_bos.size() / tts_n_embd);
        } else {
            LOG_WRN("TTS Local: failed to get text_eos embedding\n");
        }
    }
    
    // 🔧 [与 Python 对齐] 然后添加 audio_bos（总是添加）
    // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), 
                                  audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Local: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                            chunk_idx, condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Local: failed to get audio_bos embedding\n");
    }
    int n_tokens_with_bos = n_tokens + extra_tokens;  // 包含 text_eos_embed（如果有）和 audio_bos
    
    // Save condition embeddings for re-forward in sample_tts_token (包含 audio_bos)
    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd = tts_n_embd;
    ctx_omni->tts_condition_saved = true;
    // Python TTSStreamingGenerator 逻辑：
    // - idx == 0：清空 KV cache，从头开始，并拼接 spk_emb
    // - idx > 0：保持 past_key_values，继续生成
    // 这样 TTS 模型可以保持上下文连续性，避免提前生成 EOS
    int n_past_tts = 0;
    if (chunk_idx == 0) {
        // 第一个 chunk：清空 KV cache 和累积状态
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Local: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;  // 重置累计 n_past
        ctx_omni->tts_all_generated_tokens.clear();
        if (!ctx_omni->duplex_mode) {
            ctx_omni->tts_token_buffer.clear();  // 🔧 [与 Python 对齐] 新轮对话清空 buffer（仅单工模式）
        }
        print_with_timestamp("TTS Local: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        // 后续 chunk：保持 KV cache，继续使用累计的 n_past
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Local: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }
    // 使用包含 audio_bos 的 condition 进行 prefill
    if (!prefill_with_emb_tts(ctx_omni, params, 
                              condition_with_bos.data(),
                              n_tokens_with_bos, params->n_batch, &n_past_tts)) {
        LOG_ERR("TTS Local: prefill_with_emb_tts failed\n");
        return false;
    }
    print_with_timestamp("TTS Local: prefill completed, n_past_tts=%d\n", n_past_tts);
    
    // 2. Create sampler for TTS with correct TTS sampling parameters
    // 🔧 TTS流式采样参数 - 与 Python TTSStreamingGenerator 对齐
    // Python TTSSamplingParams 默认 temperature=0.8 (modeling_minicpmo.py line 75)
    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp = 0.8f;              // 🔧 [与 Python 对齐] TTSSamplingParams.temperature=0.8
    tts_sampling.top_p = 0.85f;  // 🔧 [与 Python 对齐] TTSSamplingParams.top_p=0.85             // 🔧 [与 Python streaming 对齐] top_p=0.8
    tts_sampling.top_k = 25;               // top_k = 25
    tts_sampling.penalty_repeat = 1.05f;   // repetition_penalty = 1.05
    tts_sampling.min_p = 0.01f;            // min_p = 0.01
    tts_sampling.penalty_last_n = 16;      // past_window = 16 (与Python对齐)
    
    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (!tts_sampler) {
        LOG_ERR("TTS Local: failed to create sampler\n");
        return false;
    }
    
    // 3. Generate audio tokens with streaming to T2W queue
    output_audio_tokens.clear();
    // Python: self.all_generated_tokens 是类成员变量，跨 chunk 持续累积
    // 用于：1. 正确判断是否是整个生成过程的第一个 token（re-forward condition）
    
    // 🔧 [差异3修复] 当前 chunk 的 tokens，用于 repetition penalty
    // Python generate_chunk: input_ids_sliced = new_tokens[:, 0:t]  # 只用当前 chunk 内的 tokens
    std::vector<llama_token> chunk_generated_tokens;
    
    // 🔧 [单双工适配] min_new_tokens 逻辑
    // - 双工模式: 与 Python streaming_generate 对齐
    //   max_token_per_chunk = 25 + 1  # 26
    //   min_token_per_chunk = 25 + 1  # 26
    //   if end_of_turn: min_token_per_chunk = 0
    // - 单工模式: 设置最小 100 个 tokens，防止 TTS 过早生成 EOS
    //   10 个中文字约需要 100-150 个 audio tokens (每字 10-15 tokens)
    const int min_new_tokens = ctx_omni->duplex_mode ? 
        (is_end_of_turn ? 0 : 26) : 
        100;  // 🔧 单工模式：至少生成 100 个 tokens 防止过早 EOS
    
    // 🚀 流水线优化：Token2Wav需要28个tokens (25+3 lookahead)才能输出音频
    // 首批28个tokens后推送，后续每25个tokens推送一次
    const int STREAM_CHUNK_SIZE = 25;
    const int FIRST_CHUNK_SIZE = 28;  // 首批推送阈值，减少首响时间
    bool first_chunk_pushed = false;
    std::vector<int32_t> stream_buffer;
    
    for (int t = 0; t < max_audio_tokens; ++t) {
        // 🔧 [差异1修复] 计算 force_no_eos
        // Python generate_chunk: if force_no_stop or t < min_new_tokens: logits[:, eos_token] = -torch.inf
        // 如果还没达到 min_new_tokens，阻止 EOS 被采样（在采样前设置 EOS logit 为 -inf）
        bool force_no_eos = (t < min_new_tokens);
        
        // Sample next token
        // 🔧 [差异2&3修复] 传入：
        // - all_generated_tokens: 用于判断是否是整个生成过程的第一个 token
        // - chunk_generated_tokens: 用于 repetition penalty（只用当前 chunk 的 tokens）
        // - t: token_index_in_chunk，用于判断是否跳过 sampling processors
        // - force_no_eos: 是否阻止 EOS（用于 min_new_tokens 逻辑）
        // - is_end_of_turn: 是否是 final chunk（EOS 是否应加入 KV cache）
        llama_token sampled_token_abs = sample_tts_token(
            tts_sampler,
            ctx_omni,
            params,
            &n_past_tts,
            &ctx_omni->tts_all_generated_tokens,  // 用于判断 is_first_token_overall
            &chunk_generated_tokens,               // 🔧 [差异3修复] 用于 repetition penalty
            t,                                     // token_index_in_chunk
            force_no_eos,                          // 🔧 [差异1修复] 阻止 EOS
            is_end_of_turn                         // 🔧 [与 Python 对齐] EOS 是否加入 KV cache
        );
        
        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Local: sample_tts_token failed at step %d\n", t);
            break;
        }
        
        // Convert to relative index and check EOS
        int relative_idx = sampled_token_abs - audio_bos_token_id;
        if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
            LOG_ERR("TTS Local: invalid token ID %d (relative_idx: %d) at step %d\n",
                    sampled_token_abs, relative_idx, t);
            break;
        }
        
        // Store token (as absolute ID for internal use, will convert to relative for output)
        output_audio_tokens.push_back(relative_idx);  // Store relative ID for token2wav
        stream_buffer.push_back(relative_idx);  // Also add to stream buffer
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
        // 🔧 [差异3修复] 添加到当前 chunk 的列表
        chunk_generated_tokens.push_back(sampled_token_abs);
        
        // 🔧 [与 Python 对齐] EOS token 检测 - 必须在流式推送之前
        // Python: if next_id.eq(self.eos_token).any(): finished[:] = True; else: 添加到 buffer
        // 如果是 EOS，立即从 buffer 中移除，防止被推送到 T2W
        bool is_eos = (relative_idx == num_audio_tokens - 1);
        if (is_eos) {
            output_audio_tokens.pop_back();
            stream_buffer.pop_back();
            chunk_generated_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
            // 不 break，继续执行后续逻辑（包括可能的流式推送剩余 tokens）
        }
        
        // 🔧 [与 Python 流式双工对齐] 流式推送，但保留 lookahead
        // Python _generate_waveform_from_tokens 逻辑：
        //   while len(buffer) >= CHUNK_SIZE + pre_lookahead:  # 28
        //       stream(buffer[:28])
        //       buffer = buffer[CHUNK_SIZE:]  # 只移除 25 个，保留 3 个 lookahead
        // 
        // 关键：我们推送所有可用的 tokens，但 T2W 端负责保留 lookahead
        // 这里我们每 25 个 tokens 推送一次，让 T2W 端的滑动窗口正确处理
        // 🔧 [与 Python 对齐] is_end_of_turn 时不进行流式推送，让所有 tokens 在最后一起发送
        // 这样 T2W 才能一次性 flush 所有 buffer，输出完整音频
        int push_threshold = first_chunk_pushed ? STREAM_CHUNK_SIZE : FIRST_CHUNK_SIZE;
        if ((int)stream_buffer.size() >= push_threshold && ctx_omni->t2w_thread_info && !is_end_of_turn) {
            first_chunk_pushed = true;
            T2WOut *t2w_out = new T2WOut();
            t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
            t2w_out->is_final = false;
            t2w_out->is_chunk_end = false;  // 🔧 中间推送，不是 chunk 结束
            t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
            
            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();
            stream_buffer.clear();
        }
        
        if (t < 5 || (t + 1) % 25 == 0) {
        }
        
        // 如果是 EOS，在完成上述处理后退出循环
        if (is_eos) {
            break;
        }
    }
    // 问题：如果只推送 25 tokens，T2W buffer 从 3->28，处理后剩余 3 个 tokens
    //       这 3 个 tokens 是当前 chunk 的最后 3 个，要等到下一个 chunk 才被处理
    //       导致当前 chunk 的尾音被延迟，听起来"吞字"
    // 解决：在 chunk 结束时发送 is_chunk_end=true 信号，让 T2W flush 掉 buffer 中的剩余 tokens
    if (ctx_omni->t2w_thread_info) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
        // 🔧 [与 Python 对齐] is_end_of_turn 时传 is_final=true，让 T2W 立即 flush 所有 buffer
        // Python: token2wav.stream(..., last_chunk=is_last_chunk) 其中 is_last_chunk = end_of_turn
        t2w_out->is_final = is_end_of_turn;
        t2w_out->is_chunk_end = !is_end_of_turn;  // 非 turn 结束时才用 is_chunk_end
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
        
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
    }
    
    // Cleanup sampler
    common_sampler_free(tts_sampler);
    
    // 🔧 更新累计 n_past，用于下一个 chunk 的 KV cache 位置
    // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
    ctx_omni->tts_n_past_accumulated = n_past_tts;
    
    // Save tokens to file if output_dir is specified
    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE* f = fopen(tokens_file.c_str(), "wb");
        if (f) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }
    
    return !output_audio_tokens.empty();
}

// Helper function to play WAV file
static void play_wav_file(const std::string& wav_file_path) {
#ifndef _WIN32
    // Play audio asynchronously using fork() to avoid blocking TTS thread
    pid_t pid = fork();
    if (pid == 0) {
        #ifdef __APPLE__
            execl("/usr/bin/afplay", "afplay", wav_file_path.c_str(), (char*)NULL);
        #else
            execl("/usr/bin/aplay", "aplay", wav_file_path.c_str(), (char*)NULL);
        #endif
        _exit(1);
    } else if (pid > 0) {
        // Parent process: continue without waiting
    } else {
        std::string play_cmd;
        #ifdef __APPLE__
            play_cmd = "afplay \"" + wav_file_path + "\" &";
        #else
            play_cmd = "aplay \"" + wav_file_path + "\" &";
        #endif
        LOG_WRN("TTS: fork() failed, using system() fallback for audio playback\n");
        system(play_cmd.c_str());
    }
#endif
    // Windows: no-op (audio playback handled by frontend)
}


// Helper function to move old output directory to old_output/<id>/
static void move_old_output_to_archive() {
    const std::string base_output_dir = "./tools/omni/output";
    const std::string old_output_base_dir = "./old_output";
    
    // Helper function to check if directory exists and has content
    auto dir_has_content = [](const std::string& dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            return false;  // Directory doesn't exist
        }
        if (!(info.st_mode & S_IFDIR)) {
            return false;  // Not a directory
        }
        
        // Check if directory has any files/subdirectories
#ifdef _WIN32
        std::string cmd = "dir /b \"" + dir_path + "\" 2>NUL | findstr /r \".\" >NUL 2>&1";
#else
        std::string cmd = "test -n \"$(ls -A " + dir_path + " 2>/dev/null)\"";
#endif
        int ret = system(cmd.c_str());
        return (ret == 0);  // Returns 0 if directory has content
    };
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // Helper function to find next available ID in old_output directory
    auto get_next_output_id = [](const std::string& old_output_base) -> int {
        // Ensure old_output base directory exists
        cross_platform_mkdir_p(old_output_base);
        
        // Find maximum ID in old_output directory
        int max_id = -1;
#ifdef _WIN32
        std::string find_cmd = "dir /b \"" + old_output_base + "\" 2>NUL";
#else
        std::string find_cmd = "ls -1 " + old_output_base + " 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1";
#endif
        FILE* pipe = popen(find_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
#ifdef _WIN32
            // On Windows, read all entries and find the max numeric ID
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        int id = std::stoi(result);
                        if (id > max_id) max_id = id;
                    } catch (...) {}
                }
            }
#else
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                // Remove trailing newline
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        max_id = std::stoi(result);
                    } catch (...) {
                        max_id = -1;
                    }
                }
            }
#endif
            pclose(pipe);
        }
        
        return max_id + 1;  // Next ID is max_id + 1 (or 0 if no existing IDs)
    };
    
    // Check if output directory has content
    bool output_has_content = false;
    
    // Check if base output directory exists and has content
    if (dir_has_content(base_output_dir)) {
        output_has_content = true;
    } else {
        // Check subdirectories
        if (dir_has_content(base_output_dir + "/llm_debug") || 
            dir_has_content(base_output_dir + "/tts_txt") ||
            dir_has_content(base_output_dir + "/tts_wav")) {
            output_has_content = true;
        }
    }
    
    if (output_has_content) {
        int next_id = get_next_output_id(old_output_base_dir);
        std::string old_output_dir = old_output_base_dir + "/" + std::to_string(next_id);
        
        // Create old_output/<id> directory
        if (!create_dir(old_output_dir)) {
            LOG_ERR("Failed to create old_output directory: %s\n", old_output_dir.c_str());
        } else {
            // Move entire output directory contents to old_output/<id>/
            // Use find + xargs for more reliable moving
#ifdef _WIN32
            std::string move_cmd = "robocopy \"" + base_output_dir + "\" \"" + old_output_dir + "\" /E /MOVE >NUL 2>&1";
            system(move_cmd.c_str());
            // robocopy returns non-zero on success (exit codes < 8 are success)
            // Re-create the base output dir since robocopy /MOVE removes it
            cross_platform_mkdir_p(base_output_dir);
#else
            std::string move_cmd = "find " + base_output_dir + " -mindepth 1 -maxdepth 1 -exec mv {} " + old_output_dir + "/ \\; 2>/dev/null";
            int ret = system(move_cmd.c_str());
            if (ret == 0) {
            } else {
                // Fallback: try simple mv command
                std::string fallback_cmd = "sh -c 'cd " + base_output_dir + " && mv * " + old_output_dir + "/ 2>/dev/null || true'";
                ret = system(fallback_cmd.c_str());
                if (ret == 0) {
                } else {
                    LOG_WRN("Failed to move old output directory (may be empty or already moved)\n");
                }
            }
#endif
        }
    } else {
    }
}

// Helper function to merge all WAV files into a single file
static void merge_wav_files(const std::string& output_dir, int num_chunks) {
    if (num_chunks == 0) {
        LOG_WRN("TTS: no chunks to merge\n");
        return;
    }
    
    std::string merged_file = output_dir + "/tts_output_merged.wav";
    
    // Check all chunk files exist
    std::vector<std::string> chunk_files;
    for (int i = 0; i < num_chunks; ++i) {
        std::string chunk_file = output_dir + "/tts_output_chunk_" + std::to_string(i) + ".wav";
        struct stat st;
        if (stat(chunk_file.c_str(), &st) == 0 && st.st_size > 0) {
            chunk_files.push_back(chunk_file);
        } else {
            LOG_WRN("TTS: chunk file %s does not exist or is empty\n", chunk_file.c_str());
        }
    }
    
    if (chunk_files.empty()) {
        LOG_WRN("TTS: no valid WAV files to merge\n");
        return;
    }
    
    // Method 1: Use ffmpeg with concat demuxer (most reliable)
    std::string concat_list_file = output_dir + "/concat_list.txt";
    FILE* f_list = fopen(concat_list_file.c_str(), "w");
    if (f_list) {
        for (const auto& chunk_file : chunk_files) {
            fprintf(f_list, "file '%s'\n", chunk_file.c_str());
        }
        fclose(f_list);
        
        std::string ffmpeg_cmd = "ffmpeg -f concat -safe 0 -i \"" + concat_list_file + "\" -c copy \"" + merged_file + "\" -y -loglevel error 2>&1";
        int ret = system(ffmpeg_cmd.c_str());
        unlink(concat_list_file.c_str());  // Clean up temp file
        
        if (ret == 0) {
            struct stat st;
            if (stat(merged_file.c_str(), &st) == 0 && st.st_size > 0) {
                return;  // Success
            }
        }
    }
    
    // Method 2: Fallback to sox (if available)
    std::string sox_cmd = "sox";
    for (const auto& chunk_file : chunk_files) {
        sox_cmd += " \"" + chunk_file + "\"";
    }
    sox_cmd += " \"" + merged_file + "\"";
    int ret = system(sox_cmd.c_str());
    
    if (ret == 0) {
        struct stat st;
        if (stat(merged_file.c_str(), &st) == 0 && st.st_size > 0) {
        } else {
            LOG_WRN("TTS: merged file was not created or is empty (sox)\n");
        }
    } else {
        LOG_WRN("TTS: failed to merge WAV files (tried ffmpeg and sox). Please install ffmpeg or sox.\n");
    }
}

// ==============================================================================
// TTS Thread Function - Duplex Mode
// 双工模式专用的 TTS 线程函数
// 与单工版本的主要差异：
// 1. 不需要 simplex_round_idx 管理和 round_XXX 输出目录
// 2. TTS KV cache 跨 chunk 保持（由 is_end_of_turn 控制是否重置）
// 3. 使用 generate_audio_tokens_local（双工版本，max_audio_tokens=26）
// ==============================================================================
void tts_thread_func_duplex(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> all_audio_tokens;
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 双工模式：固定输出目录（不使用 round_XXX 子目录）
    // 🔧 [多实例支持] 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    const std::string tts_output_dir = base_output_dir + "/tts_txt";
    const std::string llm_debug_output_dir = base_output_dir + "/llm_debug";
    const std::string tts_wav_output_dir = base_output_dir + "/tts_wav";
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) {
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // 创建输出目录
    create_dir(tts_output_dir);
    create_dir(llm_debug_output_dir);
    create_dir(tts_wav_output_dir);
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int audio_eos_token_id = audio_bos_token_id + 6561;
    const int text_eos_token_id = 151692;
    const int num_audio_tokens = 6562;
    
    // 🔧 [诊断] 用于追踪所有 decode 调用
    int decode_call_idx = 0;
    
    // 🔧 [修复重复生成问题] 标志位：当前 turn 是否已经执行过 turn_eos flush
    bool turn_eos_flushed = false;

    print_with_timestamp("TTS thread (duplex mode) started\n");

    // Multi Round Persistent Loop
    while(tts_thread_running) {
        if (!tts_thread_running) {
            break;
        }
        
        // 🔧 [双工模式] 打断检测
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列
            {
                std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                auto& queue = ctx_omni->tts_thread_info->queue;
                while (!queue.empty()) {
                    LLMOut *llm_out = queue.front();
                    queue.pop();
                    delete llm_out;
                }
            }
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();
            // 清除 TTS KV cache
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
                print_with_timestamp("TTS Duplex: break_event - cleared TTS KV cache\n");
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        std::string llm_text = "";
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
        
        // 🔧 [修复双工缺字问题] 从 LLMOut 获取 is_end_of_turn 状态
        bool accumulated_is_end_of_turn = false;
            
        // Wait for queue
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
            auto& queue = ctx_omni->tts_thread_info->queue;
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!tts_thread_running) {
                break;
            }
            
            // 清空 current_chunk 数据
            current_chunk_token_ids.clear();
            current_chunk_hidden_states.clear();
            current_chunk_n_embd = 0;
            accumulated_is_end_of_turn = false;
            
            // 累积所有队列中的数据
            while (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                llm_finish |= llm_out->llm_finish;
                accumulated_is_end_of_turn |= llm_out->is_end_of_turn;
                
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text += llm_out->text;
                    debug_dir = llm_out->debug_dir;
                }
                
                // 累积数据
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids.insert(current_chunk_token_ids.end(), 
                                                   llm_out->token_ids.begin(), 
                                                   llm_out->token_ids.end());
                    current_chunk_hidden_states.insert(current_chunk_hidden_states.end(), 
                                                       llm_out->hidden_states.begin(), 
                                                       llm_out->hidden_states.end());
                    current_chunk_n_embd = llm_out->n_embd;
                }
                delete llm_out;
                queue.pop();
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            
            // 双工模式：如果有新数据，继续处理
            // 🔧 [关键诊断] 每次取出 LLMOut 后都打印状态
            // 🔧 [修复双工缺字问题] 使用 accumulated_is_end_of_turn 而非全局状态
            print_with_timestamp("TTS Duplex: after queue - speek_done=%d, llm_finish=%d, token_ids.size=%zu, is_end_of_turn=%d, llm_text.len=%zu\n",
                                ctx_omni->speek_done ? 1 : 0,
                                llm_finish ? 1 : 0,
                                current_chunk_token_ids.size(),
                                accumulated_is_end_of_turn ? 1 : 0,
                                llm_text.size());
            
            if (ctx_omni->speek_done && llm_finish) {
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    ctx_omni->speek_done = false;
                } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                    ctx_omni->speek_done = false;
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true, will call TTS to flush buffer\n");
                } else {
                    // LISTEN/CHUNK_EOS 且没有实际文本
                    decode_call_idx++;
                    llm_finish = false;
                    llm_text.clear();
                    
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    continue;
                }
            }
        }
            
        std::string& response = llm_text;
        // 处理不完整的 UTF-8 字节
        if (!incomplete_bytes.empty()) {
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            response = response.substr(0, response.size() - incomplete_len);
        }
        
        // Skip empty responses
        if (response.empty() && !llm_finish) {
            if (ctx_omni->speek_done) {
                llm_finish = false;
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // 保持状态
                } else {
                    // 轮次结束，重置 TTS 状态
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }
            }
            continue;
        }
            
        // Tokenize text input
        std::vector<llama_token> text_tokens = common_tokenize(ctx_omni->ctx_tts_llama, response, false, true);
        
        if (text_tokens.empty() && !llm_finish) {
            continue;
        }
        
        // Handle empty final chunk
        if (text_tokens.empty() && response.empty() && llm_finish) {
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            speek_cv.notify_all();
            
            if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                // LISTEN/CHUNK_EOS
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = false;
                    t2w_out->is_chunk_end = true;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                continue;
            } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                // turn 结束，继续调用 TTS flush buffer
                print_with_timestamp("TTS Duplex: empty final chunk but is_end_of_turn=true, will call TTS to flush buffer\n");
            } else {
                // 非双工模式
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = true;
                    t2w_out->is_chunk_end = false;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                if (mem && !ctx_omni->duplex_mode) {
                    llama_memory_seq_rm(mem, 0, 0, -1);
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                continue;
            }
        }
        
        // Check for LLM data
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        if (has_llm_data) {
            int current_chunk_idx = chunk_idx;
            
            // 收到有效 LLM 数据，重置 turn_eos_flushed 标志位
            if (turn_eos_flushed) {
                turn_eos_flushed = false;
            }
            
            print_with_timestamp("TTS Duplex: processing chunk_idx=%d, n_tokens=%zu, is_end_of_turn=%d\n",
                                chunk_idx, current_chunk_token_ids.size(), accumulated_is_end_of_turn ? 1 : 0);
            
            // 安全检查
            if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
                LOG_ERR("TTS Duplex: invalid current_chunk_n_embd=%d\n", current_chunk_n_embd);
                continue;
            }
            
            size_t expected_hidden_size = current_chunk_token_ids.size() * current_chunk_n_embd;
            if (current_chunk_hidden_states.size() != expected_hidden_size) {
                LOG_ERR("TTS Duplex: hidden_states size mismatch\n");
                continue;
            }
            
            // Filter special tokens
            int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
            std::vector<llama_token> filtered_token_ids = current_chunk_token_ids;
            std::vector<float> filtered_hidden_states = current_chunk_hidden_states;
            filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
            int n_tokens_filtered = (int)(filtered_hidden_states.size() / current_chunk_n_embd);
            
            if (n_tokens_filtered <= 0) {
                continue;
            }
            
            // Compute merged embeddings
            const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
            std::vector<float> merged_embeddings;
            bool merged_success = false;
            
            if (ctx_omni->emb_text_weight && ctx_omni->projector_semantic_linear1_weight) {
                // Step 1: emb_text
                std::vector<float> llm_embeds(n_tokens_filtered * tts_n_embd, 0.0f);
                bool emb_text_success = true;
                for (int i = 0; i < n_tokens_filtered; i++) {
                    if (!tts_emb_text(ctx_omni, filtered_token_ids[i], llm_embeds.data() + i * tts_n_embd, tts_n_embd)) {
                        emb_text_success = false;
                        break;
                    }
                }
                
                if (emb_text_success) {
                    // Step 2: projector_semantic
                    std::vector<float> projected_hidden(n_tokens_filtered * tts_n_embd, 0.0f);
                    bool projector_success = tts_projector_semantic(ctx_omni, filtered_hidden_states.data(),
                                                                     n_tokens_filtered, current_chunk_n_embd,
                                                                     projected_hidden.data(), tts_n_embd);
                    
                    if (projector_success) {
                        // Step 3: Normalize
                        normalize_l2_per_token(projected_hidden.data(), n_tokens_filtered, tts_n_embd);
                        
                        // Step 4: Merge
                        size_t merge_size = (size_t)n_tokens_filtered * tts_n_embd;
                        merged_embeddings.resize(merge_size);
                        for (size_t i = 0; i < merge_size; i++) {
                            merged_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                        }
                        
                        // Add audio_bos_embed
                        std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
                        if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
                            merged_embeddings.insert(merged_embeddings.end(), audio_bos_embed.begin(), audio_bos_embed.end());
                            n_tokens_filtered += 1;
                        }
                        
                        merged_success = true;
                    }
                }
            }
            
            // 🔧 [双工模式] 保存 LLM debug 数据（追加模式，统一放在 llm_debug 目录）
            // Save LLM debug data: text, token_ids, hidden_states, and merged embeddings
            {
                // 1. Save LLM text output (追加模式，只记录纯文本，不记录 special tokens)
                {
                    // 从 llm_text 中过滤掉 special tokens
                    std::string clean_text = llm_text;
                    // 移除所有 [[XXX]] 格式的 special token 标记
                    size_t pos = 0;
                    while ((pos = clean_text.find("[[")) != std::string::npos) {
                        size_t end_pos = clean_text.find("]]", pos);
                        if (end_pos != std::string::npos) {
                            clean_text.erase(pos, end_pos - pos + 2);
                        } else {
                            break;
                        }
                    }
                    // 移除多余的空格
                    while (clean_text.find("  ") != std::string::npos) {
                        size_t space_pos = clean_text.find("  ");
                        clean_text.erase(space_pos, 1);
                    }
                    // 去除首尾空格
                    while (!clean_text.empty() && clean_text[0] == ' ') clean_text.erase(0, 1);
                    while (!clean_text.empty() && clean_text.back() == ' ') clean_text.pop_back();
                    
                    // 只有非空文本才记录
                    if (!clean_text.empty()) {
                        std::string text_file = llm_debug_output_dir + "/llm_text.txt";
                        FILE *f_text = fopen(text_file.c_str(), "a");
                        if (f_text) {
                            fprintf(f_text, "%s\n", clean_text.c_str());
                            fclose(f_text);
                        }
                    }
                    decode_call_idx++;
                }
                
                // 2. Save LLM token IDs (追加模式)
                std::string token_ids_file = llm_debug_output_dir + "/llm_token_ids.txt";
                FILE *f_tokens = fopen(token_ids_file.c_str(), "a");
                if (f_tokens) {
                    fprintf(f_tokens, "[chunk_%d] ", current_chunk_idx);
                    for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                        fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                    }
                    fprintf(f_tokens, "\n");
                    fclose(f_tokens);
                }
                
                // 3. Save LLM hidden states (追加模式，文本格式)
                int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
                std::string hidden_txt_file = llm_debug_output_dir + "/llm_hidden_states.txt";
                FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "a");
                if (f_hidden_txt) {
                    fprintf(f_hidden_txt, "[chunk_%d] Hidden States (shape: [%d, %d]):\n", current_chunk_idx, n_tokens_orig, current_chunk_n_embd);
                    for (int i = 0; i < n_tokens_orig; ++i) {
                        fprintf(f_hidden_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                                current_chunk_hidden_states[i * current_chunk_n_embd + 0],
                                current_chunk_hidden_states[i * current_chunk_n_embd + 1],
                                current_chunk_hidden_states[i * current_chunk_n_embd + 2]);
                    }
                    fclose(f_hidden_txt);
                }
                
                // 4. Save merged embeddings (追加模式) if successfully computed
                if (merged_success && !merged_embeddings.empty()) {
                    std::string merged_txt_file = llm_debug_output_dir + "/merged_embeddings.txt";
                    FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "a");
                    if (f_merged_txt) {
                        fprintf(f_merged_txt, "[chunk_%d] Merged Embeddings (shape: [%d, %d]):\n", current_chunk_idx, n_tokens_filtered, tts_n_embd);
                        for (int i = 0; i < n_tokens_filtered; ++i) {
                            fprintf(f_merged_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                                    merged_embeddings[i * tts_n_embd + 0],
                                    merged_embeddings[i * tts_n_embd + 1],
                                    merged_embeddings[i * tts_n_embd + 2]);
                        }
                        fclose(f_merged_txt);
                    }
                }
            }
            
            // Generate audio tokens using duplex version
            // 🔧 [修复尾音问题] 当 is_end_of_turn=true 时，即使 merged_embeddings 为空，
            // 也要调用 TTS 生成（只添加 text_eos_embed），让 TTS flush 它的 buffer
            bool should_call_tts = (merged_success && !merged_embeddings.empty()) || 
                                   (accumulated_is_end_of_turn && ctx_omni->duplex_mode);
            
            if (should_call_tts) {
                std::vector<int32_t> audio_tokens_out;
                bool is_end_of_turn = accumulated_is_end_of_turn;
                
                if (merged_embeddings.empty() && is_end_of_turn) {
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true with empty embeddings, calling TTS to flush\n");
                    n_tokens_filtered = 0;
                }
                
                bool tts_gen_success = generate_audio_tokens_local(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx,
                                                audio_tokens_out, is_end_of_turn, tts_wav_output_dir);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                    
                    if (is_end_of_turn && ctx_omni->duplex_mode) {
                        turn_eos_flushed = true;
                    }
                }
            }
            
            ++chunk_idx;
            llm_text.clear();
            response.clear();
            
            // Handle final chunk
            if (llm_finish) {
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;
                speek_cv.notify_all();
                
                merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // LISTEN/CHUNK_EOS: 保持 TTS 状态
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    llm_finish = false;
                    tts_finish = false;
                } else {
                    // 真正的轮次结束
                    if (ctx_omni->t2w_thread_info && !turn_eos_flushed) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    // 清除 TTS KV cache
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    all_audio_tokens.clear();
                    llm_finish = false;
                    tts_finish = false;
                }
            }
            continue;
        } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn && llm_finish) {
            // turn 结束但没有新数据，调用 TTS flush buffer
            if (turn_eos_flushed) {
                print_with_timestamp("TTS Duplex: turn_eos already flushed, skipping TTS generation\n");
            } else {
                print_with_timestamp("TTS Duplex: no LLM data but is_end_of_turn=true, calling TTS to flush buffer\n");
                
                const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
                std::vector<float> empty_embeddings;
                std::vector<int32_t> audio_tokens_out;
                int n_tokens_for_tts = 0;
                int current_chunk_idx = chunk_idx;
                
                bool tts_gen_success = generate_audio_tokens_local(ctx_omni, params, empty_embeddings,
                                                n_tokens_for_tts, tts_n_embd, current_chunk_idx,
                                                audio_tokens_out, true, tts_wav_output_dir);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                } else {
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->is_chunk_end = false;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                }
                
                turn_eos_flushed = true;
            }
            
            // 重置 TTS 状态
            merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            llm_finish = false;
            tts_finish = false;
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            speek_cv.notify_all();
            
            llm_text.clear();
            response.clear();
            continue;
        } else {
            llm_text.clear();
            response.clear();
            continue;
        }
    }
    
    print_with_timestamp("TTS thread (duplex mode) stopped\n");
}

void tts_thread_func(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;  // TTS model's n_past counter
    std::vector<llama_token> audio_tokens;  // Collected audio tokens
    std::vector<llama_token> all_audio_tokens;  // All audio tokens collected across all chunks
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 🔧 [多实例支持] 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // Helper function to create directory
    struct stat info;
    auto create_dir = [](const std::string& dir_path) {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // 🔧 [单工模式] Helper function to get round-specific output directory
    // 单工模式下返回 output/round_XXX，双工模式下返回 output
    auto get_round_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            // 单工模式：使用 round_XXX 子目录
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d", base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            // 双工模式：直接使用 base_output_dir
            return base_output_dir;
        }
    };
    
    // 🔧 [单工模式] 动态目录路径（每轮开始时更新）
    std::string current_round_dir = get_round_output_dir();
    std::string tts_output_dir = current_round_dir + "/tts_txt";
    std::string llm_debug_output_dir = current_round_dir + "/llm_debug";
    std::string tts_wav_output_dir = current_round_dir + "/tts_wav";
    
    // 记录上一次创建目录的 round_idx，避免重复创建
    int last_created_round_idx = -1;
    
    // 🔧 [单工模式] Helper function to update output directories for current round
    auto update_output_dirs = [&]() {
        current_round_dir = get_round_output_dir();
        tts_output_dir = current_round_dir + "/tts_txt";
        llm_debug_output_dir = current_round_dir + "/llm_debug";
        tts_wav_output_dir = current_round_dir + "/tts_wav";
        
        // 只在新的 round 时创建目录
        if (ctx_omni->simplex_round_idx != last_created_round_idx || ctx_omni->duplex_mode) {
            create_dir(tts_output_dir);
            create_dir(llm_debug_output_dir);
            create_dir(tts_wav_output_dir);
            last_created_round_idx = ctx_omni->simplex_round_idx;
            
            if (!ctx_omni->duplex_mode) {
                print_with_timestamp("TTS: 创建单工模式输出目录: %s\n", current_round_dir.c_str());
            }
        }
    };
    
    // 初始创建目录
    update_output_dirs();
    
    // TTS model constants from config.json
    const int audio_bos_token_id = 151687;
    // Audio EOS token: relative index 6561, absolute ID = 151687 + 6561 = 158248
    const int audio_eos_token_id = audio_bos_token_id + 6561;  // 158248
    const int text_eos_token_id = 151692;  // 用于文本结束检测
    const int spk_emb_token_id = 21143;
    const int num_audio_tokens = 6562;
    const int max_audio_tokens = 1000;  // Maximum audio tokens to generate per text chunk
    
    // WAV timing log file (will be created in current round directory)
    auto create_wav_timing_file = [&]() {
        std::string wav_timing_file = tts_wav_output_dir + "/wav_timing.txt";
        FILE* f_timing = fopen(wav_timing_file.c_str(), "w");
        if (f_timing) {
            fprintf(f_timing, "# WAV file generation timing log\n");
            fprintf(f_timing, "# Format: chunk_index, elapsed_time_ms (since stream_decode start), file_size_bytes, request_duration_ms\n");
            fprintf(f_timing, "# Time 0 is when stream_decode() function starts\n");
            fclose(f_timing);
        }
    };
    create_wav_timing_file();

    print_with_timestamp("TTS thread started\n");

    // 🔧 [单工模式] 标志位：当前 break_event 是否已经递增过 round_idx
    // 防止在等待 T2W 线程清除 break_event 期间重复递增
    bool break_round_incremented = false;

    // Multi Round Persistent Loop
    while(tts_thread_running) {
        
        if (!tts_thread_running) {
            break;
        }
        
        // 🔧 [P0-打断检测] 检测 break_event 并清空队列、重置状态
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列（队列元素类型是 LLMOut*）
            {
                std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                auto& queue = ctx_omni->tts_thread_info->queue;
                while (!queue.empty()) {
                    LLMOut *llm_out = queue.front();
                    queue.pop();
                    delete llm_out;
                }
            }
            
            // 🔧 [单工模式] 打断时递增 round 索引（只递增一次！）
            if (!ctx_omni->duplex_mode && !break_round_incremented) {
                ctx_omni->simplex_round_idx++;
                // 🔧 [移除] wav_turn_base 的递增移到 T2W 线程的打断处理中
                // 原因：避免竞态条件，确保 T2W 先处理完当前轮次的所有数据再递增
                // ctx_omni->wav_turn_base += 1000;  // 已移到 T2W 线程
                break_round_incremented = true;  // 标记已递增，防止重复
                print_with_timestamp("TTS: 打断触发，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
            }
            
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();  // 🔧 [多轮对话修复] 清理不完整的 UTF-8 字节
            // 🔧 [多轮对话修复] 清理 TTS 累积状态，避免混淆
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            // 不清除 break_event，让 T2W 线程也能检测到
            continue;
        }
        
        // 🔧 [单工模式] break_event 被 T2W 线程清除后，重置标志位
        if (break_round_incremented && !ctx_omni->break_event.load()) {
            break_round_incremented = false;
        }

        std::string llm_text = "";
        // 保存当前chunk的token IDs和hidden states用于TTS条件生成
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
            
        // Always wait for queue if not finished, or if finished but need to reset state
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                
            auto& queue = ctx_omni->tts_thread_info->queue;
            // 🔧 [P0-打断检测] 在等待时也监听 break_event
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            // 检测到 break_event 时跳过当前处理
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!tts_thread_running) {
                break;
            }
            // 🔧 [关键修复] 每次只处理一个 chunk，不要一次性取出所有
            // 之前的问题：while (!queue.empty()) 会取出所有 chunk，导致：
            // - llm_text 累积了多个 chunk 的文本
            // - 但 token_ids/hidden_states 被覆盖，只保留最后一个 chunk
            // - TTS 用 10 token 的 condition 去生成 20 token 文本的语音 → 内容丢失！
            // 
            // Python 逻辑：每次只处理一个 chunk，生成对应的 audio tokens
            if (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                llm_finish |= llm_out->llm_finish;
                // 只取一个 chunk 的数据
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text = llm_out->text;  // 注意：= 而不是 +=
                    debug_dir = llm_out->debug_dir;
                }
                // 保存这一个 chunk 的 token IDs 和 hidden states
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids = llm_out->token_ids;
                    current_chunk_hidden_states = llm_out->hidden_states;
                    current_chunk_n_embd = llm_out->n_embd;
                    
                    // 🔧 [诊断日志] 打印 TTS 接收到的数据
                    std::string token_ids_str = "";
                    for (size_t i = 0; i < current_chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1 && i < 19) token_ids_str += " ";
                    }
                    if (current_chunk_token_ids.size() > 20) token_ids_str += "...";
                    
                    print_with_timestamp("TTS<-LLM: chunk_idx=%d, text='%s', n_tokens=%zu, hidden_size=%zu, token_ids=[%s]\n",
                                        chunk_idx,
                                        llm_text.c_str(),
                                        current_chunk_token_ids.size(),
                                        current_chunk_hidden_states.size(),
                                        token_ids_str.c_str());
                }
                delete llm_out;
                queue.pop();
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            
            // 🔧 [诊断] 打印取出数据后的关键状态
            print_with_timestamp("TTS: after queue pop - speek_done=%d, llm_finish=%d, llm_text.empty=%d, token_ids.size=%zu\n",
                                ctx_omni->speek_done, llm_finish, llm_text.empty(), current_chunk_token_ids.size());
            
            // If speek_done is true but we received llm_finish=true, handle state transition
            if (ctx_omni->speek_done && llm_finish) {
                print_with_timestamp("TTS: speek_done=true and llm_finish=true, resetting state for next round\n");
                // 🔧 [关键修复 - 与 Python 对齐] 在双工模式下，如果有新数据，应该继续处理
                // Python: 每次 streaming_generate 调用都会独立处理，不会因为之前的状态跳过
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    // 双工模式下有新数据，重置 speek_done 并继续处理
                    ctx_omni->speek_done = false;
                    // 不执行 continue，继续后续的 TTS 处理
                } else {
                    // 没有新数据，或者非双工模式：重置状态并跳过
                    llm_finish = false;
                    llm_text.clear();
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    // 🔧 [多轮对话修复] 清理 TTS 累积状态
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    continue;  // Skip processing and wait for next input
                }
            }
        }
            
        std::string& response = llm_text;
        // 拼接上一个循环保存的不完整字节
        if (!incomplete_bytes.empty()) {
            print_with_timestamp("TTS: prepending incomplete_bytes (len=%zu) to response (len=%zu)\n", 
                                incomplete_bytes.length(), response.length());
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        // 检查当前响应是否有不完整的 UTF-8 序列
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            print_with_timestamp("TTS: detected incomplete UTF-8 sequence at end: incomplete_len=%zu, response_len=%zu\n", 
                                incomplete_len, response.length());
            // 保存不完整的字节
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            // 仅处理完整的部分
            response = response.substr(0, response.size() - incomplete_len);
            print_with_timestamp("TTS: after truncation: response_len=%zu, incomplete_bytes_len=%zu\n", 
                                response.length(), incomplete_bytes.length());
        } else {
            // 确保没有残留的不完整字节
            incomplete_bytes.clear();
        }
        
        // 🔧 [诊断] 打印 response 状态
        print_with_timestamp("TTS: before empty check - response.empty=%d, response='%s', llm_finish=%d\n",
                            response.empty(), response.substr(0, 50).c_str(), llm_finish);
        
        // Skip empty responses (but allow processing if llm_finish is true to handle end of generation)
        if (response.empty() && !llm_finish) {
            // If speek_done is true but we're still getting empty responses, just continue waiting
            // Don't reset speek_done here - let stream_prefill reset it after it wakes up
            if (ctx_omni->speek_done) {
                print_with_timestamp("TTS: speek_done=true with empty response, keeping state (waiting for stream_prefill)\n");
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                // NOTE: 不重置 speek_done，让 stream_prefill 完成等待后重置
            }
            continue;
        }
        fflush(stdout);
            
        // 🔧 [修复] 空 response + llm_finish 的提前检查
        // 必须在 tokenize 之前检查，因为空字符串 tokenize 可能返回非空结果（BOS token）
        if (response.empty() && llm_finish) {
            // 🔧 [修复] 当收到 llm_finish=true 但没有新数据时，
            // 需要 flush tts_token_buffer 中剩余的 tokens，并发送 is_final=true 到 T2W
            print_with_timestamp("TTS: received llm_finish=true with no data, finalizing (tts_token_buffer=%zu)\n",
                                ctx_omni->tts_token_buffer.size());
            
            // 🔧 [修复] Flush tts_token_buffer 中剩余的 tokens 到 T2W
            if (ctx_omni->t2w_thread_info && !ctx_omni->tts_token_buffer.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                             ctx_omni->tts_token_buffer.end());
                t2w_out->is_final = false;  // 先发送剩余 tokens
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: flushed %zu remaining tokens from tts_token_buffer\n", 
                                    ctx_omni->tts_token_buffer.size());
                ctx_omni->tts_token_buffer.clear();
            }
            
            // 🔧 [修复] 发送 is_final=true 到 T2W，让 T2W 写入 generation_done.flag
            if (ctx_omni->t2w_thread_info) {
                // 🔧 保存当前 round_idx 用于 T2W（递增前的值）
                int current_round_idx = ctx_omni->simplex_round_idx;
                
                // 单工模式：递增 round_idx
                if (!ctx_omni->duplex_mode) {
                    ctx_omni->simplex_round_idx++;
                    print_with_timestamp("TTS: 单工模式轮次结束，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
                }
                
                T2WOut *t2w_final = new T2WOut();
                t2w_final->audio_tokens.clear();
                t2w_final->is_final = true;
                t2w_final->round_idx = current_round_idx;  // 🔧 使用递增前的值
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_final);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: sent is_final=true to T2W (llm_finish with no data)\n");
            }
            
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            speek_cv.notify_all();
            print_with_timestamp("TTS: finished processing all chunks (llm_finish with no data path)\n");
            // 非双工模式或真正的轮次结束：完全重置
            tts_finish = false;
            llm_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            // 🔧 [多轮对话修复] 清理 TTS 累积状态
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }
        
        // 2. 根据Python实现，将LLM输出转换为TTS条件（hidden_text_merge模式）
        // Python流程：
        //   1. 使用TTS的emb_text层（152064词表）处理LLM token: llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
        //   2. 使用projector_semantic将LLM hidden states（4096维）投影到TTS hidden_dim（768维）:
        //      hidden_embeds = self.tts.projector_semantic(output.last_hidden_states)
        //   3. 归一化: hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        //   4. 合并: tts_embeds = llm_embeds + hidden_embeds
        //   5. 添加audio_bos_token的embedding
        //   6. 将tts_embeds作为condition传递给TTS（使用batch.embd而不是token IDs）
        fflush(stdout);
        
        // 🔧 [诊断] 打印 current_chunk_n_embd 的值，确认数据是否正确传递
        print_with_timestamp("TTS: DEBUG before has_llm_data - token_ids.size=%zu, hidden_states.size=%zu, n_embd=%d\n",
                            current_chunk_token_ids.size(), current_chunk_hidden_states.size(), current_chunk_n_embd);
        
        // 使用从队列中获取的token IDs和hidden states
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        // 🔧 [移除] 之前这里有一段"等待 50ms 并累积更多消息"的逻辑
        // 这会导致多个 chunk 被合并，破坏了"每次只处理一个 chunk"的逻辑
        // Python 不会累积多个 chunk，而是每个 chunk 独立处理
        // llm_finish 标志会在下一次从队列取数据时正确捕获
        
        // 🔧 [诊断日志] 打印关键数据状态
        print_with_timestamp("TTS: has_llm_data=%d, token_ids=%zu, hidden_states=%zu, n_embd=%d, llm_finish=%d\n",
                             has_llm_data, current_chunk_token_ids.size(), 
                             current_chunk_hidden_states.size(), current_chunk_n_embd, llm_finish);
        
        // // 🔧 [调试] 打印传给 TTS 的 token IDs
        // if (has_llm_data) {
        //     for (size_t i = 0; i < current_chunk_token_ids.size(); i++) {
        //     }
            
        //     // 打印每个 token 对应的 hidden state 前3个值
        //     for (size_t i = 0; i < current_chunk_token_ids.size(); i++) {
        //         size_t offset = i * current_chunk_n_embd;
        //     }
        // }
        
        if (has_llm_data) {
            // 🔧 [单工模式] 在每轮第一个 chunk 时更新输出目录
            // 确保每轮数据保存在独立的 round_XXX 子目录下
            if (chunk_idx == 0 && !ctx_omni->duplex_mode) {
                update_output_dirs();
                create_wav_timing_file();
            }
            
            // Save current chunk_idx to ensure consistent directory naming
            // This is important because chunk_idx will be incremented after processing
            int current_chunk_idx = chunk_idx;
            
            // 🔧 [安全检查] 防止除零和异常值导致的崩溃
            if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
                LOG_ERR("TTS: invalid current_chunk_n_embd=%d, skipping chunk %d\n", 
                        current_chunk_n_embd, current_chunk_idx);
                continue;
            }
            
            // 🔧 [安全检查] 验证 hidden_states 大小合理性
            if (current_chunk_hidden_states.size() > 100000000) {  // 100M floats = ~400MB，作为上限
                LOG_ERR("TTS: hidden_states size too large (%zu), possible corruption, skipping chunk %d\n", 
                        current_chunk_hidden_states.size(), current_chunk_idx);
                continue;
            }
            
            // 🔧 [安全检查] 验证 token_ids 和 hidden_states 的对齐关系
            size_t expected_hidden_size = current_chunk_token_ids.size() * current_chunk_n_embd;
            if (current_chunk_hidden_states.size() != expected_hidden_size) {
                LOG_ERR("TTS: hidden_states size mismatch: got %zu, expected %zu (tokens=%zu * n_embd=%d), skipping chunk %d\n",
                        current_chunk_hidden_states.size(), expected_hidden_size, 
                        current_chunk_token_ids.size(), current_chunk_n_embd, current_chunk_idx);
                continue;
            }
            
            // CRITICAL FIX: Filter special tokens before computing merged_embeddings
            // This ensures merged_embeddings match Python's computation (which filters tokens)
            // Note: We keep original data for saving debug files, but use filtered data for merged_embeddings
            int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
            
            std::vector<llama_token> filtered_token_ids = current_chunk_token_ids;
            std::vector<float> filtered_hidden_states = current_chunk_hidden_states;
            filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
            
            int n_tokens_filtered = (int)(filtered_hidden_states.size() / current_chunk_n_embd);
            
            // 🔧 [诊断日志] 打印过滤前后的 token 数量
            print_with_timestamp("TTS: n_tokens_orig=%d, n_tokens_filtered=%d (filtered %d special tokens)\n",
                                 n_tokens_orig, n_tokens_filtered, n_tokens_orig - n_tokens_filtered);
            
            // 注意：C++ 的 hidden state 收集方式和 Python 不同：
            // - Python: forward(T_{i-1}) → 收集 H_{i-1} → 采样 T_i（需要延迟调整）
            // - C++: 采样 T_i → forward(T_i) → 收集 H_i（不需要延迟！）
            // 所以 C++ 的 T_i 直接对应 H_i = forward(T_i)，对齐已经正确
            
            
            // 🔧 [安全检查] 如果所有 token 都被过滤掉了，跳过处理
            if (n_tokens_filtered <= 0) {
                LOG_WRN("TTS: all tokens filtered out, skipping chunk %d\n", current_chunk_idx);
                continue;  // 跳过这个 chunk
            }
            
            // Compute merged embeddings (llm_embeds + projected_hidden) for saving
            // This matches Python implementation: tts_embeds = llm_embeds + hidden_embeds
            // IMPORTANT: Use filtered data to match Python's behavior
            const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
            std::vector<float> merged_embeddings;  // 合并后的embedding
            bool merged_success = false;
            
            // 🔧 [与 Python 对齐] is_final_text_chunk 需要在外层声明，供后面调用使用
            // 🔧 [修复] 当 llm_finish=true 时，这是最后一个 text chunk，需要设置 is_final_text_chunk=true
            // 这样 TTS 会 flush 剩余的 tts_token_buffer，并正确处理 text_eos_embed
            bool is_final_text_chunk = llm_finish;
            
            // Try to compute merged embeddings if weights are available
            if (ctx_omni->emb_text_weight && ctx_omni->projector_semantic_linear1_weight) {
                // Step 1: Convert token IDs to embeddings using emb_text (using filtered tokens)
                std::vector<float> llm_embeds(n_tokens_filtered * tts_n_embd, 0.0f);
                bool emb_text_success = true;
                
                for (int i = 0; i < n_tokens_filtered; i++) {
                    llama_token token_id = filtered_token_ids[i];
                    float * emb = llm_embeds.data() + i * tts_n_embd;
                    if (!tts_emb_text(ctx_omni, token_id, emb, tts_n_embd)) {
                        emb_text_success = false;
                        break;
                    }
                }
                
                if (emb_text_success) {
                    // Debug: Save llm_embeds for comparison
                    {
                        std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
                        create_dir(chunk_dir);
                        std::string llm_embeds_file = chunk_dir + "/llm_embeds_cpp.txt";
                        FILE *f_llm_embeds = fopen(llm_embeds_file.c_str(), "w");
                        if (f_llm_embeds) {
                            fprintf(f_llm_embeds, "LLM Embeddings from emb_text (C++ computed, shape: [%d, %d]):\n", n_tokens_filtered, tts_n_embd);
                            for (int i = 0; i < n_tokens_filtered; ++i) {
                                fprintf(f_llm_embeds, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_llm_embeds, "%.6f", llm_embeds[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_llm_embeds, " ");
                                }
                                fprintf(f_llm_embeds, "\n");
                            }
                            fclose(f_llm_embeds);
                        }
                    }
                    
                    // Step 2: Project hidden states using projector_semantic (using filtered hidden states)
                    std::vector<float> projected_hidden(n_tokens_filtered * tts_n_embd, 0.0f);
                    bool projector_success = tts_projector_semantic(ctx_omni,
                                                                     filtered_hidden_states.data(),
                                                                     n_tokens_filtered,
                                                                     current_chunk_n_embd,
                                                                     projected_hidden.data(),
                                                                     tts_n_embd);
                    
                    if (projector_success) {
                        // Debug: Save projected_hidden before normalization for comparison
                        {
                            std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
                            create_dir(chunk_dir);
                            std::string projected_file = chunk_dir + "/projected_hidden_before_norm_cpp.txt";
                            FILE *f_projected = fopen(projected_file.c_str(), "w");
                            if (f_projected) {
                                fprintf(f_projected, "Projected Hidden States BEFORE normalization (C++ computed, shape: [%d, %d]):\n", n_tokens_filtered, tts_n_embd);
                                for (int i = 0; i < n_tokens_filtered; ++i) {
                                    fprintf(f_projected, "Token %d: ", i);
                                    for (int j = 0; j < tts_n_embd; ++j) {
                                        fprintf(f_projected, "%.6f", projected_hidden[i * tts_n_embd + j]);
                                        if (j < tts_n_embd - 1) fprintf(f_projected, " ");
                                    }
                                    fprintf(f_projected, "\n");
                                }
                                fclose(f_projected);
                            }
                        }
                        
                        // Step 3: Normalize projected hidden states (CRITICAL: must normalize before merging)
                        // Check L2 norm before normalization for debugging
                        if (n_tokens_filtered > 0) {
                            // Check all tokens, not just first one
                            float avg_norm_before = 0.0f;
                            float min_norm_before = 1e10f;
                            float max_norm_before = 0.0f;
                            for (int t = 0; t < n_tokens_filtered; t++) {
                                float * vec = projected_hidden.data() + t * tts_n_embd;
                                float norm_sq = 0.0f;
                                for (int i = 0; i < tts_n_embd; i++) {
                                    norm_sq += vec[i] * vec[i];
                                }
                                float norm = std::sqrt(norm_sq);
                                avg_norm_before += norm;
                                if (norm < min_norm_before) min_norm_before = norm;
                                if (norm > max_norm_before) max_norm_before = norm;
                            }
                            avg_norm_before /= n_tokens_filtered;
                        }
                        
                        // CRITICAL: Normalize projected_hidden before merging
                        normalize_l2_per_token(projected_hidden.data(), n_tokens_filtered, tts_n_embd);
                        
                        // Debug: Save projected_hidden after normalization for comparison
                        {
                            std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
                            create_dir(chunk_dir);
                            std::string projected_file = chunk_dir + "/projected_hidden_after_norm_cpp.txt";
                            FILE *f_projected = fopen(projected_file.c_str(), "w");
                            if (f_projected) {
                                fprintf(f_projected, "Projected Hidden States AFTER normalization (C++ computed, shape: [%d, %d]):\n", n_tokens_filtered, tts_n_embd);
                                for (int i = 0; i < n_tokens_filtered; ++i) {
                                    fprintf(f_projected, "Token %d: ", i);
                                    for (int j = 0; j < tts_n_embd; ++j) {
                                        fprintf(f_projected, "%.6f", projected_hidden[i * tts_n_embd + j]);
                                        if (j < tts_n_embd - 1) fprintf(f_projected, " ");
                                    }
                                    fprintf(f_projected, "\n");
                                }
                                fclose(f_projected);
                            }
                        }
                        
                        // Verify normalization after normalization (check all tokens)
                        if (n_tokens_filtered > 0) {
                            float avg_norm_after = 0.0f;
                            float min_norm_after = 1e10f;
                            float max_norm_after = 0.0f;
                            int norm_error_count = 0;
                            for (int t = 0; t < n_tokens_filtered; t++) {
                                float * vec = projected_hidden.data() + t * tts_n_embd;
                                float norm_sq = 0.0f;
                                for (int i = 0; i < tts_n_embd; i++) {
                                    norm_sq += vec[i] * vec[i];
                                }
                                float norm = std::sqrt(norm_sq);
                                avg_norm_after += norm;
                                if (norm < min_norm_after) min_norm_after = norm;
                                if (norm > max_norm_after) max_norm_after = norm;
                                if (std::abs(norm - 1.0f) > 0.01f) {
                                    norm_error_count++;
                                    LOG_ERR("TTS: ERROR - token %d normalization failed: norm=%.6f (expected ~1.0)\n", t, norm);
                                }
                            }
                            avg_norm_after /= n_tokens_filtered;
                            if (norm_error_count > 0) {
                                LOG_ERR("TTS: ERROR - normalization failed for %d/%d tokens! Expected all norms to be ~1.0\n", 
                                        norm_error_count, n_tokens_filtered);
                            } else {
                            }
                        }
                        
                        // Step 4: Merge: merged_embeds = llm_embeds + projected_hidden
                        // CRITICAL: Use normalized projected_hidden for merging
                        // Verify projected_hidden is normalized before merging
                        if (n_tokens_filtered > 0) {
                            float verify_norm_check = 0.0f;
                            float * vec_check = projected_hidden.data() + 0 * tts_n_embd;
                            for (int i = 0; i < tts_n_embd; i++) {
                                verify_norm_check += vec_check[i] * vec_check[i];
                            }
                            float verify_norm_val = std::sqrt(verify_norm_check);
                            if (std::abs(verify_norm_val - 1.0f) > 0.01f) {
                                LOG_ERR("TTS: CRITICAL ERROR - projected_hidden is NOT normalized before merge! norm=%.6f\n", verify_norm_val);
                            }
                        }
                        
                        // 🔧 [安全检查] 防止创建过大的 vector 导致崩溃
                        size_t merge_size = (size_t)n_tokens_filtered * tts_n_embd;
                        if (n_tokens_filtered <= 0 || n_tokens_filtered > 10000 || 
                            tts_n_embd <= 0 || tts_n_embd > 10000 ||
                            merge_size > 100000000) {  // 100M elements max
                            LOG_ERR("TTS: invalid merge size: n_tokens_filtered=%d, tts_n_embd=%d, merge_size=%zu\n",
                                    n_tokens_filtered, tts_n_embd, merge_size);
                            break;  // 跳过这个 chunk，避免崩溃
                        }
                        
                        merged_embeddings.resize(merge_size);
                        for (size_t i = 0; i < merge_size; i++) {
                            merged_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                        }
                        
                        // 🔧 [修复] 不在 merge embed 阶段添加 audio_bos
                        // Python 中 audio_bos 是在 TTS 类内部（prefill 前）添加的
                        // 由 tts_thread_func 在 prefill 之前动态添加 audio_bos
                        // 这样可以确保 audio_bos 使用正确的 embedding 权重和位置
                        
                        merged_success = true;
                        
                        // Debug: Verify merged_embeddings calculation
                        if (n_tokens_filtered > 0) {
                            float * llm_emb_check = llm_embeds.data() + 0 * tts_n_embd;
                            float * proj_hidden_check = projected_hidden.data() + 0 * tts_n_embd;
                            float * merged_check = merged_embeddings.data() + 0 * tts_n_embd;
                        }
                    } else {
                        print_with_timestamp("TTS: WARNING - projector_semantic failed, skipping merged embedding save\n");
                    }
                }
            } else {
                print_with_timestamp("TTS: WARNING - TTS weights not loaded, skipping merged embedding computation\n");
            }
            
            // Save LLM debug data: text, token_ids, hidden_states, and merged embeddings
            {
                // Create chunk-specific directory
                std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
                create_dir(chunk_dir);
                
                // 1. Save LLM text output
                std::string text_file = chunk_dir + "/llm_text.txt";
                FILE *f_text = fopen(text_file.c_str(), "w");
                if (f_text) {
                    fprintf(f_text, "%s", response.c_str());
                    fclose(f_text);
                }
                
                // 2. Save LLM token IDs (original, before filtering)
                std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
                FILE *f_tokens = fopen(token_ids_file.c_str(), "w");
                if (f_tokens) {
                    for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                        fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                    }
                    fprintf(f_tokens, "\n");
                    fclose(f_tokens);
                }
                
                // 3. Save LLM hidden states (binary format, original before filtering)
                std::string hidden_file = chunk_dir + "/llm_hidden_states.bin";
                FILE *f_hidden = fopen(hidden_file.c_str(), "wb");
                if (f_hidden) {
                    // Write header: n_tokens, n_embd
                    int32_t header[2] = {n_tokens_orig, current_chunk_n_embd};
                    fwrite(header, sizeof(int32_t), 2, f_hidden);
                    // Write hidden states data
                    fwrite(current_chunk_hidden_states.data(), sizeof(float), current_chunk_hidden_states.size(), f_hidden);
                    fclose(f_hidden);
                }
                
                // 4. Save LLM hidden states as text (for easy inspection)
                std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
                FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "w");
                if (f_hidden_txt) {
                    fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", n_tokens_orig, current_chunk_n_embd);
                    for (int i = 0; i < n_tokens_orig; ++i) {
                        fprintf(f_hidden_txt, "Token %d: ", i);
                        for (int j = 0; j < current_chunk_n_embd; ++j) {
                            fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                            if (j < current_chunk_n_embd - 1) fprintf(f_hidden_txt, " ");
                        }
                        fprintf(f_hidden_txt, "\n");
                    }
                    fclose(f_hidden_txt);
                }
                
                // 5. Save merged embeddings (binary format) if successfully computed
                if (merged_success && !merged_embeddings.empty()) {
                    std::string merged_file = chunk_dir + "/merged_embeddings.bin";
                    FILE *f_merged = fopen(merged_file.c_str(), "wb");
                    if (f_merged) {
                        // Write header: n_tokens, tts_n_embd (using filtered token count)
                        int32_t header[2] = {n_tokens_filtered, tts_n_embd};
                        fwrite(header, sizeof(int32_t), 2, f_merged);
                        // Write merged embeddings data
                        fwrite(merged_embeddings.data(), sizeof(float), merged_embeddings.size(), f_merged);
                        fclose(f_merged);
                    }
                    
                    // 6. Save merged embeddings as text (for easy inspection)
                    std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
                    FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "w");
                    if (f_merged_txt) {
                        fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", n_tokens_filtered, tts_n_embd);
                        fprintf(f_merged_txt, "# Formula: merged_embeds = emb_text(filtered_token_ids) + normalize(projector_semantic(filtered_hidden_states))\n");
                        fprintf(f_merged_txt, "# Note: Special tokens have been filtered before computation (matching Python behavior)\n");
                        for (int i = 0; i < n_tokens_filtered; ++i) {
                            fprintf(f_merged_txt, "Token %d: ", i);
                            for (int j = 0; j < tts_n_embd; ++j) {
                                fprintf(f_merged_txt, "%.6f", merged_embeddings[i * tts_n_embd + j]);
                                if (j < tts_n_embd - 1) fprintf(f_merged_txt, " ");
                            }
                            fprintf(f_merged_txt, "\n");
                        }
                        fclose(f_merged_txt);
                        print_with_timestamp("TTS: saved merged embeddings (text) to %s\n", merged_txt_file.c_str());
                    }
                } else {
                    print_with_timestamp("TTS: skipped saving merged embeddings (computation failed or weights not available)\n");
                }
            }
            
            // Generate audio tokens using local TTS model (if merged_embeddings available)
            // or fall back to TTS server
            std::string wav_file_path;
            bool is_final_chunk = llm_finish;
            bool tts_success = false;
            
            // Use local TTS model if merged_embeddings were successfully computed
            if (merged_success && !merged_embeddings.empty()) {
                std::vector<int32_t> audio_tokens;
                
                // 🔧 [单双工适配] 根据 duplex_mode 选择不同的函数
                bool tts_gen_success = false;
                // 🔧 [与 Python 对齐] 传递 is_final_text_chunk，用于 flush buffer
                tts_gen_success = generate_audio_tokens_local_simplex(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx,
                                                audio_tokens, tts_wav_output_dir, is_final_text_chunk);
                if (tts_gen_success) {
                    tts_success = true;
                    
                    // Save audio tokens for external token2wav processing (backup)
                    // token2wav expects relative token IDs (0-6561)
                    std::string tokens_txt_file = tts_wav_output_dir + "/audio_tokens_chunk_" + 
                                                  std::to_string(current_chunk_idx) + ".txt";
                    FILE* f_tokens = fopen(tokens_txt_file.c_str(), "w");
                    if (f_tokens) {
                        for (size_t i = 0; i < audio_tokens.size(); ++i) {
                            fprintf(f_tokens, "%d", audio_tokens[i]);
                            if (i < audio_tokens.size() - 1) fprintf(f_tokens, ",");
                        }
                        fprintf(f_tokens, "\n");
                        fclose(f_tokens);
                    }
                    
                    // Accumulate all audio tokens for final WAV generation
                    all_audio_tokens.insert(all_audio_tokens.end(), 
                                           audio_tokens.begin(), audio_tokens.end());
                    
                    // 🔧 [与 Python 流式双工对齐] tokens 已在 generate_audio_tokens_local 内部流式推送
                    // T2W 端的 buffer 会持续累积，并按滑动窗口处理，保留 lookahead
                    // Python 逻辑：buffer 滑动 CHUNK_SIZE (25)，保留 pre_lookahead (3)
                    
                    // Also save audio tokens to file for the chunk (for debugging)
                    // This is separate from the token2wav sliding window
                } else {
                    LOG_ERR("TTS Local: failed for chunk %d\n", current_chunk_idx);
                }
            }
            
            // Always increment chunk_idx after processing
            // This ensures each chunk gets its own directory and data is not overwritten
            ++chunk_idx;
            
            // Clear processed text
            llm_text.clear();
            response.clear();
            
            // If this is the final chunk, mark as finished and merge all WAV files
            if (llm_finish) {
                // 🔧 [与 Python 对齐] LLM 完成时，flush 剩余的 tts_token_buffer
                // 因为 is_final_text_chunk 的判断可能不准确（队列时序问题）
                if (!ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
                    print_with_timestamp("TTS: llm_finish=true, flushing remaining %zu tokens from tts_token_buffer\n",
                                        ctx_omni->tts_token_buffer.size());
                    
                    T2WOut *t2w_out = new T2WOut();
                    if (t2w_out) {
                        t2w_out->audio_tokens.assign(
                            ctx_omni->tts_token_buffer.begin(),
                            ctx_omni->tts_token_buffer.end()
                        );
                        t2w_out->is_final = false;  // 还不是最后一个，后面还有 turn_end 的 is_final
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                        
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    ctx_omni->tts_token_buffer.clear();
                }
                
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
                speek_cv.notify_all();
                print_with_timestamp("TTS: finished processing all chunks\n");
                
                // Merge all WAV files into a single file
                merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                // Python: end_of_turn = last_id in turn_terminator_token_ids
                
                // 🔧 保存当前 round_idx 用于 T2W（递增前的值）
                int current_round_idx = ctx_omni->simplex_round_idx;
                
                // 🔧 [单工模式] 先递增 round 索引，再发送 is_final
                // 这样 T2W 线程在处理完 is_final 后能立即检测到新的 round_idx
                // 避免竞态条件导致下一轮数据写入旧目录
                if (!ctx_omni->duplex_mode) {
                    ctx_omni->simplex_round_idx++;
                    print_with_timestamp("TTS: 单工模式轮次结束，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
                }
                
                // 🔧 发送 is_final=true 到 T2W 队列，通知 T2W 重置 buffer
                // 注意：T2W 端只在双工模式下调用 Token2WavSession::reset()
                // 单工模式下只重置 token_buffer 为静音 tokens，不调用 reset()
                // 🔧 [修复最后一个字没说完] 移除 !all_audio_tokens.empty() 条件
                // 原因：is_final=true 必须发送，否则 T2W 不会 flush 最后的 buffer
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                    t2w_out->is_final = true;
                    t2w_out->round_idx = current_round_idx;  // 🔧 使用递增前的值
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS: sent is_final=true to T2W queue (turn end)\n");
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                // 🔧 [修复竞态条件] 不在这里重置 tts_condition_saved
                // 原因：新一轮的 generate_audio_tokens_local 会在 chunk_idx == 0 时重新设置
                // 如果在这里重置，可能与新一轮的 TTS 处理产生竞态条件
                // 旧版单工代码也没有在这里重置
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_condition_saved = false;
                }
                
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                all_audio_tokens.clear();
                llm_finish = false;
                tts_finish = false;
                // 🔧 [移除] wav_turn_base 的递增移到 T2W 线程的 is_final 处理中
                // 原因：避免竞态条件，确保 T2W 先处理完当前轮次的所有数据再递增
                // ctx_omni->wav_turn_base += 1000;  // 已移到 T2W 线程
            }
            
            continue;  // Skip the rest of the TTS processing
        } else {
            // Clear processed text and continue
            llm_text.clear();
            response.clear();
            continue;
        }
        
        // OLD TTS PROCESSING CODE BELOW (DISABLED)
        // All code below is disabled and replaced by TTS service calls above
        #if 0
        // 准备TTS输入：使用合并后的embedding（hidden_text_merge模式）
        const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        std::vector<float> tts_condition_embeddings;  // 合并后的embedding
        std::vector<llama_token> input_tokens;  // Fallback: 如果无法生成embedding，使用token IDs
        
        if (has_llm_data) {
            // 步骤1: 使用TTS的emb_text层处理LLM token IDs
            int n_tokens = (int)current_chunk_token_ids.size();
            std::vector<float> llm_embeds(n_tokens * tts_n_embd, 0.0f);
            bool emb_text_success = true;
            
            for (int i = 0; i < n_tokens; i++) {
                llama_token token_id = current_chunk_token_ids[i];
                float * emb = llm_embeds.data() + i * tts_n_embd;
                if (!tts_emb_text(ctx_omni, token_id, emb, tts_n_embd)) {
                    emb_text_success = false;
                    break;
                }
            }
            
            if (emb_text_success) {
                // 步骤2: 使用projector_semantic投影LLM hidden states
                // 验证hidden states的shape
                
                // 打印前几个hidden state的值用于调试
                if (current_chunk_hidden_states.size() >= 5) {
                }
                
                std::vector<float> projected_hidden(n_tokens * tts_n_embd, 0.0f);
                bool projector_success = tts_projector_semantic(ctx_omni,
                                                                 current_chunk_hidden_states.data(),
                                                                 n_tokens,
                                                                 current_chunk_n_embd,
                                                                 projected_hidden.data(),
                                                                 tts_n_embd);
                
                if (projector_success) {
                    // 打印投影后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤3: 归一化projected hidden states
                    normalize_l2_per_token(projected_hidden.data(), n_tokens, tts_n_embd);
                    
                    // 打印归一化后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤4: 合并: tts_embeds = llm_embeds + hidden_embeds
                    // 🔧 [安全检查] 防止创建过大的 vector 导致崩溃
                    size_t cond_size = (size_t)n_tokens * tts_n_embd;
                    if (n_tokens <= 0 || n_tokens > 10000 || 
                        tts_n_embd <= 0 || tts_n_embd > 10000 ||
                        cond_size > 100000000) {  // 100M elements max
                        LOG_ERR("TTS: invalid condition size: n_tokens=%d, tts_n_embd=%d, cond_size=%zu\n",
                                n_tokens, tts_n_embd, cond_size);
                        break;  // 跳过这个 chunk
                    }
                    tts_condition_embeddings.resize(cond_size);
                    for (size_t i = 0; i < cond_size; i++) {
                        tts_condition_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                    }
                    
                    // 打印合并后的前几个值
                    if (tts_condition_embeddings.size() >= 5) {
                    }
                    
                    // Save LLM debug data: text, hidden states, and merged embeddings
                    {
                        // Create chunk-specific directory
                        std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(chunk_idx);
                        create_dir(chunk_dir);
                        
                        // 1. Save LLM text output
                        std::string text_file = chunk_dir + "/llm_text.txt";
                        FILE *f_text = fopen(text_file.c_str(), "w");
                        if (f_text) {
                            fprintf(f_text, "%s", response.c_str());
                            fclose(f_text);
                        }
                        
                        // 2. Save LLM token IDs
                        std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
                        FILE *f_tokens = fopen(token_ids_file.c_str(), "w");
                        if (f_tokens) {
                            fprintf(f_tokens, "Token IDs (%zu):\n", current_chunk_token_ids.size());
                            for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                                fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                                if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                            }
                            fprintf(f_tokens, "\n");
                            fclose(f_tokens);
                        }
                        
                        // 3. Save LLM hidden states (binary format for precision)
                        std::string hidden_file = chunk_dir + "/llm_hidden_states.bin";
                        FILE *f_hidden = fopen(hidden_file.c_str(), "wb");
                        if (f_hidden) {
                            // Write header: n_tokens, n_embd
                            int32_t header[2] = {n_tokens, current_chunk_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_hidden);
                            // Write hidden states data
                            fwrite(current_chunk_hidden_states.data(), sizeof(float), current_chunk_hidden_states.size(), f_hidden);
                            fclose(f_hidden);
                        }
                        
                        // 4. Save LLM hidden states as text (for easy inspection)
                        std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
                        FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "w");
                        if (f_hidden_txt) {
                            fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", n_tokens, current_chunk_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_hidden_txt, "Token %d: ", i);
                                for (int j = 0; j < current_chunk_n_embd; ++j) {
                                    fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                                    if (j < current_chunk_n_embd - 1) fprintf(f_hidden_txt, " ");
                                }
                                fprintf(f_hidden_txt, "\n");
                            }
                            fclose(f_hidden_txt);
                        }
                        
                        // 5. Save merged embeddings (binary format)
                        std::string merged_file = chunk_dir + "/merged_embeddings.bin";
                        FILE *f_merged = fopen(merged_file.c_str(), "wb");
                        if (f_merged) {
                            // Write header: n_tokens, tts_n_embd
                            int32_t header[2] = {n_tokens, tts_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_merged);
                            // Write merged embeddings data
                            fwrite(tts_condition_embeddings.data(), sizeof(float), tts_condition_embeddings.size(), f_merged);
                            fclose(f_merged);
                        }
                        
                        // 6. Save merged embeddings as text (for easy inspection)
                        std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
                        FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "w");
                        if (f_merged_txt) {
                            fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_merged_txt, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_merged_txt, "%.6f", tts_condition_embeddings[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_merged_txt, " ");
                                }
                                fprintf(f_merged_txt, "\n");
                            }
                            fclose(f_merged_txt);
                        }
                        
                        // 7. Save intermediate results: llm_embeds and projected_hidden
                        std::string llm_embeds_file = chunk_dir + "/llm_embeds.txt";
                        FILE *f_llm_embeds = fopen(llm_embeds_file.c_str(), "w");
                        if (f_llm_embeds) {
                            fprintf(f_llm_embeds, "LLM Embeddings from emb_text (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_llm_embeds, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_llm_embeds, "%.6f", llm_embeds[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_llm_embeds, " ");
                                }
                                fprintf(f_llm_embeds, "\n");
                            }
                            fclose(f_llm_embeds);
                        }
                        
                        std::string projected_file = chunk_dir + "/projected_hidden.txt";
                        FILE *f_projected = fopen(projected_file.c_str(), "w");
                        if (f_projected) {
                            fprintf(f_projected, "Projected Hidden States (shape: [%d, %d], after normalization):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_projected, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_projected, "%.6f", projected_hidden[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_projected, " ");
                                }
                                fprintf(f_projected, "\n");
                            }
                            fclose(f_projected);
                        }
                    }
                    
                    // 🔧 [修复] 不在 merge embed 阶段添加 audio_bos
                    // audio_bos 将在 prefill 之前动态添加
                } else {
                    emb_text_success = false;
                }
            }
            
            if (!emb_text_success) {
                // Fallback: 使用token IDs
                input_tokens.insert(input_tokens.end(), current_chunk_token_ids.begin(), current_chunk_token_ids.end());
                input_tokens.push_back(audio_bos_token_id);
            }
        } else {
            // 没有LLM数据，使用text_tokens作为fallback
            input_tokens.insert(input_tokens.end(), text_tokens.begin(), text_tokens.end());
            input_tokens.push_back(audio_bos_token_id);
        }
        
        // 3. Reset TTS KV cache ONLY when starting a completely new round (speek_done was true and we're starting fresh)
        // IMPORTANT: Python's TTSStreamingGenerator maintains past_key_values across chunks:
        //   - First call: past_key_values = None (reset)
        //   - Subsequent chunks: past_key_values = self.past_key_values (continue)
        //   - Only reset on end_of_turn or new round
        // We should NOT clear KV cache for each new chunk from LLM if LLM is still generating
        // Only clear when speek_done was true (indicating a new round)
        fflush(stdout);
        
        // Only reset if we're starting a new round (speek_done was true, meaning previous round finished)
        // AND we have previous state (tts_n_past > 0)
        // This matches Python behavior: reset only on end_of_turn or new round
        static bool last_speek_done = false;
        bool should_reset = (last_speek_done && ctx_omni->speek_done == false && tts_n_past > 0);
        last_speek_done = ctx_omni->speek_done;
        
        if (should_reset) {
            fflush(stdout);
            // Clear KV cache to start fresh for new round
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            } else {
            }
            fflush(stdout);
            tts_n_past = 0;
        } else {
            fflush(stdout);
        }
        
        // 4. Evaluate input (使用embedding或token IDs)
        // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
        // Python: condition_length = current_condition.shape[1]
        // Python: pos_ids = torch.arange(self.text_start_pos, self.text_start_pos + condition_length)
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        if (!tts_condition_embeddings.empty()) {
            // 🔧 [修复] 在 prefill 之前动态添加 audio_bos embedding
            // Python 中 audio_bos 是在 TTS 类内部（TTSStreamingGenerator.generate_with_buffer）添加的
            // 每个 chunk 都会加 audio_bos: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
            std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
            if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
                // Append audio_bos to tts_condition_embeddings
                tts_condition_embeddings.insert(tts_condition_embeddings.end(), 
                                                audio_bos_embed.begin(), audio_bos_embed.end());
                print_with_timestamp("TTS: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                                    chunk_idx, tts_condition_embeddings.size() / tts_n_embd);
            } else {
                LOG_ERR("TTS: failed to get audio_bos embedding for prefill\n");
            }
            
            // 使用合并后的embedding作为输入（正确的实现方式）
            int condition_length = (int)(tts_condition_embeddings.size() / tts_n_embd);
            fflush(stdout);
            
            // 使用prefill_emb_with_hidden类似的逻辑，但针对TTS模型
            // Python: prefill时，position_ids从text_start_pos开始，到text_start_pos + condition_length
            // C++: batch.pos[j] = tts_n_past + j，其中tts_n_past对应text_start_pos
            int n_pos = condition_length;
            int text_start_pos_before = tts_n_past;  // 保存prefill前的text_start_pos
            if (!prefill_with_emb_tts(ctx_omni, params, tts_condition_embeddings.data(), n_pos, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input embeddings\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
                int condition_length_processed = tts_n_past - text_start_pos_before;
        } else {
            // Fallback: 使用token IDs作为输入
            fflush(stdout);
            fflush(stdout);
            if (!eval_tokens_tts(ctx_omni, params, input_tokens, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input tokens\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
        }
        
        // 4. Generate audio tokens using TTS model
        // Following Python flow: decode 1 token at a time, accumulate to buffer, yield when buffer reaches 25 tokens
        audio_tokens.clear();
        bool audio_gen_finish = false;
        int audio_token_count = 0;
        
        // Limit audio tokens per chunk: 25 tokens per chunk (each chunk corresponds to ~10 LLM tokens)
        // 每一轮的限制：LLM生成10个token → TTS生成25个token
        // 接下来LLM再生成10个token → TTS再生成25个token，以此类推
        int text_token_count = (int)text_tokens.size();
        // Fixed at 25 tokens per chunk (matching Python's audio_token_chunk_size=25)
        int max_audio_tokens_for_text = 25;
        
        // Buffer for accumulating audio tokens (yield when reaching 25 tokens, matching Python chunk_size=25)
        const int audio_token_chunk_size = 25;  // Yield size matching Python
        std::vector<llama_token> audio_token_buffer;
        
        // 🔧 [差异3修复] 当前 chunk 的 tokens，用于 repetition penalty
        std::vector<llama_token> chunk_generated_tokens_tts;
        
        // Generate audio tokens one by one (matching Python decode flow)
        while (audio_token_count < max_audio_tokens_for_text && !audio_gen_finish && !ctx_omni->speek_done) {
            fflush(stdout);
            
            // Decode 1 audio token at a time (matching Python: each decode generates 1 token)
            // 🔧 [差异2&3修复] 传入：
            // - all_audio_tokens: 用于判断 is_first_token_overall
            // - chunk_generated_tokens_tts: 用于 repetition penalty（只用当前 chunk 的 tokens）
            // - audio_token_count: token_index_in_chunk，用于判断 is_chunk_first_token
            // - force_no_eos: false（这个路径已有 25 tokens 的限制，不需要额外阻止 EOS）
            llama_token audio_token = sample_tts_token_simplex(ctx_omni->ctx_tts_sampler, ctx_omni, params, &tts_n_past, &all_audio_tokens, &chunk_generated_tokens_tts, audio_token_count, false);
            fflush(stdout);
            
            audio_token_buffer.push_back(audio_token);
            audio_tokens.push_back(audio_token);
            all_audio_tokens.push_back(audio_token);
            chunk_generated_tokens_tts.push_back(audio_token);  // 🔧 [差异3修复]
            audio_token_count++;
            
            // Check for end of audio generation
            bool is_audio = is_audio_token(audio_token, audio_bos_token_id, num_audio_tokens);
            int relative_idx = audio_token - audio_bos_token_id;
            fflush(stdout);
            
            // Check for audio EOS token (relative index 6561, absolute 158248)
            // Note: EOS token should only end audio generation for current chunk, not set speek_done
            // speek_done should only be set when llm_finish is true
            if (audio_token == audio_eos_token_id || relative_idx == 6561) {
                fflush(stdout);
                audio_gen_finish = true;
                // Only set tts_finish if LLM has also finished, otherwise continue to next chunk
                if (llm_finish) {
                    tts_finish = true;
                }
                break;
            }
            
            // Yield when buffer reaches chunk_size (25 tokens), matching Python yield logic
            if (audio_token_buffer.size() >= audio_token_chunk_size) {
                fflush(stdout);
                
                // Push audio tokens to T2W queue for token2wav processing
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens = audio_token_buffer;  // Copy the 25 tokens
                    t2w_out->is_final = false;  // Not final, more chunks may come
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                    
                    {
                        std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        // Check if queue has space
                        while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                            // Wait for space in queue (with timeout to avoid deadlock)
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            lock.lock();
                        }
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                
                // Yield this chunk (create unit_buffer and save)
                // Note: In Python, each yield creates a chunk, but here we create one buffer per 25-token chunk
                struct unit_buffer *res = new unit_buffer();
                res->text = response;
                res->unit_n_past = ctx_omni->n_past;
                res->completed = false;  // Not completed yet, more tokens may come
                
                // TODO: Convert audio tokens to audio waveform
                // For now, create a placeholder buffer
                res->buffer.resize(audio_token_chunk_size * 100);  // Placeholder size
                
                if (!res->buffer.empty()) {
                    // 读写使用 mutex 保护
                    std::unique_lock<std::mutex> lock(buffer_mutex);
                    ctx_omni->omni_output->output.push_back(res);
                    ctx_omni->omni_output->idx += 1;
                } else {
                    delete res;
                }
                
                // Clear buffer after yield, matching Python behavior
                audio_token_buffer.clear();
            }
        }
        
        // If we exited the loop because we reached max_audio_tokens_for_text (25 tokens per chunk),
        // mark audio_gen_finish as true to indicate this chunk is complete
        // This allows the next iteration to process the next chunk from LLM (if LLM is still generating)
        if (!audio_gen_finish && audio_token_count >= max_audio_tokens_for_text && !ctx_omni->speek_done) {
            fflush(stdout);
            audio_gen_finish = true;
        }
        fflush(stdout);
        
        // Save audio tokens to file for debugging/analysis
        // Note: Save relative indices (0-6561) for token2wav compatibility
        if (!audio_tokens.empty()) {
            // Always save to fixed output directory
            const int audio_bos_token_id = 151687;
            std::string token_file = tts_output_dir + "/tts_audio_tokens_chunk_" + std::to_string(chunk_idx) + ".txt";
            FILE *f = fopen(token_file.c_str(), "w");
            if (f) {
                fprintf(f, "Text: %s\n", response.c_str());
                fprintf(f, "Audio tokens (%zu) [relative_index]:\n", audio_tokens.size());
                for (size_t i = 0; i < audio_tokens.size(); ++i) {
                    int absolute_id = audio_tokens[i];
                    int relative_idx = absolute_id - audio_bos_token_id;
                    // Verify token is in valid range
                    if (absolute_id < audio_bos_token_id || relative_idx >= 6562) {
                        LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                        // Still save relative index, but mark as invalid
                        relative_idx = -1;  // Mark as invalid
                    }
                    // Save relative index only (for token2wav compatibility)
                    fprintf(f, "%d", relative_idx);
                    if (i < audio_tokens.size() - 1) fprintf(f, ",");
                }
                fprintf(f, "\n");
                fclose(f);
            } else {
                LOG_ERR("Failed to open file for writing: %s\n", token_file.c_str());
            }
        }
        
        // Create final unit_buffer for remaining tokens in buffer (if any)
        if (!audio_token_buffer.empty()) {
            // Push remaining tokens to T2W queue (final chunk)
            if (ctx_omni->t2w_thread_info) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens = audio_token_buffer;  // Copy remaining tokens
                t2w_out->is_final = (audio_gen_finish || llm_finish);  // Mark as final if generation finished
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                
                {
                    std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    // Check if queue has space
                    while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        lock.lock();
                    }
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            struct unit_buffer *res = new unit_buffer();
            res->text = response;
            res->unit_n_past = ctx_omni->n_past;
            res->completed = (audio_gen_finish || llm_finish);
            
            // TODO: Convert audio tokens to audio waveform
            res->buffer.resize(audio_token_buffer.size() * 100);  // Placeholder size
            
            if (!res->buffer.empty()) {
                // 读写使用 mutex 保护
                std::unique_lock<std::mutex> lock(buffer_mutex);
                ctx_omni->omni_output->output.push_back(res);
                ctx_omni->omni_output->idx += 1;
            } else {
                delete res;
            }
        }
        
        // Clear processed text after generating audio tokens for this chunk
        // This ensures we wait for new text in the next iteration instead of reprocessing the same text
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        // C++: tts_n_past is already updated during prefill (condition_length) and decode (len(chunk_generated_tokens))
        // So tts_n_past already equals text_start_pos after this chunk
        int condition_length_final = (int)(tts_condition_embeddings.empty() ? input_tokens.size() : (tts_condition_embeddings.size() / tts_n_embd));
        int chunk_generated_tokens_count = (int)audio_tokens.size();
        fflush(stdout);
        
        llm_text.clear();
        response.clear();
        ++chunk_idx;
        
        if (ctx_omni->speek_done) {
            // 可能是外部强行打断了，清除内部所有状态
            printf("speek done setted by outside, clear all internal states\n");
            tts_finish = true;
            // Reset TTS model state by clearing KV cache
            // Note: This might need llama_kv_cache_clear or similar function
            tts_n_past = 0;
        }
        
        // Only finish and set speek_done when BOTH LLM and TTS have finished
        // If only audio_gen_finish is true but llm_finish is false, continue to next chunk
        if (llm_finish && (tts_finish || audio_gen_finish)) {
            // Save all collected audio tokens to a summary file
            if (!all_audio_tokens.empty()) {
                // Always save to fixed output directory
                // Note: Save relative indices (0-6561) for token2wav compatibility
                const int audio_bos_token_id = 151687;
                std::string summary_file = tts_output_dir + "/tts_all_audio_tokens_summary.txt";
                FILE *f = fopen(summary_file.c_str(), "w");
                if (f) {
                    fprintf(f, "Total audio tokens: %zu\n", all_audio_tokens.size());
                    fprintf(f, "Audio token range: relative [0, %d) (absolute [%d, %d))\n", 
                            6562, audio_bos_token_id, audio_bos_token_id + 6562);
                    fprintf(f, "Audio tokens [relative_index]:\n");
                    int invalid_count = 0;
                    for (size_t i = 0; i < all_audio_tokens.size(); ++i) {
                        int absolute_id = all_audio_tokens[i];
                        int relative_idx = absolute_id - audio_bos_token_id;
                        // Verify token is in valid range
                        if (absolute_id < audio_bos_token_id || relative_idx >= 6562 || relative_idx < 0) {
                            LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                    absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                            relative_idx = -1;  // Mark as invalid
                            invalid_count++;
                        }
                        // Save relative index only (for token2wav compatibility)
                        fprintf(f, "%d", relative_idx);
                        if (i < all_audio_tokens.size() - 1) {
                            if ((i + 1) % 20 == 0) fprintf(f, "\n");
                            else fprintf(f, " ");
                        }
                    }
                    fprintf(f, "\n");
                    if (invalid_count > 0) {
                        fprintf(f, "WARNING: Found %d tokens outside valid range [%d, %d)\n", 
                                invalid_count, audio_bos_token_id, audio_bos_token_id + 6562);
                    }
                    fclose(f);
                    fflush(stdout);
                } else {
                    LOG_ERR("Failed to open summary file for writing: %s\n", summary_file.c_str());
                }
            }
            
            // 🔧 [与 Python 流式双工对齐] 发送 is_final=true 到 T2W 队列，flush 剩余 buffer
            // Python: is_last_chunk=True 时会 flush 所有剩余 tokens
            // T2W 端 buffer 已经累积了所有 tokens，这里只是通知它 flush
            if (ctx_omni->t2w_thread_info && !all_audio_tokens.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                t2w_out->is_final = true;
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 🔧 传递轮次索引
                
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            // 旧代码（已禁用，改用T2W线程处理）
            /*
            if (ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                // ... 旧的同步处理代码 ...
            }
            */
            
            // 兼容旧代码的变量定义（避免编译错误）
            bool _unused_final_wav_placeholder = false;
            if (_unused_final_wav_placeholder && ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                std::vector<float> final_wav;
                if (ctx_omni->token2wav_session->feed_window(ctx_omni->token2wav_buffer, true, final_wav)) {
                    if (!final_wav.empty()) {
                        const int sample_rate = omni::flow::Token2Wav::kSampleRate;
                        std::string wav_path = tts_wav_output_dir + "/wav_" + 
                                               std::to_string(ctx_omni->token2wav_wav_idx) + ".wav";
                        
                        // Convert float to int16 and write WAV
                        const int16_t num_channels = 1;
                        const int16_t bits_per_sample = 16;
                        const int16_t block_align = num_channels * (bits_per_sample / 8);
                        const int32_t byte_rate = sample_rate * block_align;
                        
                        std::vector<int16_t> pcm(final_wav.size());
                        for (size_t i = 0; i < final_wav.size(); ++i) {
                            float x = final_wav[i];
                            if (!std::isfinite(x)) x = 0.0f;
                            x = std::max(-1.0f, std::min(1.0f, x));
                            float y = x * 32767.0f;
                            if (y >= 32767.0f) pcm[i] = 32767;
                            else if (y <= -32768.0f) pcm[i] = -32768;
                            else pcm[i] = (int16_t)y;
                        }
                        
                        uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
                        uint32_t riff_size = 36u + data_bytes;
                        
                        FILE* f_wav = fopen(wav_path.c_str(), "wb");
                        if (f_wav) {
                            fwrite("RIFF", 1, 4, f_wav);
                            fwrite(&riff_size, 4, 1, f_wav);
                            fwrite("WAVE", 1, 4, f_wav);
                            fwrite("fmt ", 1, 4, f_wav);
                            uint32_t fmt_size = 16;
                            uint16_t audio_format = 1;
                            fwrite(&fmt_size, 4, 1, f_wav);
                            fwrite(&audio_format, 2, 1, f_wav);
                            fwrite(&num_channels, 2, 1, f_wav);
                            fwrite(&sample_rate, 4, 1, f_wav);
                            fwrite(&byte_rate, 4, 1, f_wav);
                            fwrite(&block_align, 2, 1, f_wav);
                            fwrite(&bits_per_sample, 2, 1, f_wav);
                            fwrite("data", 1, 4, f_wav);
                            fwrite(&data_bytes, 4, 1, f_wav);
                            fwrite(pcm.data(), 1, data_bytes, f_wav);
                            fclose(f_wav);
                        }
                        ctx_omni->token2wav_wav_idx++;
                    }
                }
                
                // Reset buffer for next round (re-initialize with 3 silence tokens)
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
            }
            
            tts_finish = false;
            chunk_idx = 0;
            llm_finish = false;
            tts_n_past = 0;  // Reset TTS model state
            audio_tokens.clear();
            all_audio_tokens.clear();  // Clear for next round
            printf("\ntts finished\n");

            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            speek_cv.notify_all();
        } else if (audio_gen_finish && !llm_finish) {
            // Current chunk finished (either reached 25 tokens or EOS token) but LLM is still generating
            // Reset audio_gen_finish to allow processing next chunk from LLM
            // This ensures: LLM生成10个token → TTS生成25个token → LLM再生成10个token → TTS再生成25个token
            fflush(stdout);
            audio_gen_finish = false;  // Reset for next chunk
            // Note: llm_text and response are already cleared above, so we'll wait for next chunk in the next iteration
        }
        #endif  // End of disabled old TTS processing code
    }
}

// ======================= Python Token2Wav 服务管理函数 =======================

// 启动 Python Token2Wav 服务进程
static bool start_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_initialized) {
        return true;  // 已经启动
    }
    
    // 构建 Python 脚本路径
    std::string script_path = ctx_omni->python_t2w_script_dir + "/token2wav_service.py";
    
    // 检查脚本是否存在
    FILE* check = fopen(script_path.c_str(), "r");
    if (!check) {
        LOG_ERR("Python T2W: 脚本不存在: %s\n", script_path.c_str());
        return false;
    }
    fclose(check);
    
    print_with_timestamp("Python T2W: 启动服务进程 %s\n", script_path.c_str());
    
    // 创建管道
#ifdef _WIN32
    // Windows: use CreateProcess with pipes for bidirectional communication
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;
    
    HANDLE hStdinRead, hStdinWrite, hStdoutRead, hStdoutWrite;
    
    if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdin) 失败\n");
        return false;
    }
    SetHandleInformation(hStdinWrite, HANDLE_FLAG_INHERIT, 0);
    
    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdout) 失败\n");
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        return false;
    }
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;
    
    ZeroMemory(&pi, sizeof(pi));
    
    // Set environment if needed
    if (!ctx_omni->python_t2w_gpu_id.empty()) {
        _putenv_s("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str());
    }
    
    std::string win_cmd = "python \"" + script_path + "\"";
    char cmd_buf[2048];
    strncpy(cmd_buf, win_cmd.c_str(), sizeof(cmd_buf) - 1);
    cmd_buf[sizeof(cmd_buf) - 1] = '\0';
    
    if (!CreateProcessA(NULL, cmd_buf, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        LOG_ERR("Python T2W: CreateProcess 失败, error=%lu\n", GetLastError());
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        return false;
    }
    
    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    CloseHandle(pi.hThread);
    
    ctx_omni->python_t2w_pid = (int)(intptr_t)pi.hProcess;
    
    // Convert HANDLEs to FILE*
    int stdin_fd = _open_osfhandle((intptr_t)hStdinWrite, 0);
    int stdout_fd = _open_osfhandle((intptr_t)hStdoutRead, 0);
    
    if (stdin_fd < 0 || stdout_fd < 0) {
        LOG_ERR("Python T2W: _open_osfhandle 失败\n");
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess);
        return false;
    }
    
    ctx_omni->python_t2w_stdin = _fdopen(stdin_fd, "w");
    ctx_omni->python_t2w_stdout = _fdopen(stdout_fd, "r");
    
#else
    // POSIX implementation using fork/pipe
    int stdin_pipe[2];
    int stdout_pipe[2];
    
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        LOG_ERR("Python T2W: 创建管道失败\n");
        return false;
    }
    
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERR("Python T2W: fork 失败\n");
        return false;
    }
    
    if (pid == 0) {
        // 子进程
        close(stdin_pipe[1]);   // 关闭写端
        close(stdout_pipe[0]);  // 关闭读端
        
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        
        // 🔧 设置 CUDA_VISIBLE_DEVICES 环境变量（必须在 import torch 之前）
        // 这样 Python 子进程只能看到指定的 GPU
        if (!ctx_omni->python_t2w_gpu_id.empty()) {
            setenv("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str(), 1);
        }
        
        // 执行 Python 脚本 (使用 conda Python)
        execlp("/Users/tianchi/software/miniconda3/bin/python", "python", script_path.c_str(), (char*)NULL);
        
        // 如果 execlp 失败
        _exit(1);
    }
    
    // 父进程
    close(stdin_pipe[0]);   // 关闭读端
    close(stdout_pipe[1]);  // 关闭写端
    
    ctx_omni->python_t2w_pid = pid;
    ctx_omni->python_t2w_stdin = fdopen(stdin_pipe[1], "w");
    ctx_omni->python_t2w_stdout = fdopen(stdout_pipe[0], "r");
#endif
    
    if (!ctx_omni->python_t2w_stdin || !ctx_omni->python_t2w_stdout) {
        LOG_ERR("Python T2W: fdopen 失败\n");
        stop_python_t2w_service(ctx_omni);
        return false;
    }
    
    // 设置为行缓冲
    setvbuf(ctx_omni->python_t2w_stdin, NULL, _IOLBF, 0);
    setvbuf(ctx_omni->python_t2w_stdout, NULL, _IOLBF, 0);
    
    // 等待服务就绪
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout)) {
        print_with_timestamp("Python T2W: 服务响应: %s", buffer);
        // 检查是否就绪
        if (strstr(buffer, "\"status\":\"ready\"") || strstr(buffer, "\"status\": \"ready\"")) {
            ctx_omni->python_t2w_initialized = true;
            print_with_timestamp("Python T2W: 服务就绪\n");
            return true;
        }
    }
    
    LOG_ERR("Python T2W: 服务未能正常启动\n");
    stop_python_t2w_service(ctx_omni);
    return false;
}

// 停止 Python Token2Wav 服务进程
static void stop_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_stdin) {
        // 发送退出命令
        fprintf(ctx_omni->python_t2w_stdin, "{\"cmd\":\"quit\"}\n");
        fflush(ctx_omni->python_t2w_stdin);
        fclose(ctx_omni->python_t2w_stdin);
        ctx_omni->python_t2w_stdin = nullptr;
    }
    
    if (ctx_omni->python_t2w_stdout) {
        fclose(ctx_omni->python_t2w_stdout);
        ctx_omni->python_t2w_stdout = nullptr;
    }
    
    if (ctx_omni->python_t2w_pid > 0) {
        // 等待子进程退出
#ifdef _WIN32
        HANDLE hProcess = (HANDLE)(intptr_t)ctx_omni->python_t2w_pid;
        // Wait briefly for process to exit gracefully
        if (WaitForSingleObject(hProcess, 500) == WAIT_TIMEOUT) {
            // Force terminate if still running
            TerminateProcess(hProcess, 1);
            WaitForSingleObject(hProcess, 1000);
        }
        CloseHandle(hProcess);
#else
        int status;
        waitpid(ctx_omni->python_t2w_pid, &status, WNOHANG);
        
        // 如果还没退出，发送 SIGTERM
        if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
            kill(ctx_omni->python_t2w_pid, SIGTERM);
            usleep(100000);  // 等待 100ms
            
            // 如果还没退出，发送 SIGKILL
            if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
                kill(ctx_omni->python_t2w_pid, SIGKILL);
            }
        }
        
        waitpid(ctx_omni->python_t2w_pid, &status, 0);
#endif
        ctx_omni->python_t2w_pid = -1;
    }
    
    ctx_omni->python_t2w_initialized = false;
    print_with_timestamp("Python T2W: 服务已停止\n");
}

// 发送命令到 Python 服务并获取响应
static bool send_python_t2w_command(struct omni_context * ctx_omni, const std::string& cmd_json, std::string& response) {
    if (!ctx_omni->python_t2w_initialized || !ctx_omni->python_t2w_stdin || !ctx_omni->python_t2w_stdout) {
        return false;
    }
    
    // 发送命令
    fprintf(ctx_omni->python_t2w_stdin, "%s\n", cmd_json.c_str());
    fflush(ctx_omni->python_t2w_stdin);
    
    // 读取响应
    char buffer[8192];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout)) {
        response = buffer;
        // 去掉末尾换行
        while (!response.empty() && (response.back() == '\n' || response.back() == '\r')) {
            response.pop_back();
        }
        return true;
    }
    
    return false;
}

// 初始化 Python Token2Wav 模型
static bool init_python_t2w_model(struct omni_context * ctx_omni, const std::string& device) {
    if (!ctx_omni->python_t2w_initialized) {
        if (!start_python_t2w_service(ctx_omni)) {
            return false;
        }
    }
    
    // 🔧 设备格式转换: "gpu:0" -> "cuda:0", "gpu" -> "cuda:0", "cpu" -> "cpu"
    // 由于 CUDA_VISIBLE_DEVICES 已经在 fork 时设置，Python 只能看到一张卡，所以始终使用 cuda:0
    std::string python_device = "cuda:0";
    if (device.find("cpu") != std::string::npos) {
        python_device = "cpu";
    }
    
    // 发送 init 命令
    // 🔧 使用 float16 以节省显存（已在 token2wav_service.py 中修复 dtype bug）
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
             "{\"cmd\":\"init\",\"model_dir\":\"%s\",\"device\":\"%s\",\"float16\":true,\"n_timesteps\":5}",
             ctx_omni->python_t2w_model_dir.c_str(), python_device.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: init 命令发送失败\n");
        return false;
    }
    
    print_with_timestamp("Python T2W init 响应: %s\n", response.c_str());
    return response.find("\"status\":\"ok\"") != std::string::npos || 
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 设置参考音频
static bool set_python_t2w_ref_audio(struct omni_context * ctx_omni, const std::string& ref_audio_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"set_ref_audio\",\"ref_audio_path\":\"%s\"}", ref_audio_path.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: set_ref_audio 命令发送失败\n");
        return false;
    }
    
    print_with_timestamp("Python T2W set_ref_audio 响应: %s\n", response.c_str());
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 处理 tokens 并生成 WAV
static bool process_python_t2w_tokens(struct omni_context * ctx_omni, 
                                const std::vector<int32_t>& tokens, 
                                bool last_chunk, 
                                const std::string& output_path,
                                double& inference_time_ms,
                                double& audio_duration) {
    // 构建 tokens JSON 数组
    std::string tokens_json = "[";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) tokens_json += ",";
        tokens_json += std::to_string(tokens[i]);
    }
    tokens_json += "]";
    
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), 
             "{\"cmd\":\"process\",\"tokens\":%s,\"last_chunk\":%s,\"output_path\":\"%s\"}",
             tokens_json.c_str(), 
             last_chunk ? "true" : "false",
             output_path.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: process 命令发送失败\n");
        return false;
    }
    
    // 解析响应中的时间信息
    // 简单解析，不使用 JSON 库
    size_t pos = response.find("\"inference_time_ms\":");
    if (pos != std::string::npos) {
        inference_time_ms = atof(response.c_str() + pos + 20);
    }
    pos = response.find("\"audio_duration\":");
    if (pos != std::string::npos) {
        audio_duration = atof(response.c_str() + pos + 17);
    }
    
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 重置 Python T2W 缓存
static bool reset_python_t2w_cache(struct omni_context * ctx_omni) {
    std::string response;
    if (!send_python_t2w_command(ctx_omni, "{\"cmd\":\"reset\"}", response)) {
        LOG_ERR("Python T2W: reset 命令发送失败\n");
        return false;
    }
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// ======================= Token2Wav 线程函数 =======================

// Python Token2Wav thread function
void t2w_thread_func_python(struct omni_context * ctx_omni, common_params *params) {
    print_with_timestamp("T2W thread (Python) started\n");
    fflush(stdout);
    
    auto& queue = ctx_omni->t2w_thread_info->queue;
    auto& mtx = ctx_omni->t2w_thread_info->mtx;
    auto& cv = ctx_omni->t2w_thread_info->cv;
    
    // Token2Wav sliding window parameters
    constexpr int32_t CHUNK_SIZE = 25;      // Main chunk size (25 tokens = 1s audio)
    constexpr int32_t PRE_LOOKAHEAD = 3;    // Lookahead for overlap
    constexpr int32_t WINDOW_SIZE = CHUNK_SIZE + PRE_LOOKAHEAD;  // 28
    
    // Buffer to accumulate tokens (for sliding window)
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    
    // 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // Helper function to get round-specific output directory
    auto get_wav_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d/tts_wav", 
                     base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            return base_output_dir + "/tts_wav";
        }
    };
    
    int last_round_idx = ctx_omni->simplex_round_idx;
    std::string tts_wav_output_dir = get_wav_output_dir();
    int wav_idx = 0;
    const int sample_rate = 24000;  // Python Token2Wav 输出采样率
    
    // 确保输出目录存在
    {
        cross_platform_mkdir_p(tts_wav_output_dir);
    }
    
    while (t2w_thread_running) {
        // 检测打断事件
        if (ctx_omni->break_event.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            while (!queue.empty()) {
                T2WOut *t2w_out = queue.front();
                queue.pop();
                delete t2w_out;
            }
            ctx_omni->break_event = false;
            token_buffer = {4218, 4218, 4218};
            wav_idx = 0;
            
            if (!ctx_omni->duplex_mode) {
                ctx_omni->wav_turn_base += 1000;
            }
            
            if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                last_round_idx = ctx_omni->simplex_round_idx;
                tts_wav_output_dir = get_wav_output_dir();
                cross_platform_mkdir_p(tts_wav_output_dir);
            }
            
            // 重置 Python 缓存
            reset_python_t2w_cache(ctx_omni);
            continue;
        }
        
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !queue.empty() || !t2w_thread_running; });
        
        if (!t2w_thread_running && queue.empty()) {
            break;
        }
        
        if (ctx_omni->break_event.load()) {
            lock.unlock();
            continue;
        }
        
        // Get all available tokens from queue
        std::vector<llama_token> new_tokens;
        bool is_final = false;
        bool is_chunk_end = false;
        int received_round_idx = -1;  // 🔧 从队列中获取的 round_idx
        
        while (!queue.empty()) {
            T2WOut *t2w_out = queue.front();
            queue.pop();
            new_tokens.insert(new_tokens.end(), t2w_out->audio_tokens.begin(), t2w_out->audio_tokens.end());
            is_final = is_final || t2w_out->is_final;
            is_chunk_end = is_chunk_end || t2w_out->is_chunk_end;
            // 🔧 使用最新的 round_idx（最后一个有效的值）
            if (t2w_out->round_idx >= 0) {
                received_round_idx = t2w_out->round_idx;
            }
            delete t2w_out;
        }
        
        lock.unlock();
        
        if (new_tokens.empty() && !is_chunk_end && !is_final) {
            continue;
        }
        
        // 🔧 [通过 T2WOut 传递 round_idx] 使用传入的 round_idx 确定输出目录
        // 这比从 ctx_omni->simplex_round_idx 读取更可靠，避免竞态条件
        if (!ctx_omni->duplex_mode && received_round_idx >= 0 && received_round_idx != last_round_idx) {
            print_with_timestamp("T2W(Python): round_idx 变化 %d -> %d（来自T2WOut），更新输出目录\n",
                                last_round_idx, received_round_idx);
            last_round_idx = received_round_idx;
            tts_wav_output_dir = get_wav_output_dir();
            cross_platform_mkdir_p(tts_wav_output_dir);
            // 重置 wav 索引，因为是新的轮次
            wav_idx = 0;
        }
        
        // Add new tokens to buffer
        token_buffer.insert(token_buffer.end(), new_tokens.begin(), new_tokens.end());
        
        // 检查 Python 服务是否初始化
        if (!ctx_omni->python_t2w_initialized) {
            continue;
        }
        
        bool need_flush = false;
        size_t min_process_threshold = WINDOW_SIZE;
        
        if (!ctx_omni->duplex_mode) {
            need_flush = is_final || is_chunk_end;
        } else {
            need_flush = is_final;
        }

        // 🔧 [修复双工模式最后一个字没说完] 当 is_final=true 但 token_buffer 为空时
        // 也需要调用 reset_python_t2w_cache，否则 T2W 的流式缓存不会被重置
        // 这会导致下一个 turn 的音频和上一个 turn 的尾音混在一起
        if (is_final && token_buffer.empty()) {
            print_with_timestamp("T2W(Python): is_final=true but token_buffer empty, calling reset directly\n");
            reset_python_t2w_cache(ctx_omni);
            // 不需要处理 token_buffer，直接继续等待下一个消息
            continue;
        }
        
        // Process windows using sliding window
        while (token_buffer.size() >= min_process_threshold || (need_flush && !token_buffer.empty())) {
            size_t process_size = std::min(token_buffer.size(), (size_t)WINDOW_SIZE);
            bool is_last_window = need_flush && (token_buffer.size() <= WINDOW_SIZE);
            
            std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);
            
            // 生成 WAV 输出路径
            std::string wav_path = tts_wav_output_dir + "/wav_" + std::to_string(ctx_omni->wav_turn_base + wav_idx) + ".wav";
            
            double inference_time_ms = 0;
            double audio_duration = 0;
            
            if (process_python_t2w_tokens(ctx_omni, window, is_last_window, wav_path, inference_time_ms, audio_duration)) {
                if (audio_duration > 0) {
                    auto wav_complete_time = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        wav_complete_time - ctx_omni->stream_decode_start_time).count();
                    
                    if (wav_idx == 0) {
                        print_with_timestamp("🎉 首响时间 (First Audio Response): %lldms\n", (long long)elapsed_ms);
                    }
                    
                    float rtf = (float)(inference_time_ms / 1000.0) / audio_duration;
                    print_with_timestamp("T2W(Python): wav_%d.wav | %.2fs audio | %.1fms inference | RTF=%.2f | t=%lldms\n",
                                        ctx_omni->wav_turn_base + wav_idx, audio_duration, inference_time_ms, rtf, (long long)elapsed_ms);
                    wav_idx++;
                    
                    // 注意：不要在中途重置缓存！Python 原版在整个对话中保持缓存连续
                    // 只在对话/轮次结束时（is_final=true）才重置缓存
                    // 使用独立 GPU 后，显存不再是问题
                }
            } else {
                LOG_ERR("T2W(Python): process 失败\n");
            }
            
            // Slide window
            if (!ctx_omni->duplex_mode) {
                if (token_buffer.size() > CHUNK_SIZE) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + CHUNK_SIZE);
                } else {
                    token_buffer.clear();
                }
            } else {
                size_t slide_amount;
                if (is_last_window) {
                    slide_amount = token_buffer.size();
                } else if (token_buffer.size() > CHUNK_SIZE) {
                    slide_amount = CHUNK_SIZE;
                } else if (token_buffer.size() > PRE_LOOKAHEAD) {
                    slide_amount = token_buffer.size() - PRE_LOOKAHEAD;
                } else {
                    slide_amount = 0;
                }
                
                if (slide_amount > 0 && slide_amount <= token_buffer.size()) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide_amount);
                } else if (slide_amount > token_buffer.size()) {
                    token_buffer.clear();
                }
            }
            
            if (is_last_window) {
                if (is_final) {
                    // 写入结束标记
                    std::string done_flag_path = tts_wav_output_dir + "/generation_done.flag";
                    FILE* flag_file = fopen(done_flag_path.c_str(), "w");
                    if (flag_file) {
                        int last_wav_idx = (wav_idx > 0) ? (ctx_omni->wav_turn_base + wav_idx - 1) : 0;
                        fprintf(flag_file, "%d\n", last_wav_idx);
                        fclose(flag_file);
                    }
                    
                    token_buffer = {4218, 4218, 4218};
                    
                    // 重置 Python 缓存
                    reset_python_t2w_cache(ctx_omni);
                    
                    if (!ctx_omni->duplex_mode) {
                        wav_idx = 0;
                        ctx_omni->wav_turn_base += 1000;
                    }
                    
                    if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                        last_round_idx = ctx_omni->simplex_round_idx;
                        tts_wav_output_dir = get_wav_output_dir();
                        cross_platform_mkdir_p(tts_wav_output_dir);
                    }
                }
                break;
            }
        }
    }
    
    print_with_timestamp("T2W(Python) 线程: 停止\n");
    fflush(stdout);
}

// C++ Token2Wav thread function (原实现，保留作为备选)
// ==============================================================================
// T2W Thread Function (C++ Token2Wav)
// ==============================================================================
// 
// 📌 关于轮次管理的说明（单工模式专用）：
// 
// 单工模式下，每轮对话的 WAV 输出保存在不同的 round_XXX 目录中。
// 统一使用 ctx_omni->simplex_round_idx 作为轮次索引的唯一来源。
// 
// 变量说明：
// 1. ctx_omni->simplex_round_idx（全局变量，唯一来源）
//    - 存储在 omni_context 结构体中的「当前轮次索引」
//    - 更新时机：
//      a) stream_decode 开始时，通过传入的 round_idx 参数同步
//      b) TTS 线程在每轮结束时递增（在发送 is_final=true 之前）
//    - 初始值为 0，每轮对话结束后 +1
// 
// 2. last_round_idx（T2W 线程本地变量）
//    - T2W 线程内部记录的「上一次使用的轮次索引」
//    - 用于检测 simplex_round_idx 是否发生变化
//    - 当 simplex_round_idx != last_round_idx 时，说明进入了新轮次，需要更新输出目录
// 
// 轮次同步流程：
//   Python调用 stream_decode(round_idx) -> 同步 simplex_round_idx
//                                              ↓
//   T2W线程检测到 simplex_round_idx != last_round_idx -> 更新 tts_wav_output_dir
// ==============================================================================
void t2w_thread_func_cpp(struct omni_context * ctx_omni, common_params *params) {
    print_with_timestamp("T2W thread (C++) started\n");
    fflush(stdout);
    
    auto& queue = ctx_omni->t2w_thread_info->queue;
    auto& mtx = ctx_omni->t2w_thread_info->mtx;
    auto& cv = ctx_omni->t2w_thread_info->cv;
    
    // Token2Wav sliding window parameters
    constexpr int32_t CHUNK_SIZE = 25;      // Main chunk size (25 tokens = 1s audio)
    constexpr int32_t PRE_LOOKAHEAD = 3;    // Lookahead for overlap
    constexpr int32_t WINDOW_SIZE = CHUNK_SIZE + PRE_LOOKAHEAD;  // 28
    
    // Buffer to accumulate tokens (for sliding window)
    // Python: buffer = [4218] * 3  # 预先放入3个前缀静音token
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    
    // 🔧 [多实例支持] 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // 🔧 [单工模式] Helper function to get round-specific output directory
    auto get_wav_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            // 单工模式：使用 round_XXX 子目录
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d/tts_wav", 
                     base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            // 双工模式：直接使用 base_output_dir
            return base_output_dir + "/tts_wav";
        }
    };
    
    // 📌 last_round_idx：T2W 线程本地记录的「上一次使用的轮次」
    // 用于检测 simplex_round_idx 是否变化，如果变化则更新输出目录
    // 🔧 [修复第一轮无输出] 初始化为 -1，确保第一轮也会触发目录创建
    int last_round_idx = -1;
    
    // WAV output settings (与 tts_thread_func 保持一致)
    std::string tts_wav_output_dir = get_wav_output_dir();
    int wav_idx = 0;
    const int sample_rate = omni::flow::Token2Wav::kSampleRate;
    
    while (t2w_thread_running) {
        // 🔧 [P0-打断检测] 检测 break_event 并清空队列
        if (ctx_omni->break_event.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            while (!queue.empty()) {
                T2WOut *t2w_out = queue.front();
                queue.pop();
                delete t2w_out;
            }
            // 重置 break_event 后继续等待新任务
            ctx_omni->break_event = false;
            token_buffer = {4218, 4218, 4218};  // 重置 buffer
            wav_idx = 0;  // 重置 wav index
            
            // 🔧 [修复竞态条件] 在 T2W 线程处理完打断后递增 wav_turn_base
            // 原因：确保当前轮次的所有 wav 文件使用旧的编号，然后再切换到新编号
            if (!ctx_omni->duplex_mode) {
                ctx_omni->wav_turn_base += 1000;
            }
            
            // 🔧 [单工模式] 打断后更新输出目录（TTS 线程已经递增了 simplex_round_idx）
            if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                last_round_idx = ctx_omni->simplex_round_idx;
                tts_wav_output_dir = get_wav_output_dir();
                print_with_timestamp("T2W线程: 打断后更新输出目录为 %s\n", tts_wav_output_dir.c_str());
            }
            continue;
        }
        
        std::unique_lock<std::mutex> lock(mtx);
        
        // Wait for queue to have data or thread to stop
        cv.wait(lock, [&] { return !queue.empty() || !t2w_thread_running || ctx_omni->break_event.load(); });
        
        if (!t2w_thread_running && queue.empty()) {
            break;
        }
        
        // 🔧 [P0-打断检测] 检测到 break_event 时跳过当前数据
        if (ctx_omni->break_event.load()) {
            lock.unlock();
            continue;
        }
        
        // Get all available tokens from queue
        std::vector<llama_token> new_tokens;
        bool is_final = false;
        bool is_chunk_end = false;  // 标记 TTS chunk 结束
        int received_round_idx = -1;  // 🔧 保存传入的 round_idx
        
        while (!queue.empty()) {
            T2WOut *t2w_out = queue.front();
            queue.pop();
            
            new_tokens.insert(new_tokens.end(), t2w_out->audio_tokens.begin(), t2w_out->audio_tokens.end());
            is_final = is_final || t2w_out->is_final;  // 任何一个是 final 就是 final
            is_chunk_end = is_chunk_end || t2w_out->is_chunk_end;  // 任何一个是 chunk_end 就是 chunk_end
            // 🔧 保存最后一个有效的 round_idx（优先使用非负值）
            if (t2w_out->round_idx >= 0) {
                received_round_idx = t2w_out->round_idx;
            }
            delete t2w_out;
        }
        
        lock.unlock();
        
        if (new_tokens.empty() && !is_chunk_end && !is_final) {
            continue;
        }
        
        // 📌 [轮次切换检测] 使用 T2WOut 传入的 round_idx 判断，而不是直接读取 ctx_omni->simplex_round_idx
        // 原因：TTS 线程在发送 is_final 之前会递增 simplex_round_idx，导致竞态条件
        // 现在 T2WOut.round_idx 保存的是递增前的值，确保 WAV 写入正确的目录
        int effective_round_idx = (received_round_idx >= 0) ? received_round_idx : ctx_omni->simplex_round_idx;
        
        if (!ctx_omni->duplex_mode && effective_round_idx != last_round_idx) {
            print_with_timestamp("T2W线程(C++): 轮次切换 (%d -> %d)，更新输出目录\n",
                                last_round_idx, effective_round_idx);
            
            // 更新本地记录的轮次
            last_round_idx = effective_round_idx;
            
            // 更新输出目录（基于 effective_round_idx）
            tts_wav_output_dir = base_output_dir + "/round_" + 
                                 (effective_round_idx < 100 ? (effective_round_idx < 10 ? "00" : "0") : "") + 
                                 std::to_string(effective_round_idx) + "/tts_wav";
            
            // 重置轮次相关状态
            wav_idx = 0;                                                    // WAV 文件编号从 0 开始
            ctx_omni->wav_turn_base = effective_round_idx * 1000;           // 更新全局 WAV 编号基数
            token_buffer = {4218, 4218, 4218};                              // 重置 token buffer（3个静音前缀）
            
            print_with_timestamp("T2W线程(C++): 新输出目录 %s\n", tts_wav_output_dir.c_str());
            
            // 确保目录存在
            cross_platform_mkdir_p(tts_wav_output_dir);
        }
        
        // Add new tokens to buffer
        size_t buffer_before = token_buffer.size();
        token_buffer.insert(token_buffer.end(), new_tokens.begin(), new_tokens.end());
        
        // 🔧 [DEBUG] 打印收到的 token IDs (只打印前10个和后3个)
        if (new_tokens.size() > 0) {
            std::string tokens_str = "[";
            for (size_t i = 0; i < std::min(new_tokens.size(), (size_t)10); i++) {
                tokens_str += std::to_string(new_tokens[i]);
                if (i < std::min(new_tokens.size(), (size_t)10) - 1) tokens_str += ",";
            }
            if (new_tokens.size() > 10) {
                tokens_str += "...";
                for (size_t i = new_tokens.size() - 3; i < new_tokens.size(); i++) {
                    tokens_str += "," + std::to_string(new_tokens[i]);
                }
            }
            tokens_str += "]";
        }
        
        // Check if token2wav is initialized
        if (!ctx_omni->token2wav_initialized || !ctx_omni->token2wav_session) {
            continue;
        }
        
        // 处理逻辑（单工/双工分开）
        bool need_flush = false;
        size_t min_process_threshold = WINDOW_SIZE;
        
        if (!ctx_omni->duplex_mode) {
            need_flush = is_final || is_chunk_end;
        } else {
            need_flush = is_final;  // 完全 flush 只在轮次结束时
        }
        
        // Process windows using sliding window
        int process_count = 0;
        while (token_buffer.size() >= min_process_threshold || (need_flush && !token_buffer.empty())) {
            // Determine how many tokens to process
            size_t process_size = std::min(token_buffer.size(), (size_t)WINDOW_SIZE);
            // 🔧 is_last_window: 当是 final 或 chunk_end，且 buffer 中的 tokens 不足一个完整 window 时
            bool is_last_window = need_flush && (token_buffer.size() <= WINDOW_SIZE);
            
            std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);
            
            // Time the inference
            auto t2w_start = std::chrono::high_resolution_clock::now();
            
            std::vector<float> chunk_wav;
            if (ctx_omni->token2wav_session->feed_window(window, is_last_window, chunk_wav)) {
                auto t2w_end = std::chrono::high_resolution_clock::now();
                double t2w_ms = std::chrono::duration<double, std::milli>(t2w_end - t2w_start).count();
                
                if (!chunk_wav.empty()) {
                    // Write WAV file
                    std::string wav_path = tts_wav_output_dir + "/wav_" + std::to_string(ctx_omni->wav_turn_base + wav_idx) + ".wav";
                    
                    const int16_t num_channels = 1;
                    const int16_t bits_per_sample = 16;
                    const int16_t block_align = num_channels * (bits_per_sample / 8);
                    const int32_t byte_rate = sample_rate * block_align;
                    
                    std::vector<int16_t> pcm(chunk_wav.size());
                    for (size_t i = 0; i < chunk_wav.size(); ++i) {
                        float x = chunk_wav[i];
                        if (!std::isfinite(x)) x = 0.0f;
                        x = std::max(-1.0f, std::min(1.0f, x));
                        pcm[i] = (int16_t)(x * 32767.0f);
                    }
                    
                    uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
                    uint32_t riff_size = 36u + data_bytes;
                    
                    FILE* f_wav = fopen(wav_path.c_str(), "wb");
                    if (f_wav) {
                        fwrite("RIFF", 1, 4, f_wav);
                        fwrite(&riff_size, 4, 1, f_wav);
                        fwrite("WAVE", 1, 4, f_wav);
                        fwrite("fmt ", 1, 4, f_wav);
                        uint32_t fmt_size = 16;
                        uint16_t audio_format = 1;
                        fwrite(&fmt_size, 4, 1, f_wav);
                        fwrite(&audio_format, 2, 1, f_wav);
                        fwrite(&num_channels, 2, 1, f_wav);
                        fwrite(&sample_rate, 4, 1, f_wav);
                        fwrite(&byte_rate, 4, 1, f_wav);
                        fwrite(&block_align, 2, 1, f_wav);
                        fwrite(&bits_per_sample, 2, 1, f_wav);
                        fwrite("data", 1, 4, f_wav);
                        fwrite(&data_bytes, 4, 1, f_wav);
                        fwrite(pcm.data(), 1, data_bytes, f_wav);
                        fclose(f_wav);
                        
                        float audio_duration = chunk_wav.size() / (float)sample_rate;
                        float rtf = (float)(t2w_ms / 1000.0) / audio_duration;
                        
                        auto wav_complete_time = std::chrono::high_resolution_clock::now();
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            wav_complete_time - ctx_omni->stream_decode_start_time).count();
                        
                        if (wav_idx == 0) {
                            print_with_timestamp("🎉 首响时间 (First Audio Response): %lldms\n", (long long)elapsed_ms);
                        }
                        print_with_timestamp("T2W线程: wav_%d.wav | %.2fs audio | %.1fms inference | RTF=%.2f | t=%lldms\n",
                                            ctx_omni->wav_turn_base + wav_idx, audio_duration, t2w_ms, rtf, (long long)elapsed_ms);
                        wav_idx++;
                    }
                }
            } else {
                LOG_ERR("T2W线程: feed_window 失败\n");
            }
            
            // Slide window by CHUNK_SIZE (25), keep last PRE_LOOKAHEAD (3) for overlap
            size_t buffer_before_slide = token_buffer.size();
            
            if (!ctx_omni->duplex_mode) {
                // 🔧 [单工模式] 保持原有逻辑，绝对不改动
                // Slide window by CHUNK_SIZE (25), keep last PRE_LOOKAHEAD (3) for overlap
                if (token_buffer.size() > CHUNK_SIZE) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + CHUNK_SIZE);
                } else {
                    token_buffer.clear();
                }
            } else {
                // 🔧 [双工模式-与 Python 对齐]
                // Python: self.token2wav_buffer = self.token2wav_buffer[min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead):]
                size_t slide_amount;
                if (is_last_window) {
                    // 最后一个 window，清空 buffer
                    slide_amount = token_buffer.size();
                } else if (token_buffer.size() > CHUNK_SIZE) {
                    // 正常情况：滑动 CHUNK_SIZE
                    slide_amount = CHUNK_SIZE;
                } else if (token_buffer.size() > PRE_LOOKAHEAD) {
                    // 有 chunk_eos 时，保留 pre_lookahead
                    slide_amount = token_buffer.size() - PRE_LOOKAHEAD;
                } else {
                    slide_amount = 0;  // 太少了，不滑动
                }
                
                if (slide_amount > 0 && slide_amount <= token_buffer.size()) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide_amount);
                } else if (slide_amount > token_buffer.size()) {
                    token_buffer.clear();
                }
            }
            process_count++;
            
            if (is_last_window) {
                // 🔧 [与 Python 对齐] 只有 is_final（轮次结束）时才重置
                // Python: if is_last_chunk: stream(..., last_chunk=True); buffer = []
                // 普通 chunk 结束时，剩余 tokens 保留在 buffer 中等待下一个 chunk
                if (is_final) {
                    // 🚀 [优化] 写入结束标记文件，通知 Python 立即结束（无需等待超时）
                    // 标记文件包含最后一个 wav 的编号，方便 Python 验证
                    {
                        std::string done_flag_path = tts_wav_output_dir + "/generation_done.flag";
                        FILE* flag_file = fopen(done_flag_path.c_str(), "w");
                        if (flag_file) {
                            // 写入最后一个 wav 的编号（wav_idx - 1，因为 wav_idx 已经指向下一个）
                            int last_wav_idx = (wav_idx > 0) ? (ctx_omni->wav_turn_base + wav_idx - 1) : 0;
                            fprintf(flag_file, "%d\n", last_wav_idx);
                            fclose(flag_file);
                            print_with_timestamp("T2W线程: 写入结束标记 %s (last_wav=%d)\n", done_flag_path.c_str(), last_wav_idx);
                        }
                    }
                    
                    // 🔧 [关键] 不调用 Token2WavSession::reset()
                    // 原因：reset() 会把 stream_started_=false，导致下一轮 feed_window 失败
                    // 单工和双工模式都只重置 token_buffer，保持 Token2Wav 的 stream 状态
                    // 重新初始化buffer（3个静音token作为前缀）
                    token_buffer = {4218, 4218, 4218};
                    
                    // 🔧 [修复竞态条件] 在 T2W 线程处理完 is_final 后递增 wav_turn_base
                    // 原因：确保当前轮次的所有 wav 文件使用旧的编号，然后再切换到新编号
                    // 这样避免了 TTS 线程提前递增导致最后几个 wav 文件编号跳跃的问题
                    if (!ctx_omni->duplex_mode) {
                        wav_idx = 0;  // 🔧 [单工模式] 重置 wav_idx 用于下一轮
                        ctx_omni->wav_turn_base += 1000;
                    }
                    
                    // 🔧 [单工模式] 在 is_final 后检查是否需要更新目录
                    // simplex_round_idx 已经在 TTS 线程中递增
                    if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                        last_round_idx = ctx_omni->simplex_round_idx;
                        tts_wav_output_dir = get_wav_output_dir();
                        print_with_timestamp("T2W线程: 轮次结束后更新输出目录为 %s\n", tts_wav_output_dir.c_str());
                        // 确保目录存在
                        cross_platform_mkdir_p(tts_wav_output_dir);
                    }
                }
                // 注意：is_chunk_end 时不重置 buffer，剩余 tokens 保留给下一个 chunk
                break;
            }
        }
        
    }
    
    print_with_timestamp("T2W(C++) 线程: 停止\n");
    fflush(stdout);
}

// Token2Wav 线程入口函数（根据配置选择 Python 或 C++ 实现）
void t2w_thread_func(struct omni_context * ctx_omni, common_params *params) {
    if (ctx_omni->use_python_token2wav) {
        t2w_thread_func_python(ctx_omni, params);
    } else {
        t2w_thread_func_cpp(ctx_omni, params);
    }
}

bool stream_prefill(struct omni_context * ctx_omni, std::string aud_fname, std::string img_fname, int index, int max_slice_nums) {
    
    // 只有在新一轮开始时 (index == 0) 才需要等待上一轮 TTS 完成
    // 同一轮内的后续 prefill (index >= 1) 不需要等待
    if (ctx_omni->use_tts && index == 0 && ctx_omni->warmup_done.load() && !ctx_omni->duplex_mode) {
        // 🔧 如果 break_event 已触发，跳过等待（上一轮已被打断）
        if (ctx_omni->break_event.load()) {
            print_with_timestamp("TTS: break_event active, skipping wait for previous round\n");
            ctx_omni->speek_done = true;
            ctx_omni->break_event.store(false);
            speek_cv.notify_all();
        }
        print_with_timestamp("TTS: 等待上一轮语音生成完成\n");
        std::unique_lock<std::mutex> lock(speek_mtx);
        // 添加超时等待，避免永久卡住
        auto wait_result = speek_cv.wait_for(lock, std::chrono::seconds(5), [&]{return ctx_omni->speek_done || ctx_omni->break_event.load(); });
        if (!wait_result) {
            // 强制设置为 true 以继续
            ctx_omni->speek_done = true;
        }
        // 等待完成后重置 speek_done，为下一轮做准备
        ctx_omni->speek_done = false;
        
        // 🔧 [多轮对话修复] 清理 TTS 队列中的残留数据，避免混淆
        if (ctx_omni->tts_thread_info && !ctx_omni->duplex_mode) {
            std::lock_guard<std::mutex> tts_lock(ctx_omni->tts_thread_info->mtx);
            auto& tts_queue = ctx_omni->tts_thread_info->queue;
            while (!tts_queue.empty()) {
                LLMOut* old_out = tts_queue.front();
                tts_queue.pop();
                delete old_out;
            }
            print_with_timestamp("stream_prefill: cleared TTS queue for new turn\n");
        }
    } else if (ctx_omni->use_tts && index == 0 && !ctx_omni->duplex_mode) {
        // 否则 LLM 输出会被丢弃（因为 speek_done 初始值为 true）
        ctx_omni->speek_done = false;
        
        // 🔧 [多轮对话修复] 首次初始化时也要清理队列
        if (ctx_omni->tts_thread_info) {
            std::lock_guard<std::mutex> tts_lock(ctx_omni->tts_thread_info->mtx);
            auto& tts_queue = ctx_omni->tts_thread_info->queue;
            while (!tts_queue.empty()) {
                LLMOut* old_out = tts_queue.front();
                tts_queue.pop();
                delete old_out;
            }
        }
    } else if (ctx_omni->use_tts) {
    }
    
    // ctx_omni->need_speek = false;
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    
    std::string voice_clone_prompt = "";
    std::string assistant_prompt = "";
    
    if (ctx_omni->media_type == 1){ // audio
        // 如果 audio_voice_clone_prompt 以 "<|" 开头（特殊 token），不添加前缀
        // 这允许完全控制 prompt 格式（例如使用 system 而不是 user）
        if (ctx_omni->audio_voice_clone_prompt.substr(0, 2) == "<|") {
            voice_clone_prompt = ctx_omni->audio_voice_clone_prompt;
        } else {
            voice_clone_prompt = "<|im_start|>user\n" + ctx_omni->audio_voice_clone_prompt;
        }
        // 如果 audio_assistant_prompt 以 "<|" 开头（特殊 token），不添加前缀
        if (ctx_omni->audio_assistant_prompt.substr(0, 2) == "<|") {
            assistant_prompt = ctx_omni->audio_assistant_prompt;
        } else {
            assistant_prompt = "<|im_start|>user\n" + ctx_omni->audio_assistant_prompt;
        }
    }
    else if (ctx_omni->media_type == 2){ // omni
        if (ctx_omni->omni_voice_clone_prompt.substr(0, 2) == "<|") {
            voice_clone_prompt = ctx_omni->omni_voice_clone_prompt;
        } else {
            voice_clone_prompt = "<|im_start|>user\n" + ctx_omni->omni_voice_clone_prompt;
        }
        if (ctx_omni->omni_assistant_prompt.substr(0, 2) == "<|") {
            assistant_prompt = ctx_omni->omni_assistant_prompt;
        } else {
            assistant_prompt = "<|im_start|>user\n" + ctx_omni->omni_assistant_prompt;
        }
    }
    // 这是因为 omni_init 中可能会调用 stream_prefill(voice_audio, "", 0)，
    // 然后测试脚本又会调用 stream_prefill(audio_0, "", 0)
    // 如果不检查这个标志，系统 prompt 会被评估两次，导致格式混乱
    if (index == 0 && !ctx_omni->system_prompt_initialized) {
        print_with_timestamp("stream_prefill: n_past = %d\n voice_clone_prompt = %s\n assistant_prompt = %s\n", ctx_omni->n_past, voice_clone_prompt.c_str(), assistant_prompt.c_str());
        // tc-todo
        // llama_kv_cache_clear(ctx_omni->ctx_llama);
        
        // 🔧 [对齐 Python 双工模型] 初始化格式
        // Python 双工模型的 _init_duplex_session：
        //   1. feed prefix_system_prompt（包含 <|audio_start|>）
        //   2. feed ref_audio 的 APM embedding
        //   3. feed suffix_system_prompt（包含 <|audio_end|><|im_end|>）
        // 
        // 完整格式：
        //   <|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>
        //   [ref_audio APM embedding]
        //   <|audio_end|><|im_end|>
        
        if (ctx_omni->duplex_mode && aud_fname.length() > 0) {
            // 双工模式：参考音频需要送入 LLM
            
            // Step 1: 评估 prefix (voice_clone_prompt，包含 <|audio_start|>)
            eval_string(ctx_omni, ctx_omni->params, voice_clone_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
            
            // Step 2: 获取并 prefill 参考音频的 APM embedding
            auto * audio_embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
            if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
                prefill_with_emb(ctx_omni, ctx_omni->params, audio_embeds->embed, audio_embeds->n_pos, 
                                ctx_omni->params->n_batch, &ctx_omni->n_past);
                omni_embed_free(audio_embeds);
            } else {
            }
            
            // Step 3: 评估 suffix (assistant_prompt，包含 <|audio_end|><|im_end|>)
            eval_string(ctx_omni, ctx_omni->params, assistant_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        } else {
            // 🔧 [与 Python 对齐] 非双工模式也需要在 system prompt 中插入 ref_audio embedding
            // Python: sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            // 格式: <|im_start|>system\n{vc_prompt_prefix}\n<|audio_start|>[ref_audio_embed]<|audio_end|>{vc_prompt_suffix}<|im_end|>\n
            
            // 确定 ref_audio 路径：优先使用配置的路径，否则使用默认路径
            std::string system_ref_audio = ctx_omni->ref_audio_path.empty() 
                ? "tools/omni/assets/default_ref_audio/default_ref_audio.wav" 
                : ctx_omni->ref_audio_path;
            print_with_timestamp("system prompt ref_audio: %s\n", system_ref_audio.c_str());
            
            // Step 1: 评估 prefix (voice_clone_prompt，包含 <|audio_start|>)
            eval_string(ctx_omni, ctx_omni->params, voice_clone_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
            
            // Step 2: 获取并 prefill 参考音频的 APM embedding
            auto * ref_audio_embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, system_ref_audio);
            if (ref_audio_embeds != nullptr && ref_audio_embeds->n_pos > 0) {
                print_with_timestamp("system prompt ref_audio embedding: n_pos=%d\n", ref_audio_embeds->n_pos);
                prefill_with_emb(ctx_omni, ctx_omni->params, ref_audio_embeds->embed, ref_audio_embeds->n_pos, 
                                ctx_omni->params->n_batch, &ctx_omni->n_past);
                omni_embed_free(ref_audio_embeds);
            } else {
                print_with_timestamp("WARNING: failed to load system prompt ref_audio: %s\n", system_ref_audio.c_str());
            }
            
            // Step 3: 评估 suffix (assistant_prompt，包含 <|audio_end|><|im_end|>)
            eval_string(ctx_omni, ctx_omni->params, assistant_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        }
        
        // 标记系统 prompt 已初始化
        ctx_omni->system_prompt_initialized = true;

        //把这步完成再开llm线程以防冲突
        ctx_omni->n_keep = ctx_omni->n_past;
        print_with_timestamp("🔒 n_keep 设置为 %d (system prompt tokens)，这部分永远不会被滑动窗口删除\n", ctx_omni->n_keep);
        eval_prefix(ctx_omni, ctx_omni->params);
        
        // 🔧 [说明] index=0 时，aud_fname 通常是 ref_audio（用于 voice cloning）
        // ref_audio 已经在上面的 system prompt 初始化中被正确 prefill 了
        // 这里不需要再处理 aud_fname，因为：
        // 1. 如果 aud_fname 是 ref_audio，它已经作为 system prompt 的一部分被处理了
        // 2. 如果 aud_fname 是用户音频，用户音频应该从 index >= 1 开始传入
        // 所以 index=0 阶段只负责 system prompt 初始化，不处理额外的音频输入
        print_with_timestamp("stream_prefill(index=0): system prompt 初始化完成，ref_audio 已在其中 prefill\n");
        
        // 🔧 [#39 滑动窗口] 注册 system prompt 保护长度
        sliding_window_register_system_prompt(ctx_omni);

        print_with_timestamp("n_past = %d\n", ctx_omni->n_past);
        
        if (ctx_omni->async){
            //create llm thread
            print_with_timestamp("create llm & tts thread\n");
            if (!ctx_omni->llm_thread.joinable()) {
                llm_thread_running = true;
                ctx_omni->llm_thread = std::thread(llm_thread_func, ctx_omni, ctx_omni->params);
                print_with_timestamp("create llm thread success\n");
            }
            if (ctx_omni->use_tts && !ctx_omni->tts_thread.joinable()) {
                tts_thread_running = true;
                // 🔧 [双工模式] 根据 duplex_mode 选择不同的 TTS 线程函数
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_thread = std::thread(tts_thread_func_duplex, ctx_omni, ctx_omni->params);
                    print_with_timestamp("create tts thread (duplex mode) success\n");
                } else {
                    ctx_omni->tts_thread = std::thread(tts_thread_func, ctx_omni, ctx_omni->params);
                    print_with_timestamp("create tts thread (simplex mode) success\n");
                }
            }
            
            // Start T2W thread if TTS is enabled and thread is not already running
            if (ctx_omni->use_tts && ctx_omni->t2w_thread_info && !ctx_omni->t2w_thread.joinable()) {
                t2w_thread_running = true;
                ctx_omni->t2w_thread = std::thread(t2w_thread_func, ctx_omni, ctx_omni->params);
                print_with_timestamp("create t2w thread success\n");
            }
        }

    }
    else {
        if (!ctx_omni->async) {
            if (img_fname.length() > 0) {
                // 🔧 [高清模式] 使用 V2.6 slice schema
                // 如果指定了 max_slice_nums，临时设置（用于高清+高刷组合模式）
                if (max_slice_nums >= 1 && ctx_omni->ctx_vision) {
                    vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
                    LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
                }
                std::vector<std::vector<float>> vision_chunks;
                if (!omni_image_embed_make_chunks_with_filename(ctx_omni->ctx_vision, 
                        ctx_omni->params->cpuparams.n_threads, img_fname, vision_chunks)) {
                    LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
                    return false;
                }
                
                int n_chunks = (int)vision_chunks.size();
                int tokens_per_chunk = (int)vision_chunks[0].size() / hidden_size;
                bool has_slices = (n_chunks > 1);
                
                std::string prefix = "<unit>";
                eval_string(ctx_omni, ctx_omni->params, prefix.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                
                // Overview
                eval_string(ctx_omni, ctx_omni->params, "<image>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                prefill_with_emb(ctx_omni, ctx_omni->params, vision_chunks[0].data(), tokens_per_chunk, ctx_omni->params->n_batch, &ctx_omni->n_past);
                eval_string(ctx_omni, ctx_omni->params, "</image>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                
                // Slices (V2.6 schema)
                if (has_slices) {
                    for (int i = 1; i < n_chunks; i++) {
                        eval_string(ctx_omni, ctx_omni->params, "<slice>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                        prefill_with_emb(ctx_omni, ctx_omni->params, vision_chunks[i].data(), tokens_per_chunk, ctx_omni->params->n_batch, &ctx_omni->n_past);
                        eval_string(ctx_omni, ctx_omni->params, "</slice>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    }
                    eval_string(ctx_omni, ctx_omni->params, "\n", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                }
                LOG_INF("%s: prefilled %d vision chunks (%d tokens each)\n", __func__, n_chunks, tokens_per_chunk);
            }
            if (aud_fname.length() > 0) {
                print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
                auto * embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
                // 🔧 [修复] 音频太短时会在 audition_audio_preprocess 中自动 pad 静音到 100ms
                // 这里做安全检查，如果仍然失败则跳过该帧音频
                if (embeds != nullptr && embeds->n_pos > 0) {
                    print_with_timestamp("stream_prefill(index=%d): user audio embedding: n_pos=%d\n", index, embeds->n_pos);
                    // 🔧 添加音频标记，与 index=0 保持一致
                    eval_string(ctx_omni, ctx_omni->params, "<|audio_start|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    prefill_with_emb(ctx_omni, ctx_omni->params, embeds->embed, embeds->n_pos, ctx_omni->params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, ctx_omni->params, "<|audio_end|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    omni_embed_free(embeds);
                } else {
                    LOG_WRN("%s: audio encoding failed, skipping audio for this frame\n", __func__);
                }
            }
        }
        else {
            // async 模式：将 embeds 加入队列，由 LLM 线程处理
            
            const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
            omni_embeds * omni_embeds = new struct omni_embeds();
            //video
            if (img_fname.length() > 0) {
                LOG_INF("%s: img_fname:%s\n", __func__, img_fname.c_str());
                // 🔧 [高清模式] 如果指定了 max_slice_nums，临时设置（用于高清+高刷组合模式）
                if (max_slice_nums >= 1 && ctx_omni->ctx_vision) {
                    vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
                    LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
                }
                // 🔧 [高清模式] 使用新的 chunks 接口，支持 V2.6 slice schema
                if (!omni_image_embed_make_chunks_with_filename(ctx_omni->ctx_vision, 
                        ctx_omni->params->cpuparams.n_threads, img_fname, omni_embeds->vision_embed)) {
                    LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
                    delete omni_embeds;
                    return false;
                }
                LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int)omni_embeds->vision_embed.size());
            }
            //audio
            // 只有在音频路径非空时才处理音频
            if (aud_fname.length() > 0) {
                LOG_INF("%s: aud_fname:%s\n", __func__, aud_fname.c_str());
                auto * audio_embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
                // 🔧 [修复] 音频太短时会在 audition_audio_preprocess 中自动 pad 静音到 100ms
                // 这里做安全检查，如果仍然失败则跳过该帧音频（保持 audio_embed 为空）
                if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
                    //save to buffer
                    LOG_INF("%s: audio_embeds->n_pos: %d ,hidden_size: %d\n", __func__, audio_embeds->n_pos, hidden_size);
                    omni_embeds->audio_embed.resize(audio_embeds->n_pos * hidden_size);
                    std::memcpy(omni_embeds->audio_embed.data(), audio_embeds->embed, omni_embeds->audio_embed.size() * sizeof(float));
                    omni_embed_free(audio_embeds);
                } else {
                    LOG_WRN("%s: audio encoding failed, skipping audio for this frame: %s\n", __func__, aud_fname.c_str());
                }
            }
            omni_embeds->index = index;
            // 🔧 [整合] <|im_start|>user\n 已在 sys prompt 末尾添加，后续轮次在 stream_decode 结束时添加
            // 不再需要在这里设置 is_round_start 标记
            
            std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
            ctx_omni->llm_thread_info->cv.wait(lock, [&] { return ctx_omni->llm_thread_info->queue.size() < ctx_omni->llm_thread_info->MAX_QUEUE_SIZE; });
            ctx_omni->llm_thread_info->queue.push(omni_embeds);

            //notify the llm
            lock.unlock();
            ctx_omni->llm_thread_info->cv.notify_all();
        }
    }
    // 🔧 [诊断] 打印 stream_prefill 结束时的状态
    print_with_timestamp("\n\nc++ finish stream_prefill(index=%d). n_past=%d, n_keep=%d, n_ctx=%d\n\n",
                         index, ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx);
    return true;
}

bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx) {
    // NOTE: 不再自动归档旧输出目录，因为这会导致同一 session 中每轮对话的输出被移走
    // 如果需要归档，可以在新 session 开始时（omni_init）手动调用
    // move_old_output_to_archive();
    
    // 🔧 [轮次同步] 如果调用方指定了 round_idx，立即同步 simplex_round_idx
    // 这解决了 TTS 线程异步递增 round_idx 导致的竞态条件问题
    // 场景：Python 端在 streaming_generate 结束后立即递增 current_round_number，
    //       但 C++ 的 TTS 线程可能还没处理完上一轮，导致 simplex_round_idx 滞后
    // 
    // 注意：新 session 时的 KV cache 清理在 update_session_config 中处理
    // 这里只处理同一 session 内的轮次同步
    if (round_idx >= 0 && !ctx_omni->duplex_mode) {
        if (ctx_omni->simplex_round_idx != round_idx) {
            print_with_timestamp("📍 [轮次同步] 调用方指定 round_idx=%d，当前 simplex_round_idx=%d，强制同步\n",
                                round_idx, ctx_omni->simplex_round_idx);
            ctx_omni->simplex_round_idx = round_idx;
            // 同时更新 wav_turn_base 以保持一致性
            ctx_omni->wav_turn_base = round_idx * 1000;
        }
    }
    
    // 🔧 [已禁用] 不再清空 llm_debug/chunk_* 目录
    // 原因：这个清空操作和 TTS 线程的写入操作存在竞态条件
    // 场景：
    //   1. 第一轮 stream_decode 完成，LLM 返回，但 TTS 线程还在处理 chunk_0-9
    //   2. 第二轮 stream_decode 开始（TTS 还没递增 simplex_round_idx）
    //   3. 第二轮 stream_decode 清空了 round_XXX/llm_debug/chunk_*
    //   4. 导致 TTS 已经写入的 chunk_0-9 被删除，只剩下后续的 chunk_10 等
    // 现在每个 round 有独立的目录（round_000, round_001...），不需要清空旧数据
    
    // Record start time (t=0) for WAV file naming
    ctx_omni->stream_decode_start_time = std::chrono::high_resolution_clock::now();
    
    // 🔧 [诊断] 打印 stream_decode 开始时的关键状态
    print_with_timestamp("📍 stream_decode 开始: n_past=%d, n_keep=%d, n_ctx=%d, duplex_mode=%d\n",
                         ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx, ctx_omni->duplex_mode);
    
    // 🔧 [双工模式] 重置 ended_with_listen 标志
    // 每次 decode 开始时，假设会以非 listen 结束（需要清理 KV cache）
    // 如果 LLM 线程检测到 <|listen|>，会设置为 true
    
    // 🔧 [与 Python 对齐] 重置 llm_generation_done 标志
    // 每次新的 decode 开始时重置，TTS 线程会检查此标志来决定是否添加 text_eos_embed
    if (!ctx_omni->duplex_mode) ctx_omni->llm_generation_done.store(false);
    ctx_omni->ended_with_listen = false;
    
    // 🔧 [关键修复] 在 decode 开始时重置 break_event
    // 问题：break_event 只在 T2W 线程中被重置，但 T2W 可能还在等待数据
    //       导致新的 decode 检测到 break_event=true 后立即退出，不生成任何 token
    // 解决：在 decode 开始时立即重置 break_event，确保新一轮生成可以正常进行
    if (ctx_omni->duplex_mode && ctx_omni->break_event.load()) {
        ctx_omni->break_event.store(false);
        print_with_timestamp("📍 stream_decode: reset break_event from true to false\n");
    }
    
    //eval_string(ctx_omni->ctx_llama, std::string("hello").c_str(), params->n_batch, &ctx_omni->n_past, false);
    
    // 🔧 [修复多轮对话] 清空上一轮的文本队列，重置状态标志
    if (ctx_omni->duplex_mode) {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->text_done_flag = false;
        ctx_omni->text_streaming = true;
    }
    
    if (ctx_omni->async && ctx_omni->duplex_mode){
        ctx_omni->need_speek = true;
        //ctx_omni->llm_thread.join();
        ctx_omni->llm_thread_info->cv.notify_all();
        print_with_timestamp("wait prefill done\n");
    }
    
    //eval_string(ctx_omni->ctx_llama, std::string("hello").c_str(), params->n_batch, &ctx_omni->n_past, false);
    
    // 🔧 [修复多轮对话] 清空上一轮的文本队列，重置状态标志
    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->text_done_flag = false;
        ctx_omni->text_streaming = true;
    }
    
    if (ctx_omni->async){
        // 🔧 确保线程已启动（如果 prefill 是同步模式执行的，线程可能还没启动）
        if (!ctx_omni->tts_thread.joinable() && ctx_omni->use_tts) {
            tts_thread_running = true;
            if (ctx_omni->duplex_mode) {
                ctx_omni->tts_thread = std::thread(tts_thread_func_duplex, ctx_omni, ctx_omni->params);
                print_with_timestamp("stream_decode: create tts thread (duplex mode)\n");
            } else {
                ctx_omni->tts_thread = std::thread(tts_thread_func, ctx_omni, ctx_omni->params);
                print_with_timestamp("stream_decode: create tts thread (simplex mode)\n");
            }
        }
        if (!ctx_omni->t2w_thread.joinable() && ctx_omni->use_tts && ctx_omni->t2w_thread_info) {
            t2w_thread_running = true;
            ctx_omni->t2w_thread = std::thread(t2w_thread_func, ctx_omni, ctx_omni->params);
            print_with_timestamp("stream_decode: create t2w thread\n");
        }
        
        ctx_omni->need_speek = true;
        //ctx_omni->llm_thread.join();
        ctx_omni->llm_thread_info->cv.notify_all();
        print_with_timestamp("wait prefill done\n");
        std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        g_decode_cv.wait(lock, []{ return prefill_done; });
        prefill_done = false;
    }
    // 只有启用 TTS 时才设置 speek_done 为 false
    if (ctx_omni->use_tts) {
        ctx_omni->speek_done = false;
    }
    
    // 🔧 [对齐 Python MiniCPM-o-4_5-latest] 根据模式设置不同的 assistant generation prompt
    // 
    // === 非双工模式 (Simplex) ===
    // Python default_tts_chat_template:
    //   {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' + think_str + '<|tts_bos|>' }}{% endif %}
    // 其中 think_str = "<think>\n\n</think>\n\n"
    // 
    // 注意：stream_prefill 已经添加了 <|audio_start|>[audio]<|audio_end|>
    //       这里只需要添加关闭用户消息的 <|im_end|> 和 assistant generation prompt
    // 
    // 完整的 assistant generation prompt (非双工 TTS):
    //   <|im_end|>\n             (关闭用户消息，stream_prefill 已添加 <|audio_end|>)
    //   <|im_start|>assistant\n  (开始 assistant turn)
    //   <think>\n\n</think>\n\n  (think 标记，注意换行符)
    //   <|tts_bos|>              (TTS 开始标记)
    //
    // === 双工模式 (Duplex) ===
    // 双工模式使用 <unit> 标记，不使用标准 chat template
    // stream_prefill 添加 <unit>[audio_embed] (无 audio_start/end)
    // 模型自动输出 <|speak|> 或 <|listen|> 来控制对话流程
    
    if (ctx_omni->duplex_mode) {
        // 🔧 [双工模式] 不需要添加 assistant prompt
        // 双工模型会根据上下文自动决定说话还是继续监听
        // stream_prefill 已添加 <unit>[audio_embed]
        // 模型会输出 <|speak|>xxx<|chunk_eos|> 或 <|listen|><|chunk_eos|>
        print_with_timestamp("stream_decode: 双工模式，跳过 assistant prompt\n");
    } else if (ctx_omni->use_tts) {
        // 🔧 [非双工 TTS 模式] 需要包含 <|tts_bos|>，告诉模型开始生成 TTS 文本
        // stream_prefill 已添加 <|audio_start|>[audio]<|audio_end|>，这里关闭用户消息并添加 assistant prompt
        // 格式: <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
        std::string prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>";
        print_with_timestamp("📍 [单工TTS] 添加 assistant prompt: \"%s\", n_past=%d\n", 
                            prompt.c_str(), ctx_omni->n_past);
        {
            eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        }
        print_with_timestamp("📍 [单工TTS] assistant prompt 完成, n_past=%d\n", ctx_omni->n_past);
    } else {
        // 🔧 [非双工纯 LLM 模式] 只使用标准的 assistant prompt（无 TTS 标记，无 think 标记）
        // 格式: <|im_end|>\n<|im_start|>assistant\n
        std::string prompt = "<|im_end|>\n<|im_start|>assistant\n";
        {
            eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        }
    }
    LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
    LOG_INF("<assistant>");
    const int max_tgt_len = ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n", 
                         max_tgt_len, ctx_omni->params->n_predict, ctx_omni->params->n_ctx);
    // LLM chunk size: 每chunk推送给TTS的LLM tokens数量
    // 原始Python: generate_chunk_size=10
    // 注意：step_size影响TTS条件长度，可能影响音质
    // step_size=5: 首响更快(612ms)但可能影响音质
    // step_size=10: 首响稍慢(791ms)但音质更稳定
    int step_size = 10;  // 恢复原始值
    std::string response = "";
    
    // tts streaming memory
    std::string tts_txt = "";
    int chunk_idx = 0;
    std::vector<llama_token> audio_input_ids;
    // TODO write to specific buffers
    std::vector<float> tts_output;
    tts_output.resize(1/* batch_size */ * (ctx_omni->params->n_ctx /* seq_len */ * 2) * 256);
    bool llm_finish = false;
    bool llm_first_token_logged = false;
    
    // 🔧 [修复双工缺字问题] 记录当前 chunk 是否是 turn 的结束
    // 此变量随 LLMOut 一起传递给 TTS 线程，避免全局状态的时序问题
    bool local_is_end_of_turn = false;
    // 🔧 [P0-打断检测] 双工模式下记录当前 chunk 生成的 token 数
    int current_chunk_tokens = 0;
    for (int il = 0; il < max_tgt_len; ) {
        // 🔧 [P0-打断检测] 外层循环也检测 break_event
        if (ctx_omni->break_event.load()) {
            llm_finish = true;
            break;
        }
        fflush(stdout);
        response = "";
        fflush(stdout);
        
        // 注意: speek_done=true 现在只表示"TTS 完成，可接受新 prefill"
        // 不再用于控制 LLM 退出。LLM 应该正常完成直到 EOS 或达到最大长度。
        // 打断逻辑通过 need_speek 或其他机制处理。
        fflush(stdout);
        
        int jl = 0;  // 计数有效的 TTS token 数量
        int total_tokens_generated = 0;  // 计数总共生成的 token 数量（包括被过滤的）
        // 收集当前chunk的token IDs和hidden states用于TTS条件生成
        // 🔧 [优化] 只收集有效的 TTS token，确保每次给 TTS 的都是 step_size 个有效 token
        std::vector<llama_token> chunk_token_ids;
        std::vector<float> chunk_hidden_states;
        int llm_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
        
        // 🔧 [修复双工缺字问题] 每个 chunk 开始时重置 is_end_of_turn 状态
        // 只有当检测到 TURN_EOS/TTS_EOS/EOS 时才会在下面设置为 true
        local_is_end_of_turn = false;
        
        // 🔧 [单双工适配] chunk 限制只在双工模式下生效
        // - 双工模式: 每个 chunk 最多 max_new_speak_tokens_per_chunk 个 tokens，便于及时响应打断
        // - 单工模式: 无限制，LLM 生成直到 EOS
        int max_chunk_tokens = ctx_omni->duplex_mode ? ctx_omni->max_new_speak_tokens_per_chunk : 0;
        bool chunk_limit_reached = (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens);
        {
            fflush(stdout);
            // 🔧 [重要] 循环直到收集到 step_size 个有效 token，而不是生成 step_size 个 token
            // 🔧 [P0-打断检测] 检测 break_event，支持双工模式下的打断
            // 🔧 [P2-chunk限制] 检测 max_new_speak_tokens_per_chunk，便于及时响应打断
            while (jl < step_size && !llm_finish && !ctx_omni->break_event.load() && !chunk_limit_reached) {
                // streaming llm
                const char * tmp = nullptr;
                float * hidden_states = nullptr;

                llama_token sampled_token = 0;
                {
                    std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                    // 使用新函数获取token文本、hidden state和token ID
                    tmp = llama_loop_with_hidden_and_token(ctx_omni, ctx_omni->params, ctx_omni->ctx_sampler, ctx_omni->n_past, hidden_states, sampled_token);
                }
                
                total_tokens_generated++;
                
                // 🔧 [过滤逻辑] 只收集有效的 TTS token
                // 特殊 token（如 <think>, </think>, 换行等）不计入 step_size
                if (tmp != nullptr && hidden_states != nullptr) {
                    if (is_valid_tts_token(sampled_token)) {
                        // 有效 token：收集并计入计数
                        chunk_token_ids.push_back(sampled_token);
                        chunk_hidden_states.insert(chunk_hidden_states.end(), hidden_states, hidden_states + llm_n_embd);
                        jl++;  // 只有有效 token 才增加计数
                        
                        // 🔧 [调试] 打印收集的 token 和 hidden states 摘要
                        
                        // 🔧 [P2-chunk限制] 更新当前 chunk 的 token 计数
                        current_chunk_tokens++;
                        
                        // 检查是否达到 chunk 限制
                        if (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens) {
                            chunk_limit_reached = true;
                        }
                    } else {
                        // 🔧 [调试] 打印被过滤的 token
                    }
                }
                
                // if (hidden_states != nullptr) {
                //     int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
                //     // 打印第一个 embedding 的前5个数字
                //     printf("First embedding (first 5): ");
                //     for (int i = 0; i < 5 && i < n_embd; i++) {
                //         printf("%.6f ", hidden_states[i]);
                //     }
                //     printf("\n");
                //     // 打印最后一个 embedding 的后5个数字 (这里只有1个token，所以第一个和最后一个是同一个)
                //     printf("Last embedding (last 5): ");
                //     for (int i = n_embd - 5; i < n_embd; i++) {
                //         if (i >= 0) {
                //             printf("%.6f ", hidden_states[i]);
                //         }
                //     }
                //     printf("\n");
                //     free(hidden_states);
                // }
                if (!llm_first_token_logged) {
                    llm_first_token_logged = true;
                }
                if (tmp == nullptr) {
                    LOG_ERR("llama_loop returned nullptr!");
                    break;
                }
                
                // 🔧 [调试日志] 记录每个生成的 token 到文件
                
                // 🔧 [使用 token ID 检测] 使用缓存的 token ID 进行检测，比字符串比较更高效
                OmniTokenType token_type = get_token_type(ctx_omni, sampled_token);
                if (token_type != OmniTokenType::NORMAL) {
                }

                if (ctx_omni->duplex_mode) {
                    // 🔧 [与 Python 对齐] turn_eos 处理：
                    // Python 中 turn_eos 不触发 LLM 跳出，它只是标记 is_end_of_turn。
                    // LLM 继续生成直到 chunk_eos/listen 通过 is_end_token() 正常跳出。
                    // turn_eos 本身作为 special token 被过滤掉（不加入文本 response）。
                    // is_end_of_turn 传递给 TTS 线程，让 TTS 知道这是最后一个 chunk。
                    if (token_type == OmniTokenType::TURN_EOS || 
                        token_type == OmniTokenType::TTS_EOS ||
                        token_type == OmniTokenType::EOS) {
                        local_is_end_of_turn = true;
                        ctx_omni->current_turn_ended = true;
                        print_with_timestamp("LLM Duplex: turn_eos detected (type=%d), "
                                            "set is_end_of_turn=true (not breaking, wait for chunk_eos)\n",
                                            (int)token_type);
                        // 不 break，不设 llm_finish，继续生成直到 chunk_eos/listen
                    }
                }
                
                if (is_end_token(ctx_omni, sampled_token)){
                    llm_finish = true;
                    
                    // 🔧 [与 Python 对齐] 设置 llm_generation_done 标志
                    // TTS 线程会检查这个标志来决定是否添加 text_eos_embed
                    if (!ctx_omni->duplex_mode) ctx_omni->llm_generation_done.store(true);
                    print_with_timestamp("LLM: detected end token, set llm_generation_done=true\n");
                    
                    // 🔧 [P1-双工模式] 设置 current_turn_ended 状态
                    // Python: end_of_turn = last_id in turn_terminator_token_ids (只有 turn_eos)
                    // 只有 TURN_EOS 和 TTS_EOS 才标记轮次真正结束
                    // CHUNK_EOS/CHUNK_TTS_EOS 只是 chunk 结束，轮次未结束
                    // LISTEN 只是暂时切换到听状态，用户说完后模型还要继续回复
                    // 这样 TTS KV cache 才能在多轮 speak-listen-speak 中保持连续
                    if (token_type == OmniTokenType::TURN_EOS || 
                        token_type == OmniTokenType::TTS_EOS ||
                        token_type == OmniTokenType::EOS) {
                        ctx_omni->current_turn_ended = true;
                    } else if (token_type == OmniTokenType::LISTEN) {
                        // LISTEN: 不设置 current_turn_ended，保持 TTS 状态连续
                    }
                    
                    // 🔧 [P1-双工模式] <|listen|> token 特殊处理：
                    // - 在双工模式下，<|listen|> 表示模型主动切换到听状态
                    // - 需要通过 text_queue 通知 SSE 客户端
                    // - 设置 ended_with_listen 标志，让 stream_decode 末尾不清理 KV cache
                    if (token_type == OmniTokenType::LISTEN && ctx_omni->duplex_mode) {
                        
                        // 🔧 [关键] 标记以 listen 结束，不清理 KV cache
                        ctx_omni->ended_with_listen = true;
                        
                        // 推送一个特殊的 JSON 标记到 text_queue，SSE 会转发给客户端
                        if (ctx_omni->async) {
                            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
                            // 使用特殊前缀标记这是状态消息而非文本
                            ctx_omni->text_queue.push_back("__IS_LISTEN__");
                            ctx_omni->text_cv.notify_all();
                        }
                    }
                    
                    // Don't add end tokens to response
                    break;
                }

                // Copy tmp to a local string immediately to avoid issues with static string
                std::string tmp_str(tmp);
                response += tmp_str;
                fflush(stdout);
            }
            fflush(stdout);
            fflush(stdout);
        }
        fflush(stdout);
        
        // 🔧 [P2-chunk限制] 如果达到 chunk 限制，结束当前 decode
        // 这与 Python 双工 server 行为一致：每次 generate 只返回一个 chunk
        // 客户端需要再次调用 stream_decode 获取下一个 chunk
        if (chunk_limit_reached) {
            
            // 🔧 [P0-修复] 与 Python 对齐：达到 chunk 限制时，强制添加 <|chunk_eos|> token
            // Python: self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                // Feed chunk_eos token to model (update KV cache)
                std::vector<llama_token> chunk_eos_tokens = {ctx_omni->special_token_chunk_eos};
                eval_tokens(ctx_omni, ctx_omni->params, chunk_eos_tokens, 
                           ctx_omni->params->n_batch, &ctx_omni->n_past);
            }
            // 这样 SSE 流会结束，客户端可以再次调用 decode
            llm_finish = true;
            // 注意：不重置 current_chunk_tokens，下次 decode 会从 0 开始
            current_chunk_tokens = 0;
        }
        
        // add </unit> token after each chunk
        if (ctx_omni->duplex_mode && ctx_omni->special_token_unit_end >= 0) {
            std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
            // Feed </unit> token to model (update KV cache)
            std::vector<llama_token> unit_end_tokens = {ctx_omni->special_token_unit_end};
            eval_tokens(ctx_omni, ctx_omni->params, unit_end_tokens, 
                       ctx_omni->params->n_batch, &ctx_omni->n_past);
        }
        fflush(stdout);
        if (!response.empty()) {
            fflush(stdout);
        } else {
            fflush(stdout);
        }
        fflush(stdout);
        if (il == 0){
        }
        // 🔧 使用总生成的 token 数量更新 il（用于和 max_tgt_len 比较）
        il += total_tokens_generated;
        
        // 🔧 [统一处理] 移除响应中的所有特殊结束 token
        // 注意: <|speak|> 不是结束 token，而是开始说话的标记，应该被移除但不截断后面内容
        {
            static const std::vector<std::string> end_token_strings = {
                "<|tts_eos|>",
                "</s>",
                "<|listen|>",
                "<|turn_eos|>",
                "<|chunk_eos|>",
                "<|chunk_tts_eos|>"
                // 注意：移除了 <|speak|>，它不是结束 token
            };
            
            // 对于结束 token，截断其后的内容
            for (const auto& delimiter : end_token_strings) {
                size_t end = response.find(delimiter);
                if (end != std::string::npos) {
                    response = response.substr(0, end);
                }
            }
            
            // 🔧 [特殊处理] <|speak|> 是开始标记，直接移除它（不截断后面内容）
            size_t speak_pos = response.find("<|speak|>");
            while (speak_pos != std::string::npos) {
                response.erase(speak_pos, std::string("<|speak|>").length());
                speak_pos = response.find("<|speak|>");
            }
        }
        fflush(stdout);
        if (ctx_omni->async){
            fflush(stdout);
            
            // push text fragment for server stream
            if (!response.empty()) {
                fflush(stdout);
                std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
                ctx_omni->text_queue.push_back(response);
                ctx_omni->text_cv.notify_all();
                fflush(stdout);
            }
            fflush(stdout);
            
            if (ctx_omni->use_tts && ctx_omni->tts_thread_info && (!response.empty() || llm_finish)) {
                // LLM chunk timing
                auto llm_chunk_time = std::chrono::high_resolution_clock::now();
                auto llm_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    llm_chunk_time - ctx_omni->stream_decode_start_time).count();
                fflush(stdout);
                LLMOut * llm_out = new LLMOut();
                llm_out->text = response;
                llm_out->n_past = ctx_omni->n_past;
                llm_out->llm_finish = llm_finish;
                llm_out->debug_dir = debug_dir;
                // 填充token IDs和hidden states用于TTS条件生成
                llm_out->token_ids = chunk_token_ids;
                llm_out->hidden_states = chunk_hidden_states;
                llm_out->n_embd = llm_n_embd;
                // 🔧 [修复双工缺字问题] 传递 is_end_of_turn 状态
                // 此状态随数据一起传递，确保 TTS 处理的是与当前 chunk 对应的状态
                llm_out->is_end_of_turn = local_is_end_of_turn;
                
                // 🔧 [诊断日志] 打印 LLM 推送给 TTS 的数据
                {
                    std::string token_ids_str = "";
                    for (size_t i = 0; i < chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(chunk_token_ids[i]);
                        if (i < chunk_token_ids.size() - 1 && i < 19) token_ids_str += " ";
                    }
                    if (chunk_token_ids.size() > 20) token_ids_str += "...";
                    
                    print_with_timestamp("LLM->TTS: text='%s', n_tokens=%zu, hidden_size=%zu, n_embd=%d, token_ids=[%s]\n",
                                        response.c_str(),
                                        chunk_token_ids.size(),
                                        chunk_hidden_states.size(),
                                        llm_n_embd,
                                        token_ids_str.c_str());
                }
                
                fflush(stdout);
                fflush(stdout);
                std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                fflush(stdout);
                ctx_omni->tts_thread_info->cv.wait(lock, [&] { return ctx_omni->tts_thread_info->queue.size() < ctx_omni->tts_thread_info->MAX_QUEUE_SIZE; });
                fflush(stdout);
                fflush(stdout);
                // 🔧 [关键修复 - 与 Python 对齐] 在双工模式下，LLM 始终推送数据到 TTS 队列
                // 因为双工模式下每个 chunk 都需要独立处理，不能因为上一个 chunk 完成（speek_done=true）就丢弃新数据
                // Python 双工模型：每次 streaming_generate 调用都会独立处理 LLM 输出并生成 TTS
                // 只有在非双工模式下，才检查 speek_done 来避免重复处理
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode){
                    fflush(stdout);
                    ctx_omni->tts_thread_info->queue.push(llm_out);
                    fflush(stdout);
                    //notify the tts thread
                    ctx_omni->tts_thread_info->cv.notify_all();
                    fflush(stdout);
                } else {
                    // speek_done is true, delete llm_out to prevent memory leak
                    fflush(stdout);
                    delete llm_out;
                    llm_out = nullptr;
                    fflush(stdout);
                }
                fflush(stdout);
            }
            fflush(stdout);
            if (llm_finish) break;
            fflush(stdout);
        }else{
            fflush(stdout);
        }
        fflush(stdout);
    }
    fflush(stdout);
    // 🔧 [P1-SSE响应] 推送轮次结束标记
    // mark text done
    {
        std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
        // 推送 end_of_turn 标记，让客户端知道当前轮次结束
        if (!ctx_omni->duplex_mode || !ctx_omni->ended_with_listen) {
            ctx_omni->text_queue.push_back("__END_OF_TURN__");
        }

        ctx_omni->text_done_flag = true;
        ctx_omni->text_cv.notify_all();
        ctx_omni->text_streaming = false;
    }
    // Safety checks before cleanup
    if (ctx_omni == nullptr) {
        LOG_ERR("stream_decode: ctx_omni is nullptr in cleanup!");
        return false;
    }
    if (ctx_omni->ctx_llama == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->ctx_llama is nullptr in cleanup!");
        return false;
    }
    if (ctx_omni->params == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->params is nullptr in cleanup!");
        return false;
    }
    
    // ==================== 轮次边界记录与滑窗检查 ====================
    // 🔧 [单工多轮对话] 在 decode 结束时：
    // 1. 先检查并执行滑窗（基于之前的轮次边界）
    // 2. 再记录当前轮次的结束边界（作为下一轮的开始位置）
    if (!ctx_omni->duplex_mode) {
        // 🔧 [滑窗检查] 检查是否需要执行滑窗，确保下一轮有足够空间
        // 当 n_past > n_ctx - reserved_space 时执行滑窗
        const int reserved_space = 1024;  // 预留空间
        const int n_ctx = ctx_omni->params->n_ctx;
        
        if (ctx_omni->n_past > n_ctx - reserved_space) {
            print_with_timestamp("⚠️ Decode 结束滑窗检查: n_past=%d > n_ctx-reserved=%d，需要滑窗\n",
                                 ctx_omni->n_past, n_ctx - reserved_space);
            
            // 调用滑窗函数，传入 reserved_space 作为需要腾出的空间
            kv_cache_slide_window(ctx_omni, ctx_omni->params, reserved_space);
        } else {
            print_with_timestamp("📍 Decode 结束: n_past=%d, 剩余空间=%d, 无需滑窗\n",
                                 ctx_omni->n_past, n_ctx - ctx_omni->n_past);
        }
        
        // 🔧 [轮次边界] 记录当前轮次的结束位置（也是下一轮的开始位置）
        // 一个完整轮次 = 用户提问（可能多个 audio prefill）+ 模型回答
        // 这样按轮次删除时，可以删除完整的"问答对"
        ctx_omni->round_start_positions.push_back(ctx_omni->n_past);
        print_with_timestamp("📍 轮次 %zu 结束，记录边界于 n_past=%d\n",
                             ctx_omni->round_start_positions.size(), ctx_omni->n_past);
        
        // 🔧 [整合] 为下一轮准备 <|im_end|>\n<|im_start|>user\n
        // 第一轮的 <|im_start|>user\n 在 sys prompt 末尾
        // 后续轮次需要在 decode 结束时添加，结束当前 assistant 回复并开始新一轮 user 输入
        eval_string(ctx_omni, ctx_omni->params, "<|im_end|>\n<|im_start|>user\n", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        print_with_timestamp("📍 为下一轮准备: eval <|im_end|>\\n<|im_start|>user\\n, n_past=%d\n", ctx_omni->n_past);
    }
    
    // 🔧 [双工模式] 在双工模式下永远不清理 KV cache
    // Python 双工模型的 llm_past_key_values 一直累积，只在 reset_session() 时清空
    // C++ 已有滑动窗口机制 (kv_cache_slide_window)，会在上下文满时自动滑动
    if (ctx_omni->duplex_mode) {
        // 不调用 clean_kvcache 和 eval_prefix，保留当前上下文
        // 滑动窗口机制会在 prefill_with_emb/eval_tokens 时自动触发
    } else {
        // 非双工模式（单工），每轮对话后清理 KV cache
        // clean_kvcache(ctx_omni);
        // eval_prefix(ctx_omni, ctx_omni->params);
    }
    
    return true;
}

bool stop_speek(struct omni_context * ctx_omni){
    ctx_omni->speek_done = true;
    if (ctx_omni->use_tts && ctx_omni->tts_thread_info) {
        std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
        while(!ctx_omni->tts_thread_info->queue.empty()){
            LLMOut *llm_out = ctx_omni->tts_thread_info->queue.front();
            if (llm_out) {
                delete llm_out;
                llm_out = nullptr;
            }
            ctx_omni->tts_thread_info->queue.pop();
        }
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    return true;
}

bool clean_kvcache(struct omni_context * ctx_omni) {
    
    if (ctx_omni->clean_kvcache) {
        print_with_timestamp("🧹 clean_kvcache: 清理 KV cache, 删除范围=[%d, %d), n_keep=%d\n",
                             ctx_omni->n_keep, ctx_omni->n_past, ctx_omni->n_keep);
        
        // 获取 memory 对象并清理 KV cache
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
        if (mem) {
            // 删除 [n_keep, n_past) 范围的所有 token，保留 system prompt 等
            bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, ctx_omni->n_past);
            if (!rm_ok) {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 失败\n");
            } else {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 成功\n");
            }
        } else {
            print_with_timestamp("🧹 clean_kvcache: 无法获取 memory 对象\n");
        }
        
        // 重置 n_past 到 n_keep
        int old_n_past = ctx_omni->n_past;
        ctx_omni->n_past = ctx_omni->n_keep;
        print_with_timestamp("🧹 clean_kvcache: n_past 从 %d 重置到 %d\n", old_n_past, ctx_omni->n_past);
        
        // 🔧 [#39 滑动窗口] 重置滑窗状态
        sliding_window_reset(ctx_omni);
        // 但保留 system_preserve_length，因为 n_keep 部分仍然保留
        ctx_omni->system_preserve_length = ctx_omni->n_keep;
        print_with_timestamp("🧹 clean_kvcache: 滑窗状态已重置, system_preserve_length=%d\n", ctx_omni->system_preserve_length);
    } else {
        print_with_timestamp("🧹 clean_kvcache: clean_kvcache=false, 跳过清理\n");
    }
    
    return true;
}
