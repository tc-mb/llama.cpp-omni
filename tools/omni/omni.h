#include "ggml.h"
#include "llama.h"

#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <atomic>

struct vision_ctx;
struct audition_ctx;
struct audition_audio_f32;

// Forward declaration for C++ Token2Wav
namespace omni {
namespace flow {
class Token2WavSession;
}
}

//
// omni ctx
//
struct omni_embed {
    float * embed;
    int n_pos;
};
struct omni_embeds{
    // 🔧 [高清模式] vision_embed 改为二维 vector
    // vision_embed[0] = overview embed (64 tokens * hidden_size)
    // vision_embed[1..n] = slice embeds (各 64 tokens * hidden_size)
    std::vector<std::vector<float>> vision_embed;
    std::vector<float> audio_embed;
    int index = 0;
    int end_flag = false;
};

struct LLMThreadInfo {
    int MAX_QUEUE_SIZE;
    std::queue<omni_embeds*> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    LLMThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};

struct T2WOut {
    std::vector<llama_token> audio_tokens;  // Audio token IDs (25 tokens per chunk)
    bool is_final = false;  // Whether this is the final chunk (turn end)
    bool is_chunk_end = false;  // Whether this is the end of a TTS chunk (flush buffer, but not final)
    int round_idx = -1;  // 🔧 [修复目录同步] 轮次索引，由 TTS 线程设置，T2W 线程使用此值确定输出目录
};

struct T2WThreadInfo {
    int MAX_QUEUE_SIZE;
    std::queue<T2WOut*> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    T2WThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};

// Projector Semantic: 2-layer MLP (LLM hidden states -> TTS embedding)
// forward(x): relu(linear1(x)) -> linear2
// ==================== 滑动窗口配置 ====================
// 🔧 [#39] 基于 Python stream_decoder.py 的 DuplexWindowConfig
struct SlidingWindowConfig {
    // 滑窗模式: "off" / "basic" / "context"
    // - "off": 禁用滑窗
    // - "basic": 基础滑窗（按 cache 长度触发）
    // - "context": 带 context 的滑窗（保留生成文本到 previous）
    std::string mode = "off";
    
    // 基础滑窗参数
    int high_water_tokens = 4000;  // 高水位线：超过此值触发滑窗
    int low_water_tokens = 3500;   // 低水位线：滑窗后保留到此值
    
    // RoPE 参数
    float rope_theta = 10000.0f;   // RoPE base frequency
};

// Unit 历史记录条目
struct UnitEntry {
    int unit_id = -1;              // Unit ID
    int length = 0;                // 该 unit 在 cache 中的长度（tokens 数）
    std::string type;              // 类型: "audio" / "video" / "omni" / "system"
    std::vector<llama_token> generated_tokens;  // 生成的 tokens
    bool is_listen = false;        // 是否是 listen 状态
};

struct projector_hparams {
    int32_t in_dim  = 4096;  // 输入维度 (LLM hidden size)
    int32_t out_dim = 768;   // 输出维度 (TTS embedding size)
};

struct projector_layer {
    struct ggml_tensor * linear1_weight = nullptr;  // [in_dim, out_dim]
    struct ggml_tensor * linear1_bias   = nullptr;  // [out_dim]
    struct ggml_tensor * linear2_weight = nullptr;  // [out_dim, out_dim]
    struct ggml_tensor * linear2_bias   = nullptr;  // [out_dim]
};

struct projector_model {
    projector_hparams hparams;
    projector_layer layer;
    
    struct ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buf_type = nullptr;
    bool initialized = false;
};

struct omni_context {
    struct vision_ctx * ctx_vision = NULL;
    struct audition_ctx * ctx_audio = NULL;
    
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
    struct common_sampler * ctx_sampler = NULL;
    
    // 🔧 [单双工适配] 是否拥有模型（用于 omni_free 时决定是否释放模型）
    // true: omni_init 内部加载的模型，omni_free 时需要释放
    // false: 外部传入的已有模型（模型复用），omni_free 时不释放
    bool owns_model = true;

    // 🔧 [Length Penalty] 用于调整 EOS token 的采样概率
    // length_penalty > 1.0 会降低 EOS 概率，让模型生成更长的输出
    // length_penalty < 1.0 会增加 EOS 概率，让模型更早结束
    float length_penalty = 1.0f;

    struct llama_context * ctx_tts_llama = NULL;
    struct llama_model * model_tts = NULL;
    struct common_sampler * ctx_tts_sampler = NULL;
    
    // struct TTSContext * ctx_tts = NULL;
    struct vocal_ctx * vocal = NULL;
    std::shared_ptr<std::vector<float>> spk_embeds;
    std::vector<float> audio_emb;
    std::vector<float> omni_emb;    
    int output_audio_round_per_text[5] = {16, 8, 4, 2, 2};
    int output_audio_chunk_size[5] = {5, 10, 20, 40, 40};
    
    struct omni_output *omni_output = NULL;
    int n_past = 0;
    int n_keep = 0;
    
    // ==================== 轮次边界管理（用于智能滑动窗口） ====================
    // 每轮对话开始时的 n_past 位置
    // round_start_positions[i] 表示第 i 轮开始的 n_past 位置
    // 第 i 轮的范围是 [round_start_positions[i], round_start_positions[i+1])
    // 最后一轮的结束位置是当前 n_past
    std::vector<int> round_start_positions;
    
    // 滑动窗口保留的最大上下文长度（不包括 n_keep）
    // 设置为 0 表示使用旧的按比例删除策略
    int max_preserved_context = 2048;
    
    // ==================== 滑动窗口状态 (#39) ====================
    SlidingWindowConfig sliding_window_config;
    
    // Unit 历史管理（用于按 unit 粒度删除）
    std::vector<UnitEntry> unit_history;
    int next_unit_id = 0;
    int pending_unit_id = -1;           // 当前正在处理的 unit ID
    int pending_unit_start_cache_len = 0;  // pending unit 开始时的 cache 长度
    
    // System prompt 保护长度（这部分永远不会被滑窗删除）
    int system_preserve_length = 0;
    
    // RoPE 位置偏移（用于 RoPE 位置重对齐后的 position_ids 计算）
    int position_offset = 0;
    
    // 滑窗统计
    int sliding_event_count = 0;    // 滑窗触发次数
    int total_dropped_tokens = 0;   // 总共丢弃的 token 数
    int total_dropped_units = 0;    // 总共丢弃的 unit 数
    
    bool async = false;
    std::thread llm_thread;
    std::thread tts_thread;
    std::thread t2w_thread;
    struct LLMThreadInfo *llm_thread_info = NULL;
    struct TTSThreadInfo *tts_thread_info = NULL;
    struct T2WThreadInfo *t2w_thread_info = NULL;
    
    volatile bool need_speek = false;
    volatile bool speek_done = true;
    
    // 预热标志：第一轮对话视为预热（例如音色克隆参考音频），完成后设为 true
    std::atomic<bool> warmup_done{false};
    
    // ==================== 双工模式状态 ====================
    // 当前轮次是否已结束（用于决策是否允许切换到 listen 状态）
    // Python: self.current_turn_ended
    bool current_turn_ended = true;
    
    // 打断事件标志
    // break_event: 打断当前生成，但保持会话活跃（用于双工模式的用户打断）
    //              打断后可继续调用 prefill/decode
    std::atomic<bool> break_event{false};
    
    // session_stop_event: 终止整个会话（预留，目前未使用）
    //                     用于彻底关闭当前会话，需要重新 omni_init
    std::atomic<bool> session_stop_event{false};
    
    // 🔧 [双工模式] 记录当前 decode 是否以 <|listen|> 结束
    // 如果是，则不清理 KV cache，让下一个音频片段可以累积上下文
    std::atomic<bool> ended_with_listen{false};
    
    // 🔧 [与 Python 对齐] LLM 生成结束标志
    // 当 LLM 检测到 end token 时设置为 true
    // TTS 线程检查此标志来决定是否添加 text_eos_embed
    std::atomic<bool> llm_generation_done{false};
    
    // ==================== 双工模式参数 ====================
    // 每个 chunk 最大生成 token 数（用于限制单次 speak 长度，便于及时响应打断）
    // 设置为 0 表示无限制
    int max_new_speak_tokens_per_chunk = 26;
    
    // listen_prob_scale: 调整 <|listen|> token 的采样概率
    // 1.0: Python 默认
    float listen_prob_scale = 1.0f;
    
    // 是否启用双工模式
    // simplex: 单工模式，用户说完后模型回复，回复完用户再说
    // duplex: 双工模式，模型可以在任意时刻决定听/说切换
    bool duplex_mode = false;
    
    // 系统 prompt 是否已初始化（防止 stream_prefill index=0 被重复调用导致 prompt 重复）
    bool system_prompt_initialized = false;
    
    class AudioInputManager * audio_input_manager = NULL;
    
    // models path and other configs
    struct common_params * params = NULL;
    
    // 当前是以「语音通话」还是「视频通话」模式进入的，0 = 语音，1 = 视频；
    int media_type = 0;
    int use_tts = false;
    std::string tts_bin_dir = "";
    std::string ref_audio_path = "";  // 参考音频路径（用于音色克隆）
    
    // 🔧 [高清/高刷模式] 
    // high_image: 高清模式，max_slice_nums 设置为 2，vision 可以看到更多细节
    // high_refresh: 高刷模式，1秒5帧，第1帧作为主图，后4帧stack合并成一张图
    //               注意：stack 处理在 Python server 层实现，C++ 只是标记
    bool high_image = false;
    bool high_refresh = false;
    
    // 🔧 [多实例支持] 可配置的输出目录，避免多个服务实例冲突
    std::string base_output_dir = "./tools/omni/output";
    
    // 每次会话，是否清除 kv cache（默认开启自动清理 kv cache）
    bool clean_kvcache = true;
    
    std::string omni_voice_clone_prompt = "";
    std::string omni_assistant_prompt = "";
    std::string audio_voice_clone_prompt = "";
    std::string audio_assistant_prompt = "";
    
    // 语言设置 (用于 prompt 生成)
    std::string language = "zh";

    // text streaming queue for server
    std::mutex text_mtx;
    std::condition_variable text_cv;
    std::deque<std::string> text_queue;
    bool text_streaming = false;
    bool text_done_flag = false;

    // llama inference mutex - 保护 ctx_llama 的推理操作
    std::mutex llama_mtx;
    
    // TTS weights loaded from GGUF file
    // emb_code: (num_audio_tokens=6562, hidden_size=768) - for converting audio token IDs to embeddings
    float * emb_code_weight = nullptr;
    int emb_code_vocab_size = 0;  // num_audio_tokens = 6562
    int emb_code_hidden_size = 0; // hidden_size = 768
    bool emb_code_stored_as_transposed = false; // true if stored as [hidden_size, num_audio_tokens] = [768, 6562]
    
    // emb_text: (vocab_size=152064, hidden_size=768)
    float * emb_text_weight = nullptr;
    int emb_text_vocab_size = 0;
    int emb_text_hidden_size = 0;
    
    // projector_semantic: two-layer MLP (4096 -> 768 -> 768)
    // Legacy float* weights (kept for backward compatibility)
    float * projector_semantic_linear1_weight = nullptr;  // (4096, 768)
    float * projector_semantic_linear1_bias = nullptr;   // (768,)
    float * projector_semantic_linear2_weight = nullptr; // (768, 768)
    float * projector_semantic_linear2_bias = nullptr;  // (768,)
    int projector_semantic_input_dim = 0;  // 4096
    int projector_semantic_output_dim = 0;  // 768
    
    // New ggml-based projector model (精度验证版本)
    struct projector_model projector;
    
    // head_code: Linear layer (hidden_size=768 -> num_audio_tokens=6562)
    // Note: num_vq=1, so we only need one head_code layer
    float * head_code_weight = nullptr;  // (768, 6562) - stored as (hidden_size, num_audio_tokens)
    int head_code_hidden_size = 0;  // 768
    int head_code_num_audio_tokens = 0;  // 6562
    
    // TTS condition embeddings (for first audio token re-forward)
    // Used to store the condition embeddings so we can re-forward them for the first audio token
    // This ensures KV cache state matches Python's behavior (past_key_values=None on first forward)
    std::vector<float> tts_condition_embeddings;  // Condition embeddings (n_tokens * n_embd)
    int tts_condition_length = 0;  // Number of tokens in condition
    int tts_condition_n_embd = 0;  // Embedding dimension (768)
    bool tts_condition_saved = false;  // Whether condition has been saved
    
    // 🔧 TTS KV cache 累计位置（用于保持跨 chunk 的上下文连续性）
    // Python TTSStreamingGenerator 使用 text_start_pos 来跟踪位置
    int tts_n_past_accumulated = 0;
    
    // 🔧 [关键修复] TTS 已生成的所有 audio tokens（跨 chunk 累积）
    // Python: self.all_generated_tokens 是类成员变量，跨 chunk 持续累积
    // 用于：1. RAS 重复检测（需要完整历史）2. 正确判断 audio_bos（只有第一个 token 才是）
    std::vector<llama_token> tts_all_generated_tokens;
    
    // 🔧 [与 Python 对齐] TTS audio token buffer（跨 text chunk 累积）
    // Python: self._token_buffer 是类成员变量，用于累积 audio token
    // 只有满足 chunk_size (25) 才会 yield，不足的保留到下一个 text chunk
    std::vector<int32_t> tts_token_buffer;
    
    // Timestamp for stream_decode start (used for WAV file naming)
    std::chrono::high_resolution_clock::time_point stream_decode_start_time;
    
    // C++ Token2Wav session for audio synthesis
    std::unique_ptr<omni::flow::Token2WavSession> token2wav_session;
    bool token2wav_initialized = false;
    std::string token2wav_model_dir;  // Directory containing token2wav GGUF models
    
    // 🔧 [Python Token2Wav] 使用 Python stepaudio2 库实现的 Token2Wav
    // 设置为 true 时使用 Python 实现（精度更高），false 时使用 C++ 实现
    // macOS 上默认使用 C++ 实现（无 CUDA）
    bool use_python_token2wav = false;
    std::string python_t2w_script_dir;  // Python Token2Wav 脚本目录
    std::string python_t2w_model_dir;   // Python Token2Wav 模型目录
    
    // Python Token2Wav 服务进程 (通过 popen 启动)
    FILE* python_t2w_stdin = nullptr;   // 写入命令
    FILE* python_t2w_stdout = nullptr;  // 读取响应
    pid_t python_t2w_pid = -1;          // 进程 ID
    bool python_t2w_initialized = false;
    std::string python_t2w_gpu_id;      // GPU ID (如 "0", "1")
    
    // 🔧 Python T2W 独立 GPU 配置
    // C++ LLM+TTS 占用约 22GB，Python T2W 占用约 3.3GB
    // 单卡 24GB 放不下，需要使用独立 GPU
    // 设置为空字符串表示使用与 C++ 相同的 GPU
    std::string python_t2w_dedicated_gpu = "";  // 独立 GPU ID，如 "1"
    
    // Token2Wav sliding window buffer (跨 chunk 保持状态)
    // Python 逻辑: buffer 初始填充 3 个静音 token (4218)
    // 每次取 28 个 tokens (25 main + 3 lookahead)，处理后移动 25 个，保留 3 个重叠
    std::vector<int32_t> token2wav_buffer;
    int token2wav_wav_idx = 0;  // 输出 WAV 文件计数器
    int wav_turn_base = 0;      // 每轮对话结束时 +1000，用于区分不同轮次的 WAV 文件
    
    // 🔧 [单工模式] 当前轮次索引（用于创建 round_000、round_001 等子目录）
    int simplex_round_idx = 0;
    
    // ==================== 特殊 Token ID ====================
    // 在 omni_init 时从词表查找并缓存
    llama_token special_token_speak = -1;        // <|speak|>: 模型开始说话
    llama_token special_token_listen = -1;       // <|listen|>: 模型开始听（双工）
    llama_token special_token_chunk_eos = -1;    // <|chunk_eos|>: 语义 chunk 结束
    llama_token special_token_chunk_tts_eos = -1;// <|chunk_tts_eos|>: TTS chunk 结束
    llama_token special_token_turn_eos = -1;     // <|turn_eos|>: 轮次结束
    llama_token special_token_tts_eos = -1;      // <|tts_eos|>: 旧版 TTS 结束
    llama_token special_token_eos = -1;          // </s>: 序列结束
    llama_token tts_bos_token_id = -1;           // <|tts_bos|>: TTS 开始（用于双工强制继续说话）
    llama_token special_token_unit_end = -1;     // </unit>: unit 结束标记（双工 chunk 边界）
    llama_token special_token_tts_pad = -1;      // <|tts_pad|>: TTS 填充（双工模式下禁止采样）
};

//
// omni embed
//
bool prefill_with_emb(struct omni_context * ctx_omni, struct common_params * params, float* embed, int n_pos, int n_batch, int* n_past);
bool prefill_emb_with_hidden(struct omni_context * ctx_omni, struct common_params * params, float* embed, int n_pos, int n_batch, int* n_past, float *& hidden_states);
bool omni_eval_embed(struct llama_context * ctx_llama, const struct omni_embed * embed, int n_batch, int * n_past);
void omni_embed_free(struct omni_embed * embed);
struct omni_embed * omni_image_embed_make_with_bytes(struct vision_ctx * ctx_vision, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision, int n_threads, std::string image_path);
struct omni_embed * omni_audio_embed_make_with_bytes(struct audition_ctx * ctx_audition, int n_threads, audition_audio_f32 * audio);
struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audition, int n_threads, std::string audio_path);

//
// omni main
//
struct omni_context * omni_init(struct common_params * params, int media_type, bool use_tts, std::string tts_bin_dir,
                                int tts_gpu_layers = -1, const std::string & token2wav_device = "gpu:0",
                                bool duplex_mode = false,
                                llama_model * existing_model = nullptr, llama_context * existing_ctx = nullptr,
                                const std::string & base_output_dir = "./tools/omni/output");

void omni_free(struct omni_context * ctx_omni);

// ANE/CoreML warmup — call once after omni_init to pre-load models into NPU
void omni_warmup_ane(struct omni_context * ctx_omni);

// 停止所有线程（在 join 之前调用）
void omni_stop_threads(struct omni_context * ctx_omni);

bool stream_prefill(struct omni_context * ctx_omni,
                            std::string aud_fname,
                            std::string img_fname = "",
                            int index = 0,
                            int max_slice_nums = -1);  // -1 表示使用全局设置，>=1 表示本次 prefill 的 slice 数量

bool stream_decode(struct omni_context * ctx_omni,
                        std::string debug_dir,
                        int round_idx = -1);  // round_idx: 由调用方指定的轮次索引，-1 表示使用内部计数

bool stop_speek(struct omni_context * ctx_omni);

bool clean_kvcache(struct omni_context * ctx_omni);

// TTS 推理函数声明（用于 test_tts_inference.cpp）
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path);
bool prefill_with_emb_tts(struct omni_context* ctx_omni, common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts);
// sample_tts_token 参数说明：
// - all_generated_tokens: 跨 chunk 累积的所有 tokens（用于判断是否是整个过程的第一个 token，即 re-forward condition）
// - chunk_generated_tokens: 当前 chunk 内已生成的 tokens（用于 repetition penalty，与 Python generate_chunk 对齐）
// - token_index_in_chunk: 当前 chunk 内的 token 索引（用于判断是否跳过 sampling processors）
// - force_no_eos: 是否强制阻止 EOS token 被采样（用于 min_new_tokens 逻辑，与 Python generate_chunk 对齐）
llama_token sample_tts_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past_tts, const std::vector<llama_token> * all_generated_tokens = nullptr, const std::vector<llama_token> * chunk_generated_tokens = nullptr, int token_index_in_chunk = 0, bool force_no_eos = false);

// Projector 函数声明（精度验证版本）
bool projector_init(projector_model & model, const std::string & fname, bool use_cuda);
void projector_free(projector_model & model);
std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens);

// ==================== 滑动窗口函数声明 (#39) ====================
// Unit 管理
int sliding_window_register_unit_start(struct omni_context * ctx_omni);
void sliding_window_register_unit_end(struct omni_context * ctx_omni, const std::string & input_type, 
                                      const std::vector<llama_token> & generated_tokens = {}, bool is_listen = false);
void sliding_window_register_system_prompt(struct omni_context * ctx_omni);

// 滑窗执行
bool sliding_window_enforce(struct omni_context * ctx_omni);
bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length);
void sliding_window_reset(struct omni_context * ctx_omni);

// ==================== 高清模式函数声明 ====================
// 设置 vision max_slice_nums 覆盖值，用于高清模式
void vision_set_max_slice_nums(struct vision_ctx * ctx_vision, int max_slice_nums);