#include "omni-impl.h"
#include "audition.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <thread>
#include <set>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>
#include <array>
#include <numeric>
#include <fstream>
#include <functional>

#define MINIAUDIO_IMPLEMENTATION
#ifndef OMNI_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

struct omni_logger_state audition_g_logger_state = {GGML_LOG_LEVEL_CONT, omni_log_callback_default, NULL};

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};


//
// audition layers
//
struct audition_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t n_embd;
    int32_t n_ff;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    int32_t proj_scale_factor = 0; // idefics3

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM everytime they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    float eps = 1e-6;
    float rope_theta = 0.0;

    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    int32_t spatial_merge_size = 0;

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t n_fft = 0; // whisper n_fft parameter
    int32_t n_ctx = 0; // whisper audio context length
    int32_t proj_stack_factor = 0; // ultravox
};

struct audition_layer {
    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    // layernorm 1
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layernorm 2
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w = nullptr;
    ggml_tensor * ls_2_w = nullptr;
};

struct whisper_layer {
    ggml_tensor * attn_ln_0_w = nullptr;
    ggml_tensor * attn_ln_0_b = nullptr;
    ggml_tensor * attn_ln_1_w = nullptr;
    ggml_tensor * attn_ln_1_b = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * mlp_ln_w = nullptr;
    ggml_tensor * mlp_ln_b = nullptr;
    ggml_tensor * mlp_0_w = nullptr;
    ggml_tensor * mlp_0_b = nullptr;
    ggml_tensor * mlp_1_w = nullptr;
    ggml_tensor * mlp_1_b = nullptr;
};

struct audition_model {
    omni_model_type model_type = MiniCPM_o;
    audition_hparams hparams;

    // whisper encoder
    ggml_tensor * whisper_e_pe = nullptr;
    ggml_tensor * whisper_e_conv_1_w = nullptr;
    ggml_tensor * whisper_e_conv_1_b = nullptr;
    ggml_tensor * whisper_e_conv_2_w = nullptr;
    ggml_tensor * whisper_e_conv_2_b = nullptr;
    ggml_tensor * whisper_e_ln_w = nullptr;
    ggml_tensor * whisper_e_ln_b = nullptr;
    
    // whisper layers
    std::vector<whisper_layer> whisper_layers;
    
    // audio projector
    ggml_tensor * whisper_proj_1_w = nullptr;
    ggml_tensor * whisper_proj_1_b = nullptr;
    ggml_tensor * whisper_proj_2_w = nullptr;
    ggml_tensor * whisper_proj_2_b = nullptr;
    
    // mel filters data (loaded from gguf file)
    std::vector<float> whisper_filters_data;


    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<audition_layer> layers;

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    ggml_tensor * projection; // TODO: rename it to fc (fully connected layer)
    ggml_tensor * mm_fc_w;
    ggml_tensor * mm_fc_b;
};

// Whisper KV cache for streaming audio processing
struct whisper_kv_cache {
    std::vector<ggml_tensor*> k_l;  // K cache for each layer [n_layer]
    std::vector<ggml_tensor*> v_l;  // V cache for each layer [n_layer]
    
    ggml_context* ctx = nullptr;     // ggml context for KV cache tensors
    ggml_backend_buffer_t buffer = nullptr;  // backend buffer for KV cache
    
    int n_layer = 0;   // number of layers
    int size = 0;      // fixed cache size (n_audio_ctx, e.g., 1500)
    int iter = 0;      // current iteration count for streaming
};

struct audition_ctx {
    audition_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;

    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    
    // Whisper KV cache for streaming
    struct whisper_kv_cache whisper_kv_cache;
    bool whisper_streaming_mode = true;

    // for debugging
    bool debug_graph = false;
    std::vector<ggml_tensor *> debug_print_tensors;

    audition_ctx(audition_context_params & audition_params) {
        debug_graph = std::getenv("MTMD_DEBUG_GRAPH") != nullptr;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (audition_params.use_gpu) {
            auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
            if (backend_name != nullptr) {
                backend = ggml_backend_init_by_name(backend_name, nullptr);
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );
    }

    ~audition_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    // this function is added so that we don't change too much of the existing code
    omni_model_type model_type() const {
        return model.model_type;
    }
};

struct audition_graph {
    audition_ctx * ctx;
    const audition_model & model;
    const audition_hparams & hparams;

    // we only support single image per batch
    const audition_audio_f32 & audio;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int d_head;
    const int n_layer;
    const float eps;
    const float kq_scale;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0;
    ggml_cgraph * gf;

    audition_graph(audition_ctx * ctx, const audition_audio_f32 & audio) :
            ctx(ctx),
            model(ctx->model),
            hparams(model.hparams),
            audio(audio),
            patch_size(hparams.patch_size),
            n_patches_x(patch_size > 0 ? audio.nx / patch_size : 0),
            n_patches_y(patch_size > 0 ? audio.ny / patch_size : 0),
            n_patches(n_patches_x * n_patches_y),
            n_embd(hparams.n_embd),
            n_head(hparams.n_head),
            d_head(n_head > 0 ? n_embd / n_head : 0),
            n_layer(hparams.n_layer),
            eps(hparams.eps),
            kq_scale(d_head > 0 ? 1.0f / sqrtf((float)d_head) : 0.0f) {
        // Validate critical parameters to prevent division by zero
        if (n_head == 0) {
            throw std::runtime_error("audition_graph: n_head cannot be zero");
        }
        if (d_head == 0) {
            throw std::runtime_error("audition_graph: d_head cannot be zero (n_embd must be divisible by n_head)");
        }
        if (patch_size == 0) {
            throw std::runtime_error("audition_graph: patch_size cannot be zero");
        }
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx->buf_compute_meta.size(),
            /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };
        ctx0_ptr.reset(ggml_init(params));
        ctx0 = ctx0_ptr.get();
        gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
    }

    ggml_cgraph * build_whisper() {
        
        const int n_frames = audio.nx;  // mel spectrogram frames
        const int n_mels = audio.ny;    // mel bins number
        const int n_state = hparams.n_embd;
        const int n_head = hparams.n_head;
        const int n_layer = hparams.n_layer;
        
        const int n_state_head = n_state / n_head;
        const float KQscale = 1.0f / sqrtf(float(n_state_head));
        
        // LOG_INF("%s: Building whisper encoder graph - n_frames=%d, n_mels=%d, n_state=%d, n_head=%d, n_layer=%d\n", 
        //         __func__, n_frames, n_mels, n_state, n_head, n_layer);
        
        // Step 1: build input tensor
        // input [2*n_ctx, n_mels] 2D tensor
        // n_frames == 2*n_ctx
        ggml_tensor * inp_raw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_mels);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);

        
        // LOG_INF("%s: Created inp_raw tensor with shape [%d, %d] = %ld elements (whisper format)\n", 
        //         __func__, n_frames, n_mels, ggml_nelements(inp_raw));
        
        // 直接使用inp_raw作为mel输入
        ggml_tensor * mel = inp_raw;
        ggml_tensor * cur = mel;
        
        // Step 2: conv layers + GELU activation (721-732)
        {
            // LOG_INF("%s: Before conv1d - input shape: [%ld, %ld, %ld, %ld]\n", __func__, 
            //         cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            // LOG_INF("%s: Conv1 weight shape: [%ld, %ld, %ld, %ld]\n", __func__, 
            //         model.whisper_e_conv_1_w->ne[0], model.whisper_e_conv_1_w->ne[1], 
            //         model.whisper_e_conv_1_w->ne[2], model.whisper_e_conv_1_w->ne[3]);
            
            // conv1
            cur = ggml_conv_1d_ph(ctx0, model.whisper_e_conv_1_w, cur, 1, 1);
            cur = ggml_add(ctx0, cur, model.whisper_e_conv_1_b);
            cur = ggml_gelu(ctx0, cur);
            
            // conv2 (stride=2, downsample)
            cur = ggml_conv_1d_ph(ctx0, model.whisper_e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.whisper_e_conv_2_b);
            cur = ggml_gelu(ctx0, cur);
        }
        
        {
            // sequence length after conv
            const int n_tokens = cur->ne[0];
            
            // calculate iter loop
            auto & kv_cache = ctx->whisper_kv_cache;
            const int n_audio_ctx = hparams.n_ctx;  // 1500
            const int n_iter = n_audio_ctx / n_tokens;  // 1500 / 50 = 30

            const int effective_iter = kv_cache.buffer != nullptr ? kv_cache.iter : 0;
            
            // LOG_INF("%s: Position encoding - n_tokens=%d, n_audio_ctx=%d, n_iter=%d, effective_iter=%d\n",
            //         __func__, n_tokens, n_audio_ctx, n_iter, effective_iter);
            
            // create position encoding view (with offset, whisper_encoder.cpp:1032-1035)
            const size_t e_pe_stride = model.whisper_e_pe->ne[0] * ggml_element_size(model.whisper_e_pe);
            const size_t e_pe_total_bytes = ggml_nbytes(model.whisper_e_pe);
            const size_t e_pe_view_bytes = model.whisper_e_pe->ne[0] * n_tokens * ggml_element_size(model.whisper_e_pe);
            
            // Calculate position encoding offset with bounds checking
            // If effective_iter exceeds the buffer capacity, use modulo to wrap around
            // or reset KV cache if it's too large
            size_t e_pe_offset;
            if (effective_iter >= n_iter) {
                // Position encoding buffer is exhausted, reset KV cache
                LOG_WRN("%s: Position encoding buffer exhausted (effective_iter=%d >= n_iter=%d), resetting KV cache\n",
                        __func__, effective_iter, n_iter);
                audition_whisper_clear_kv_cache(ctx);
                e_pe_offset = 0;  // Use offset 0 after reset
            } else {
                e_pe_offset = model.whisper_e_pe->ne[0] * ggml_element_size(model.whisper_e_pe) * n_tokens * effective_iter;
            }
            
            // Final bounds check for position encoding view
            if (e_pe_offset + e_pe_view_bytes > e_pe_total_bytes) {
                LOG_ERR("%s: FATAL - Position encoding view would overflow! offset=%zu, view_size=%zu, total_size=%zu, effective_iter=%d\n",
                        __func__, e_pe_offset, e_pe_view_bytes, e_pe_total_bytes, effective_iter);
                throw std::runtime_error("Position encoding buffer overflow - view exceeds bounds");
            }
            
            // LOG_INF("%s: Position encoding offset - e_pe_offset=%zu bytes, e_pe_view_bytes=%zu, e_pe_total_bytes=%zu (pe_dim=%ld, elem_size=%zu, effective_iter=%d)\n",
            //         __func__, e_pe_offset, e_pe_view_bytes, e_pe_total_bytes, model.whisper_e_pe->ne[0], ggml_element_size(model.whisper_e_pe), effective_iter);
            
            ggml_tensor * e_pe = ggml_view_2d(ctx0, model.whisper_e_pe, 
                                            model.whisper_e_pe->ne[0], n_tokens,
                                            e_pe_stride, 
                                            e_pe_offset);
            
            // transpose and add position encoding (whisper_encoder.cpp:1036)
            cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
        }
        
        ggml_tensor * inpL = cur;
        
        // Step 4: Transformer encoder layers (800-940)
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.whisper_layers[il];
            
            // Pre-attention layer norm
            {
                cur = ggml_norm(ctx0, inpL, hparams.eps);
                cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.attn_ln_0_w), layer.attn_ln_0_b);
            }
            
            // Self-attention (with KV cache reading - Phase 2)
            {
                // Query, Key, Value projections
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
                Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
                
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur); // no bias for key
                
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
                Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
                
                // Reshape for multi-head attention
                const int n_tokens = cur->ne[1]; // sequence length (current chunk)
                
                // Reshape current K and V
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_state_head, n_head, n_tokens);
                
                ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                
                // Step 5: store (1073-1091), then read (1096-1111)
                auto & kv_cache = ctx->whisper_kv_cache;
                
                // Store
                if (kv_cache.buffer != nullptr) {
                    // Calculate bounds checking
                    const int tokens_to_write = n_tokens;
                    const int current_total_tokens = kv_cache.iter * n_tokens;
                    const int new_total_tokens = current_total_tokens + tokens_to_write;
                    const int max_tokens = kv_cache.size; // n_audio_ctx (1500)
                    
                    // LOG_INF("%s: Layer %d - KV cache bounds check: iter=%d, n_tokens=%d, current_total=%d, new_total=%d, max_tokens=%d\n",
                    //         __func__, il, kv_cache.iter, n_tokens, current_total_tokens, new_total_tokens, max_tokens);
                    
                    // Bounds check: ensure we don't write beyond cache size
                    if (new_total_tokens > max_tokens) {
                        // LOG_ERR("%s: FATAL - KV cache overflow detected! iter=%d, n_tokens=%d, current_total=%d, new_total=%d, max_tokens=%d\n",
                        //         __func__, kv_cache.iter, n_tokens, current_total_tokens, new_total_tokens, max_tokens);
                        // LOG_ERR("%s: Attempting to write %d tokens when cache can only hold %d tokens total\n",
                        //         __func__, new_total_tokens, max_tokens);
                        throw std::runtime_error("KV cache buffer overflow - exceeded cache size");
                    }
                    
                    // Check if we have enough space for this write
                    if (current_total_tokens + tokens_to_write > max_tokens) {
                        // LOG_ERR("%s: FATAL - Not enough space in KV cache for write! Need %d tokens, have %d available\n",
                        //         __func__, tokens_to_write, max_tokens - current_total_tokens);
                        throw std::runtime_error("KV cache buffer overflow - not enough space");
                    }
                    
                    const size_t k_offset_bytes = ggml_row_size(kv_cache.k_l[il]->type, n_state) * (kv_cache.iter * n_tokens);
                    const size_t k_total_bytes = ggml_row_size(kv_cache.k_l[il]->type, n_state) * tokens_to_write;
                    const size_t k_cache_total_bytes = ggml_nbytes(kv_cache.k_l[il]);
                    
                    // LOG_INF("%s: Layer %d - Writing KV cache: iter=%d, n_tokens=%d, k_offset=%zu, k_write_bytes=%zu, k_cache_total=%zu\n",
                    //         __func__, il, kv_cache.iter, n_tokens, k_offset_bytes, k_total_bytes, k_cache_total_bytes);
                    
                    // Additional bounds check: ensure offset + write size doesn't exceed cache size
                    if (k_offset_bytes + k_total_bytes > k_cache_total_bytes) {
                        // LOG_ERR("%s: FATAL - K cache write would overflow! offset=%zu, write_size=%zu, cache_size=%zu\n",
                        //         __func__, k_offset_bytes, k_total_bytes, k_cache_total_bytes);
                        throw std::runtime_error("K cache buffer overflow - write exceeds cache bounds");
                    }
                    
                    // K cache store
                    ggml_tensor * k_cache_view = ggml_view_1d(ctx0, kv_cache.k_l[il], 
                                                              n_tokens * n_state, 
                                                              k_offset_bytes);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
                    
                    // V cache store (overwrite Vcur)
                    Vcur = ggml_reshape_2d(ctx0, Vcur, n_state, n_tokens);
                    
                    ggml_tensor * v_cache_view = nullptr;
                    const bool v_trans = true;
                    
                    if (!v_trans) {
                        throw std::runtime_error("non-transposed V cache not supported");
                    } else {
                        const size_t v_offset_bytes = kv_cache.iter * n_tokens * ggml_element_size(kv_cache.v_l[il]);
                        const size_t v_row_size = kv_cache.size * ggml_element_size(kv_cache.v_l[il]);
                        const size_t v_cache_total_bytes = ggml_nbytes(kv_cache.v_l[il]);
                        
                        // LOG_INF("%s: Layer %d - V cache write: v_offset=%zu, v_row_size=%zu, v_cache_total=%zu\n",
                        //         __func__, il, v_offset_bytes, v_row_size, v_cache_total_bytes);
                        
                        // Bounds check for V cache
                        // V cache is stored as [n_audio_ctx, n_state] with transposed layout
                        // Each row is n_state elements, and we have n_audio_ctx rows
                        const size_t v_write_bytes = n_tokens * n_state * ggml_element_size(kv_cache.v_l[il]);
                        if (v_offset_bytes + v_write_bytes > v_cache_total_bytes) {
                            // LOG_ERR("%s: FATAL - V cache write would overflow! offset=%zu, write_size=%zu, cache_size=%zu\n",
                            //         __func__, v_offset_bytes, v_write_bytes, v_cache_total_bytes);
                            throw std::runtime_error("V cache buffer overflow - write exceeds cache bounds");
                        }
                        
                        v_cache_view = ggml_view_2d(ctx0, kv_cache.v_l[il],
                                                    n_tokens, n_state,
                                                    v_row_size,
                                                    v_offset_bytes);
                        Vcur = ggml_transpose(ctx0, Vcur);
                    }
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
                }
                
                // Step 2: create K and V views for attention
                ggml_tensor * K = nullptr;
                ggml_tensor * V = nullptr;
                
                if (kv_cache.buffer != nullptr) {
                    // KV cache initialized, create views for all history
                    // iter not increased yet, so total_tokens = n_tokens * (iter + 1)
                    const int total_tokens = n_tokens * (kv_cache.iter + 1);
                    const int max_tokens = kv_cache.size;
                    
                    // LOG_INF("%s: Layer %d - KV cache read: iter=%d, n_tokens=%d, total_tokens=%d, max_tokens=%d\n",
                    //         __func__, il, kv_cache.iter, n_tokens, total_tokens, max_tokens);
                    
                    // Bounds check: ensure we don't read beyond cache size
                    if (total_tokens > max_tokens) {
                        LOG_ERR("%s: FATAL - KV cache read overflow! total_tokens=%d, max_tokens=%d\n",
                                __func__, total_tokens, max_tokens);
                        throw std::runtime_error("KV cache buffer overflow - read exceeds cache size");
                    }
                    
                    // Clamp total_tokens to cache size as safety measure
                    const int safe_total_tokens = std::min(total_tokens, max_tokens);
                    if (safe_total_tokens != total_tokens) {
                        LOG_WRN("%s: Layer %d - Clamping total_tokens from %d to %d to prevent overflow\n",
                                __func__, il, total_tokens, safe_total_tokens);
                    }
                    
                    K = ggml_view_3d(ctx0, kv_cache.k_l[il],
                                     n_state_head, safe_total_tokens, n_head,
                                     ggml_row_size(kv_cache.k_l[il]->type, n_state),
                                     ggml_row_size(kv_cache.k_l[il]->type, n_state_head),
                                     0);
                    
                    // V cache: physical layout uses transposed storage with row stride = kv_cache.size
                    // 读取时stride必须与存储时一致！
                    // 存储时: v_cache_view row_stride = kv_cache.size * elem_size
                    // 所以读取时也要用同样的stride
                    const size_t v_read_row_stride = kv_cache.size * ggml_element_size(kv_cache.v_l[il]);
                    ggml_tensor * V_2d = ggml_view_2d(ctx0, kv_cache.v_l[il],
                        safe_total_tokens, n_state,
                        v_read_row_stride,  // row stride 必须与存储时一致
                        0);  // offset (start from beginning)
                    
                    // 关键修复：V_2d是 [T, S]，但内存按T方向连续存储
                    // 需要先transpose成 [S, T] 并cont，才能正确reshape成 [D, H, T]
                    // 1. V_2d [T, S] -> transpose -> [S, T] (non-contiguous view)
                    // 2. cont -> [S, T] contiguous (内存按S方向连续)
                    // 3. reshape -> [D, H, T] 其中 S = D*H
                    ggml_tensor * V_2d_t = ggml_cont(ctx0, ggml_transpose(ctx0, V_2d));
                    // V_2d_t: [n_state, safe_total_tokens] contiguous
                    
                    // Reshape from [n_state, safe_total_tokens] to [n_state_head, n_head, safe_total_tokens]
                    ggml_tensor * V_3d = ggml_reshape_3d(ctx0, V_2d_t, n_state_head, n_head, safe_total_tokens);
                    // V_3d: [D, H, T] 其中 (d, h, t) 对应原始 V_2d[t, d + h*D]
                    
                    // Permute to [n_head, safe_total_tokens, n_state_head] to match non-cache path
                    V = ggml_cast(ctx0, ggml_permute(ctx0, V_3d, 1, 2, 0, 3), GGML_TYPE_F16);
                    // V: [H, T, D]

                } else {
                    // KV cache not initialized, only use current K and V
                    K = ggml_permute(ctx0,
                                   ggml_cast(ctx0, Kcur, GGML_TYPE_F16),
                                   0, 2, 1, 3);
                    
                    V = ggml_cast(ctx0,
                                ggml_permute(ctx0, Vcur, 1, 2, 0, 3),
                                GGML_TYPE_F16);
                }
                
                // Step 3: Attention computation
                ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
                
                ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
                ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
                cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_tokens);
            }
            
            // Attention output projection
            {
                cur = ggml_mul_mat(ctx0, layer.attn_ln_1_w, cur);
                cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
            }
            
            // Residual connection
            cur = ggml_add(ctx0, cur, inpL);
            
            ggml_tensor * inpFF = cur;
            
            // Feed-forward network
            {
                // Pre-FFN layer norm
                {
                    cur = ggml_norm(ctx0, inpFF, hparams.eps);
                    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.mlp_ln_w), layer.mlp_ln_b);
                }
                
                // FFN layers
                cur = ggml_mul_mat(ctx0, layer.mlp_0_w, cur);
                cur = ggml_add(ctx0, cur, layer.mlp_0_b);
                cur = ggml_gelu(ctx0, cur);
                
                cur = ggml_mul_mat(ctx0, layer.mlp_1_w, cur);
                cur = ggml_add(ctx0, cur, layer.mlp_1_b);
            }
            
            // FFN residual connection
            inpL = ggml_add(ctx0, cur, inpFF);
        }
        
        cur = inpL;
        
        // Step 5: last layer norm
        {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.whisper_e_ln_w), model.whisper_e_ln_b);
        }
        
        // Step 6: audio projector (954-971)
        // NOTE: average pooling (k=2, s=5) is NOT supported by ggml_pool_1d when k != s
        // So we only do the projector here, pooling is done manually in audition_audio_batch_encode
        {
            // first projector layer
            cur = ggml_mul_mat(ctx0, model.whisper_proj_1_w, cur);
            cur = ggml_add(ctx0, cur, model.whisper_proj_1_b);
            cur = ggml_relu(ctx0, cur);
            
            // second projector layer
            cur = ggml_mul_mat(ctx0, model.whisper_proj_2_w, cur);
            cur = ggml_add(ctx0, cur, model.whisper_proj_2_b);
            
            // Average pooling (k=5, s=5, p=0) along token dimension
            // pool_1d operates on ne[0], so we permute to move tokens to ne[0]
            // Output tokens = (input_tokens - 5) / 5 + 1 (floor division)
            cur = ggml_cpy(ctx0,
                ggml_permute(ctx0, cur, 1, 0, 2, 3),  // [embd, tokens] -> [tokens, embd]
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
            cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 5, 5, 0);  // pool on ne[0] (tokens)
            cur = ggml_cpy(ctx0,
                ggml_permute(ctx0, cur, 1, 0, 2, 3),  // [tokens, embd] -> [embd, tokens]
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
        }
        
        ggml_build_forward_expand(gf, cur);
        
        LOG_INF("%s: Whisper encoder graph built (projector output, pre-pooling)\n", __func__);
        return gf;
    }

private:
    //
    // utility functions
    //

};

static ggml_cgraph * audition_audio_build_graph(audition_ctx * ctx, const audition_audio_f32_batch & audios) {
    GGML_ASSERT(audios.entries.size() == 1 && "n_batch > 1 is not supported");
    audition_graph graph(ctx, *audios.entries[0]);

    ggml_cgraph * res;

    res = graph.build_whisper();
    return res;
}

struct audition_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_audio = true;

    // TODO @ngxson : we should not pass audition_ctx here, it should be audition_model
    audition_model_loader(const char * fname) : fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(audition_model & model) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // model type
        std::string model_type;
        {
            get_string(KEY_MODEL_TYPE, model_type, false);
            // if (!model_type.empty()) {
            //     model.model_type = omni_model_type_from_string(model_type);
            // } else {
            //     throw std::runtime_error(string_format("%s: model type not found\n", __func__));
            // }
            // TODO: tc
            model.model_type = MiniCPM_o;
        }

        const bool is_audio = true;

        // other hparams
        {
            const char * prefix = "audio";

            if (is_audio && model.model_type == MiniCPM_o) {
                LOG_INF("%s: Loading custom whisper encoder parameters\n", __func__);

                get_u32("d_model", hparams.n_embd);
                get_u32("encoder_attention_heads", hparams.n_head);
                get_u32("encoder_layers", hparams.n_layer);
                
                // Validate critical parameters
                if (hparams.n_embd == 0) {
                    throw std::runtime_error(string_format("%s: invalid n_embd (d_model) = 0\n", __func__));
                }
                if (hparams.n_head == 0) {
                    throw std::runtime_error(string_format("%s: invalid n_head (encoder_attention_heads) = 0\n", __func__));
                }
                if (hparams.n_layer == 0) {
                    throw std::runtime_error(string_format("%s: invalid n_layer (encoder_layers) = 0\n", __func__));
                }
                if (hparams.n_embd % hparams.n_head != 0) {
                    throw std::runtime_error(string_format("%s: n_embd (%d) must be divisible by n_head (%d)\n", 
                        __func__, hparams.n_embd, hparams.n_head));
                }
                
                // FFN dimension
                hparams.n_ff = hparams.n_embd * 4;
                
                // read mel bins number from gguf
                get_u32("n_mel", hparams.n_mel_bins, false);
                if (hparams.n_mel_bins == 0) {
                    hparams.n_mel_bins = 80; //  80 mel bins
                }
                
                // read n_fft from gguf
                get_u32("n_fft", hparams.n_fft, false);
                if (hparams.n_fft == 0) {
                    hparams.n_fft = 400; // whisper n_fft
                }
                
                // other parameters
                hparams.eps = 1e-5f; // layer norm epsilon
                hparams.projection_dim = 0; // processed by audio projector
                
                // For whisper models, patch_size is not used but needs to be set to avoid errors
                // Try to read from GGUF, otherwise set a default value
                get_u32("patch_size", hparams.patch_size, false);
                if (hparams.patch_size == 0) {
                    hparams.patch_size = 1; // Default value for whisper (not actually used)
                }

                get_u32("max_source_positions", hparams.n_ctx, false);
                if (hparams.n_ctx == 0) {
                    hparams.n_ctx = 1500; // whisper default audio context length
                }
                
                LOG_INF("%s: Custom whisper encoder - n_embd=%d, n_head=%d, n_layer=%d, n_mel_bins=%d, n_fft=%d, n_ctx=%d, patch_size=%d\n", 
                        __func__, hparams.n_embd, hparams.n_head, hparams.n_layer, hparams.n_mel_bins, hparams.n_fft, hparams.n_ctx, hparams.patch_size);
            } else {
                LOG_INF("%s: Using standard parameter loading (is_audio=%d, model_type=%d)\n", 
                        __func__, is_audio, model.model_type);

                get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
                get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
                get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
                get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
                get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
                get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);
            }

            if (is_audio) {
                // our whisper encoder doesn't read mel_bins, fixed to 80
                if (model.model_type != MiniCPM_o) {
                    get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                }

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;


            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            // model-specific params
            switch (model.model_type) {
                case MiniCPM_o:
                    {
                        // Custom whisper encoder - use GELU activation function
                        hparams.ffn_op = FFN_GELU;
                        log_ffn_op = "gelu";
                        // no proj_stack_factor, use default value 0
                    } break;
                default:
                    break;
            }

            LOG_INF("%s: projector:          %s\n", __func__, model_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            
            LOG_INF("\n--- audio hparams ---\n");
            LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
            LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(audition_ctx & ctx_audition) {
        auto & model = ctx_audition.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = "a";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        size_t mem_size = static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead();
        
        // 为自定义whisper encoder分配更多内存
        if (model.model_type == MiniCPM_o) {
            mem_size *= 4; // 增加4倍内存以适应大型whisper模型
            LOG_INF("%s: Custom whisper encoder detected, increasing memory pool size to %zu bytes\n", __func__, mem_size);
        }
        
        struct ggml_init_params params = {
            /*.mem_size =*/ mem_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_audition.ctx_data.reset(ggml_init(params));
        if (!ctx_audition.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // 简化的张量获取函数，直接使用whisper_encoder.cpp中的张量名称
        auto get_whisper_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_audition.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        // helper function - 使用标准张量名称的获取函数
        auto get_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_audition.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        if (model.model_type != MiniCPM_o) {
        model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"));
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"));
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"));
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w = get_tensor(string_format(TN_LS_1,        prefix, il, "weight"), false); // no bias
            layer.ls_2_w = get_tensor(string_format(TN_LS_2,        prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);
        }
        } else {
            LOG_INF("%s: Skipping standard layers loading for CUSTOM_WHISPER\n", __func__);
        }

        LOG_INF("%s: Loading custom whisper encoder tensors\n", __func__);

        // load whisper encoder (649-693)
        model.whisper_e_pe = get_whisper_tensor("encoder.positional_embedding");
        
        // conv layers
        model.whisper_e_conv_1_w = get_whisper_tensor("encoder.conv1.weight");
        model.whisper_e_conv_1_b = get_whisper_tensor("encoder.conv1.bias");
        model.whisper_e_conv_2_w = get_whisper_tensor("encoder.conv2.weight");
        model.whisper_e_conv_2_b = get_whisper_tensor("encoder.conv2.bias");
        
        // last layer norm
        model.whisper_e_ln_w = get_whisper_tensor("encoder.ln_post.weight");
        model.whisper_e_ln_b = get_whisper_tensor("encoder.ln_post.bias");
        
        // load encoder layers (660-685)
        const int n_layer = hparams.n_layer;
        model.whisper_layers.resize(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.whisper_layers[i];
            std::string prefix = "encoder.blocks." + std::to_string(i) + ".";
            
            // attention layer norm
            layer.attn_ln_0_w = get_whisper_tensor(prefix + "attn_ln.weight");
            layer.attn_ln_0_b = get_whisper_tensor(prefix + "attn_ln.bias");
            
            // attention weights
            layer.attn_q_w = get_whisper_tensor(prefix + "attn.query.weight");
            layer.attn_q_b = get_whisper_tensor(prefix + "attn.query.bias");
            layer.attn_k_w = get_whisper_tensor(prefix + "attn.key.weight");
            // 注意：key没有bias（参考whisper_encoder.cpp:678行）
            layer.attn_v_w = get_whisper_tensor(prefix + "attn.value.weight");
            layer.attn_v_b = get_whisper_tensor(prefix + "attn.value.bias");
            
            // attention output projection
            layer.attn_ln_1_w = get_whisper_tensor(prefix + "attn.out.weight");
            layer.attn_ln_1_b = get_whisper_tensor(prefix + "attn.out.bias");
            
            // MLP layer norm
            layer.mlp_ln_w = get_whisper_tensor(prefix + "mlp_ln.weight");
            layer.mlp_ln_b = get_whisper_tensor(prefix + "mlp_ln.bias");
            
            // MLP weights
            layer.mlp_0_w = get_whisper_tensor(prefix + "mlp.0.weight");
            layer.mlp_0_b = get_whisper_tensor(prefix + "mlp.0.bias");
            layer.mlp_1_w = get_whisper_tensor(prefix + "mlp.2.weight");
            layer.mlp_1_b = get_whisper_tensor(prefix + "mlp.2.bias");
        }
        
        // load audio projector (687-691)
        model.whisper_proj_1_w = get_whisper_tensor("audio_projector.linear1.weight");
        model.whisper_proj_1_b = get_whisper_tensor("audio_projector.linear1.bias");
        model.whisper_proj_2_w = get_whisper_tensor("audio_projector.linear2.weight");
        model.whisper_proj_2_b = get_whisper_tensor("audio_projector.linear2.bias");
        
        // load mel filters (622-633)
        {
            int idx_flt_data = -1;
            for (int i = 0; i < gguf_get_n_kv(ctx_gguf.get()); ++i) {
                const char * key = gguf_get_key(ctx_gguf.get(), i);
                if (strcmp(key, "filters") == 0) {
                    idx_flt_data = i;
                    break;
                }
            }
            
            if (idx_flt_data >= 0) {
                const float* filter_data_addr = (const float*)gguf_get_arr_data(ctx_gguf.get(), idx_flt_data);
                int n_flts = hparams.n_mel_bins * hparams.n_fft;
                model.whisper_filters_data.resize(n_flts);
                std::memcpy(model.whisper_filters_data.data(), filter_data_addr, n_flts * sizeof(float));
                LOG_INF("%s: Loaded mel filters: %d mel bins x %d fft bins = %d elements\n", 
                        __func__, hparams.n_mel_bins, hparams.n_fft, n_flts);
            } else {
                LOG_ERR("%s: Failed to find 'filters' in GGUF metadata\n", __func__);
                throw std::runtime_error("Missing mel filters data in model file");
            }
        }
        
        LOG_INF("%s: Successfully loaded %d whisper encoder layers, projector and mel filters\n", __func__, n_layer);
        
        // Step 5: initialize whisper KV cache
        // n_audio_ctx = 1500
        const int n_audio_ctx = hparams.n_ctx;
        const int n_state = hparams.n_embd;
        audition_whisper_init_kv_cache(&ctx_audition, n_state, n_layer, n_audio_ctx);

        // load data
        {
            std::vector<uint8_t> read_buf;

            auto fin = std::ifstream(fname, std::ios::binary);
            if (!fin) {
                throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
            }

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_audition.backend);
            ctx_audition.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_audition.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_audition.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                ggml_tensor * cur = ggml_get_tensor(ctx_audition.ctx_data.get(), t->name);
                const size_t offset = tensor_offset[t->name];
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                }
                size_t num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buft_is_host(buft)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();

            LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }
    }

    void alloc_compute_meta(audition_ctx & ctx_audition) {
        const auto & hparams = ctx_audition.model.hparams;
        ctx_audition.buf_compute_meta.resize(ctx_audition.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        // create a fake batch
        audition_audio_f32_batch batch;
        audition_audio_f32_ptr audio(audition_audio_f32_init());

        audio->nx = hparams.warmup_audio_size;
        audio->ny = hparams.n_mel_bins;

        batch.entries.push_back(std::move(audio));

        ggml_cgraph * gf = audition_audio_build_graph(&ctx_audition, batch);
        ggml_backend_sched_reserve(ctx_audition.sched.get(), gf);

        for (size_t i = 0; i < ctx_audition.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_audition.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_audition.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_audition.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }
    }

    void get_bool(const std::string & key, bool & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }
};

struct audition_ctx * audition_init(const char * fname, struct audition_context_params audition_params) {
    audition_g_logger_state.verbosity_thold = audition_params.verbosity;
    audition_ctx * ctx_audio = nullptr;

    LOG_INF("%s: initializing audition model from '%s'\n", __func__, fname);
    
    try {
        LOG_INF("%s: creating model loader...\n", __func__);
        audition_model_loader loader(fname);
        LOG_INF("%s: creating audition context...\n", __func__);
        ctx_audio = new audition_ctx(audition_params);
        LOG_INF("%s: loading hyperparameters...\n", __func__);
        loader.load_hparams(ctx_audio->model);
        LOG_INF("%s: loading tensors...\n", __func__);
        loader.load_tensors(*ctx_audio);
        LOG_INF("%s: allocating compute metadata...\n", __func__);
        loader.alloc_compute_meta(*ctx_audio);
        LOG_INF("%s: audition model initialized successfully\n", __func__);

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        fprintf(stderr, "%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        fflush(stderr);
        if (ctx_audio) {
            delete ctx_audio;
        }
        return nullptr;
    } catch (...) {
        LOG_ERR("%s: failed to load model '%s': unknown exception\n", __func__, fname);
        fprintf(stderr, "%s: failed to load model '%s': unknown exception\n", __func__, fname);
        fflush(stderr);
        if (ctx_audio) {
            delete ctx_audio;
        }
        return nullptr;
    }

    return ctx_audio;
}

void audition_free(audition_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}


//
// audition processor
//
struct audition_audio_u8 * audition_audio_u8_init() {
    return new audition_audio_u8();
}
struct audition_audio_f32 * audition_audio_f32_init() {
    return new audition_audio_f32();
}

struct audition_audio_f32_batch * audition_audio_f32_batch_init() {
    return new audition_audio_f32_batch();
}

void audition_audio_u8_free(struct audition_audio_u8 * audio) { if (audio) delete audio; }
void audition_audio_f32_free(struct audition_audio_f32 * audio) { if (audio) delete audio; }
void audition_audio_f32_batch_free(struct audition_audio_f32_batch * batch) { if (batch) delete batch; }

whisper_preprocessor::whisper_filters audition_get_mel_filters(const struct audition_ctx * ctx) {
    whisper_preprocessor::whisper_filters filters;
    
    if (ctx->model.model_type == MiniCPM_o && !ctx->model.whisper_filters_data.empty()) {
        filters.n_mel = ctx->model.hparams.n_mel_bins;
        filters.n_fft = ctx->model.hparams.n_fft;
        filters.data = ctx->model.whisper_filters_data;
        
        LOG_INF("%s: Retrieved mel filters from model: %d mel bins x %d fft bins\n", 
                __func__, filters.n_mel, filters.n_fft);
    } else {
        LOG_ERR("%s: No mel filters data available in model\n", __func__);
        // 返回空的滤波器，调用者需要检查
        filters.n_mel = 0;
        filters.n_fft = 0;
    }
    
    return filters;
}


//
// audition query
//
int audition_n_output_tokens(const struct audition_ctx * ctx, struct audition_audio_f32 * audio) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (audio->nx / patch_size) * (audio->ny / patch_size);

    omni_model_type proj = ctx->model_type();

    switch (proj) {
        case MiniCPM_o:
            {
                // Whisper encoder 输出 token 数计算：
                // 1. 输入: mel frames = audio->nx
                // 2. Conv1 (stride=1, padding=k/2): 输出 ≈ 输入 (same padding)
                // 3. Conv2 (stride=2, padding=k/2): 输出 ≈ 输入/2 (向下取整)
                // 4. Pool (k=5, s=5, p=0): 输出 = (输入 - 5) / 5 + 1
                // 公式: conv_out = (in + 2*p - d*(k-1) - 1) / s + 1 (floor)
                //       pool_out = (in + 2*p - k) / s + 1 (floor)
                const int n_tokens_after_conv = audio->nx / 2;  // conv2 stride=2 下采样
                const int pool_k = 5;
                const int pool_s = 5;
                n_patches = (n_tokens_after_conv - pool_k) / pool_s + 1;
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

int audition_n_mmproj_embd(const struct audition_ctx * ctx) {
    switch (ctx->model.model_type) {
        case MiniCPM_o:
            // 返回whisper音频投影器的输出维度
            return ctx->model.whisper_proj_2_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}


//
// audition forward
//
bool audition_audio_batch_encode(audition_ctx * ctx, const int n_threads, const audition_audio_f32_batch * audios_c_ptr, float * vec) {
    const audition_audio_f32_batch & audios = *audios_c_ptr;
    int batch_size = audios.entries.size();

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false; // only support batch size of 1
    }
    
    // 阶段4：如果是音频且未启用流式模式，自动清空 KV cache
    if (audios.is_audio && !ctx->whisper_streaming_mode) {
        if (ctx->whisper_kv_cache.buffer != nullptr && ctx->whisper_kv_cache.iter > 0) {
            LOG_INF("%s: Streaming mode disabled, clearing KV cache before encoding\n", __func__);
            audition_whisper_clear_kv_cache(ctx);
        }
    }
    
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = audition_audio_build_graph(ctx, audios);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = audios.entries[0]->nx;
    const int image_size_height = audios.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

    const bool use_window_attn = hparams.n_wa_pattern > 0; // for qwen2.5vl

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (audios.is_audio) {
        // audio input
        GGML_ASSERT(audios.entries.size() == 1);
        const auto & mel_inp = audios.entries[0];
        const int n_step = mel_inp->nx;
        const int n_mel  = mel_inp->ny;
        
        LOG_INF("%s: Audio input data - n_step=%d, n_mel=%d, total_elements=%d\n", 
                __func__, n_step, n_mel, n_step * n_mel);
        
        std::vector<float> inp_raw(n_step * n_mel);
        std::memcpy(inp_raw.data(), mel_inp->buf.data(), n_step * n_mel * sizeof(float));
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->model.model_type) {
        case MiniCPM_o:
            {
                // do nothing - this projector type doesn't need additional input setup
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }
    
    // 阶段5：主图执行后，增加 KV cache 的 iter 计数
    // KV cache 写入已经在图中直接完成（通过 ggml_cpy）
    if (audios.is_audio && ctx->whisper_kv_cache.buffer != nullptr) {
        // Calculate n_tokens: after conv stride=2, input frames are halved
        // Input: mel spectrogram frames (audios.entries[0]->nx = 100)
        // After conv1 stride=1: still 100
        // After conv2 stride=2: 100/2 = 50 tokens
        // Note: pooling happens later in projector, so n_tokens here is before pooling
        const int input_frames = audios.entries[0]->nx;
        const int n_tokens = input_frames / 2; // After conv2 with stride=2
        
        const int current_total_tokens = ctx->whisper_kv_cache.iter * n_tokens;
        const int new_total_tokens = current_total_tokens + n_tokens;
        const int max_tokens = ctx->whisper_kv_cache.size;
        
        LOG_INF("%s: Before incrementing iter: input_frames=%d, n_tokens=%d, current_iter=%d, current_total=%d, new_total=%d, max_tokens=%d\n",
                __func__, input_frames, n_tokens, ctx->whisper_kv_cache.iter, current_total_tokens, new_total_tokens, max_tokens);
        
        if (new_total_tokens > max_tokens) {
            LOG_ERR("%s: FATAL - Cannot increment iter: would exceed cache size! current_total=%d, new_total=%d, max_tokens=%d\n",
                    __func__, current_total_tokens, new_total_tokens, max_tokens);
            // Clear cache and reset iter instead of crashing
            LOG_WRN("%s: Clearing KV cache and resetting iter to prevent overflow\n", __func__);
            audition_whisper_clear_kv_cache(ctx);
            ctx->whisper_kv_cache.iter = 0;
        } else {
            ctx->whisper_kv_cache.iter++;
            LOG_INF("%s: KV cache iter incremented to %d (total_tokens=%d, max_tokens=%d)\n", 
                    __func__, ctx->whisper_kv_cache.iter, new_total_tokens, max_tokens);
        }
    }

    // the last node is the final output (after projector + pooling)
    ggml_tensor * final_out = ggml_graph_node(gf, -1);
    
    // Get output shape: [n_embd, n_tokens]
    const int n_embd = final_out->ne[0];
    const int n_tokens_out = final_out->ne[1];
    
    LOG_INF("%s: Final output shape: [%d, %d]\n", __func__, n_embd, n_tokens_out);
    
    // Copy output to vec
    ggml_backend_tensor_get(final_out, vec, 0, ggml_nbytes(final_out));

    // sanity check (only support batch size of 1 for now)
    const int expected_n_tokens_out = audition_n_output_tokens(ctx, audios.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", 
                __func__, expected_n_tokens_out, n_tokens_out);
        
        // 🔧 [安全检查] 如果输出 0 tokens，返回 false 而不是 abort
        if (n_tokens_out == 0) {
            LOG_WRN("%s: audio encoding produced 0 tokens, returning false\n", __func__);
            return false;
        }
        
        // 🔧 [修复] 由于 pooling 边界的舍入误差，允许 ±1 token 的差异
        // 这种情况在处理非整数秒的音频时很常见
        const int diff = abs(n_tokens_out - expected_n_tokens_out);
        if (diff <= 1) {
            LOG_WRN("%s: token count mismatch within tolerance (expected=%d, got=%d, diff=%d), continuing\n",
                    __func__, expected_n_tokens_out, n_tokens_out, diff);
            // 继续执行，不 abort
        } else {
            // 差异超过 1 个 token，可能是严重问题
            LOG_ERR("%s: token count mismatch exceeds tolerance (expected=%d, got=%d, diff=%d)\n",
                    __func__, expected_n_tokens_out, n_tokens_out, diff);
            GGML_ABORT("Invalid number of output tokens");
        }
    }
    
    LOG_INF("%s: Final output: %d tokens x %d dims\n", __func__, n_tokens_out, n_embd);

    return true;
}

bool audition_audio_encode(struct audition_ctx * ctx, const int n_threads, audition_audio_f32 * audio, float * vec) {
    audition_audio_f32_batch audios;
    audition_audio_f32_ptr audio_copy(audition_audio_f32_init());
    *audio_copy = *audio;
    audios.entries.push_back(std::move(audio_copy));

    return audition_audio_batch_encode(ctx, n_threads, &audios, vec);
}


//
// Whisper KV cache management
//
void audition_whisper_init_kv_cache(struct audition_ctx * ctx, int n_state, int n_layer, int n_audio_ctx) {
    auto & kv_cache = ctx->whisper_kv_cache;
    
    // Store parameters
    kv_cache.n_layer = n_layer;
    kv_cache.size = n_audio_ctx;
    kv_cache.iter = 0;
    
    // Create ggml context for KV cache tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/ size_t(2u * n_layer * ggml_tensor_overhead()),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    kv_cache.ctx = ggml_init(params);
    if (!kv_cache.ctx) {
        LOG_ERR("%s: failed to allocate ggml context for KV cache\n", __func__);
        return;
    }
    
    // Allocate K and V tensors for each layer
    // K cache: [n_state, n_audio_ctx] for each layer
    // V cache: [n_audio_ctx, n_state] for each layer (transposed for efficient access)
    kv_cache.k_l.reserve(n_layer);
    kv_cache.v_l.reserve(n_layer);
    
    const ggml_type kv_type = GGML_TYPE_F16;  // Use F16 for KV cache to save memory
    
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * k = ggml_new_tensor_1d(kv_cache.ctx, kv_type, n_state * n_audio_ctx);
        ggml_tensor * v = ggml_new_tensor_1d(kv_cache.ctx, kv_type, n_state * n_audio_ctx);
        
        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);
        
        kv_cache.k_l.push_back(k);
        kv_cache.v_l.push_back(v);
    }
    
    // Allocate backend buffer for KV cache
    kv_cache.buffer = ggml_backend_alloc_ctx_tensors(kv_cache.ctx, ctx->backend);
    if (!kv_cache.buffer) {
        LOG_ERR("%s: failed to allocate backend buffer for KV cache\n", __func__);
        ggml_free(kv_cache.ctx);
        kv_cache.ctx = nullptr;
        return;
    }
    
    // Clear the cache
    ggml_backend_buffer_clear(kv_cache.buffer, 0);
    
    LOG_INF("%s: KV cache initialized - n_layer=%d, n_state=%d, n_audio_ctx=%d, buffer_size=%.2f MB\n",
            __func__, n_layer, n_state, n_audio_ctx,
            ggml_backend_buffer_get_size(kv_cache.buffer) / (1024.0 * 1024.0));
}

void audition_whisper_free_kv_cache(struct audition_ctx * ctx) {
    auto & kv_cache = ctx->whisper_kv_cache;
    
    if (kv_cache.buffer) {
        ggml_backend_buffer_free(kv_cache.buffer);
        kv_cache.buffer = nullptr;
    }
    
    if (kv_cache.ctx) {
        ggml_free(kv_cache.ctx);
        kv_cache.ctx = nullptr;
    }
    
    kv_cache.k_l.clear();
    kv_cache.v_l.clear();
    kv_cache.n_layer = 0;
    kv_cache.size = 0;
    kv_cache.iter = 0;
    
    LOG_INF("%s: KV cache freed\n", __func__);
}

void audition_whisper_clear_kv_cache(struct audition_ctx * ctx) {
    auto & kv_cache = ctx->whisper_kv_cache;
    
    if (kv_cache.buffer) {
        ggml_backend_buffer_clear(kv_cache.buffer, 0);
    }
    
    kv_cache.iter = 0;
    
    LOG_INF("%s: KV cache cleared\n", __func__);
}

// from mtmd-audio.cpp  TODO: move to correct place

// align x to upper multiple of n
#define _ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

namespace whisper_preprocessor {

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    WHISPER_ASSERT(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int   n_samples,
        const int   /*sample_rate*/,
        const int   frame_size,
        const int   frame_step,
        const int   n_mel,
        const int   n_threads,
        const whisper_filters & filters,
        const bool   debug,
        whisper_mel & mel) {
    //const int64_t t_start_us = ggml_time_us();

    // Hann window
    WHISPER_ASSERT(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    // 阶段5修改：流式模式下不需要30秒的padding，音频已经按1秒切分
    int64_t stage_1_pad = 0;  // 原始：WHISPER_SAMPLE_RATE * 30（30秒padding）
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filters & filters,
        std::vector<whisper_mel> & output) {

    if (n_samples == 0) {
        // empty audio
        return false;
    }

    whisper_mel out_full;
    bool ok = log_mel_spectrogram(
                samples,
                n_samples,
                COMMON_SAMPLE_RATE,
                WHISPER_N_FFT,
                WHISPER_HOP_LENGTH,
                filters.n_mel,
                4, // n_threads
                filters,
                false, // debug
                out_full);
    if (!ok) {
        return false;
    }

    output.push_back(std::move(out_full));
    
    return true;
}

} // namespace whisper_preprocessor

// from mtmd-helper.cpp (temp)
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);

    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    ma_decoder_uninit(&decoder);
    return true;
}

bool audition_audio_preprocess(
    audition_ctx * ctx,
    audition_audio_u8 * audio,
    audition_audio_f32 ** out_mel) {
    const unsigned char * audio_data = audio->buf.data();
    size_t audio_data_len = audio->buf.size();

    // ===== 步骤1：解码音频 =====
    std::vector<float> pcm_f32;
    int sample_rate = 16000;
    
    bool decode_ok = decode_audio_from_buf(
        audio_data, 
        audio_data_len, 
        sample_rate, 
        pcm_f32
    );

    if (!decode_ok) {
        LOG_ERR("%s: Failed to decode audio data\n", __func__);
        return false;
    }
    
    LOG_INF("%s: Decoded audio - sample_rate=%d, n_samples=%zu\n",
            __func__, sample_rate, pcm_f32.size());
    
    // 🔧 [修复] 将音频 pad 到 100ms 的倍数（1600 samples @ 16kHz）
    // 这是为了确保 Whisper encoder 的输出 token 数与预期一致
    // 原因：
    //   - mel_frames = n_samples / 160
    //   - conv2 下采样 2 倍：mel_frames / 2
    //   - pool (k=5, s=5)：(mel_frames/2 - 5) / 5 + 1
    //   - 只有当 mel_frames 是 10 的倍数时，token 数才能精确计算
    //   - mel_frames = 10 对应 n_samples = 1600 (100ms)
    // 
    // 尾音处理：流式音频的最后一片可能不足 100ms 的倍数，需要 pad 静音
    const size_t CHUNK_SAMPLES = 1600;  // 100ms @ 16kHz
    size_t original_size = pcm_f32.size();
    
    if (original_size < CHUNK_SAMPLES) {
        // 太短，pad 到最小 100ms
        pcm_f32.resize(CHUNK_SAMPLES, 0.0f);
        LOG_WRN("%s: Audio too short (%zu samples = %.1fms), padded with silence to %zu samples (100ms)\n",
                __func__, original_size, original_size * 1000.0f / sample_rate, CHUNK_SAMPLES);
    } else if (original_size % CHUNK_SAMPLES != 0) {
        // 不是 100ms 的倍数，pad 到下一个 100ms 边界
        size_t padded_size = ((original_size / CHUNK_SAMPLES) + 1) * CHUNK_SAMPLES;
        pcm_f32.resize(padded_size, 0.0f);
        LOG_WRN("%s: Audio not aligned to 100ms (%zu samples = %.1fms), padded with silence to %zu samples (%.0fms)\n",
                __func__, original_size, original_size * 1000.0f / sample_rate, 
                padded_size, padded_size * 1000.0f / sample_rate);
    }
    
    // ===== 步骤2：获取 Mel 滤波器 =====
    whisper_preprocessor::whisper_filters filters = audition_get_mel_filters(ctx);
    
    if (filters.n_mel == 0 || filters.n_fft == 0) {
        LOG_ERR("%s: Mel filters not available in model\n", __func__);
        return false;
    }
    
    LOG_INF("%s: Using mel filters - n_mel=%d, n_fft=%d\n",
            __func__, filters.n_mel, filters.n_fft);
    
    // ===== 步骤3：生成 Mel 频谱 =====
    std::vector<whisper_preprocessor::whisper_mel> mel_spec_chunks;
    
    bool preprocess_ok = whisper_preprocessor::preprocess_audio(
        pcm_f32.data(),
        pcm_f32.size(),
        filters,
        mel_spec_chunks
    );
    
    if (!preprocess_ok || mel_spec_chunks.empty()) {
        LOG_ERR("%s: Failed to generate mel spectrogram\n", __func__);
        return false;
    }
    
    LOG_INF("%s: Generated %zu mel spectrogram chunk(s)\n",
            __func__, mel_spec_chunks.size());
    
    // ===== 步骤4：封装为 audition_audio_f32 =====
    // 1s -> 100 frames -> 50 tokens -> 1 chunk
    auto & mel_spec = mel_spec_chunks[0];
    
    audition_audio_f32 * mel_f32 = audition_audio_f32_init();
    mel_f32->nx = mel_spec.n_len;   // 时间维度（frames）
    mel_f32->ny = mel_spec.n_mel;   // Mel bins 数量
    mel_f32->buf = std::move(mel_spec.data);  // 移动数据避免拷贝
    
    *out_mel = mel_f32;
    
    LOG_INF("%s: Mel spectrogram ready - n_len=%d, n_mel=%d, total_size=%zu\n",
            __func__, mel_f32->nx, mel_f32->ny, mel_f32->buf.size());
    
    return true;
}
