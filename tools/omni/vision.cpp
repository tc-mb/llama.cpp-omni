#include "omni-impl.h"
#include "vision.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#if defined(ENABLE_COREML)
#include "coreml/omni_coreml.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>
#include <array>
#include <numeric>
#include <functional>


struct omni_logger_state g_logger_state = {GGML_LOG_LEVEL_CONT, omni_log_callback_default, NULL};

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
// vision layers
//
struct vision_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t n_embd;
    int32_t n_ff;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    // idefics3
    int32_t preproc_image_size = 0;
    int32_t proj_scale_factor = 0;

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM everytime they load the model
    int32_t warmup_image_size = 0;

    ffn_op_type ffn_op = FFN_GELU;

    float eps = 1e-6;
    float rope_theta = 0.0;

    std::vector<vision_image_size> image_res_candidates; // for llava-uhd style models
    int32_t image_crop_resolution;
    std::unordered_set<int32_t> vision_feature_layer;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    int32_t spatial_merge_size = 0;

    // legacy;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number
    int minicpmv_max_slice_nums = 0;
    int32_t insert_layer_id = 0;            // MiniCPM-o 4.6 ViT merger insertion layer
};

struct vision_layer {
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

struct vision_model {
    omni_model_type model_type = MiniCPM_o;
    vision_hparams hparams;

    // embeddings
    ggml_tensor * patch_embeddings = nullptr;
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<vision_layer> layers;

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    ggml_tensor * projection; // TODO: rename it to fc (fully connected layer)
    ggml_tensor * mm_fc_w;
    ggml_tensor * mm_fc_b;
    ggml_tensor * mm_ffn_up_w = nullptr;
    ggml_tensor * mm_ffn_up_b = nullptr;
    ggml_tensor * mm_ffn_down_w = nullptr;
    ggml_tensor * mm_ffn_down_b = nullptr;
    ggml_tensor * mm_input_norm_w = nullptr;
    ggml_tensor * mm_input_norm_b = nullptr;

    // MINICPMV projection
    ggml_tensor * mm_model_pos_embed_k = nullptr;
    ggml_tensor * mm_model_query = nullptr;
    ggml_tensor * mm_model_proj = nullptr;
    ggml_tensor * mm_model_kv_proj = nullptr;
    ggml_tensor * mm_model_attn_q_w = nullptr;
    ggml_tensor * mm_model_attn_q_b = nullptr;
    ggml_tensor * mm_model_attn_k_w = nullptr;
    ggml_tensor * mm_model_attn_k_b = nullptr;
    ggml_tensor * mm_model_attn_v_w = nullptr;
    ggml_tensor * mm_model_attn_v_b = nullptr;
    ggml_tensor * mm_model_attn_o_w = nullptr;
    ggml_tensor * mm_model_attn_o_b = nullptr;
    ggml_tensor * mm_model_ln_q_w = nullptr;
    ggml_tensor * mm_model_ln_q_b = nullptr;
    ggml_tensor * mm_model_ln_kv_w = nullptr;
    ggml_tensor * mm_model_ln_kv_b = nullptr;
    ggml_tensor * mm_model_ln_post_w = nullptr;
    ggml_tensor * mm_model_ln_post_b = nullptr;

    // MiniCPM-o 4.6 ViT merger (ported from MiniCPM-V 4.6 vision tower)
    ggml_tensor * vit_merger_ln1_w     = nullptr;
    ggml_tensor * vit_merger_ln1_b     = nullptr;
    ggml_tensor * vit_merger_attn_q_w  = nullptr;
    ggml_tensor * vit_merger_attn_q_b  = nullptr;
    ggml_tensor * vit_merger_attn_k_w  = nullptr;
    ggml_tensor * vit_merger_attn_k_b  = nullptr;
    ggml_tensor * vit_merger_attn_v_w  = nullptr;
    ggml_tensor * vit_merger_attn_v_b  = nullptr;
    ggml_tensor * vit_merger_attn_o_w  = nullptr;
    ggml_tensor * vit_merger_attn_o_b  = nullptr;
    ggml_tensor * vit_merger_ds_ln_w   = nullptr;
    ggml_tensor * vit_merger_ds_ln_b   = nullptr;
    ggml_tensor * vit_merger_ds_up_w   = nullptr;
    ggml_tensor * vit_merger_ds_up_b   = nullptr;
    ggml_tensor * vit_merger_ds_down_w = nullptr;
    ggml_tensor * vit_merger_ds_down_b = nullptr;

};

struct vision_ctx {
    vision_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;

    int max_nodes = 8192;
    
    // 🔧 [高清模式] 运行时覆盖 max_slice_nums，-1 表示使用模型默认值
    int max_slice_nums_override = -1;

    // 🔧 [batch encode 开关] 是否对多个尺寸相同的 slice 启用批量编码优化。
    // 默认 false（关闭）—— 该优化只在大图/高清高刷等 slice 数较多时收益明显，
    // 并非通用场景，因此需要显式开启。
    bool batch_encode = false;

    // CoreML / ANE model path
    std::string coreml_model_path;

    ggml_backend_sched_ptr sched;

    // for debugging
    bool debug_graph = false;
    std::vector<ggml_tensor *> debug_print_tensors;

    vision_ctx(vision_context_params & ctx_params) {
        debug_graph = std::getenv("Omni_DEBUG_GRAPH") != nullptr;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            auto backend_name = std::getenv("Omni_BACKEND_DEVICE");
            if (backend_name != nullptr) {
                backend = ggml_backend_init_by_name(backend_name, nullptr);
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: vision using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: vision using CPU backend\n", __func__);
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );
    }

    ~vision_ctx() {
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

static bool is_minicpm_o_4_6(const vision_model & model) {
    return model.model_type == MiniCPM_o_4_6;
}

struct vision_graph {
    vision_ctx * ctx;
    const vision_model & model;
    const vision_hparams & hparams;

    const vision_image_f32 & img; // reference image (all batched images must share dimensions)
    const int batch_size;

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

    vision_graph(vision_ctx * ctx, const vision_image_f32 & img, int batch_size = 1) :
            ctx(ctx),
            model(ctx->model),
            hparams(model.hparams),
            img(img),
            batch_size(batch_size),
            patch_size(hparams.patch_size),
            n_patches_x(img.nx / patch_size),
            n_patches_y(img.ny / patch_size),
            n_patches(n_patches_x * n_patches_y),
            n_embd(hparams.n_embd),
            n_head(hparams.n_head),
            d_head(n_embd / n_head),
            n_layer(hparams.n_layer),
            eps(hparams.eps),
            kq_scale(1.0f / sqrtf((float)d_head)) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx->buf_compute_meta.size(),
            /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };
        ctx0_ptr.reset(ggml_init(params));
        ctx0 = ctx0_ptr.get();
        gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
    }

    ggml_cgraph * build_minicpmv() {
        const int n_pos = n_patches;

        // position embeddings for the projector (not for ViT)
        // keep as [n_output_dim, n_pos, 1] even for batch>1, ggml_add broadcasts
        int n_output_dim = vision_n_mmproj_embd(ctx);
        ggml_tensor * pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_output_dim, n_pos, 1);
        ggml_set_name(pos_embed, "pos_embed");
        ggml_set_input(pos_embed);

        // for selecting learned pos embd, used by ViT
        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

        ggml_tensor * inp = build_inp();
        ggml_tensor * embeddings = build_vit(
                                inp, n_patches,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                learned_pos_embd,
                                nullptr);
        // embeddings shape: [n_embd, n_patches] (batch=1) or [n_embd, n_patches, batch_size] (batch>1)

        // resampler projector (it is just another transformer)

        ggml_tensor * q = model.mm_model_query;
        ggml_tensor * v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);

        // norm
        q = build_norm(q, model.mm_model_ln_q_w, model.mm_model_ln_q_b, NORM_TYPE_NORMAL, eps, -1);
        v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

        // k = v + pos_embed (pos_embed broadcasts over batch dim)
        ggml_tensor * k = ggml_add(ctx0, v, pos_embed);

        // attention
        {
            int n_embd = vision_n_mmproj_embd(ctx);
            const int d_head = 128;
            int n_head = n_embd/d_head;
            int num_query = ctx->model.hparams.minicpmv_query_num;
            ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q),
                model.mm_model_attn_q_b);
            ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k),
                model.mm_model_attn_k_b);
            ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v),
                model.mm_model_attn_v_b);

            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
            if (batch_size > 1) {
                K = ggml_reshape_4d(ctx0, K, d_head, n_head, n_pos, batch_size);
                V = ggml_reshape_4d(ctx0, V, d_head, n_head, n_pos, batch_size);
                // Q has no batch dim (shared query), expand via repeat for mul_mat broadcast compatibility
                ggml_tensor * Q_target = ggml_new_tensor_4d(ctx0, Q->type, d_head, n_head, num_query, batch_size);
                Q = ggml_repeat(ctx0, Q, Q_target);
            } else {
                K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
                V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);
            }

            cb(Q, "resampler_Q", -1);
            cb(K, "resampler_K", -1);
            cb(V, "resampler_V", -1);

            const float resampler_kq_scale = 1.0f / sqrtf((float)d_head);  

            embeddings = build_attn(
                model.mm_model_attn_o_w,
                model.mm_model_attn_o_b,
                Q, K, V, nullptr, resampler_kq_scale, -1);
            cb(embeddings, "resampler_attn_out", -1);
        }
        // layernorm
        embeddings = build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);

        // projection
        embeddings = ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    // ViT graph for MiniCPM-o 4.6: SigLIP-style ViT with an inserted window-attention
    // merger (2x2 spatial downsample) at hparams.insert_layer_id, followed by the
    // remaining ViT layers on the downsampled tokens and a final 2x2 downsample
    // merger that projects into the LLM embedding space.
    // Structure originally introduced in MiniCPM-V 4.6 (PROJECTOR_TYPE_MINICPMV4_6
    // in upstream llama.cpp / tools/mtmd); reused here for MiniCPM-o 4.6.
    ggml_cgraph * build_minicpm_o_4_6() {
        GGML_ASSERT(n_patches_x % 4 == 0 && n_patches_y % 4 == 0);
        const int insert_lid = hparams.insert_layer_id;
        GGML_ASSERT(insert_lid >= 0 && insert_lid < n_layer);
        const int n_pos      = n_patches;
        const int half_h     = n_patches_y / 2;
        const int half_w     = n_patches_x / 2;
        const int n_ds       = half_h * half_w;
        const int qh         = half_h / 2;
        const int qw         = half_w / 2;
        const int n_ds2      = qh * qw;

        auto add_i32_input = [&](const char * name, int n) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
            ggml_set_name(t, name);
            ggml_set_input(t);
            return t;
        };

        // position indices for ViT learned positional embeddings
        ggml_tensor * positions = add_i32_input("positions", n_pos);
        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

        // ViT merger window reorder indices + block-diagonal mask
        // (mask layout follows qwen2vl: -inf except for 4x4 blocks on the diagonal,
        // so each window-major group of 4 tokens only attends to itself)
        // TODO: when vision graph gains flash-attn support, cast mask to F16 here
        //       (see llama.cpp tools/mtmd/models/minicpmv.cpp::clip_graph_minicpmv4_6::build)
        ggml_tensor * vit_merger_window_idx     = add_i32_input("vit_merger_window_idx", n_pos);
        ggml_tensor * vit_merger_inv_window_idx = add_i32_input("vit_merger_inv_window_idx", n_pos);
        ggml_tensor * vit_merger_window_mask    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(vit_merger_window_mask, "vit_merger_window_mask");
        ggml_set_input(vit_merger_window_mask);

        // ViT merger 2x2 downsample gather indices
        ggml_tensor * vit_merger_ds_idx_0 = add_i32_input("vit_merger_ds_idx_0", n_ds);
        ggml_tensor * vit_merger_ds_idx_1 = add_i32_input("vit_merger_ds_idx_1", n_ds);
        ggml_tensor * vit_merger_ds_idx_2 = add_i32_input("vit_merger_ds_idx_2", n_ds);
        ggml_tensor * vit_merger_ds_idx_3 = add_i32_input("vit_merger_ds_idx_3", n_ds);

        // final merger 2x2 downsample gather indices
        ggml_tensor * merger_ds_idx_0 = add_i32_input("merger_ds_idx_0", n_ds2);
        ggml_tensor * merger_ds_idx_1 = add_i32_input("merger_ds_idx_1", n_ds2);
        ggml_tensor * merger_ds_idx_2 = add_i32_input("merger_ds_idx_2", n_ds2);
        ggml_tensor * merger_ds_idx_3 = add_i32_input("merger_ds_idx_3", n_ds2);

        // patch embedding + positional embedding
        ggml_tensor * inp = build_inp();
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);

        ggml_tensor * inpL = inp;
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        // ViT layers 0..insert_layer_id (inclusive)
        // Mirrors the separate-qkv path of build_vit() so the two manually
        // unrolled segments around the ViT merger read like build_vit() expansions.
        for (int il = 0; il <= insert_lid; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "layer_inp_normed", il);

            {
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                cb(cur, "attn_out_scaled", il);
            }
            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;
            cb(cur, "ffn_inp", il);

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "ffn_inp_normed", il);

            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                            layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
            cb(cur, "ffn_out", il);

            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                cb(cur, "ffn_out_scaled", il);
            }
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // ViT merger: window self-attention
        // Tokens are reordered to window-major (4 tokens per window are contiguous),
        // and a block-diagonal mask restricts attention to within each window. This
        // mirrors the qwen2vl windowed-attention pattern so build_attn() can pick the
        // flash-attention path when available.
        {
            ggml_tensor * residual = inpL;
            ggml_tensor * cur = build_norm(inpL,
                model.vit_merger_ln1_w, model.vit_merger_ln1_b,
                NORM_TYPE_NORMAL, eps, -1);
            cb(cur, "vit_merger_attn_inp_normed", -1);

            cur = ggml_get_rows(ctx0, cur, vit_merger_window_idx);
            cb(cur, "vit_merger_window_reorder", -1);

            ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.vit_merger_attn_q_w, cur);
            if (model.vit_merger_attn_q_b) {
                Qcur = ggml_add(ctx0, Qcur, model.vit_merger_attn_q_b);
            }
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.vit_merger_attn_k_w, cur);
            if (model.vit_merger_attn_k_b) {
                Kcur = ggml_add(ctx0, Kcur, model.vit_merger_attn_k_b);
            }
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.vit_merger_attn_v_w, cur);
            if (model.vit_merger_attn_v_b) {
                Vcur = ggml_add(ctx0, Vcur, model.vit_merger_attn_v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
            cb(Qcur, "vit_merger_Qcur", -1);
            cb(Kcur, "vit_merger_Kcur", -1);
            cb(Vcur, "vit_merger_Vcur", -1);

            cur = build_attn(model.vit_merger_attn_o_w, model.vit_merger_attn_o_b,
                             Qcur, Kcur, Vcur, vit_merger_window_mask, kq_scale, -1);
            cb(cur, "vit_merger_attn_out", -1);

            cur = ggml_get_rows(ctx0, cur, vit_merger_inv_window_idx);
            inpL = ggml_add(ctx0, cur, residual);
            cb(inpL, "vit_merger_attn_residual", -1);
        }

        // ViT merger: 2x2 spatial downsample + MLP (4 tokens -> 1)
        {
            ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_0);
            ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_1);
            ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_2);
            ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_3);

            ggml_tensor * mean_res = ggml_add(ctx0, p0, p1);
            mean_res = ggml_add(ctx0, mean_res, p2);
            mean_res = ggml_add(ctx0, mean_res, p3);
            mean_res = ggml_scale(ctx0, mean_res, 0.25f);
            cb(mean_res, "vit_merger_ds_mean_res", -1);

            ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
            cat = ggml_concat(ctx0, cat, p2, 0);
            cat = ggml_concat(ctx0, cat, p3, 0);

            ggml_tensor * cur = build_norm(cat,
                model.vit_merger_ds_ln_w, model.vit_merger_ds_ln_b,
                NORM_TYPE_NORMAL, eps, -1);
            cb(cur, "vit_merger_ds_normed", -1);

            // MiniCPMV4_6ViTWindowAttentionMerger downsample MLP uses gelu_pytorch_tanh (FFN_GELU)
            cur = build_ffn(cur,
                model.vit_merger_ds_up_w,   model.vit_merger_ds_up_b,
                nullptr, nullptr,
                model.vit_merger_ds_down_w, model.vit_merger_ds_down_b,
                FFN_GELU, -1);
            cb(cur, "vit_merger_ds_mlp_out", -1);

            inpL = ggml_add(ctx0, cur, mean_res);
            cb(inpL, "vit_merger_ds_out", -1);
        }

        // ViT layers (insert_layer_id+1)..n_layer-1, operating on the downsampled tokens
        {
            const int64_t n_pos_ds = n_ds;
            for (int il = insert_lid + 1; il < n_layer; il++) {
                auto & layer = model.layers[il];
                ggml_tensor * cur = inpL;

                cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
                cb(cur, "layer_inp_normed", il);

                {
                    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                    if (layer.q_b) {
                        Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                    }
                    ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                    if (layer.k_b) {
                        Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                    }
                    ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                    if (layer.v_b) {
                        Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                    }

                    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos_ds);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos_ds);
                    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos_ds);
                    cb(Qcur, "Qcur", il);
                    cb(Kcur, "Kcur", il);
                    cb(Vcur, "Vcur", il);

                    cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                    cb(cur, "attn_out", il);
                }

                if (layer.ls_1_w) {
                    cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                    cb(cur, "attn_out_scaled", il);
                }
                cur = ggml_add(ctx0, cur, inpL);
                inpL = cur;
                cb(cur, "ffn_inp", il);

                cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
                cb(cur, "ffn_inp_normed", il);

                cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b,
                                layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
                cb(cur, "ffn_out", il);

                if (layer.ls_2_w) {
                    cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                    cb(cur, "ffn_out_scaled", il);
                }
                cur = ggml_add(ctx0, inpL, cur);
                cb(cur, "layer_out", il);

                inpL = cur;
            }
        }

        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
            cb(inpL, "post_ln", -1);
        }

        // Final Merger (DownsampleMLP): another 2x2 spatial merge -> projector embedding
        {
            ggml_tensor * p0 = ggml_get_rows(ctx0, inpL, merger_ds_idx_0);
            ggml_tensor * p1 = ggml_get_rows(ctx0, inpL, merger_ds_idx_1);
            ggml_tensor * p2 = ggml_get_rows(ctx0, inpL, merger_ds_idx_2);
            ggml_tensor * p3 = ggml_get_rows(ctx0, inpL, merger_ds_idx_3);

            ggml_tensor * cat = ggml_concat(ctx0, p0, p1, 0);
            cat = ggml_concat(ctx0, cat, p2, 0);
            cat = ggml_concat(ctx0, cat, p3, 0);

            ggml_tensor * cur = build_norm(cat,
                model.mm_input_norm_w, model.mm_input_norm_b,
                NORM_TYPE_NORMAL, eps, -1);
            cb(cur, "merger_normed", -1);

            // MiniCPMV4_6DownsampleMLP uses nn.GELU() (erf-based, FFN_GELU_ERF)
            cur = build_ffn(cur,
                model.mm_ffn_up_w,   model.mm_ffn_up_b,
                nullptr, nullptr,
                model.mm_ffn_down_w, model.mm_ffn_down_b,
                FFN_GELU_ERF, -1);
            cb(cur, "merger_out", -1);

            inpL = cur;
        }

        ggml_build_forward_expand(gf, inpL);
        return gf;
    }

private:
    //
    // utility functions
    //

    void cb(ggml_tensor * cur0, const char * name, int il) const {
        if (ctx->debug_graph) {
            ggml_tensor * cur = ggml_cpy(ctx0, cur0, ggml_dup_tensor(ctx0, cur0));
            std::string cur_name = il >= 0 ? std::string(name) + "_" + std::to_string(il) : name;
            ggml_set_name(cur, cur_name.c_str());
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
            ctx->debug_print_tensors.push_back(cur);
        }
    }

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(
                ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                ggml_tensor * learned_pos_embd,
                std::function<ggml_tensor *(ggml_tensor *, const vision_layer &)> add_pos
            ) {
        if (learned_pos_embd) {
            inp = ggml_add(ctx0, inp, learned_pos_embd);
            cb(inp, "pos_embed", -1);
        }

        ggml_tensor * inpL = inp;

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
            cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                if (layer.q_norm) {
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

                if (batch_size > 1) {
                    Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_pos, batch_size);
                    Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head, n_pos, batch_size);
                    Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head, n_pos, batch_size);
                } else {
                    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                if (add_pos) {
                    Qcur = add_pos(Qcur, layer);
                    Kcur = add_pos(Kcur, layer);
                    cb(Qcur, "Qcur_pos", il);
                    cb(Kcur, "Kcur_pos", il);
                }

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                cb(cur, "attn_out_scaled", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur; // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                ffn_t, il);

            cb(cur, "ffn_out", il);

            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                cb(cur, "ffn_out_scaled", il);
            }

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
        }
        return inpL;
    }

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches] (batch=1) or [n_embd, n_patches, batch_size] (batch>1)
    ggml_tensor * build_inp() {
        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        if (batch_size > 1) {
            inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, batch_size);
        } else {
            inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
        }
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (model.patch_bias) {
            inp = ggml_add(ctx0, inp, model.patch_bias);
            cb(inp, "patch_bias", -1);
        }
        return inp;
    }

    ggml_tensor * build_inp_raw(int channels = 3) {
        ggml_tensor * inp_raw;
        if (batch_size > 1) {
            inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels, batch_size);
        } else {
            inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels);
        }
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);
        return inp_raw;
    }

    ggml_tensor * build_norm(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const {

        cur = type == NORM_TYPE_RMS
            ? ggml_rms_norm(ctx0, cur, norm_eps)
            : ggml_norm(ctx0, cur, norm_eps);

        if (mw || mb) {
            cb(cur, "norm", il);
        }

        if (mw) {
            cur = ggml_mul(ctx0, cur, mw);
            if (mb) {
                cb(cur, "norm_w", il);
            }
        }

        if (mb) {
            cur = ggml_add(ctx0, cur, mb);
        }

        return cur;
    }

    ggml_tensor * build_ffn(
            ggml_tensor * cur,
            ggml_tensor * up,
            ggml_tensor * up_b,
            ggml_tensor * gate,
            ggml_tensor * gate_b,
            ggml_tensor * down,
            ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const {

        ggml_tensor * tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
        cb(tmp, "ffn_up", il);

        if (up_b) {
            tmp = ggml_add(ctx0, tmp, up_b);
            cb(tmp, "ffn_up_b", il);
        }

        if (gate) {
            cur = ggml_mul_mat(ctx0, gate, cur);
            cb(cur, "ffn_gate", il);

            if (gate_b) {
                cur = ggml_add(ctx0, cur, gate_b);
                cb(cur, "ffn_gate_b", il);
            }
        } else {
            cur = tmp;
        }

        // we only support parallel ffn for now
        switch (type_op) {
            case FFN_SILU:
                if (gate) {
                    cur = ggml_swiglu_split(ctx0, cur, tmp);
                    cb(cur, "ffn_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_silu", il);
                } break;
            case FFN_GELU:
                if (gate) {
                    cur = ggml_geglu_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_gelu", il);
                } break;
            case FFN_GELU_ERF:
                if (gate) {
                    cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu_erf", il);
                } else {
                    cur = ggml_gelu_erf(ctx0, cur);
                    cb(cur, "ffn_gelu_erf", il);
                } break;
            case FFN_GELU_QUICK:
                if (gate) {
                    cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu_quick", il);
                } else {
                    cur = ggml_gelu_quick(ctx0, cur);
                    cb(cur, "ffn_gelu_quick", il);
                } break;
        }

        if (down) {
            cur = ggml_mul_mat(ctx0, down, cur);
        }

        if (down_b) {
            cb(cur, "ffn_down", il);
        }

        if (down_b) {
            cur = ggml_add(ctx0, cur, down_b);
        }

        return cur;
    }

    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float kq_scale,
            int il) const {
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        ggml_build_forward_expand(gf, q_cur);
        ggml_build_forward_expand(gf, k_cur);
        ggml_build_forward_expand(gf, v_cur);

        ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        //cb(q, "q", il);

        ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        //cb(k, "k", il);

        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);
        //cb(k, "v", il);

        ggml_tensor * cur;

        // TODO @ngxson : support flash attention
        {
            const auto n_tokens = q->ne[1];
            const auto n_head   = q->ne[2];
            const auto n_batch  = q->ne[3];
            // const auto n_kv     = k->ne[1]; // for flash attention

            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            // F32 may not needed for vision encoders?
            // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            if (n_batch > 1) {
                cur = ggml_cont(ctx0, cur);
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0]*n_head, n_tokens, n_batch);
            } else {
                cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);
            }
        }

        cb(cur, "kqv_out", il);

        if (wo) {
            cur = ggml_mul_mat(ctx0, wo, cur);
        }

        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
        }

        return cur;
    }

};

static ggml_cgraph * vision_image_build_graph(vision_ctx * ctx, const vision_image_f32_batch & imgs) {
    const int n_batch = (int)imgs.entries.size();
    GGML_ASSERT(n_batch >= 1);
    vision_graph graph(ctx, *imgs.entries[0], n_batch);

    ggml_cgraph * res;

    switch (ctx->model_type()) {
        case MiniCPM_o:
            {
                res = graph.build_minicpmv();
            } break;
        case MiniCPM_o_4_6:
            {
                res = graph.build_minicpm_o_4_6();
            } break;
        default:
            {
                res = graph.build_minicpmv();
            } break;
    }
    return res;
}

struct vision_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision = false;
    bool has_audio  = false;

    vision_model_loader(const char * fname) : fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load vision model from %s. Does this file exist?\n", __func__, fname));
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

    void load_hparams(vision_model & model) {
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

            std::string proj_type;
            get_string(KEY_PROJ_TYPE, proj_type, false);
            // MiniCPM-o 4.6 currently reuses the MiniCPM-V 4.6 vision tower verbatim,
            // so we recognize V4_6 GGUF identifiers here. Add the o-4.6 model_type
            // string(s) once the MiniCPM-o 4.6 checkpoints are published.
            if (proj_type == "minicpmv4_6" || model_type == "MiniCPM-V-4_6" || model_type == "minicpmv4_6") {
                model.model_type = MiniCPM_o_4_6;
            }
        }

        // other hparams
        {
            const char * prefix = "vision";
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            get_u32(KEY_IMAGE_SIZE, hparams.image_size);
            get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.preproc_image_size, false);
            get_u32(KEY_PATCH_SIZE, hparams.patch_size);
            get_u32(KEY_IMAGE_CROP_RESOLUTION, hparams.image_crop_resolution, false);
            get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false);
            get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
            if (hparams.minicpmv_query_num == 0) {
                // Fallback to hardcoded values for legacy models
                if (hparams.minicpmv_version == 20) {
                    hparams.minicpmv_query_num = 96;
                } else if (hparams.minicpmv_version == 25) {
                    hparams.minicpmv_query_num = 64;
                } else if (hparams.minicpmv_version == 26) {
                    hparams.minicpmv_query_num = 64;
                } else if (hparams.minicpmv_version == 40) {
                    hparams.minicpmv_query_num = 64;
                } else if (hparams.minicpmv_version == 45) {
                    hparams.minicpmv_query_num = 64;
                } else if (hparams.minicpmv_version == 100045) {
                    hparams.minicpmv_query_num = 64;
                } else {
                    hparams.minicpmv_query_num = 64;
                }
            }
            
            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
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
            
            int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
            int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
            GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
            GGML_ASSERT(idx_std >= 0  && "image_std not found");
            const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
            const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
            for (int i = 0; i < 3; ++i) {
                hparams.image_mean[i] = mean_data[i];
                hparams.image_std[i]  = std_data[i];
            }

            // Load the vision feature layer indices if they are explicitly provided;
            // if multiple vision feature layers are present, the values will be concatenated
            // to form the final visual features.
            // NOTE: gguf conversions should standardize the values of the vision feature layer to
            // be non-negative, since we use -1 to mark values as unset here.
            std::vector<int> vision_feature_layer;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer, false);
            // convert std::vector to std::unordered_set
            for (auto & layer : vision_feature_layer) {
                hparams.vision_feature_layer.insert(layer);
            }

            // model-specific params
            switch (model.model_type) {
                case MiniCPM_o:
                    {
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 25; // default to 20 if not set
                        }
                    } break;
                case MiniCPM_o_4_6:
                    {
                        hparams.minicpmv_version = 100046;
                        hparams.proj_scale_factor = 4;
                        hparams.minicpmv_max_slice_nums = 9;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.proj_scale_factor, false);

                        std::vector<int> wa_layer_indexes;
                        get_arr_int(KEY_WIN_ATTN_LAYER_INDEXES, wa_layer_indexes, false);
                        if (!wa_layer_indexes.empty()) {
                            hparams.insert_layer_id = wa_layer_indexes[0];
                        }
                    } break;
                default:
                    break;
            }

            LOG_INF("%s: model_type:         %d\n", __func__, model.model_type);
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            
            LOG_INF("\n--- vision hparams ---\n");
            LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
            LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
            LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
            LOG_INF("%s: proj_scale_factor:  %d\n", __func__, hparams.proj_scale_factor);
            LOG_INF("%s: n_wa_pattern:       %d\n", __func__, hparams.n_wa_pattern);
            LOG_INF("%s: insert_layer_id:     %d\n", __func__, hparams.insert_layer_id);
            
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(vision_ctx & ctx_vision) {
        auto & model = ctx_vision.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;
        const char * prefix = "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_vision.ctx_data.reset(ggml_init(params));
        if (!ctx_vision.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        auto get_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_vision.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings = get_tensor(TN_PATCH_EMBD,   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        // layers
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

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.model_type == MiniCPM_o || model.model_type == MiniCPM_o_4_6
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }

        switch (model.model_type) {
            case MiniCPM_o:
                {
                    // model.mm_model_pos_embed = get_tensor(new_vision->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case MiniCPM_o_4_6:
                {
                    model.vit_merger_ln1_w     = get_tensor(string_format(TN_VIT_MERGER_LN1, "weight"));
                    model.vit_merger_ln1_b     = get_tensor(string_format(TN_VIT_MERGER_LN1, "bias"));
                    model.vit_merger_attn_q_w  = get_tensor(string_format(TN_VIT_MERGER_ATTN_Q, "weight"));
                    model.vit_merger_attn_q_b  = get_tensor(string_format(TN_VIT_MERGER_ATTN_Q, "bias"), false);
                    model.vit_merger_attn_k_w  = get_tensor(string_format(TN_VIT_MERGER_ATTN_K, "weight"));
                    model.vit_merger_attn_k_b  = get_tensor(string_format(TN_VIT_MERGER_ATTN_K, "bias"), false);
                    model.vit_merger_attn_v_w  = get_tensor(string_format(TN_VIT_MERGER_ATTN_V, "weight"));
                    model.vit_merger_attn_v_b  = get_tensor(string_format(TN_VIT_MERGER_ATTN_V, "bias"), false);
                    model.vit_merger_attn_o_w  = get_tensor(string_format(TN_VIT_MERGER_ATTN_O, "weight"));
                    model.vit_merger_attn_o_b  = get_tensor(string_format(TN_VIT_MERGER_ATTN_O, "bias"), false);

                    model.vit_merger_ds_ln_w   = get_tensor(string_format(TN_VIT_MERGER_DS_LN, "weight"));
                    model.vit_merger_ds_ln_b   = get_tensor(string_format(TN_VIT_MERGER_DS_LN, "bias"));
                    model.vit_merger_ds_up_w   = get_tensor(string_format(TN_VIT_MERGER_DS_UP, "weight"));
                    model.vit_merger_ds_up_b   = get_tensor(string_format(TN_VIT_MERGER_DS_UP, "bias"), false);
                    model.vit_merger_ds_down_w = get_tensor(string_format(TN_VIT_MERGER_DS_DOWN, "weight"));
                    model.vit_merger_ds_down_b = get_tensor(string_format(TN_VIT_MERGER_DS_DOWN, "bias"), false);

                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B, false);
                    model.mm_ffn_up_w     = get_tensor(string_format(TN_MM_UP,   "weight"));
                    model.mm_ffn_up_b     = get_tensor(string_format(TN_MM_UP,   "bias"), false);
                    model.mm_ffn_down_w   = get_tensor(string_format(TN_MM_DOWN, "weight"));
                    model.mm_ffn_down_b   = get_tensor(string_format(TN_MM_DOWN, "bias"), false);
                } break;
            default:
                GGML_ASSERT(false && "unknown model type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            auto fin = std::ifstream(fname, std::ios::binary);
            if (!fin) {
                throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
            }

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_vision.backend);
            ctx_vision.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_vision.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_vision.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                ggml_tensor * cur = ggml_get_tensor(ctx_vision.ctx_data.get(), t->name);
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

    void alloc_compute_meta(vision_ctx & ctx_vision) {
        const auto & hparams = ctx_vision.model.hparams;
        ctx_vision.buf_compute_meta.resize(ctx_vision.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        // create a fake batch
        vision_image_f32_batch batch;
        vision_image_f32_ptr img(vision_image_f32_init());
        img->nx = hparams.warmup_image_size;    
        img->ny = hparams.warmup_image_size;
        batch.entries.push_back(std::move(img));

        ggml_cgraph * gf = vision_image_build_graph(&ctx_vision, batch);
        ggml_backend_sched_reserve(ctx_vision.sched.get(), gf);

        for (size_t i = 0; i < ctx_vision.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_vision.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_vision.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_vision.sched.get(), backend);
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

    void set_llava_uhd_res_candidates(vision_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(vision_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }
};

struct vision_ctx * vision_init(const char * fname, struct vision_context_params ctx_params) {
    g_logger_state.verbosity_thold = ctx_params.verbosity;
    vision_ctx * ctx_vision = nullptr;

    try {
        vision_model_loader loader(fname);
        ctx_vision = new vision_ctx(ctx_params);
        loader.load_hparams(ctx_vision->model);
        loader.load_tensors(*ctx_vision);
        loader.alloc_compute_meta(*ctx_vision);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        if (ctx_vision) {
            delete ctx_vision;
        }
        return nullptr;
    }
    return ctx_vision;
}

void vision_free(vision_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}


//
// vision processor
//
struct vision_image_u8 * vision_image_u8_init() {
    return new vision_image_u8();
}

struct vision_image_f32 * vision_image_f32_init() {
    return new vision_image_f32();
}

struct vision_image_f32_batch * vision_image_f32_batch_init() {
    return new vision_image_f32_batch();
}

void vision_image_u8_free(struct vision_image_u8  * img) { if (img) delete img; }
void vision_image_f32_free(struct vision_image_f32 * img) { if (img) delete img; }

static void normalize_image_u8_to_f32(const vision_image_u8 & src, vision_image_f32 & dst, const float mean[3], const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

struct image_manipulation {
    // Bilinear resize function
    static void bilinear_resize(const vision_image_u8& src, vision_image_u8& dst, int target_width, int target_height) {
        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float x_ratio = static_cast<float>(src.nx - 1) / target_width;
        float y_ratio = static_cast<float>(src.ny - 1) / target_height;

        for (int y = 0; y < target_height; y++) {
            for (int x = 0; x < target_width; x++) {
                float px = x_ratio * x;
                float py = y_ratio * y;
                int x_floor = static_cast<int>(px);
                int y_floor = static_cast<int>(py);
                float x_lerp = px - x_floor;
                float y_lerp = py - y_floor;

                for (int c = 0; c < 3; c++) {
                    float top = lerp(
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    float bottom = lerp(
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool bicubic_resize(const vision_image_u8 & img, vision_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5] = {};
        float d0, d2, d3, a0, a1, a2, a3;
        int i, j, k, jj;
        int x, y;
        float dx, dy;
        float tx, ty;

        tx = (float)nx / (float)target_width;
        ty = (float)ny / (float)target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int)(tx * j);
                y = (int)(ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                    }
                }
            }
        }

        return true;
    }

    // llava-1.6 type of resize_and_pad
    // if the ratio is not 1:1, padding with pad_color will be applied
    // pad_color is single channel, default is 0 (black)
    static void resize_and_pad_image(const vision_image_u8 & image, vision_image_u8 & dst, const vision_image_size & target_resolution, std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        int target_width  = target_resolution.width;
        int target_height = target_resolution.height;

        float scale_w = static_cast<float>(target_width) / image.nx;
        float scale_h = static_cast<float>(target_height) / image.ny;

        int new_width, new_height;

        if (scale_w < scale_h) {
            new_width  = target_width;
            new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
        } else {
            new_height = target_height;
            new_width  = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
        }

        vision_image_u8 resized_image;
        bicubic_resize(image, resized_image, new_width, new_height);

        vision_image_u8 padded_image;
        padded_image.nx = target_width;
        padded_image.ny = target_height;
        padded_image.buf.resize(3 * target_width * target_height);

        // Fill the padded image with the fill color
        for (size_t i = 0; i < padded_image.buf.size(); i += 3) {
            padded_image.buf[i]     = pad_color[0];
            padded_image.buf[i + 1] = pad_color[1];
            padded_image.buf[i + 2] = pad_color[2];
        }

        // Calculate padding offsets
        int pad_x = (target_width  - new_width)  / 2;
        int pad_y = (target_height - new_height) / 2;

        // Copy the resized image into the center of the padded buffer
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
                }
            }
        }
        dst = std::move(padded_image);
    }

    static void crop_image(const vision_image_u8 & image, vision_image_u8 & dst, int x, int y, int w, int h) {
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = 3 * ((y + i)*image.nx + (x + j));
                int dst_idx = 3 * (i*w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than max_dimension, it will be resized to max_dimension
    static vision_image_size calc_size_preserved_ratio(const vision_image_size & inp_size, const int align_size, const int max_dimension) {
        if (inp_size.width <= 0 || inp_size.height <= 0 || align_size <= 0 || max_dimension <= 0) {
            return {0, 0};
        }

        float scale = std::min(1.0f, std::min(static_cast<float>(max_dimension) / inp_size.width,
                                              static_cast<float>(max_dimension) / inp_size.height));

        float target_width_f  = static_cast<float>(inp_size.width)  * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        int aligned_width  = VISION_ALIGN((int)target_width_f,  align_size);
        int aligned_height = VISION_ALIGN((int)target_height_f, align_size);

        return {aligned_width, aligned_height};
    }

private:
    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};

struct llava_uhd {
    struct slice_coordinates {
        int x;
        int y;
        vision_image_size size;
    };

    struct slice_instructions {
        vision_image_size overview_size; // size of downscaled image
        vision_image_size refined_size;  // size of image right before slicing (must be multiple of slice size)
        vision_image_size grid_size;     // grid_size.width * grid_size.height = number of slices
        std::vector<slice_coordinates> slices;
        bool padding_refined = false;  // if true, refine image will be padded to the grid size (e.g. llava-1.6)
    };

    static slice_instructions get_slice_instructions(struct vision_ctx * ctx, const vision_image_size & original_size) {
        slice_instructions res;
        const int merge_factor    = is_minicpm_o_4_6(ctx->model) ? 4 : 1;
        const int patch_size      = ctx->model.hparams.patch_size * merge_factor;
        const int slice_size      = ctx->model.hparams.image_size;
        const int original_width  = original_size.width;
        const int original_height = original_size.height;

        const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;


        auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
        res.overview_size = best_size;

        {
            // 🔧 [高清模式] 优先使用运行时覆盖值，否则使用模型默认值
            const int max_slice_nums = (ctx->max_slice_nums_override >= 0) 
                                       ? ctx->max_slice_nums_override 
                                       : ctx->model.hparams.minicpmv_max_slice_nums;
            const float log_ratio = log((float)original_width / original_height);
            const float ratio = (float)original_width * original_height / (slice_size * slice_size);
            const int multiple = fmin(ceil(ratio), max_slice_nums);

            auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
            auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
            res.grid_size    = best_grid;
            res.refined_size = refine_size;

            LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                    __func__, original_width, original_height,
                    res.overview_size.width, res.overview_size.height,
                    res.refined_size.width, res.refined_size.height,
                    res.grid_size.width, res.grid_size.height);

            if (!has_slices || max_slice_nums == 0) {
                return res;
            }

            int width  = refine_size.width;
            int height = refine_size.height;
            int grid_x = int(width  / best_grid.width);
            int grid_y = int(height / best_grid.height);
            for (int patches_y = 0,                    ic = 0;
                    patches_y < refine_size.height && ic < best_grid.height;
                    patches_y += grid_y,              ic += 1) {
                for (int patches_x = 0,                   jc = 0;
                        patches_x < refine_size.width && jc < best_grid.width;
                        patches_x += grid_x,             jc += 1) {
                    slice_coordinates slice;
                    slice.x = patches_x;
                    slice.y = patches_y;
                    slice.size.width  = grid_x;
                    slice.size.height = grid_y;
                    res.slices.push_back(slice);
                    LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                            __func__, (int)res.slices.size() - 1,
                            slice.x, slice.y, slice.size.width, slice.size.height);
                }
            }
        }

        return res;
    }

    static std::vector<vision_image_u8_ptr> slice_image(const vision_image_u8 * img, const slice_instructions & inst) {
        std::vector<vision_image_u8_ptr> output;

        // resize to overview size
        vision_image_u8_ptr resized_img(vision_image_u8_init());
        image_manipulation::bicubic_resize(*img, *resized_img, inst.overview_size.width, inst.overview_size.height);
        output.push_back(std::move(resized_img));
        if (inst.slices.empty()) {
            // no slices, just return the resized image
            return output;
        }

        // resize to refined size
        vision_image_u8_ptr refined_img(vision_image_u8_init());
        if (inst.padding_refined) {
            image_manipulation::resize_and_pad_image(*img, *refined_img, inst.refined_size);
        } else {
            image_manipulation::bilinear_resize(*img, *refined_img, inst.refined_size.width, inst.refined_size.height);
        }

        // create slices
        for (const auto & slice : inst.slices) {
            int x = slice.x;
            int y = slice.y;
            int w = slice.size.width;
            int h = slice.size.height;

            vision_image_u8_ptr img_slice(vision_image_u8_init());
            image_manipulation::crop_image(*refined_img, *img_slice, x, y, w, h);
            output.push_back(std::move(img_slice));
        }

        return output;
    }

private:
    static vision_image_size get_best_resize(const vision_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
            float r = static_cast<float>(width) / height;
            height  = static_cast<int>(scale_resolution / std::sqrt(r));
            width   = static_cast<int>(height * r);
        }
        vision_image_size res;
        res.width  = ensure_divide(width,  patch_size);
        res.height = ensure_divide(height, patch_size);
        return res;
    }

    static vision_image_size resize_maintain_aspect_ratio(const vision_image_size & orig, const vision_image_size & target_max) {
        float scale_width  = static_cast<float>(target_max.width)  / orig.width;
        float scale_height = static_cast<float>(target_max.height) / orig.height;
        float scale = std::min(scale_width, scale_height);
        return vision_image_size{
            static_cast<int>(orig.width  * scale),
            static_cast<int>(orig.height * scale),
        };
    }

    /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * For example, when given a list of resolutions:
     *  - 100x100
     *  - 200x100
     *  - 100x200
     *  - 200x200
     *
     * And an input image of size 111x200, then 100x200 is the best fit (least wasted resolution).
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
    static vision_image_size select_best_resolution(const vision_image_size & original_size, const std::vector<vision_image_size> & possible_resolutions) {
        vision_image_size best_fit;
        int min_wasted_area = std::numeric_limits<int>::max();
        int max_effective_resolution = 0;

        for (const vision_image_size & candidate : possible_resolutions) {
            auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
            int effective_resolution = std::min(
                target_size.width * target_size.height,
                original_size.width * original_size.height);
            int wasted_area = (candidate.width * candidate.height) - effective_resolution;

            if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
                max_effective_resolution = effective_resolution;
                min_wasted_area = wasted_area;
                best_fit = candidate;
            }

            LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
        }

        return best_fit;
    }

    static int ensure_divide(int length, int patch_size) {
        return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
    }

    static vision_image_size get_refine_size(const vision_image_size & original_size, const vision_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        int grid_x = grid.width;
        int grid_y = grid.height;

        int refine_width  = ensure_divide(width, grid_x);
        int refine_height = ensure_divide(height, grid_y);

        vision_image_size grid_size;
        grid_size.width  = refine_width  / grid_x;
        grid_size.height = refine_height / grid_y;

        auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
        int best_grid_width  = best_grid_size.width;
        int best_grid_height = best_grid_size.height;

        vision_image_size refine_size;
        refine_size.width  = best_grid_width  * grid_x;
        refine_size.height = best_grid_height * grid_y;
        return refine_size;
    }

    static vision_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        std::vector<vision_image_size> candidate_grids;
        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.push_back(vision_image_size{m, split_grids_nums / m});
                }
                ++m;
            }
        }

        vision_image_size best_grid{1, 1};
        float min_error = std::numeric_limits<float>::infinity();
        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        return best_grid;
    }
};

bool vision_image_preprocess(struct vision_ctx * ctx, const vision_image_u8 * img, struct vision_image_f32_batch * res_imgs) {
    vision_image_size original_size{img->nx, img->ny};
    auto & params = ctx->model.hparams;

    switch (ctx->model.model_type) {
        case MiniCPM_o:
        case MiniCPM_o_4_6: {
            // TODO(o4.6): preprocess should follow mtmd llava-uhd (bilinear, (src-1)/(dst-1))
            //             see llama.cpp tools/mtmd/mtmd.cpp::PROJECTOR_TYPE_MINICPMV4_6
            //             (PR currently shares MiniCPM-o's bicubic path for the o-4.6 vit)
            auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
            std::vector<vision_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

            for (size_t i = 0; i < imgs.size(); ++i) {
                // vision_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
                vision_image_f32_ptr res(vision_image_f32_init());
                normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
                res_imgs->entries.push_back(std::move(res));
            }

            res_imgs->grid_x = inst.grid_size.width;
            res_imgs->grid_y = inst.grid_size.height;
            return true;
        }
        default:
            GGML_ABORT("Unknown image preprocessing type");
    }
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>> & pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}


//
// vision query
//
static int vision_n_output_tokens_for_image(const struct vision_ctx * ctx, const vision_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    int n_patches = 64;
    omni_model_type proj = ctx->model.model_type;

    switch (proj) {
        case MiniCPM_o:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 20) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 25) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 26) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 40) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 45) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else if (params.minicpmv_version == 100045) {
                        // MiniCPM-o 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case MiniCPM_o_4_6:
            {
                // ViT merger 4x + final merger 4x = 16x total spatial downsample
                const int image_width  = img ? img->nx : params.image_size;
                const int image_height = img ? img->ny : params.image_size;
                n_patches = (image_width / params.patch_size) * (image_height / params.patch_size) / 16;
            } break;
        default:
            GGML_ABORT("unsupported model type");
    }

    return n_patches;
}

int vision_n_output_tokens(const struct vision_ctx * ctx) {
    return vision_n_output_tokens_for_image(ctx, nullptr);
}

int vision_n_mmproj_embd(const struct vision_ctx * ctx) {
    switch (ctx->model.model_type) {
        case MiniCPM_o:
            return ctx->model.mm_model_proj->ne[0];
        case MiniCPM_o_4_6:
            return ctx->model.mm_ffn_down_w->ne[1];
        default:
            GGML_ABORT("Unknown model type");
    }
}


//
// vision forward
//
bool vision_image_batch_encode(vision_ctx * ctx, const int n_threads, const vision_image_f32_batch * imgs_c_ptr, float * vec) {
    const vision_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();

    if (batch_size < 1) {
        return false;
    }

    // verify all images in batch have the same dimensions
    if (batch_size > 1) {
        const int ref_nx = imgs.entries[0]->nx;
        const int ref_ny = imgs.entries[0]->ny;
        for (int b = 1; b < batch_size; b++) {
            if (imgs.entries[b]->nx != ref_nx || imgs.entries[b]->ny != ref_ny) {
                LOG_ERR("%s: batch image %d has size %dx%d, expected %dx%d\n",
                        __func__, b, imgs.entries[b]->nx, imgs.entries[b]->ny, ref_nx, ref_ny);
                return false;
            }
        }
        LOG_INF("%s: batched encoding %d images of size %dx%d\n", __func__, batch_size, ref_nx, ref_ny);
    }

    // build the inference graph
    ctx->debug_print_tensors.clear();
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = vision_image_build_graph(ctx, imgs);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches;
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

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

    // set input pixel values for all images in the batch
    {
        const int nx = imgs.entries[0]->nx;
        const int ny = imgs.entries[0]->ny;
        const int n = nx * ny;
        std::vector<float> inp_raw(3 * n * batch_size);

        for (int b = 0; b < batch_size; b++) {
            float * batch_entry = inp_raw.data() + b * (3 * n);
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    size_t base_src = 3 * (y * nx + x);
                    size_t base_dst =      y * nx + x;
                    batch_entry[          base_dst] = imgs.entries[b]->buf[base_src    ];
                    batch_entry[1 * n + base_dst]   = imgs.entries[b]->buf[base_src + 1];
                    batch_entry[2 * n + base_dst]   = imgs.entries[b]->buf[base_src + 2];
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);
    } 

    // set input per projector
    switch (ctx->model.model_type) {
        case MiniCPM_o:
            {
                std::vector<int32_t> positions(pos_h * pos_w);
                int bucket_coords_h[1024];
                int bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                int embed_dim = vision_n_mmproj_embd(ctx);

                auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

                // pos_embed is [embed_dim, n_pos, 1] — same for all batch elements (broadcast)
                std::vector<float> pos_embed(embed_dim * pos_w * pos_h);
                for(int i = 0; i < pos_w * pos_h; ++i){
                    for(int j = 0; j < embed_dim; ++j){
                        pos_embed[i * embed_dim + j] = pos_embed_t[i][j];
                    }
                }

                set_input_f32("pos_embed", pos_embed);
            } break;
        case MiniCPM_o_4_6:
            {
                std::vector<int32_t> positions(pos_h * pos_w);
                int bucket_coords_h[1024];
                int bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                const int half_h = pos_h / 2;
                const int half_w = pos_w / 2;

                std::vector<int32_t> window_idx(n_pos);
                std::vector<int32_t> inv_window_idx(n_pos);
                {
                    int k = 0;
                    for (int wi = 0; wi < half_h; wi++) {
                        for (int wj = 0; wj < half_w; wj++) {
                            window_idx[k++] = (2*wi    ) * pos_w + (2*wj    );
                            window_idx[k++] = (2*wi    ) * pos_w + (2*wj + 1);
                            window_idx[k++] = (2*wi + 1) * pos_w + (2*wj    );
                            window_idx[k++] = (2*wi + 1) * pos_w + (2*wj + 1);
                        }
                    }
                    for (int i = 0; i < n_pos; i++) {
                        inv_window_idx[window_idx[i]] = i;
                    }
                }
                set_input_i32("vit_merger_window_idx",     window_idx);
                set_input_i32("vit_merger_inv_window_idx", inv_window_idx);

                std::vector<float> window_mask_data(n_pos * n_pos, std::numeric_limits<float>::lowest());
                for (int wi = 0; wi < n_pos / 4; wi++) {
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            window_mask_data[(wi*4 + i) * n_pos + (wi*4 + j)] = 0.0f;
                        }
                    }
                }
                set_input_f32("vit_merger_window_mask", window_mask_data);

                auto make_ds_idx = [](int off_r, int off_c, int ds_h, int ds_w, int stride_w) {
                    std::vector<int32_t> idx(ds_h * ds_w);
                    for (int i = 0; i < ds_h; i++) {
                        for (int j = 0; j < ds_w; j++) {
                            idx[i * ds_w + j] = (2*i + off_r) * stride_w + (2*j + off_c);
                        }
                    }
                    return idx;
                };

                auto vit_merger_ds_0 = make_ds_idx(0, 0, half_h, half_w, pos_w);
                auto vit_merger_ds_1 = make_ds_idx(0, 1, half_h, half_w, pos_w);
                auto vit_merger_ds_2 = make_ds_idx(1, 0, half_h, half_w, pos_w);
                auto vit_merger_ds_3 = make_ds_idx(1, 1, half_h, half_w, pos_w);
                set_input_i32("vit_merger_ds_idx_0", vit_merger_ds_0);
                set_input_i32("vit_merger_ds_idx_1", vit_merger_ds_1);
                set_input_i32("vit_merger_ds_idx_2", vit_merger_ds_2);
                set_input_i32("vit_merger_ds_idx_3", vit_merger_ds_3);

                const int qh = half_h / 2;
                const int qw = half_w / 2;
                auto merger_ds_0 = make_ds_idx(0, 0, qh, qw, half_w);
                auto merger_ds_1 = make_ds_idx(0, 1, qh, qw, half_w);
                auto merger_ds_2 = make_ds_idx(1, 0, qh, qw, half_w);
                auto merger_ds_3 = make_ds_idx(1, 1, qh, qw, half_w);
                set_input_i32("merger_ds_idx_0", merger_ds_0);
                set_input_i32("merger_ds_idx_1", merger_ds_1);
                set_input_i32("merger_ds_idx_2", merger_ds_2);
                set_input_i32("merger_ds_idx_3", merger_ds_3);
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

    // print debug nodes
    if (ctx->debug_graph) {
        LOG_INF("\n\n---\n\n");
        LOG_INF("\n\nDebug graph:\n\n");
        for (ggml_tensor * t : ctx->debug_print_tensors) {
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        }
    }

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check: output shape should be [embd_dim, n_tokens, batch_size]
    const int n_tokens_out = embeddings->ne[1];
    const int expected_n_tokens_out = vision_n_output_tokens_for_image(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    // for batch>1, output is contiguous: [embd * n_tokens * batch_size]
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));

    return true;
}

bool vision_image_encode(struct vision_ctx * ctx, const int n_threads, vision_image_f32 * img, float * vec) {
    vision_image_f32_batch imgs;
    vision_image_f32_ptr img_copy(vision_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

#if defined(ENABLE_COREML)
    if (!ctx->coreml_model_path.empty()) {
        LOG_INF("vision use CoreML (ANE)\n");
        return vision_image_batch_encode_coreml(ctx, &imgs, vec);
    }
#endif
    return vision_image_batch_encode(ctx, n_threads, &imgs, vec);
}

// 🔧 [高清模式] 设置 max_slice_nums 覆盖值
void vision_set_max_slice_nums(struct vision_ctx * ctx, int max_slice_nums) {
    if (ctx) {
        ctx->max_slice_nums_override = max_slice_nums;
        LOG_INF("%s: max_slice_nums_override set to %d\n", __func__, max_slice_nums);
    }
}

// 🔧 [batch encode 开关] 设置/读取多 slice 批量编码优化
void vision_set_batch_encode(struct vision_ctx * ctx, bool enable) {
    if (ctx) {
        ctx->batch_encode = enable;
        LOG_INF("%s: vision batch encode %s\n", __func__, enable ? "enabled" : "disabled");
    }
}

bool vision_get_batch_encode(const struct vision_ctx * ctx) {
    return ctx ? ctx->batch_encode : false;
}

//
// CoreML / ANE support
//
void vision_set_coreml_model_path(struct vision_ctx * ctx, const char * coreml_model_path) {
    if (ctx && coreml_model_path) {
        ctx->coreml_model_path = std::string(coreml_model_path);
        LOG_INF("%s: CoreML model path set to: %s\n", __func__, coreml_model_path);
    }
}

// forward declaration of the static coreml encode function (defined below)
#if defined(__APPLE__) && defined(ENABLE_COREML)
static bool vision_image_encode_coreml(float * pixel_values, int32_t * position_ids, float * pos_embed_2d, float * vec, const char * coreml_model_path);
#endif

bool vision_coreml_warmup(struct vision_ctx * ctx) {
#if defined(__APPLE__) && defined(ENABLE_COREML)
    if (!ctx || ctx->coreml_model_path.empty()) {
        return false;
    }

    LOG_INF("%s: warming up vision CoreML/ANE model...\n", __func__);
    const int64_t t_start_us = ggml_time_us();

    // ANE model fixed input shapes:
    //   pixel_values:  [1, 3, 14, 14336]
    //   position_ids:  [1, 1024]
    //   pos_embed_2d:  [1024, 1, embed_dim]
    //   output:        [1, n_query, embed_dim]
    const int embed_dim = vision_n_mmproj_embd(ctx);
    const int n_query   = vision_n_output_tokens(ctx);
    const int n_pixels  = 3 * 14 * 14336;
    const int n_pos     = 1024;

    std::vector<float>   dummy_pixels(n_pixels, 0.0f);
    std::vector<int32_t> dummy_pos_ids(n_pos, 0);
    std::vector<float>   dummy_pos_embed(n_pos * embed_dim, 0.0f);
    std::vector<float>   dummy_output(n_query * embed_dim, 0.0f);

    // run a full dummy inference to trigger ANE model loading & compilation
    bool ok = vision_image_encode_coreml(
        dummy_pixels.data(), dummy_pos_ids.data(), dummy_pos_embed.data(),
        dummy_output.data(), ctx->coreml_model_path.c_str()
    );

    const int64_t t_end_us = ggml_time_us();
    if (ok) {
        LOG_INF("%s: vision CoreML/ANE warmup done in %.2f ms\n", __func__, (t_end_us - t_start_us) / 1000.0);
    } else {
        LOG_WRN("%s: vision CoreML/ANE warmup failed\n", __func__);
    }
    return ok;
#else
    (void)ctx;
    return false;
#endif
}

#if defined(ENABLE_COREML)
static bool vision_image_encode_coreml(float * pixel_values, int32_t * position_ids, float * pos_embed_2d, float * vec, const char * coreml_model_path) {
    static int flag = 0;
    static const void* coremlEncoder = NULL;
    static std::string cached_model_path = "";

    // Check if we need to load a new model
    if (flag == 0 || (coreml_model_path && cached_model_path != coreml_model_path)) {
        if (coremlEncoder) {
            omni_coreml_closeModel(coremlEncoder);
        }
        coremlEncoder = omni_coreml_loadModel(coreml_model_path);
        if (!coremlEncoder) {
            printf("Failed to load CoreML model from: %s\n", coreml_model_path ? coreml_model_path : "null");
            return false;
        }
        cached_model_path = coreml_model_path ? coreml_model_path : "";
        flag = 1;
    }
    omni_coreml_predictWith(coremlEncoder, pixel_values, position_ids, pos_embed_2d, vec);
    return true;
}
#endif

bool vision_image_batch_encode_coreml(vision_ctx * ctx, const vision_image_f32_batch * imgs_c_ptr, float * vec) {
#if defined(ENABLE_COREML)
    const vision_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();

    if (batch_size != 1) {
        return false; // only support batch size of 1
    }

    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;
    const int patch_size    = hparams.patch_size;
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

    std::vector<float> inp_raw;
    std::vector<int32_t> positions;
    std::vector<float> pos_embed;

    // prepare inp_raw: rearrange image pixels into patch layout for ANE
    // ANE model expects [1, 3, 14, 14336] where patches are laid out horizontally
    {
        const int max_patches = 1024;
        const int nx = max_patches * patch_size;
        const int ny = patch_size;
        const int n = nx * ny;
        inp_raw.assign(3 * n, 0.0f);

        int patch_index = 0;
        for (int i = 0; i < image_size_height && patch_index < max_patches; i += patch_size) {
            for (int j = 0; j < image_size_width && patch_index < max_patches; j += patch_size) {
                for (int pi = 0; pi < patch_size; ++pi) {
                    for (int pj = 0; pj < patch_size; ++pj) {
                        int src_index = ((i + pi) * image_size_width + (j + pj)) * 3;
                        int dst_index = nx * pi + patch_index * patch_size + pj;
                        inp_raw[dst_index]         = imgs.entries[0]->buf[src_index];
                        inp_raw[n + dst_index]     = imgs.entries[0]->buf[src_index + 1];
                        inp_raw[2 * n + dst_index] = imgs.entries[0]->buf[src_index + 2];
                    }
                }
                patch_index++;
            }
        }
    }

    // prepare position_ids
    {
        positions.assign(std::max(pos_h * pos_w, 1024), 0);
        int bucket_coords_h[1024];
        int bucket_coords_w[1024];
        for (int i = 0; i < pos_h; i++){
            bucket_coords_h[i] = std::floor(70.0*i/pos_h);
        }
        for (int i = 0; i < pos_w; i++){
            bucket_coords_w[i] = std::floor(70.0*i/pos_w);
        }
        for (int i = 0, id = 0; i < pos_h; i++){
            for (int j = 0; j < pos_w; j++){
                positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
            }
        }
    }

    // prepare pos_embed_2d
    {
        int embed_dim = vision_n_mmproj_embd(ctx);

        auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

        pos_embed.assign(embed_dim * std::max(pos_w * pos_h, 1024), 0.0f);
        for(int i = 0; i < pos_w * pos_h; ++i){
            for(int j = 0; j < embed_dim; ++j){
                pos_embed[i * embed_dim + j] = pos_embed_t[i][j];
            }
        }
    }

    return vision_image_encode_coreml(inp_raw.data(), positions.data(), pos_embed.data(), vec, ctx->coreml_model_path.c_str());
#else
    (void)ctx; (void)imgs_c_ptr; (void)vec;
    LOG_ERR("%s: CoreML support not compiled. Rebuild with -DENABLE_COREML=ON\n", __func__);
    return false;
#endif
}
