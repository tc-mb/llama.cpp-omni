

#include "token2wav-impl.h"

#include <cstdio>
#include <string>
#include <cmath>
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <vector>
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstring>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include "ggml-alloc.h"
#include <algorithm>
#include <random>

#ifndef ENABLE_STDERR_LOG
#define ENABLE_STDERR_LOG 0
#endif

#if ENABLE_STDERR_LOG
#ifndef LOG_ERROR
#define LOG_ERROR(...) std::fprintf(stderr, __VA_ARGS__)
#endif
#ifndef LOG_INFO
#define LOG_INFO(...) std::fprintf(stderr, __VA_ARGS__)
#endif
#else
#ifndef LOG_ERROR
#define LOG_ERROR(...) \
    do {               \
    } while (0)
#endif
#ifndef LOG_INFO
#define LOG_INFO(...) \
    do {              \
    } while (0)
#endif
#endif

namespace omni {
namespace flow {
namespace {
// 用于根据token ids生成嵌入特征(CTB)
ggml_tensor * flow_build_token_embedding_ctb(ggml_context * ctx,
                                        ggml_tensor *  token_embedding_weight,
                                        ggml_tensor *  token_ids_tb_i32,
                                        int32_t        input_size) {
    if (ctx == nullptr || token_embedding_weight == nullptr || token_ids_tb_i32 == nullptr) {
        return nullptr;
    }
    const int64_t T = token_ids_tb_i32->ne[0];
    const int64_t B = token_ids_tb_i32->ne[1];
    ggml_tensor * ids_1d = ggml_reshape_1d(ctx, token_ids_tb_i32, T * B);
    ggml_tensor * emb_2d = ggml_get_rows(ctx, token_embedding_weight, ids_1d);
    ggml_tensor * emb_3d = ggml_reshape_3d(ctx, emb_2d, input_size, T, B);
    return ggml_cont(ctx, emb_3d);
}
// 用于对spk向量做L2归一化(CB)
ggml_tensor * flow_build_l2_normalize_cb(ggml_context * ctx, ggml_tensor * x_cb, float eps) {
    if (ctx == nullptr || x_cb == nullptr) {
        return nullptr;
    }
    const int64_t C = x_cb->ne[0];
    const int64_t B = x_cb->ne[1];
    ggml_tensor * sq     = ggml_sqr(ctx, x_cb);
    ggml_tensor * sum_1b = ggml_sum_rows(ctx, sq);
    ggml_tensor * eps_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (!ggml_get_no_alloc(ctx)) {
        *static_cast<float *>(eps_1->data) = eps;
    }
    ggml_tensor * eps_11   = ggml_reshape_2d(ctx, eps_1, 1, 1);
    ggml_tensor * eps_1b   = ggml_repeat(ctx, eps_11, sum_1b);
    ggml_tensor * denom_1b = ggml_sqrt(ctx, ggml_add(ctx, sum_1b, eps_1b));
    ggml_tensor * denom_cb = ggml_repeat(ctx, denom_1b, x_cb);
    ggml_tensor * y        = ggml_div(ctx, x_cb, denom_cb);
    return ggml_cont(ctx, y);
}
// 用于给CB矩阵加bias
ggml_tensor * flow_build_add_bias_2d(ggml_context * ctx, ggml_tensor * y_cb, ggml_tensor * bias_c) {
    if (ctx == nullptr || y_cb == nullptr) {
        return nullptr;
    }
    if (bias_c == nullptr) {
        return y_cb;
    }
    ggml_tensor * bias_2d  = ggml_reshape_2d(ctx, bias_c, bias_c->ne[0], 1);
    ggml_tensor * tmpl     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y_cb->ne[0], y_cb->ne[1]);
    ggml_tensor * bias_rep = ggml_repeat(ctx, bias_2d, tmpl);
    return ggml_add(ctx, y_cb, bias_rep);
}
ggml_tensor * flow_build_linear_ctb(ggml_context * ctx, ggml_tensor * x_ctb, ggml_tensor * w_in_out, ggml_tensor * b_out) {
    // 用于线性投影(CTB)
    ggml_tensor * y = flow_matching::build_linear(ctx, x_ctb, w_in_out, b_out);
    return y ? ggml_cont(ctx, y) : nullptr;
}
ggml_tensor * flow_build_linear_cb(ggml_context * ctx, ggml_tensor * x_cb, ggml_tensor * w_in_out, ggml_tensor * b_out) {
    // 用于线性投影(CB)
    ggml_tensor * y = flow_matching::build_linear(ctx, x_cb, w_in_out, b_out);
    return y ? ggml_cont(ctx, y) : nullptr;
}
}  // namespace
flowCausalMaskedDiffWithXvec::flowCausalMaskedDiffWithXvec(
    int32_t                                                            input_size,
    int32_t                                                            output_size,
    int32_t                                                            spk_embed_dim,
    int32_t                                                            vocab_size,
    std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> encoder,
    std::shared_ptr<flow_matching::fmCausalConditionalCFM>             decoder) :
    input_size_(input_size),
    output_size_(output_size),
    spk_embed_dim_(spk_embed_dim),
    vocab_size_(vocab_size),
    encoder_(std::move(encoder)),
    decoder_(std::move(decoder)) {}
void flowCausalMaskedDiffWithXvec::set_parameters(ggml_tensor * token_embedding_weight,
                                                  ggml_tensor * spk_affine_weight,
                                                  ggml_tensor * spk_affine_bias,
                                                  ggml_tensor * encoder_proj_weight,
                                                  ggml_tensor * encoder_proj_bias) {
    token_embedding_weight_ = token_embedding_weight;
    spk_affine_weight_      = spk_affine_weight;
    spk_affine_bias_        = spk_affine_bias;
    encoder_proj_weight_    = encoder_proj_weight;
    encoder_proj_bias_      = encoder_proj_bias;
}
flowSetupCacheOut flowCausalMaskedDiffWithXvec::build_setup_cache_graph(
    ggml_context *              ctx,
    ggml_tensor *               token_ids_tb_i32,
    ggml_tensor *               mel_ctb_f32,
    ggml_tensor *               spk_cb_f32,
    int                         n_timesteps,
    float                       temperature,
    flow_matching::fmCFMCache * estimator_cache_out) const {
    flowSetupCacheOut out{};
    if (ctx == nullptr || token_ids_tb_i32 == nullptr || mel_ctb_f32 == nullptr || spk_cb_f32 == nullptr) {
        return out;
    }
    if (!encoder_ || !decoder_) {
        return out;
    }
    // setup_cache: 先跑encoder得到mu，再用decoder生成feat和缓存
    ggml_tensor * xs_ctb = flow_build_token_embedding_ctb(ctx, token_embedding_weight_, token_ids_tb_i32, input_size_);
    ggml_tensor * spk_norm_cb = flow_build_l2_normalize_cb(ctx, spk_cb_f32, 1e-12f);
    ggml_tensor * spk_proj_cb = flow_build_linear_cb(ctx, spk_norm_cb, spk_affine_weight_, spk_affine_bias_);
    auto enc_out = encoder_->forward_chunk(ctx, xs_ctb, false, nullptr, nullptr);
    ggml_tensor * mu_ctb = flow_build_linear_ctb(ctx, enc_out.ys_ctb, encoder_proj_weight_, encoder_proj_bias_);
    out.mu_ctb = mu_ctb;
    flow_matching::fmCFMCache   tmp_cache;
    flow_matching::fmCFMCache * cache_out_ptr = estimator_cache_out ? estimator_cache_out : &tmp_cache;
    if (cache_out_ptr) {
        cache_out_ptr->clear();
    }
    ggml_tensor * feat_ctb = decoder_->build_forward_chunk_graph(ctx, mu_ctb, spk_proj_cb, mel_ctb_f32, n_timesteps,
                                                                 temperature, nullptr, cache_out_ptr);
    out.feat_ctb            = ggml_cont(ctx, feat_ctb);
    out.conformer_cnn_cache = enc_out.new_cnn_cache_ctb;
    out.conformer_att_cache = enc_out.new_att_cache;
    out.estimator_cache     = cache_out_ptr;
    return out;
}
flowInferenceChunkOut flowCausalMaskedDiffWithXvec::build_inference_chunk_graph(
    ggml_context *                    ctx,
    ggml_tensor *                     token_ids_tb_i32,
    ggml_tensor *                     spk_cb_f32,
    bool                              last_chunk,
    ggml_tensor *                     conformer_cnn_cache_in,
    ggml_tensor *                     conformer_att_cache_in,
    const flow_matching::fmCFMCache * estimator_cache_in,
    int                               n_timesteps,
    float                             temperature,
    flow_matching::fmCFMCache *       estimator_cache_out) const {
    flowInferenceChunkOut out{};
    if (ctx == nullptr || token_ids_tb_i32 == nullptr || spk_cb_f32 == nullptr) {
        return out;
    }
    if (!encoder_ || !decoder_) {
        return out;
    }
    // inference_chunk: 复用conformer/CFM缓存进行流式推理
    ggml_tensor * xs_ctb = flow_build_token_embedding_ctb(ctx, token_embedding_weight_, token_ids_tb_i32, input_size_);
    ggml_tensor * spk_norm_cb = flow_build_l2_normalize_cb(ctx, spk_cb_f32, 1e-12f);
    ggml_tensor * spk_proj_cb = flow_build_linear_cb(ctx, spk_norm_cb, spk_affine_weight_, spk_affine_bias_);
    auto enc_out = encoder_->forward_chunk(ctx, xs_ctb, last_chunk, conformer_cnn_cache_in, conformer_att_cache_in);
    ggml_tensor * mu_ctb = flow_build_linear_ctb(ctx, enc_out.ys_ctb, encoder_proj_weight_, encoder_proj_bias_);
    ggml_tensor * cond_ctb = ggml_scale(ctx, mu_ctb, 0.0f);
    flow_matching::fmCFMCache   tmp_cache;
    flow_matching::fmCFMCache * cache_out_ptr = estimator_cache_out ? estimator_cache_out : &tmp_cache;
    if (cache_out_ptr) {
        cache_out_ptr->clear();
    }
    ggml_tensor * feat_ctb = decoder_->build_forward_chunk_graph(ctx, mu_ctb, spk_proj_cb, cond_ctb, n_timesteps,
                                                                 temperature, estimator_cache_in, cache_out_ptr);
    out.feat_ctb            = ggml_cont(ctx, feat_ctb);
    out.conformer_cnn_cache = enc_out.new_cnn_cache_ctb;
    out.conformer_att_cache = enc_out.new_att_cache;
    out.estimator_cache     = cache_out_ptr;
    return out;
}
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
// 读取 attention 缓存的最大长度配置
int64_t fm_attn_cache_max_t_default() {
    const char * env = ::getenv("TOKEN2WAV_FM_MAX_T_CACHE");
    if (env && env[0] != '\0') {
        const long v = ::strtol(env, nullptr, 10);
        if (v > 0) {
            return (int64_t) v;
        }
    }
    return 600;
}
// 从时间维尾部截取一段 view
ggml_tensor * fm_attn_tail_view_time_dim(ggml_context * ctx, ggml_tensor * x_dhtb, int64_t t_keep) {
    if (!x_dhtb) {
        return nullptr;
    }
    const int64_t D = x_dhtb->ne[0];
    const int64_t H = x_dhtb->ne[1];
    const int64_t T = x_dhtb->ne[2];
    const int64_t B = x_dhtb->ne[3];
    if (t_keep <= 0 || t_keep >= T) {
        return x_dhtb;
    }
    const int64_t drop = T - t_keep;
    const size_t  off  = (size_t) drop * x_dhtb->nb[2];
    return ggml_view_4d(ctx, x_dhtb, D, H, t_keep, B, x_dhtb->nb[1], x_dhtb->nb[2], x_dhtb->nb[3], off);
}
// 把通道维 reshape 成 head 维度
ggml_tensor * fm_attn_reshape_heads_4d(ggml_context * ctx, ggml_tensor * x, int head_dim, int num_heads, int64_t T, int64_t B) {
    if (!x) {
        return nullptr;
    }
    return ggml_reshape_4d(ctx, x, head_dim, num_heads, T, B);
}
ggml_tensor * fm_attn_flatten_heads_qk(ggml_context * ctx,
                               ggml_tensor *  heads,
                               int            head_dim,
                               int64_t        T,
                               int64_t        B,
                               int            num_heads) {
    if (!heads) {
        return nullptr;
    }
    ggml_tensor * permuted   = ggml_permute(ctx, heads, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, head_dim, T, (int64_t) num_heads * B);
}
ggml_tensor * fm_attn_flatten_heads_v(ggml_context * ctx,
                              ggml_tensor *  heads,
                              int            head_dim,
                              int64_t        T,
                              int64_t        B,
                              int            num_heads) {
    if (!heads) {
        return nullptr;
    }
    ggml_tensor * flat_qk    = fm_attn_flatten_heads_qk(ctx, heads, head_dim, T, B, num_heads);
    ggml_tensor * permuted   = ggml_permute(ctx, flat_qk, 1, 0, 2, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, T, head_dim, (int64_t) num_heads * B);
}
ggml_tensor * fm_attn_merge_heads_to_channels(ggml_context * ctx,
                                      ggml_tensor *  heads_flat,
                                      int            head_dim,
                                      int            num_heads,
                                      int64_t        T,
                                      int64_t        B) {
    if (!heads_flat) {
        return nullptr;
    }
    ggml_tensor * view4d     = ggml_reshape_4d(ctx, heads_flat, head_dim, T, num_heads, B);
    ggml_tensor * permuted   = ggml_permute(ctx, view4d, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, (int64_t) head_dim * num_heads, T, B);
}
ggml_tensor * fm_attn_apply_qk_norm(ggml_context * ctx,
                            ggml_tensor *  heads,
                            float          eps,
                            ggml_tensor *  weight,
                            ggml_tensor *  bias) {
    if (!heads) {
        return nullptr;
    }
    if (!weight || !bias) {
        return heads;
    }
    return build_layer_norm(ctx, heads, weight, bias, eps);
}
ggml_tensor * fm_attn_prepare_attn_mask(ggml_context * ctx,
                                ggml_tensor *  mask,
                                int64_t        T_k,
                                int64_t        T_q,
                                int64_t        B,
                                int            num_heads) {
    if (!mask) {
        return nullptr;
    }
    ggml_tensor * mask_4d = ggml_reshape_4d(ctx, mask, T_k, T_q, 1, B);
    ggml_tensor * tmpl_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T_k, T_q, num_heads, B);
    ggml_tensor * mask_hb = ggml_repeat(ctx, mask_4d, tmpl_4d);
    ggml_tensor * mask_hb_cont = ggml_cont(ctx, mask_hb);
    ggml_tensor * mask_bh      = ggml_reshape_3d(ctx, mask_hb_cont, T_k, T_q, B * num_heads);
    return mask_bh;
}
// 从缓存里裁出最近的片段
ggml_tensor * fm_attn_slice_att_cache(ggml_context * ctx, ggml_tensor * cache, int head_dim, bool value_slice) {
    if (!cache) {
        return nullptr;
    }
    const int64_t t_cache   = cache->ne[1];
    const int64_t num_heads = cache->ne[2];
    const int64_t B         = cache->ne[3];
    const size_t offset = value_slice ? (size_t) head_dim * cache->nb[0] : 0;
    return ggml_view_4d(ctx, cache, head_dim, t_cache, num_heads, B, cache->nb[1], cache->nb[2], cache->nb[3], offset);
}
ggml_tensor * fm_attn_permute_cache_to_heads(ggml_context * ctx, ggml_tensor * cache_slice) {
    if (!cache_slice) {
        return nullptr;
    }
    return ggml_permute(ctx, cache_slice, 0, 2, 1, 3);
}
ggml_tensor * fm_attn_build_new_att_cache(ggml_context * ctx, ggml_tensor * k_heads, ggml_tensor * v_heads) {
    if (!k_heads || !v_heads) {
        return nullptr;
    }
    ggml_tensor * k_perm = ggml_permute(ctx, k_heads, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx, v_heads, 0, 2, 1, 3);
    ggml_tensor * k_cont = ggml_cont(ctx, k_perm);
    ggml_tensor * v_cont = ggml_cont(ctx, v_perm);
    ggml_tensor * kv     = ggml_concat(ctx, k_cont, v_cont, 0);
    return ggml_cont(ctx, kv);
}
ggml_tensor * fm_attn_mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, const char * label) {
    const bool ok = (a && b) && (a->ne[0] == b->ne[0]) && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0);
    if (!ok) {
        LOG_ERROR(
                     "[fmAttention] mul_mat mismatch (%s): "
                     "a.ne=(%lld,%lld,%lld,%lld) b.ne=(%lld,%lld,%lld,%lld)\n",
                     label, (a ? a->ne[0] : -1LL), (a ? a->ne[1] : -1LL), (a ? a->ne[2] : -1LL), (a ? a->ne[3] : -1LL),
                     (b ? b->ne[0] : -1LL), (b ? b->ne[1] : -1LL), (b ? b->ne[2] : -1LL), (b ? b->ne[3] : -1LL));
    }
    return ggml_mul_mat(ctx, a, b);
}
inline float fm_attn_attn_scale(int head_dim) {
    return 1.0f / std::sqrt((float) head_dim);
}
}  // namespace
fmAttention::fmAttention(int   dim,
                         int   num_heads,
                         int   head_dim,
                         bool  qkv_bias,
                         bool  qk_norm,
                         float attn_drop,
                         float proj_drop,
                         float norm_eps) :
    dim_(dim),
    num_heads_(num_heads),
    head_dim_(head_dim),
    qkv_bias_(qkv_bias),
    qk_norm_(qk_norm),
    attn_drop_(attn_drop),
    proj_drop_(proj_drop),
    norm_eps_(norm_eps) {}
void fmAttention::set_parameters(ggml_tensor * to_q_weight,
                                 ggml_tensor * to_q_bias,
                                 ggml_tensor * to_k_weight,
                                 ggml_tensor * to_k_bias,
                                 ggml_tensor * to_v_weight,
                                 ggml_tensor * to_v_bias,
                                 ggml_tensor * q_norm_weight,
                                 ggml_tensor * q_norm_bias,
                                 ggml_tensor * k_norm_weight,
                                 ggml_tensor * k_norm_bias,
                                 ggml_tensor * proj_weight,
                                 ggml_tensor * proj_bias) {
    to_q_weight_ = to_q_weight;
    to_q_bias_   = to_q_bias;
    to_k_weight_ = to_k_weight;
    to_k_bias_   = to_k_bias;
    to_v_weight_ = to_v_weight;
    to_v_bias_   = to_v_bias;
    q_norm_weight_ = q_norm_weight;
    q_norm_bias_   = q_norm_bias;
    k_norm_weight_ = k_norm_weight;
    k_norm_bias_   = k_norm_bias;
    proj_weight_ = proj_weight;
    proj_bias_   = proj_bias;
}
// 构建注意力计算图并维护缓存
ggml_tensor * fmAttention::build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * attn_mask) const {
    if (!ctx || !x) {
        return nullptr;
    }
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    const int H = num_heads_;
    const int D = head_dim_;
    ggml_tensor * q = build_linear(ctx, x, to_q_weight_, to_q_bias_);
    ggml_tensor * k = build_linear(ctx, x, to_k_weight_, to_k_bias_);
    ggml_tensor * v = build_linear(ctx, x, to_v_weight_, to_v_bias_);
    ggml_tensor * q_heads = fm_attn_reshape_heads_4d(ctx, q, D, H, T, B);
    ggml_set_name(q_heads, "fm_att_q_heads");
    ggml_tensor * k_heads = fm_attn_reshape_heads_4d(ctx, k, D, H, T, B);
    ggml_set_name(k_heads, "fm_att_k_heads");
    ggml_tensor * v_heads = fm_attn_reshape_heads_4d(ctx, v, D, H, T, B);
    if (qk_norm_) {
        q_heads = fm_attn_apply_qk_norm(ctx, q_heads, norm_eps_, q_norm_weight_, q_norm_bias_);
        k_heads = fm_attn_apply_qk_norm(ctx, k_heads, norm_eps_, k_norm_weight_, k_norm_bias_);
    }
    ggml_tensor * q_flat = fm_attn_flatten_heads_qk(ctx, q_heads, D, T, B, H);
    ggml_tensor * k_flat = fm_attn_flatten_heads_qk(ctx, k_heads, D, T, B, H);
    ggml_tensor * v_flat = fm_attn_flatten_heads_v(ctx, v_heads, D, T, B, H);
    ggml_tensor * mask_bh = fm_attn_prepare_attn_mask(ctx, attn_mask, T, T, B, H);
    ggml_tensor * scores = fm_attn_mul_mat_checked(ctx, k_flat, q_flat, "forward/kq");
    scores               = ggml_scale(ctx, scores, fm_attn_attn_scale(D));
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask_bh, 1.0f, 0.0f);
    ggml_tensor * context = fm_attn_mul_mat_checked(ctx, v_flat, probs, "forward/context");
    ggml_set_name(context, "fm_att_context");
    ggml_tensor * merged = fm_attn_merge_heads_to_channels(ctx, context, D, H, T, B);
    ggml_tensor * y      = build_linear(ctx, merged, proj_weight_, proj_bias_);
    return y;
}
ggml_tensor * fmAttention::build_forward_chunk_graph(ggml_context * ctx,
                                                     ggml_tensor *  x,
                                                     ggml_tensor *  att_cache,
                                                     ggml_tensor *  attn_mask,
                                                     ggml_tensor ** new_att_cache) const {
    if (!ctx || !x) {
        if (new_att_cache) {
            *new_att_cache = nullptr;
        }
        return nullptr;
    }
    const int64_t C  = x->ne[0];
    const int64_t dt = x->ne[1];
    const int64_t B  = x->ne[2];
    const int H = num_heads_;
    const int D = head_dim_;
    ggml_tensor * q = build_linear(ctx, x, to_q_weight_, to_q_bias_);
    ggml_tensor * k = build_linear(ctx, x, to_k_weight_, to_k_bias_);
    ggml_tensor * v = build_linear(ctx, x, to_v_weight_, to_v_bias_);
    ggml_tensor * q_heads = fm_attn_reshape_heads_4d(ctx, q, D, H, dt, B);
    ggml_tensor * k_heads = fm_attn_reshape_heads_4d(ctx, k, D, H, dt, B);
    ggml_tensor * v_heads = fm_attn_reshape_heads_4d(ctx, v, D, H, dt, B);
    if (qk_norm_) {
        q_heads = fm_attn_apply_qk_norm(ctx, q_heads, norm_eps_, q_norm_weight_, q_norm_bias_);
        k_heads = fm_attn_apply_qk_norm(ctx, k_heads, norm_eps_, k_norm_weight_, k_norm_bias_);
    }
    int64_t       t_cache = 0;
    ggml_tensor * k_total = k_heads;
    ggml_tensor * v_total = v_heads;
    if (att_cache != nullptr) {
        t_cache = att_cache->ne[1];
        ggml_tensor * k_cache_slice = fm_attn_slice_att_cache(ctx, att_cache, D, false);
        ggml_tensor * v_cache_slice = fm_attn_slice_att_cache(ctx, att_cache, D, true);
        ggml_tensor * k_cache_heads = fm_attn_permute_cache_to_heads(ctx, k_cache_slice);
        ggml_tensor * v_cache_heads = fm_attn_permute_cache_to_heads(ctx, v_cache_slice);
        k_total = ggml_concat(ctx, k_heads, k_cache_heads, 2);
        v_total = ggml_concat(ctx, v_heads, v_cache_heads, 2);
    }
    const int64_t T_total     = dt + t_cache;
    const int64_t max_t_cache = fm_attn_cache_max_t_default();
    if (max_t_cache > 0 && T_total > max_t_cache) {
        k_total = fm_attn_tail_view_time_dim(ctx, k_total, max_t_cache);
        v_total = fm_attn_tail_view_time_dim(ctx, v_total, max_t_cache);
    }
    ggml_tensor * q_flat = fm_attn_flatten_heads_qk(ctx, q_heads, D, dt, B, H);
    const int64_t T_eff  = k_total->ne[2];
    ggml_tensor * k_flat = fm_attn_flatten_heads_qk(ctx, k_total, D, T_eff, B, H);
    ggml_tensor * v_flat = fm_attn_flatten_heads_v(ctx, v_total, D, T_eff, B, H);
    ggml_tensor * mask_bh = nullptr;
    if (attn_mask != nullptr) {
        mask_bh = fm_attn_prepare_attn_mask(ctx, attn_mask, T_eff, dt, B, H);
    }
    ggml_tensor * scores = fm_attn_mul_mat_checked(ctx, k_flat, q_flat, "chunk/kq");
    scores               = ggml_scale(ctx, scores, fm_attn_attn_scale(D));
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask_bh, 1.0f, 0.0f);
    ggml_tensor * context = fm_attn_mul_mat_checked(ctx, v_flat, probs, "chunk/context");
    ggml_tensor * merged = fm_attn_merge_heads_to_channels(ctx, context, D, H, dt, B);
    ggml_tensor * y      = build_linear(ctx, merged, proj_weight_, proj_bias_);
    if (new_att_cache) {
        *new_att_cache = fm_attn_build_new_att_cache(ctx, k_total, v_total);
    }
    return y;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
constexpr float kPi = 3.14159265358979323846f;
ggml_tensor * fm_cfm_view_att_cache_packed(ggml_context * ctx,
                                    ggml_tensor *  att_cache_packed,
                                    int64_t        head_dim,
                                    int64_t        num_heads,
                                    int64_t        t_cache,
                                    int64_t        B,
                                    int            step,
                                    int            depth,
                                    int            block_idx) {
    if (att_cache_packed == nullptr) {
        return nullptr;
    }
    const int64_t D2 = 2 * head_dim;
    const int64_t slot        = static_cast<int64_t>(step) * depth + block_idx;
    const int64_t head_offset = slot * num_heads;
    const size_t offset_bytes = static_cast<size_t>(head_offset) * att_cache_packed->nb[2];
    return ggml_view_4d(ctx, att_cache_packed, D2, t_cache, num_heads, B, att_cache_packed->nb[1],
                        att_cache_packed->nb[2], att_cache_packed->nb[3], offset_bytes);
}
ggml_tensor * fm_cfm_view_cnn_cache_packed(ggml_context * ctx,
                                    ggml_tensor *  cnn_cache_packed,
                                    int64_t        C_cache,
                                    int64_t        pad,
                                    int64_t        B,
                                    int            step,
                                    int            depth,
                                    int            block_idx) {
    if (cnn_cache_packed == nullptr) {
        return nullptr;
    }
    const int64_t slot = static_cast<int64_t>(step) * depth + block_idx;
    const size_t offset_bytes = static_cast<size_t>(slot) * cnn_cache_packed->nb[2];
    return ggml_view_3d(ctx, cnn_cache_packed, C_cache, pad, B, cnn_cache_packed->nb[1], cnn_cache_packed->nb[3],
                        offset_bytes);
}
ggml_tensor * fm_cfm_cnn_slot_to_4d(ggml_context * ctx,
                                           ggml_tensor *  cache3d,
                                           int64_t        C_cache,
                                           int64_t        pad,
                                           int64_t        B) {
    ggml_tensor * cont = ggml_cont(ctx, cache3d);
    return ggml_reshape_4d(ctx, cont, C_cache, pad, 1, B);
}
}  // namespace
fmCausalConditionalCFM::fmCausalConditionalCFM(std::shared_ptr<fmDiT> estimator, float inference_cfg_rate) :
    estimator_(std::move(estimator)),
    inference_cfg_rate_(inference_cfg_rate) {
    if (estimator_) {
        out_channels_ = estimator_->out_channels();
    }
}
int fmCausalConditionalCFM::out_channels() const {
    return out_channels_;
}
// 重置流式推理状态
void fmCausalConditionalCFM::reset_stream_state() const {
    chunk_call_id_ = 0;
}
// 生成噪声张量（使用真随机数）
ggml_tensor * fmCausalConditionalCFM::deterministic_noise(ggml_context * ctx,
                                                          int64_t        C,
                                                          int64_t        T,
                                                          int64_t        B,
                                                          float          temperature,
                                                          int64_t        offset_ct) {
    if (ctx == nullptr || C <= 0 || T <= 0 || B <= 0) {
        return nullptr;
    }
    ggml_tensor * noise = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, B);
    if (!ggml_get_no_alloc(ctx)) {
        float *       data  = static_cast<float *>(noise->data);
        const int64_t total = C * T * B;
        // 🔧 使用真随机数替代周期性伪噪声，避免音频伪影
        static std::mt19937 gen(42);  // 固定种子保证可复现
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int64_t i = 0; i < total; ++i) {
            data[i] = temperature * dist(gen);
        }
    }
    return noise;
}
// 生成填充常数时间步的张量
ggml_tensor * fmCausalConditionalCFM::build_timestep_tensor(ggml_context * ctx, int64_t B_total, float t_value) {
    if (ctx == nullptr || B_total <= 0) {
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, B_total);
    if (!ggml_get_no_alloc(ctx)) {
        float * data = static_cast<float *>(t->data);
        for (int64_t i = 0; i < B_total; ++i) {
            data[i] = t_value;
        }
    }
    return t;
}
// 生成推理用的时间步序列
void fmCausalConditionalCFM::build_cosine_t_span(int n_timesteps, std::vector<float> & t_span_out) {
    const int steps = n_timesteps > 0 ? n_timesteps : 1;
    t_span_out.resize(static_cast<std::size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i) {
        const float u                           = static_cast<float>(i) / static_cast<float>(steps);
        t_span_out[static_cast<std::size_t>(i)] = 1.0f - std::cos(u * 0.5f * kPi);
    }
}
// 构建 CFM 推理计算图
ggml_tensor * fmCausalConditionalCFM::build_forward_graph(ggml_context * ctx,
                                                          ggml_tensor *  mu,
                                                          ggml_tensor *  mask,
                                                          ggml_tensor *  spks,
                                                          ggml_tensor *  cond,
                                                          int            n_timesteps,
                                                          float          temperature) const {
    if (ctx == nullptr || estimator_ == nullptr || mu == nullptr) {
        return nullptr;
    }
    const int64_t C = mu->ne[0];
    const int64_t T = mu->ne[1];
    const int64_t B = mu->ne[2];
    ggml_tensor * x = deterministic_noise(ctx, C, T, B, temperature, 0);
    ggml_set_name(x, "fm_cfm_noise_forward");
    ggml_tensor * zeros_mu = ggml_scale(ctx, mu, 0.0f);
    ggml_tensor * mu_in    = ggml_concat(ctx, mu, zeros_mu, 2);
    ggml_set_name(mu_in, "fm_cfm_mu_in_forward");
    ggml_tensor * spks_in = nullptr;
    if (spks != nullptr) {
        const int n_dims = ggml_n_dims(spks);
        if (n_dims == 2) {
            const int64_t B_spk = spks->ne[1];
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in                 = ggml_concat(ctx, spks, zeros_spk, 1);
        } else if (n_dims == 3) {
            const int64_t one   = spks->ne[1];
            const int64_t B_spk = spks->ne[2];
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in                 = ggml_concat(ctx, spks, zeros_spk, 2);
        } else if (n_dims == 1) {
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in = ggml_concat(ctx, spks, zeros_spk, 1);
        }
    }
    ggml_tensor * cond_in = nullptr;
    if (cond != nullptr) {
        ggml_tensor * zeros_cond = ggml_scale(ctx, cond, 0.0f);
        cond_in                  = ggml_concat(ctx, cond, zeros_cond, 2);
        ggml_set_name(cond_in, "fm_cfm_cond_in_forward");
    }
    ggml_tensor * mask_in = nullptr;
    if (mask != nullptr) {
        mask_in = ggml_concat(ctx, mask, mask, 2);
    }
    std::vector<float> t_span;
    build_cosine_t_span(n_timesteps, t_span);
    const int steps = n_timesteps > 0 ? n_timesteps : 1;
    float t_scalar = t_span[0];
    float dt       = t_span.size() > 1 ? (t_span[1] - t_span[0]) : 1.0f;
    const int64_t B_total = 2 * B;
    for (int step_idx = 1; step_idx <= steps; ++step_idx) {
        ggml_tensor * x_in = ggml_concat(ctx, x, x, 2);
        ggml_tensor * t_in = build_timestep_tensor(ctx, B_total, t_scalar);
        {
            char name_buf[64];
            std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_t_in_forward_step%d", step_idx - 1);
            ggml_set_name(t_in, name_buf);
        }
        ggml_tensor * dphi_all = estimator_->build_forward_graph(ctx, x_in, mask_in, mu_in, t_in, spks_in, cond_in);
        ggml_tensor * dphi_main = ggml_view_3d(ctx, dphi_all, C, T, B, dphi_all->nb[1], dphi_all->nb[2], 0);
        const size_t  offset_cfg = static_cast<size_t>(B) * dphi_all->nb[2];
        ggml_tensor * dphi_cfg   = ggml_view_3d(ctx, dphi_all, C, T, B, dphi_all->nb[1], dphi_all->nb[2], offset_cfg);
        const float   cfg       = inference_cfg_rate_;
        ggml_tensor * term_main = ggml_scale(ctx, dphi_main, 1.0f + cfg);
        ggml_tensor * term_cfg  = ggml_scale(ctx, dphi_cfg, cfg);
        ggml_tensor * dphi      = ggml_sub(ctx, term_main, term_cfg);
        ggml_tensor * dphi_scaled = ggml_scale(ctx, dphi, dt);
        x                         = ggml_add(ctx, x, dphi_scaled);
        t_scalar += dt;
        if (step_idx < steps) {
            const float t_target = t_span[static_cast<std::size_t>(step_idx)];
            dt                   = t_span[static_cast<std::size_t>(step_idx + 1)] - t_target;
        }
    }
    return x;
}
ggml_tensor * fmCausalConditionalCFM::build_forward_chunk_graph(ggml_context *     ctx,
                                                                ggml_tensor *      mu,
                                                                ggml_tensor *      spks,
                                                                ggml_tensor *      cond,
                                                                int                n_timesteps,
                                                                float              temperature,
                                                                const fmCFMCache * cache_in,
                                                                fmCFMCache *       cache_out) const {
    if (ctx == nullptr || estimator_ == nullptr || mu == nullptr) {
        if (cache_out) {
            cache_out->clear();
        }
        return nullptr;
    }
    const int64_t C = mu->ne[0];
    const int64_t T = mu->ne[1];
    const int64_t B = mu->ne[2];
    const int depth = estimator_->depth();
    const int H     = estimator_->num_heads();
    const int D     = estimator_->head_dim();
    const int64_t B_total = 2 * B;
    ggml_tensor * att_cache_packed = (cache_in != nullptr ? cache_in->att_cache : nullptr);
    ggml_tensor * cnn_cache_packed = (cache_in != nullptr ? cache_in->cnn_cache : nullptr);
    int64_t last_att_len = 0;
    if (att_cache_packed != nullptr) {
        const int64_t n_time_in = att_cache_packed->ne[2] / (int64_t) (H * depth);
        last_att_len = att_cache_packed->ne[1];
    }
    if (cnn_cache_packed != nullptr) {
        const int64_t n_time_in = cnn_cache_packed->ne[2] / (int64_t) depth;
    }
    const int64_t offset_ct = last_att_len * C * B;
    ggml_tensor * x = deterministic_noise(ctx, C, T, B, temperature, offset_ct);
    const int call_id = chunk_call_id_++;
    {
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_noise_chunk%d", call_id);
        ggml_set_name(x, name_buf);
    }
    ggml_tensor * zeros_mu = ggml_scale(ctx, mu, 0.0f);
    ggml_tensor * mu_in    = ggml_concat(ctx, mu, zeros_mu, 2);
    if (cnn_cache_packed == nullptr && att_cache_packed == nullptr) {
        ggml_set_name(mu_in, "fm_cfm_mu_in_chunk0");
    }
    ggml_tensor * spks_in = nullptr;
    if (spks != nullptr) {
        const int n_dims = ggml_n_dims(spks);
        if (n_dims == 2) {
            const int64_t B_spk = spks->ne[1];
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in                 = ggml_concat(ctx, spks, zeros_spk, 1);
        } else if (n_dims == 3) {
            const int64_t one   = spks->ne[1];
            const int64_t B_spk = spks->ne[2];
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in                 = ggml_concat(ctx, spks, zeros_spk, 2);
        } else if (n_dims == 1) {
            ggml_tensor * zeros_spk = ggml_scale(ctx, spks, 0.0f);
            spks_in                 = ggml_concat(ctx, spks, zeros_spk, 1);
        }
    }
    ggml_tensor * cond_in = nullptr;
    if (cond != nullptr) {
        ggml_tensor * zeros_cond = ggml_scale(ctx, cond, 0.0f);
        cond_in                  = ggml_concat(ctx, cond, zeros_cond, 2);
    }
    std::vector<float> t_span;
    build_cosine_t_span(n_timesteps, t_span);
    const int steps = n_timesteps > 0 ? n_timesteps : 1;
    float t_scalar = t_span[0];
    float dt       = t_span.size() > 1 ? (t_span[1] - t_span[0]) : 1.0f;
    if (cache_out != nullptr) {
        cache_out->clear();
    }
    ggml_tensor * new_att_cache_packed = nullptr;
    ggml_tensor * new_cnn_cache_packed = nullptr;
    for (int step_idx = 1; step_idx <= steps; ++step_idx) {
        const int     step = step_idx - 1;
        ggml_tensor * x_in = ggml_concat(ctx, x, x, 2);
        ggml_tensor * t_in = build_timestep_tensor(ctx, B_total, t_scalar);
        {
            char name_buf[80];
            std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_t_in_chunk%d_step%d", call_id, step);
            ggml_set_name(t_in, name_buf);
        }
        std::vector<ggml_tensor *> prev_cnn_cache;
        std::vector<ggml_tensor *> prev_att_cache;
        prev_cnn_cache.assign((std::size_t) depth, nullptr);
        prev_att_cache.assign((std::size_t) depth, nullptr);
        if (att_cache_packed != nullptr) {
            const int64_t t_cache = last_att_len;
            for (int bi = 0; bi < depth; ++bi) {
                prev_att_cache[(std::size_t) bi] =
                    fm_cfm_view_att_cache_packed(ctx, att_cache_packed, D, H, t_cache, B_total, step, depth, bi);
            }
        }
        if (cnn_cache_packed != nullptr) {
            const int64_t C_cache = cnn_cache_packed->ne[0];
            const int64_t pad     = cnn_cache_packed->ne[1];
            for (int bi = 0; bi < depth; ++bi) {
                prev_cnn_cache[(std::size_t) bi] =
                    fm_cfm_view_cnn_cache_packed(ctx, cnn_cache_packed, C_cache, pad, B_total, step, depth, bi);
            }
        }
        std::vector<ggml_tensor *> new_cnn_vec;
        std::vector<ggml_tensor *> new_att_vec;
        ggml_tensor * dphi_all = estimator_->build_forward_chunk_graph(
            ctx, x_in, mu_in, t_in, spks_in, cond_in, prev_cnn_cache, prev_att_cache, new_cnn_vec, new_att_vec);
        ggml_tensor * dphi_main = ggml_view_3d(ctx, dphi_all, C, T, B, dphi_all->nb[1], dphi_all->nb[2], 0);
        const size_t  offset_cfg = static_cast<size_t>(B) * dphi_all->nb[2];
        ggml_tensor * dphi_cfg   = ggml_view_3d(ctx, dphi_all, C, T, B, dphi_all->nb[1], dphi_all->nb[2], offset_cfg);
        const float   cfg       = inference_cfg_rate_;
        ggml_tensor * term_main = ggml_scale(ctx, dphi_main, 1.0f + cfg);
        ggml_tensor * term_cfg  = ggml_scale(ctx, dphi_cfg, cfg);
        ggml_tensor * dphi      = ggml_sub(ctx, term_main, term_cfg);
        ggml_tensor * dphi_scaled = ggml_scale(ctx, dphi, dt);
        x                         = ggml_add(ctx, x, dphi_scaled);
        if (cache_out != nullptr) {
            ggml_tensor * step_att_pack = nullptr;
            for (int bi = 0; bi < depth; ++bi) {
                ggml_tensor * ac = new_att_vec[(std::size_t) bi];
                if (step_att_pack == nullptr) {
                    step_att_pack = ac;
                } else {
                    step_att_pack = ggml_concat(ctx, step_att_pack, ac, 2);
                }
            }
            step_att_pack = ggml_cont(ctx, step_att_pack);
            if (new_att_cache_packed == nullptr) {
                new_att_cache_packed = step_att_pack;
            } else {
                new_att_cache_packed = ggml_concat(ctx, new_att_cache_packed, step_att_pack, 2);
                new_att_cache_packed = ggml_cont(ctx, new_att_cache_packed);
            }
            const int64_t C_cache       = new_cnn_vec[0]->ne[0];
            const int64_t pad           = new_cnn_vec[0]->ne[1];
            ggml_tensor * step_cnn_pack = nullptr;
            for (int bi = 0; bi < depth; ++bi) {
                ggml_tensor * cc = new_cnn_vec[(std::size_t) bi];
                ggml_tensor * cc4 = fm_cfm_cnn_slot_to_4d(ctx, cc, C_cache, pad, B_total);
                if (step_cnn_pack == nullptr) {
                    step_cnn_pack = cc4;
                } else {
                    step_cnn_pack = ggml_concat(ctx, step_cnn_pack, cc4, 2);
                }
            }
            step_cnn_pack = ggml_cont(ctx, step_cnn_pack);
            if (new_cnn_cache_packed == nullptr) {
                new_cnn_cache_packed = step_cnn_pack;
            } else {
                new_cnn_cache_packed = ggml_concat(ctx, new_cnn_cache_packed, step_cnn_pack, 2);
                new_cnn_cache_packed = ggml_cont(ctx, new_cnn_cache_packed);
            }
        }
        t_scalar += dt;
        if (step_idx < steps) {
            const float t_target = t_span[static_cast<std::size_t>(step_idx)];
            dt                   = t_span[static_cast<std::size_t>(step_idx + 1)] - t_target;
        }
    }
    if (cache_out != nullptr) {
        cache_out->n_time    = steps;
        cache_out->depth     = depth;
        cache_out->num_heads = H;
        cache_out->head_dim  = D;
        cache_out->cnn_cache = new_cnn_cache_packed;
        cache_out->att_cache = new_att_cache_packed;
    }
    return x;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
// 用 im2col 计算一维卷积
static ggml_tensor * fm_causal_conv1d_im2col_f32_n1(ggml_context * ctx,
                                          ggml_tensor *  w_kic_oc,
                                          ggml_tensor *  x_tcb,
                                          int            stride,
                                          int            padding,
                                          int            dilation) {
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cin  = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    ggml_tensor * im2col = ggml_im2col(ctx, w_kic_oc, x_tcb, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, K * Cin, Cout);
    ggml_tensor * mm = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, im2col->ne[1], Cout, im2col->ne[2]);
    return y_tcb;
}
}  // namespace
fmCausalConv1d::fmCausalConv1d(int in_channels, int out_channels, int kernel_size) :
    in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size) {}
// 绑定卷积层参数
void fmCausalConv1d::set_parameters(ggml_tensor * weight, ggml_tensor * bias) {
    weight_ = weight;
    bias_   = bias;
}
// 构建整段卷积计算图
ggml_tensor * fmCausalConv1d::build_forward_graph(ggml_context * ctx, ggml_tensor * x) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    const int64_t Cin = x->ne[0];
    const int64_t B   = x->ne[2];
    const int64_t K     = weight_->ne[0];
    const int64_t Cin_w = weight_->ne[1];
    const int64_t Cout  = weight_->ne[2];
    ggml_tensor * x_tcb = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    const int     pad_left = static_cast<int>(K - 1);
    ggml_tensor * x_pad    = ggml_pad_ext(ctx, x_tcb, pad_left, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y_tcb = nullptr;
    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
        const size_t  offset = x_pad->nb[2] * static_cast<size_t>(b_idx);
        ggml_tensor * x_pad_b =
            ggml_view_3d(ctx, x_pad, x_pad->ne[0], x_pad->ne[1], 1, x_pad->nb[1], x_pad->nb[2], offset);
        ggml_tensor * y_tcb_b = fm_causal_conv1d_im2col_f32_n1(ctx, weight_, x_pad_b, 1, 0, 1);
        if (y_tcb == nullptr) {
            y_tcb = y_tcb_b;
        } else {
            y_tcb = ggml_concat(ctx, y_tcb, y_tcb_b, 2);
        }
    }
    ggml_tensor * y = ggml_permute(ctx, y_tcb, 1, 0, 2, 3);
    y               = ggml_cont(ctx, y);
    if (bias_ != nullptr) {
        ggml_tensor * bias_broadcast = ggml_reshape_3d(ctx, bias_, Cout, 1, 1);
        y                            = ggml_add(ctx, y, bias_broadcast);
    }
    return y;
}
// 构建分块卷积并更新缓存
ggml_tensor * fmCausalConv1d::build_forward_chunk_graph(ggml_context * ctx,
                                                        ggml_tensor *  x,
                                                        ggml_tensor *  cnn_cache,
                                                        ggml_tensor ** new_cache) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    const int64_t Cin = x->ne[0];
    const int64_t dt  = x->ne[1];
    const int64_t B   = x->ne[2];
    const int64_t K     = weight_->ne[0];
    const int64_t Cin_w = weight_->ne[1];
    const int64_t Cout  = weight_->ne[2];
    const int64_t pad = kernel_size_ - 1;
    ggml_tensor * cache_in = cnn_cache;
    if (cache_in == nullptr) {
        ggml_tensor * zero_x = ggml_scale(ctx, x, 0.0f);
        ggml_tensor * zero_x_pad = zero_x;
        if (dt < pad) {
            const int rp1 = (int) (pad - dt);
            zero_x_pad    = ggml_pad_ext(ctx, zero_x, 0, 0, 0, rp1, 0, 0, 0, 0);
        }
        cache_in = ggml_view_3d(ctx, zero_x_pad, Cin, pad, B, zero_x_pad->nb[1], zero_x_pad->nb[2], 0);
    } else {
    }
    cache_in = ggml_cont(ctx, cache_in);
    ggml_tensor * cache_tcb = ggml_permute(ctx, cache_in, 1, 0, 2, 3);
    cache_tcb               = ggml_cont(ctx, cache_tcb);
    ggml_tensor * x_tcb = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    ggml_tensor * x_cat_tcb = ggml_concat(ctx, cache_tcb, x_tcb, 0);
    x_cat_tcb               = ggml_cont(ctx, x_cat_tcb);
    ggml_tensor * y_tcb = nullptr;
    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
        const size_t  offset  = x_cat_tcb->nb[2] * static_cast<size_t>(b_idx);
        ggml_tensor * x_cat_b = ggml_view_3d(ctx, x_cat_tcb, x_cat_tcb->ne[0], x_cat_tcb->ne[1], 1, x_cat_tcb->nb[1],
                                             x_cat_tcb->nb[2], offset);
        ggml_tensor * y_tcb_b = fm_causal_conv1d_im2col_f32_n1(ctx, weight_, x_cat_b, 1, 0, 1);
        if (y_tcb == nullptr) {
            y_tcb = y_tcb_b;
        } else {
            y_tcb = ggml_concat(ctx, y_tcb, y_tcb_b, 2);
        }
    }
    ggml_tensor * y = ggml_permute(ctx, y_tcb, 1, 0, 2, 3);
    y               = ggml_cont(ctx, y);
    if (bias_ != nullptr) {
        ggml_tensor * bias_broadcast = ggml_reshape_3d(ctx, bias_, Cout, 1, 1);
        y                            = ggml_add(ctx, y, bias_broadcast);
    }
    if (new_cache != nullptr) {
        ggml_tensor * x_cont = x;
        if (x_cont->op != GGML_OP_RESHAPE && x_cont->op != GGML_OP_NONE) {
            x_cont = ggml_cont(ctx, x_cont);
        }
        ggml_tensor * x_cat_ctb = ggml_concat(ctx, cache_in, x_cont, 1);
        x_cat_ctb               = ggml_cont(ctx, x_cat_ctb);
        const size_t nb1    = x_cat_ctb->nb[1];
        const size_t nb2    = x_cat_ctb->nb[2];
        const size_t offset = nb1 * static_cast<size_t>(dt);
        ggml_tensor * tail_view = ggml_view_3d(ctx, x_cat_ctb, Cin, pad, B, nb1, nb2, offset);
        ggml_tensor * tail_ctb = ggml_cont(ctx, tail_view);
        *new_cache = tail_ctb;
    }
    return y;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
fmCausalConvBlock::fmCausalConvBlock(int in_channels, int out_channels, int kernel_size) :
    in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    transpose_ctb_btc_(0, 2),
    transpose_btc_ctb_(0, 2) {}
fmCausalConvBlock::~fmCausalConvBlock() {
    delete conv1_;
    delete conv2_;
}
// 绑定卷积块参数
void fmCausalConvBlock::set_parameters(ggml_tensor * conv1_weight,
                                       ggml_tensor * conv1_bias,
                                       ggml_tensor * conv2_weight,
                                       ggml_tensor * conv2_bias,
                                       ggml_tensor * ln_weight,
                                       ggml_tensor * ln_bias) {
    if (conv1_ == nullptr) {
        conv1_ = new fmCausalConv1d(in_channels_, out_channels_, kernel_size_);
    }
    if (conv2_ == nullptr) {
        conv2_ = new fmCausalConv1d(out_channels_, out_channels_, kernel_size_);
    }
    conv1_->set_parameters(conv1_weight, conv1_bias);
    conv2_->set_parameters(conv2_weight, conv2_bias);
    ln_weight_ = ln_weight;
    ln_bias_   = ln_bias;
}
// 构建卷积块计算图
ggml_tensor * fmCausalConvBlock::build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    ggml_tensor * cur = x;
    auto apply_mask_with_transpose = [&](ggml_tensor * tensor) -> ggml_tensor * {
        ggml_tensor * tensor_btc = transpose_ctb_btc_.build_forward_graph(ctx, tensor);
        ggml_tensor * mask_btc   = transpose_ctb_btc_.build_forward_graph(ctx, mask);
        ggml_tensor * masked_btc = ggml_mul(ctx, tensor_btc, mask_btc);
        return transpose_btc_ctb_.build_forward_graph(ctx, masked_btc);
    };
    if (mask != nullptr) {
        cur = apply_mask_with_transpose(cur);
    }
    cur = conv1_->build_forward_graph(ctx, cur);
    constexpr float kLnEps = 1e-5f;
    cur                    = build_layer_norm(ctx, cur, ln_weight_, ln_bias_, kLnEps);
    cur                    = build_mish(ctx, cur);
    cur = conv2_->build_forward_graph(ctx, cur);
    if (mask != nullptr) {
        cur = apply_mask_with_transpose(cur);
    }
    return cur;
}
// 构建分块卷积块并更新缓存
ggml_tensor * fmCausalConvBlock::build_forward_chunk_graph(ggml_context * ctx,
                                                           ggml_tensor *  x,
                                                           ggml_tensor *  cnn_cache,
                                                           ggml_tensor ** new_cnn_cache) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    const int64_t Cin  = in_channels_;
    const int64_t Cout = out_channels_;
    const int64_t dt   = x->ne[1];
    const int64_t B    = x->ne[2];
    const int64_t pad  = kernel_size_ - 1;
    (void) dt;
    ggml_tensor * cache1 = nullptr;
    ggml_tensor * cache2 = nullptr;
    if (cnn_cache != nullptr) {
        const size_t nb0 = cnn_cache->nb[0];
        const size_t nb1 = cnn_cache->nb[1];
        const size_t nb2 = cnn_cache->nb[2];
        cache1 = ggml_view_3d(ctx, cnn_cache, Cin, pad, B, nb1, nb2, 0);
        const size_t offset2 = nb0 * static_cast<size_t>(Cin);
        cache2               = ggml_view_3d(ctx, cnn_cache, Cout, pad, B, nb1, nb2, offset2);
    } else {
        cache1 = nullptr;
        cache2 = nullptr;
    }
    ggml_tensor * new_cache1 = nullptr;
    ggml_tensor * cur        = conv1_->build_forward_chunk_graph(ctx, x, cache1, &new_cache1);
    constexpr float kLnEps = 1e-5f;
    cur                    = build_layer_norm(ctx, cur, ln_weight_, ln_bias_, kLnEps);
    cur                    = build_mish(ctx, cur);
    ggml_tensor * new_cache2 = nullptr;
    cur                      = conv2_->build_forward_chunk_graph(ctx, cur, cache2, &new_cache2);
    if (new_cnn_cache != nullptr) {
        ggml_tensor * merged_cache = ggml_concat(ctx, new_cache1, new_cache2, 0);
        *new_cnn_cache             = merged_cache;
    }
    return cur;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
static thread_local fmDebugContext g_fm_debug_ctx;
void fm_debug_set_context(int call_id, int step, int block) {
    g_fm_debug_ctx.call_id = call_id;
    g_fm_debug_ctx.step    = step;
    g_fm_debug_ctx.block   = block;
}
void fm_debug_clear_context() {
    g_fm_debug_ctx.call_id = -1;
    g_fm_debug_ctx.step    = -1;
    g_fm_debug_ctx.block   = -1;
}
fmDebugContext fm_debug_get_context() {
    return g_fm_debug_ctx;
}
// 构建线性层计算
ggml_tensor * build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    if (weight == nullptr) {
        return x;
    }
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (bias != nullptr) {
        y = ggml_add(ctx, y, bias);
    }
    return y;
}
// 构建 layer norm 计算
ggml_tensor * build_layer_norm(ggml_context * ctx,
                               ggml_tensor *  x,
                               ggml_tensor *  weight,
                               ggml_tensor *  bias,
                               float          eps) {
    if (x == nullptr) {
        return nullptr;
    }
    ggml_tensor * cur = ggml_norm(ctx, x, eps);
    if (weight != nullptr) {
        cur = ggml_mul(ctx, cur, weight);
    }
    if (bias != nullptr) {
        cur = ggml_add(ctx, cur, bias);
    }
    return cur;
}
ggml_tensor * build_mish(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    ggml_tensor * zeros          = ggml_sub(ctx, x, x);
    ggml_tensor * ones           = ggml_exp(ctx, zeros);
    ggml_tensor * exp_x          = ggml_exp(ctx, x);
    ggml_tensor * exp_x_plus_one = ggml_add(ctx, exp_x, ones);
    ggml_tensor * softplus_x     = ggml_log(ctx, exp_x_plus_one);
    ggml_tensor * tanh_softplus  = ggml_tanh(ctx, softplus_x);
    ggml_tensor * y              = ggml_mul(ctx, x, tanh_softplus);
    return y;
}
ggml_tensor * build_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    ggml_tensor * cur = x;
    if (scale != nullptr) {
        ggml_tensor * scaled = ggml_mul(ctx, x, scale);
        cur                  = ggml_add(ctx, cur, scaled);
    }
    if (shift != nullptr) {
        cur = ggml_add(ctx, cur, shift);
    }
    return cur;
}
ggml_tensor * build_silu(ggml_context * ctx, ggml_tensor * x) {
    if (x == nullptr) {
        return nullptr;
    }
    return ggml_silu(ctx, x);
}
ggml_tensor * build_gelu(ggml_context * ctx, ggml_tensor * x) {
    if (x == nullptr) {
        return nullptr;
    }
    return ggml_gelu(ctx, x);
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
// 把 spks 扩展到时间维
ggml_tensor * fm_dit_broadcast_spks_over_time(ggml_context * ctx, ggml_tensor * spks, int64_t T, int64_t B) {
    if (ctx == nullptr || spks == nullptr) {
        return nullptr;
    }
    const int     n_dims  = ggml_n_dims(spks);
    ggml_tensor * spks_3d = nullptr;
    if (n_dims == 2) {
        spks_3d = ggml_reshape_3d(ctx, spks, spks->ne[0], 1, spks->ne[1]);
    } else if (n_dims == 3) {
        if (spks->ne[1] == 1 || spks->ne[1] == T) {
            spks_3d = spks;
        } else {
        }
    } else {
    }
    if (spks_3d->ne[1] == T) {
        return spks_3d;
    }
    const int64_t C_spk = spks_3d->ne[0];
    ggml_tensor * tmpl  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C_spk, T, B);
    return ggml_repeat(ctx, spks_3d, tmpl);
}
}  // namespace
fmDiT::fmDiT(int   in_channels,
             int   out_channels,
             float mlp_ratio,
             int   depth,
             int   num_heads,
             int   head_dim,
             int   hidden_size) :
    in_channels_(in_channels),
    out_channels_(out_channels),
    mlp_ratio_(mlp_ratio),
    depth_(depth),
    num_heads_(num_heads),
    head_dim_(head_dim),
    hidden_size_(hidden_size) {
    t_embedder_ = new fmTimestepEmbedder(hidden_size_, 256);
    blocks_.reserve(static_cast<std::size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        blocks_.push_back(new fmDiTBlock(hidden_size_, num_heads_, head_dim_, mlp_ratio_));
    }
    final_layer_ = new fmFinalLayer(hidden_size_, out_channels_);
}
fmDiT::~fmDiT() {
    delete t_embedder_;
    t_embedder_ = nullptr;
    for (fmDiTBlock * blk : blocks_) {
        delete blk;
    }
    blocks_.clear();
    delete final_layer_;
    final_layer_ = nullptr;
}
// 绑定 DiT 输入投影参数
void fmDiT::set_parameters(ggml_tensor * in_proj_weight, ggml_tensor * in_proj_bias) {
    in_proj_weight_ = in_proj_weight;
    in_proj_bias_   = in_proj_bias;
}
// 构建 DiT blocks 的计算图
ggml_tensor * fmDiT::build_blocks_forward_graph(ggml_context * ctx,
                                                ggml_tensor *  x,
                                                ggml_tensor *  t_embed,
                                                ggml_tensor *  attn_mask) const {
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    (void) B;
    ggml_tensor * x_hidden = build_linear(ctx, x, in_proj_weight_, in_proj_bias_);
    ggml_set_name(x_hidden, "fm_dit_x_after_in_proj");
    for (fmDiTBlock * blk : blocks_) {
        x_hidden = blk->build_forward_graph(ctx, x_hidden, t_embed, attn_mask);
    }
    ggml_set_name(x_hidden, "fm_dit_x_after_blocks");
    ggml_tensor * y = final_layer_->build_forward_graph(ctx, x_hidden, t_embed);
    ggml_set_name(y, "fm_dit_y");
    return y;
}
// 构建 DiT 前向计算图
ggml_tensor * fmDiT::build_forward_graph(ggml_context * ctx,
                                         ggml_tensor *  x,
                                         ggml_tensor *  mask,
                                         ggml_tensor *  mu,
                                         ggml_tensor *  t,
                                         ggml_tensor *  spks,
                                         ggml_tensor *  cond) const {
    if (ctx == nullptr || x == nullptr || mu == nullptr || t == nullptr) {
        return nullptr;
    }
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    (void) B;
    ggml_tensor * t_embed = t_embedder_->build_forward_graph(ctx, t);
    ggml_tensor * t_embed_3d = ggml_reshape_3d(ctx, t_embed, hidden_size_, 1, t_embed->ne[1]);
    ggml_set_name(t_embed_3d, "fm_dit_t_embed");
    ggml_tensor * x_cat = ggml_concat(ctx, x, mu, 0);
    ggml_set_name(x_cat, "fm_dit_x_cat");
    if (spks != nullptr) {
        ggml_tensor * spks_bt = fm_dit_broadcast_spks_over_time(ctx, spks, T, t_embed->ne[1]);
        x_cat                 = ggml_concat(ctx, x_cat, spks_bt, 0);
    }
    if (cond != nullptr) {
        x_cat = ggml_concat(ctx, x_cat, cond, 0);
    }
    return build_blocks_forward_graph(ctx, x_cat, t_embed_3d, mask);
}
ggml_tensor * fmDiT::build_blocks_forward_chunk_graph(ggml_context *                     ctx,
                                                      ggml_tensor *                      x,
                                                      ggml_tensor *                      t_embed,
                                                      ggml_tensor *                      mask,
                                                      const std::vector<ggml_tensor *> & prev_cnn_cache,
                                                      const std::vector<ggml_tensor *> & prev_att_cache,
                                                      std::vector<ggml_tensor *> &       new_cnn_cache,
                                                      std::vector<ggml_tensor *> &       new_att_cache) const {
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    (void) B;
    new_cnn_cache.clear();
    new_att_cache.clear();
    new_cnn_cache.reserve(blocks_.size());
    new_att_cache.reserve(blocks_.size());
    ggml_tensor * x_hidden = build_linear(ctx, x, in_proj_weight_, in_proj_bias_);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        fmDiTBlock *  blk      = blocks_[i];
        ggml_tensor * prev_cnn = nullptr;
        ggml_tensor * prev_att = nullptr;
        if (i < prev_cnn_cache.size()) {
            prev_cnn = prev_cnn_cache[i];
        }
        if (i < prev_att_cache.size()) {
            prev_att = prev_att_cache[i];
        }
        ggml_tensor * this_new_cnn = nullptr;
        ggml_tensor * this_new_att = nullptr;
        x_hidden = blk->build_forward_chunk_graph(ctx, x_hidden, t_embed, prev_cnn, prev_att, mask, &this_new_cnn,
                                                  &this_new_att);
        new_cnn_cache.push_back(this_new_cnn);
        new_att_cache.push_back(this_new_att);
    }
    ggml_tensor * y = final_layer_->build_forward_graph(ctx, x_hidden, t_embed);
    return y;
}
// 构建 DiT 分块前向并更新缓存
ggml_tensor * fmDiT::build_forward_chunk_graph(ggml_context *                     ctx,
                                               ggml_tensor *                      x,
                                               ggml_tensor *                      mu,
                                               ggml_tensor *                      t,
                                               ggml_tensor *                      spks,
                                               ggml_tensor *                      cond,
                                               const std::vector<ggml_tensor *> & prev_cnn_cache,
                                               const std::vector<ggml_tensor *> & prev_att_cache,
                                               std::vector<ggml_tensor *> &       new_cnn_cache,
                                               std::vector<ggml_tensor *> &       new_att_cache) const {
    if (ctx == nullptr || x == nullptr || mu == nullptr || t == nullptr) {
        new_cnn_cache.clear();
        new_att_cache.clear();
        return nullptr;
    }
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    (void) B;
    ggml_tensor * t_embed = t_embedder_->build_forward_graph(ctx, t);
    ggml_tensor * t_embed_3d = ggml_reshape_3d(ctx, t_embed, hidden_size_, 1, t_embed->ne[1]);
    ggml_tensor * x_cat = ggml_concat(ctx, x, mu, 0);
    if (spks != nullptr) {
        ggml_tensor * spks_bt = fm_dit_broadcast_spks_over_time(ctx, spks, T, t_embed->ne[1]);
        x_cat                 = ggml_concat(ctx, x_cat, spks_bt, 0);
    }
    if (cond != nullptr) {
        x_cat = ggml_concat(ctx, x_cat, cond, 0);
    }
    ggml_tensor * mask = nullptr;
    return build_blocks_forward_chunk_graph(ctx, x_cat, t_embed_3d, mask, prev_cnn_cache, prev_att_cache, new_cnn_cache,
                                            new_att_cache);
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
constexpr float kLnEps = 1e-6f;
// 从张量里取出一个标量 view
static ggml_tensor * fm_dit_blk_view_scalar_3d(ggml_context * ctx, ggml_tensor * t3d, int64_t i0, int64_t i1, int64_t i2) {
    if (ctx == nullptr || t3d == nullptr) {
        return nullptr;
    }
    const size_t offset = static_cast<size_t>(i0) * t3d->nb[0] + static_cast<size_t>(i1) * t3d->nb[1] +
                          static_cast<size_t>(i2) * t3d->nb[2];
    return ggml_view_1d(ctx, t3d, 1, offset);
}
// 把输入沿通道切成九段
void fm_dit_blk_split_into_9_chunks(ggml_context * ctx, ggml_tensor * input, int chunk_size, ggml_tensor ** out_chunks) {
    const int64_t ne1 = input->ne[1];
    const int64_t ne2 = input->ne[2];
    for (int i = 0; i < 9; ++i) {
        const size_t offset = static_cast<size_t>(i * chunk_size) * input->nb[0];
        out_chunks[i]       = ggml_view_3d(ctx, input, chunk_size, ne1, ne2, input->nb[1], input->nb[2], offset);
    }
}
}  // namespace
fmDiTBlock::fmDiTBlock(int hidden_size, int num_heads, int head_dim, float mlp_ratio) :
    hidden_size_(hidden_size),
    num_heads_(num_heads),
    head_dim_(head_dim),
    mlp_ratio_(mlp_ratio) {
    attn_ = new fmAttention(hidden_size, num_heads, head_dim, true, true, 0.0f, 0.0f, 1e-5f);
    conv_ = new fmCausalConvBlock(hidden_size, hidden_size, 3);
    int mlp_hidden = static_cast<int>(hidden_size * mlp_ratio);
    mlp_           = new fmMLP(hidden_size, mlp_hidden, hidden_size, "gelu", nullptr, true, 0.0f);
}
fmDiTBlock::~fmDiTBlock() {
    delete attn_;
    delete conv_;
    delete mlp_;
}
// 绑定 block 的归一化和调制参数
void fmDiTBlock::set_parameters(ggml_tensor * norm1_weight,
                                ggml_tensor * norm1_bias,
                                ggml_tensor * norm2_weight,
                                ggml_tensor * norm2_bias,
                                ggml_tensor * norm3_weight,
                                ggml_tensor * norm3_bias,
                                ggml_tensor * ada_weight,
                                ggml_tensor * ada_bias) {
    norm1_weight_ = norm1_weight;
    norm1_bias_   = norm1_bias;
    norm2_weight_ = norm2_weight;
    norm2_bias_   = norm2_bias;
    norm3_weight_ = norm3_weight;
    norm3_bias_   = norm3_bias;
    ada_weight_   = ada_weight;
    ada_bias_     = ada_bias;
}
void fmDiTBlock::set_attention_parameters(ggml_tensor * to_q_weight,
                                          ggml_tensor * to_q_bias,
                                          ggml_tensor * to_k_weight,
                                          ggml_tensor * to_k_bias,
                                          ggml_tensor * to_v_weight,
                                          ggml_tensor * to_v_bias,
                                          ggml_tensor * q_norm_weight,
                                          ggml_tensor * q_norm_bias,
                                          ggml_tensor * k_norm_weight,
                                          ggml_tensor * k_norm_bias,
                                          ggml_tensor * proj_weight,
                                          ggml_tensor * proj_bias) {
    if (attn_ != nullptr) {
        attn_->set_parameters(to_q_weight, to_q_bias, to_k_weight, to_k_bias, to_v_weight, to_v_bias, q_norm_weight,
                              q_norm_bias, k_norm_weight, k_norm_bias, proj_weight, proj_bias);
    }
}
void fmDiTBlock::set_conv_parameters(ggml_tensor * conv1_weight,
                                     ggml_tensor * conv1_bias,
                                     ggml_tensor * conv2_weight,
                                     ggml_tensor * conv2_bias,
                                     ggml_tensor * ln_weight,
                                     ggml_tensor * ln_bias) {
    if (conv_ != nullptr) {
        conv_->set_parameters(conv1_weight, conv1_bias, conv2_weight, conv2_bias, ln_weight, ln_bias);
    }
}
void fmDiTBlock::set_mlp_parameters(ggml_tensor * fc1_weight,
                                    ggml_tensor * fc1_bias,
                                    ggml_tensor * fc2_weight,
                                    ggml_tensor * fc2_bias) {
    if (mlp_ != nullptr) {
        mlp_->set_parameters(fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    }
}
// 构建 DiT block 的前向计算图
ggml_tensor * fmDiTBlock::build_forward_graph(ggml_context * ctx,
                                              ggml_tensor *  x,
                                              ggml_tensor *  c,
                                              ggml_tensor *  attn_mask) const {
    if (x == nullptr || c == nullptr) {
        return nullptr;
    }
    ggml_tensor * c_silu  = build_silu(ctx, c);
    ggml_tensor * ada_out = build_linear(ctx, c_silu, ada_weight_, ada_bias_);
    ggml_tensor * chunks[9];
    fm_dit_blk_split_into_9_chunks(ctx, ada_out, hidden_size_, chunks);
    ggml_tensor * shift_msa  = chunks[0];
    ggml_tensor * scale_msa  = chunks[1];
    ggml_tensor * gate_msa   = chunks[2];
    ggml_tensor * shift_mlp  = chunks[3];
    ggml_tensor * scale_mlp  = chunks[4];
    ggml_tensor * gate_mlp   = chunks[5];
    ggml_tensor * shift_conv = chunks[6];
    ggml_tensor * scale_conv = chunks[7];
    ggml_tensor * gate_conv  = chunks[8];
    ggml_tensor * x_norm1 = build_layer_norm(ctx, x, norm1_weight_, norm1_bias_, kLnEps);
    ggml_tensor * x_mod1  = build_modulate(ctx, x_norm1, shift_msa, scale_msa);
    ggml_tensor * x_attn  = attn_->build_forward_graph(ctx, x_mod1, attn_mask);
    ggml_tensor * x_attn_gated = ggml_mul(ctx, x_attn, gate_msa);
    x                          = ggml_add(ctx, x, x_attn_gated);
    ggml_tensor * x_norm3      = build_layer_norm(ctx, x, norm3_weight_, norm3_bias_, kLnEps);
    ggml_tensor * x_mod3       = build_modulate(ctx, x_norm3, shift_conv, scale_conv);
    ggml_tensor * x_conv       = conv_->build_forward_graph(ctx, x_mod3, nullptr);
    ggml_tensor * x_conv_gated = ggml_mul(ctx, x_conv, gate_conv);
    x                          = ggml_add(ctx, x, x_conv_gated);
    ggml_tensor * x_norm2     = build_layer_norm(ctx, x, norm2_weight_, norm2_bias_, kLnEps);
    ggml_tensor * x_mod2      = build_modulate(ctx, x_norm2, shift_mlp, scale_mlp);
    ggml_tensor * x_mlp       = mlp_->build_forward_graph(ctx, x_mod2);
    ggml_tensor * x_mlp_gated = ggml_mul(ctx, x_mlp, gate_mlp);
    x                         = ggml_add(ctx, x, x_mlp_gated);
    return x;
}
// 构建 DiT block 分块前向并更新缓存
ggml_tensor * fmDiTBlock::build_forward_chunk_graph(ggml_context * ctx,
                                                    ggml_tensor *  x,
                                                    ggml_tensor *  c,
                                                    ggml_tensor *  cnn_cache,
                                                    ggml_tensor *  att_cache,
                                                    ggml_tensor *  mask,
                                                    ggml_tensor ** new_cnn_cache,
                                                    ggml_tensor ** new_att_cache) const {
    if (x == nullptr || c == nullptr) {
        return nullptr;
    }
    ggml_tensor * c_silu  = build_silu(ctx, c);
    ggml_tensor * ada_out = build_linear(ctx, c_silu, ada_weight_, ada_bias_);
    ggml_tensor * chunks[9];
    fm_dit_blk_split_into_9_chunks(ctx, ada_out, hidden_size_, chunks);
    ggml_tensor * shift_msa  = chunks[0];
    ggml_tensor * scale_msa  = chunks[1];
    ggml_tensor * gate_msa   = chunks[2];
    ggml_tensor * shift_mlp  = chunks[3];
    ggml_tensor * scale_mlp  = chunks[4];
    ggml_tensor * gate_mlp   = chunks[5];
    ggml_tensor * shift_conv = chunks[6];
    ggml_tensor * scale_conv = chunks[7];
    ggml_tensor * gate_conv  = chunks[8];
    ggml_tensor * x_norm1 = build_layer_norm(ctx, x, norm1_weight_, norm1_bias_, kLnEps);
    ggml_tensor * x_mod1  = build_modulate(ctx, x_norm1, shift_msa, scale_msa);
    ggml_tensor * local_new_att_cache = nullptr;
    ggml_tensor * x_attn = attn_->build_forward_chunk_graph(ctx, x_mod1, att_cache, mask, &local_new_att_cache);
    ggml_tensor * x_attn_gated = ggml_mul(ctx, x_attn, gate_msa);
    x                          = ggml_add(ctx, x, x_attn_gated);
    ggml_tensor * x_norm3             = build_layer_norm(ctx, x, norm3_weight_, norm3_bias_, kLnEps);
    ggml_tensor * x_mod3              = build_modulate(ctx, x_norm3, shift_conv, scale_conv);
    ggml_tensor * local_new_cnn_cache = nullptr;
    ggml_tensor * x_conv              = conv_->build_forward_chunk_graph(ctx, x_mod3, cnn_cache, &local_new_cnn_cache);
    ggml_tensor * x_conv_gated        = ggml_mul(ctx, x_conv, gate_conv);
    x                                 = ggml_add(ctx, x, x_conv_gated);
    ggml_tensor * x_norm2     = build_layer_norm(ctx, x, norm2_weight_, norm2_bias_, kLnEps);
    ggml_tensor * x_mod2      = build_modulate(ctx, x_norm2, shift_mlp, scale_mlp);
    ggml_tensor * x_mlp       = mlp_->build_forward_graph(ctx, x_mod2);
    ggml_tensor * x_mlp_gated = ggml_mul(ctx, x_mlp, gate_mlp);
    x                         = ggml_add(ctx, x, x_mlp_gated);
    if (new_cnn_cache != nullptr) {
        *new_cnn_cache = local_new_cnn_cache;
    }
    if (new_att_cache != nullptr) {
        *new_att_cache = local_new_att_cache;
    }
    return x;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
// 初始化 final layer 的尺寸信息
fmFinalLayer::fmFinalLayer(int hidden_size, int out_channels) :
    hidden_size_(hidden_size),
    out_channels_(out_channels) {}
// 绑定 final layer 的参数
void fmFinalLayer::set_parameters(ggml_tensor * ada_weight,
                                  ggml_tensor * ada_bias,
                                  ggml_tensor * ln_weight,
                                  ggml_tensor * ln_bias,
                                  ggml_tensor * linear_weight,
                                  ggml_tensor * linear_bias) {
    ada_weight_    = ada_weight;
    ada_bias_      = ada_bias;
    ln_weight_     = ln_weight;
    ln_bias_       = ln_bias;
    linear_weight_ = linear_weight;
    linear_bias_   = linear_bias;
}
// 构建 final layer 的计算图
ggml_tensor * fmFinalLayer::build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * c) const {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    (void) T;
    (void) B;
    ggml_tensor * c_silu = build_silu(ctx, c);
    ggml_tensor * ada_out = build_linear(ctx, c_silu, ada_weight_, ada_bias_);
    const int64_t half = C;
    const size_t  nb0  = ada_out->nb[0];
    const size_t  nb1  = ada_out->nb[1];
    const size_t  nb2  = ada_out->nb[2];
    ggml_tensor * shift = ggml_view_3d(ctx, ada_out, half, 1, B, nb1, nb2, 0);
    ggml_tensor * scale = ggml_view_3d(ctx, ada_out, half, 1, B, nb1, nb2, half * nb0);
    constexpr float kLnEps = 1e-6f;
    ggml_tensor *   x_norm = build_layer_norm(ctx, x, ln_weight_, ln_bias_, kLnEps);
    ggml_tensor * x_mod = build_modulate(ctx, x_norm, shift, scale);
    ggml_tensor * y = build_linear(ctx, x_mod, linear_weight_, linear_bias_);
    return y;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
static void free_gguf_context(gguf_context * ctx) {
    if (ctx) {
        gguf_free(ctx);
    }
}
static void free_ggml_context(ggml_context * ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}
static void free_backend_buffer(ggml_backend_buffer_t buf) {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
}
}  // namespace
fmFlowMatchingModelLoaderGGUF::fmFlowMatchingModelLoaderGGUF() = default;
fmFlowMatchingModelLoaderGGUF::~fmFlowMatchingModelLoaderGGUF() {
    reset();
}
// 释放旧资源并重置状态
void fmFlowMatchingModelLoaderGGUF::reset() {
    tensors_.clear();
    free_backend_buffer(buf_weights_);
    buf_weights_ = nullptr;
    free_ggml_context(ctx_data_);
    ctx_data_ = nullptr;
    free_ggml_context(ctx_meta_);
    ctx_meta_ = nullptr;
    free_gguf_context(ctx_gguf_);
    ctx_gguf_ = nullptr;
    backend_ = nullptr;
    path_.clear();
}
// 从 gguf 加载权重到后端
bool fmFlowMatchingModelLoaderGGUF::load_from_file(const std::string & gguf_path, ggml_backend_t backend) {
    reset();
    if (!backend) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: backend is null\n");
        return false;
    }
    backend_ = backend;
    path_    = gguf_path;
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;
    ctx_gguf_ = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf_) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: failed to open gguf: %s\n", gguf_path.c_str());
        return false;
    }
    ctx_meta_ = meta;
    if (!ctx_meta_) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: gguf meta ctx is null: %s\n", gguf_path.c_str());
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_);
    if (n_tensors <= 0) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: no tensors in gguf: %s\n", gguf_path.c_str());
        return false;
    }
    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        const size_t off = gguf_get_data_offset(ctx_gguf_) + gguf_get_tensor_offset(ctx_gguf_, i);
        offsets.emplace(std::string(name), off);
    }
    ggml_init_params data_params{};
    data_params.mem_size   = static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_data_              = ggml_init(data_params);
    if (!ctx_data_) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: failed to init ctx_data\n");
        return false;
    }
    tensors_.reserve(static_cast<size_t>(n_tensors));
    std::vector<ggml_tensor *> to_load;
    to_load.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta_, name);
        if (!meta_tensor) {
            LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: missing meta tensor: %s\n", name);
            return false;
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data_, meta_tensor);
        ggml_set_name(data_tensor, name);
        tensors_.emplace(std::string(name), data_tensor);
        to_load.push_back(data_tensor);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    buf_weights_                    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data_, buft);
    if (!buf_weights_) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: backend weight alloc failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buf_weights_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: failed to open file stream: %s\n", gguf_path.c_str());
        return false;
    }
    std::vector<uint8_t> staging;
    for (ggml_tensor * t : to_load) {
        if (!t || !t->name) {
            continue;
        }
        const auto it = offsets.find(std::string(t->name));
        if (it == offsets.end()) {
            LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: missing offset for tensor: %s\n", t->name);
            return false;
        }
        const size_t off = it->second;
        fin.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!fin) {
            LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: seek failed for tensor: %s\n", t->name);
            return false;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(t->data), static_cast<std::streamsize>(nbytes));
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(nbytes));
            ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        }
        if (!fin) {
            LOG_ERROR( "fmFlowMatchingModelLoaderGGUF: read failed for tensor: %s\n", t->name);
            return false;
        }
    }
    return true;
}
// 按名称获取张量
ggml_tensor * fmFlowMatchingModelLoaderGGUF::get_tensor(const std::string & name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
// 生成基于长度的 pad mask
ggml_tensor * fmFlowMatchingMaskUtils::make_pad_mask(ggml_context *           ctx,
                                                     const std::vector<int> & lengths,
                                                     int                      max_len) {
    if (ctx == nullptr || lengths.empty()) {
        return nullptr;
    }
    const int batch_size       = static_cast<int>(lengths.size());
    int       inferred_max_len = 0;
    // 推断有效的 max len
    if (max_len > 0) {
        inferred_max_len = max_len;
    } else {
        for (int len : lengths) {
            if (len > inferred_max_len) {
                inferred_max_len = len;
            }
        }
    }
    if (inferred_max_len <= 0) {
        return nullptr;
    }
    ggml_tensor * lengths_vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch_size);
    ggml_set_name(lengths_vec, "pad_mask_lengths");
    ggml_tensor * range_vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, inferred_max_len);
    ggml_set_name(range_vec, "pad_mask_range");
    ggml_tensor * lengths_2d = ggml_reshape_2d(ctx, lengths_vec, 1, batch_size);
    ggml_tensor * range_2d   = ggml_reshape_2d(ctx, range_vec, inferred_max_len, 1);
    ggml_tensor * tmpl       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inferred_max_len, batch_size);
    ggml_tensor * lengths_bt = ggml_repeat(ctx, lengths_2d, tmpl);
    ggml_tensor * range_bt   = ggml_repeat(ctx, range_2d, tmpl);
    ggml_tensor * diff   = ggml_sub(ctx, range_bt, lengths_bt);
    ggml_tensor * mask_f = ggml_step(ctx, diff);
    return mask_f;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
// 初始化 MLP 的尺寸参数
fmMLP::fmMLP(int          in_features,
             int          hidden_features,
             int          out_features,
             const char * act_layer,
             const char * norm_layer,
             bool         bias,
             float        drop) :
    in_features_(in_features),
    hidden_features_(hidden_features > 0 ? hidden_features : in_features),
    out_features_(out_features > 0 ? out_features : in_features),
    act_layer_(act_layer),
    norm_layer_(norm_layer),
    bias_(bias),
    drop_(drop) {}
// 绑定 MLP 的权重参数
void fmMLP::set_parameters(ggml_tensor * fc1_weight,
                           ggml_tensor * fc1_bias,
                           ggml_tensor * fc2_weight,
                           ggml_tensor * fc2_bias) {
    fc1_weight_ = fc1_weight;
    fc1_bias_   = fc1_bias;
    fc2_weight_ = fc2_weight;
    fc2_bias_   = fc2_bias;
}
// 构建 MLP 前向计算图
ggml_tensor * fmMLP::build_forward_graph(ggml_context * ctx, ggml_tensor * x) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    ggml_tensor * h = build_linear(ctx, x, fc1_weight_, fc1_bias_);
    h = ggml_gelu(ctx, h);
    ggml_tensor * out = build_linear(ctx, h, fc2_weight_, fc2_bias_);
    return out;
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
namespace {
// 判断 backend 是否为设备后端
bool fm_loader_backend_is_device(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return !ggml_backend_buft_is_host(buft);
}
void backend_tensor_set(ggml_backend_t backend,
                                      ggml_tensor *  tensor,
                                      const void *   data,
                                      size_t         size_bytes) {
    if (!backend || !tensor || !data) {
        return;
    }
    if (fm_loader_backend_is_device(backend)) {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size_bytes);
    } else {
        ggml_backend_tensor_set(tensor, data, 0, size_bytes);
    }
}
// 把 gguf 权重绑定到 DiT 模型
void fm_loader_bind_all_weights(const fmFlowMatchingModelLoaderGGUF & loader, fmDiT & dit) {
    fmTimestepEmbedder * te = dit.timestep_embedder();
    te->set_parameters(
        loader.get_tensor("estimator.t_embedder.mlp.0.weight"), loader.get_tensor("estimator.t_embedder.mlp.0.bias"),
        loader.get_tensor("estimator.t_embedder.mlp.2.weight"), loader.get_tensor("estimator.t_embedder.mlp.2.bias"));
    dit.set_parameters(loader.get_tensor("estimator.in_proj.weight"), loader.get_tensor("estimator.in_proj.bias"));
    auto & blocks = dit.blocks();
    for (int i = 0; i < (int) blocks.size(); ++i) {
        fmDiTBlock * blk = blocks[(size_t) i];
        blk->set_parameters(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".adaLN_modulation.1.weight"),
                            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".adaLN_modulation.1.bias"));
        blk->set_attention_parameters(
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_q.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_q.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_k.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_k.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_v.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_v.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.q_norm.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.q_norm.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.k_norm.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.k_norm.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.proj.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.proj.bias"));
        blk->set_conv_parameters(loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.1.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.1.bias"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.6.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.6.bias"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.3.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.3.bias"));
        blk->set_mlp_parameters(loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc1.weight"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc1.bias"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc2.weight"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc2.bias"));
    }
    fmFinalLayer * fl = dit.final_layer();
    fl->set_parameters(loader.get_tensor("estimator.final_layer.adaLN_modulation.1.weight"),
                       loader.get_tensor("estimator.final_layer.adaLN_modulation.1.bias"), nullptr, nullptr,
                       loader.get_tensor("estimator.final_layer.linear.weight"),
                       loader.get_tensor("estimator.final_layer.linear.bias"));
}
// 初始化 CPU backend
ggml_backend_t fm_loader_init_backend_cpu(std::string & backend_name_out) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
// 解析设备字符串中的GPU索引 (e.g., "gpu:1" -> 1, "gpu" -> 0)
int parse_gpu_index(const std::string & device) {
    if (device.find("gpu:") == 0 && device.size() > 4) {
        try {
            return std::stoi(device.substr(4));
        } catch (...) {
            return 0;
        }
    }
    return 0; // default to first GPU
}
// 初始化指定索引的 GPU backend
ggml_backend_t fm_loader_init_backend_gpu_idx(int gpu_idx, std::string & backend_name_out) {
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    backend = ggml_backend_cuda_init(gpu_idx);
#endif
    if (!backend) {
        // fallback to generic GPU init
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
// 优先初始化 GPU backend (第一个GPU)
ggml_backend_t fm_loader_init_backend_gpu_first(std::string & backend_name_out) {
    return fm_loader_init_backend_gpu_idx(0, backend_name_out);
}
}  // namespace
fmFlowMatchingGGUFModelLoader::fmFlowMatchingGGUFModelLoader() = default;
fmFlowMatchingGGUFModelLoader::~fmFlowMatchingGGUFModelLoader() {
    reset();
}
void fmFlowMatchingGGUFModelLoader::reset() {
    weights_.reset();
    estimator_.reset();
    cfm_.reset();
    if (galloc_) {
        ggml_gallocr_free(galloc_);
        galloc_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    backend_name_.clear();
    gguf_path_.clear();
    chunk_call_id_ = 0;
}
bool fmFlowMatchingGGUFModelLoader::init_backend(const std::string & device) {
    if (backend_) {
        return true;
    }
    // Support "gpu", "gpu:0", "gpu:1" etc.
    if (device.find("gpu") == 0) {
        int gpu_idx = parse_gpu_index(device);
        backend_ = fm_loader_init_backend_gpu_idx(gpu_idx, backend_name_);
        std::fprintf(stderr, "fmFlowMatchingGGUFModelLoader: init_backend device=%s, gpu_idx=%d, backend=%s\n", 
                device.c_str(), gpu_idx, backend_name_.c_str());
    } else {
        backend_ = fm_loader_init_backend_cpu(backend_name_);
    }
    if (!backend_) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader: failed to init backend\n");
        return false;
    }
    return true;
}
void fmFlowMatchingGGUFModelLoader::set_num_threads(int n_threads) {
    num_threads_ = n_threads > 0 ? n_threads : 1;
}
void fmFlowMatchingGGUFModelLoader::reset_stream() {
    chunk_call_id_ = 0;
    if (cfm_) {
        cfm_->reset_stream_state();
    }
}
// 把权重张量绑定到模型结构
bool fmFlowMatchingGGUFModelLoader::bind_parameters() {
    if (!weights_ || !estimator_) {
        return false;
    }
    fm_loader_bind_all_weights(*weights_, *estimator_);
    return true;
}
// 从 gguf 加载并初始化模型
bool fmFlowMatchingGGUFModelLoader::load_from_gguf(const std::string & gguf_path, const std::string & device) {
    reset();
    gguf_path_ = gguf_path;
    if (!init_backend(device)) {
        return false;
    }
    weights_ = std::make_unique<fmFlowMatchingModelLoaderGGUF>();
    if (!weights_->load_from_file(gguf_path, backend_)) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader: gguf weight load failed: %s\n", gguf_path.c_str());
        reset();
        return false;
    }
    estimator_ =
        std::make_shared<fmDiT>(in_channels_, out_channels_, mlp_ratio_, depth_, num_heads_, head_dim_, hidden_size_);
    cfm_ = std::make_shared<fmCausalConditionalCFM>(estimator_, inference_cfg_rate_);
    if (!bind_parameters()) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader: parameter binding failed\n");
        reset();
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    galloc_                         = ggml_gallocr_new(buft);
    if (!galloc_) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader: failed to create ggml_gallocr\n");
        reset();
        return false;
    }
    reset_stream();
    return true;
}
void fmFlowMatchingGGUFModelLoader::build_cosine_t_span(int n_timesteps, std::vector<float> & t_span_out) {
    const int steps = n_timesteps > 0 ? n_timesteps : 1;
    t_span_out.resize((size_t) steps + 1);
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 0; i <= steps; ++i) {
        const float u          = (float) i / (float) steps;
        t_span_out[(size_t) i] = 1.0f - std::cos(u * 0.5f * kPi);
    }
}
void fmFlowMatchingGGUFModelLoader::fill_noise_ctb(std::vector<float> & noise_ctb,
                                                   int64_t              C,
                                                   int64_t              T,
                                                   int64_t              B,
                                                   float                temperature,
                                                   int64_t              offset_ct) {
    noise_ctb.resize((size_t) C * (size_t) T * (size_t) B);
    const int64_t total = C * T * B;
    // 🔧 使用真随机数替代周期性伪噪声，避免音频伪影
    static std::mt19937 gen(42);  // 固定种子保证可复现
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < total; ++i) {
        noise_ctb[(size_t) i] = temperature * dist(gen);
    }
}
void fmFlowMatchingGGUFModelLoader::fill_timestep_1d(std::vector<float> & t_host, int64_t B_total, float t_value) {
    t_host.resize((size_t) B_total);
    for (int64_t i = 0; i < B_total; ++i) {
        t_host[(size_t) i] = t_value;
    }
}
void fmFlowMatchingGGUFModelLoader::bct_to_ctb(const float *        bct,
                                               int64_t              B,
                                               int64_t              C,
                                               int64_t              T,
                                               std::vector<float> & ctb_out) {
    ctb_out.resize((size_t) B * (size_t) C * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t = 0; t < T; ++t) {
                const size_t idx_bct = (size_t) b * (size_t) C * (size_t) T + (size_t) c * (size_t) T + (size_t) t;
                const size_t idx_ctb = (size_t) c + (size_t) C * ((size_t) t + (size_t) T * (size_t) b);
                ctb_out[idx_ctb]     = bct[idx_bct];
            }
        }
    }
}
void fmFlowMatchingGGUFModelLoader::bc_to_cb(const float * bc, int64_t B, int64_t C, std::vector<float> & cb_out) {
    cb_out.resize((size_t) B * (size_t) C);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            const size_t idx_bc = (size_t) b * (size_t) C + (size_t) c;
            const size_t idx_cb = (size_t) c + (size_t) C * (size_t) b;
            cb_out[idx_cb]      = bc[idx_bc];
        }
    }
}
void fmFlowMatchingGGUFModelLoader::ctb_to_bct(const std::vector<float> & ctb,
                                               int64_t                    C,
                                               int64_t                    T,
                                               int64_t                    B,
                                               std::vector<float> &       bct_out) {
    bct_out.resize((size_t) B * (size_t) C * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t = 0; t < T; ++t) {
                const size_t idx_ctb = (size_t) c + (size_t) C * ((size_t) t + (size_t) T * (size_t) b);
                const size_t idx_bct = (size_t) b * (size_t) C * (size_t) T + (size_t) c * (size_t) T + (size_t) t;
                bct_out[idx_bct]     = ctb[idx_ctb];
            }
        }
    }
}
bool fmFlowMatchingGGUFModelLoader::read_tensor_3d_ctb_f32(ggml_backend_t       backend,
                                                           ggml_tensor *        t,
                                                           std::vector<float> & out_ctb) {
    if (!backend || !t) {
        return false;
    }
    const int64_t ne0    = t->ne[0];
    const int64_t ne1    = t->ne[1];
    const int64_t ne2    = t->ne[2];
    const size_t  nb0    = t->nb[0];
    const size_t  nb1    = t->nb[1];
    const size_t  nb2    = t->nb[2];
    const size_t  nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    out_ctb.resize((size_t) ne0 * (size_t) ne1 * (size_t) ne2);
    for (int64_t b = 0; b < ne2; ++b) {
        for (int64_t tt = 0; tt < ne1; ++tt) {
            for (int64_t c = 0; c < ne0; ++c) {
                const size_t  off = (size_t) c * nb0 + (size_t) tt * nb1 + (size_t) b * nb2;
                const float * p   = reinterpret_cast<const float *>(raw.data() + off);
                out_ctb[(size_t) c + (size_t) ne0 * ((size_t) tt + (size_t) ne1 * (size_t) b)] = *p;
            }
        }
    }
    return true;
}
bool fmFlowMatchingGGUFModelLoader::read_tensor_4d_corder_f32(ggml_backend_t       backend,
                                                              ggml_tensor *        t4,
                                                              std::vector<float> & out_corder) {
    if (!backend || !t4) {
        return false;
    }
    const int64_t ne0    = t4->ne[0];
    const int64_t ne1    = t4->ne[1];
    const int64_t ne2    = t4->ne[2];
    const int64_t ne3    = t4->ne[3];
    const size_t  nb0    = t4->nb[0];
    const size_t  nb1    = t4->nb[1];
    const size_t  nb2    = t4->nb[2];
    const size_t  nb3    = t4->nb[3];
    const size_t  nbytes = ggml_nbytes(t4);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t4, raw.data(), 0, nbytes);
    out_corder.resize((size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3);
    size_t idx = 0;
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    const size_t  off = (size_t) i0 * nb0 + (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3;
                    const float * p   = reinterpret_cast<const float *>(raw.data() + off);
                    out_corder[idx++] = *p;
                }
            }
        }
    }
    return true;
}
bool fmFlowMatchingGGUFModelLoader::forward(const float *        mu_bct,
                                            const float *        spks_bc,
                                            const float *        cond_bct,
                                            int64_t              B,
                                            int64_t              C,
                                            int64_t              T,
                                            int                  n_timesteps,
                                            float                temperature,
                                            std::vector<float> & y_bct_out) const {
    if (!backend_ || !cfm_) {
        return false;
    }
    if (!mu_bct || B <= 0 || C <= 0 || T <= 0) {
        return false;
    }
    const int64_t C_out = out_channels_;
    if (C != C_out) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader.forward: expected C=%lld, got %lld\n", (long long) C_out,
                     (long long) C);
        return false;
    }
    ggml_init_params params{};
    params.mem_size       = 2048ull * 1024ull * 1024ull;
    params.mem_buffer     = nullptr;
    params.no_alloc       = true;
    ggml_context * ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }
    const bool use_device_reorder = fm_loader_backend_is_device(backend_);
    ggml_tensor * mu_upload_bct   = nullptr;
    ggml_tensor * cond_upload_bct = nullptr;
    ggml_tensor * mu_t            = nullptr;
    ggml_tensor * cond_t          = nullptr;
    mu_upload_bct = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, C, B);
    mu_t          = ggml_cont(ctx, ggml_permute(ctx, mu_upload_bct, 1, 0, 2, 3));
    if (cond_bct) {
        cond_upload_bct = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, C, B);
        cond_t          = ggml_cont(ctx, ggml_permute(ctx, cond_upload_bct, 1, 0, 2, 3));
    }
    ggml_tensor * spks_t = spks_bc ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, B) : nullptr;
    ggml_tensor * y = cfm_->build_forward_graph(ctx, mu_t, nullptr, spks_t, cond_t, n_timesteps, temperature);
    if (!y) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * y_bct = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 64, false);
    ggml_build_forward_expand(gf, y_bct);
    if (!galloc_) {
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        ggml_free(ctx);
        return false;
    }
    backend_tensor_set(backend_, mu_upload_bct, mu_bct,
                                     (size_t) B * (size_t) C * (size_t) T * sizeof(float));
    if (cond_upload_bct) {
        backend_tensor_set(backend_, cond_upload_bct, cond_bct,
                                         (size_t) B * (size_t) C * (size_t) T * sizeof(float));
    }
    if (spks_t) {
        backend_tensor_set(backend_, spks_t, spks_bc, (size_t) B * (size_t) C * sizeof(float));
    }
    {
        if (ggml_tensor * t_noise = ggml_get_tensor(ctx, "fm_cfm_noise_forward")) {
            std::vector<float> noise_ctb;
            fill_noise_ctb(noise_ctb, t_noise->ne[0], t_noise->ne[1], t_noise->ne[2], temperature, 0);
            backend_tensor_set(backend_, t_noise, noise_ctb.data(), noise_ctb.size() * sizeof(float));
        }
        std::vector<float> t_span;
        build_cosine_t_span(n_timesteps, t_span);
        const int64_t B_total = 2 * B;
        for (int step = 0; step < n_timesteps; ++step) {
            char name_buf[64];
            std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_t_in_forward_step%d", step);
            if (ggml_tensor * tt = ggml_get_tensor(ctx, name_buf)) {
                std::vector<float> t_host;
                fill_timestep_1d(t_host, B_total, t_span[(size_t) step]);
                backend_tensor_set(backend_, tt, t_host.data(), t_host.size() * sizeof(float));
            }
        }
    }
    const ggml_status st      = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    if (use_device_reorder) {
        ggml_backend_synchronize(backend_);
    }
    {
        const size_t nbytes = ggml_nbytes(y_bct);
        y_bct_out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(y_bct, y_bct_out.data(), 0, nbytes);
    }
    ggml_free(ctx);
    return true;
}
bool fmFlowMatchingGGUFModelLoader::forward_chunk(const float *        mu_bct,
                                                  const float *        spks_bc,
                                                  const float *        cond_bct,
                                                  int64_t              B,
                                                  int64_t              C,
                                                  int64_t              T_chunk,
                                                  int                  n_timesteps,
                                                  float                temperature,
                                                  fmCFMCacheHost *     cache_in_out,
                                                  std::vector<float> & y_bct_out) {
    if (!backend_ || !cfm_) {
        return false;
    }
    if (!mu_bct || B <= 0 || C <= 0 || T_chunk <= 0) {
        return false;
    }
    const int64_t C_out = out_channels_;
    if (C != C_out) {
        LOG_ERROR( "fmFlowMatchingGGUFModelLoader.forward_chunk: expected C=%lld, got %lld\n",
                     (long long) C_out, (long long) C);
        return false;
    }
    const int call_id = chunk_call_id_;
    chunk_call_id_++;
    ggml_init_params params{};
    params.mem_size       = 2048ull * 1024ull * 1024ull;
    params.mem_buffer     = nullptr;
    params.no_alloc       = true;
    ggml_context * ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }
    const bool use_device_reorder = fm_loader_backend_is_device(backend_);
    ggml_tensor * mu_upload_bct   = nullptr;
    ggml_tensor * cond_upload_bct = nullptr;
    ggml_tensor * mu_t            = nullptr;
    ggml_tensor * cond_t          = nullptr;
    mu_upload_bct = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_chunk, C, B);
    mu_t          = ggml_cont(ctx, ggml_permute(ctx, mu_upload_bct, 1, 0, 2, 3));
    if (cond_bct) {
        cond_upload_bct = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_chunk, C, B);
        cond_t          = ggml_cont(ctx, ggml_permute(ctx, cond_upload_bct, 1, 0, 2, 3));
    }
    ggml_tensor * spks_t = spks_bc ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, B) : nullptr;
    fmCFMCache cache_in;
    cache_in.clear();
    ggml_tensor * cnn_in_t     = nullptr;
    ggml_tensor * att_in_t     = nullptr;
    ggml_tensor * cnn_upload_c = nullptr;
    ggml_tensor * att_upload_c = nullptr;
    int64_t    last_att_len = 0;
    if (cache_in_out && cache_in_out->has_cache()) {
        if (cache_in_out->n_time != n_timesteps) {
            LOG_ERROR( "forward_chunk: cache n_time mismatch (cache=%d, req=%d)\n", cache_in_out->n_time,
                         n_timesteps);
            ggml_free(ctx);
            return false;
        }
        cnn_upload_c       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cache_in_out->cnn_ne[3], cache_in_out->cnn_ne[2],
                                                cache_in_out->cnn_ne[1], cache_in_out->cnn_ne[0]);
        att_upload_c       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cache_in_out->att_ne[3], cache_in_out->att_ne[2],
                                                cache_in_out->att_ne[1], cache_in_out->att_ne[0]);
        cnn_in_t           = ggml_cont(ctx, ggml_permute(ctx, cnn_upload_c, 3, 2, 1, 0));
        att_in_t           = ggml_cont(ctx, ggml_permute(ctx, att_upload_c, 3, 2, 1, 0));
        cache_in.cnn_cache = cnn_in_t;
        cache_in.att_cache = att_in_t;
        cache_in.n_time    = cache_in_out->n_time;
        cache_in.depth     = cache_in_out->depth;
        cache_in.num_heads = cache_in_out->num_heads;
        cache_in.head_dim  = cache_in_out->head_dim;
        last_att_len       = cache_in_out->att_t_cache();
    }
    fmCFMCache cache_out;
    cache_out.clear();
    ggml_tensor * y = cfm_->build_forward_chunk_graph(ctx, mu_t, spks_t, cond_t, n_timesteps, temperature,
                                                      (cnn_in_t && att_in_t) ? &cache_in : nullptr, &cache_out);
    if (!y || !cache_out.cnn_cache || !cache_out.att_cache) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * y_bct     = nullptr;
    ggml_tensor * cnn_out_c = nullptr;
    ggml_tensor * att_out_c = nullptr;
    y_bct     = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    cnn_out_c = ggml_cont(ctx, ggml_permute(ctx, cache_out.cnn_cache, 3, 2, 1, 0));
    att_out_c = ggml_cont(ctx, ggml_permute(ctx, cache_out.att_cache, 3, 2, 1, 0));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 128, false);
    ggml_build_forward_expand(gf, y_bct);
    ggml_build_forward_expand(gf, cnn_out_c);
    ggml_build_forward_expand(gf, att_out_c);
    if (!galloc_) {
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        ggml_free(ctx);
        return false;
    }
    backend_tensor_set(backend_, mu_upload_bct, mu_bct,
                                     (size_t) B * (size_t) C * (size_t) T_chunk * sizeof(float));
    if (cond_upload_bct) {
        backend_tensor_set(backend_, cond_upload_bct, cond_bct,
                                         (size_t) B * (size_t) C * (size_t) T_chunk * sizeof(float));
    }
    if (spks_t) {
        backend_tensor_set(backend_, spks_t, spks_bc, (size_t) B * (size_t) C * sizeof(float));
    }
    if (cnn_in_t && att_in_t && cache_in_out) {
        backend_tensor_set(backend_, cnn_upload_c, cache_in_out->cnn.data(),
                                         cache_in_out->cnn.size() * sizeof(float));
        backend_tensor_set(backend_, att_upload_c, cache_in_out->att.data(),
                                         cache_in_out->att.size() * sizeof(float));
    }
    {
        const int64_t offset_ct = last_att_len * C * B;
        char          noise_name[64];
        std::snprintf(noise_name, sizeof(noise_name), "fm_cfm_noise_chunk%d", call_id);
        if (ggml_tensor * t_noise = ggml_get_tensor(ctx, noise_name)) {
            std::vector<float> noise_ctb;
            fill_noise_ctb(noise_ctb, t_noise->ne[0], t_noise->ne[1], t_noise->ne[2], temperature, offset_ct);
            backend_tensor_set(backend_, t_noise, noise_ctb.data(), noise_ctb.size() * sizeof(float));
        }
        std::vector<float> t_span;
        build_cosine_t_span(n_timesteps, t_span);
        const int64_t B_total = 2 * B;
        for (int step = 0; step < n_timesteps; ++step) {
            char name_buf[80];
            std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_t_in_chunk%d_step%d", call_id, step);
            if (ggml_tensor * tt = ggml_get_tensor(ctx, name_buf)) {
                std::vector<float> t_host;
                fill_timestep_1d(t_host, B_total, t_span[(size_t) step]);
                backend_tensor_set(backend_, tt, t_host.data(), t_host.size() * sizeof(float));
            }
        }
    }
    const ggml_status st      = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    if (use_device_reorder) {
        ggml_backend_synchronize(backend_);
    }
    {
        const size_t nbytes = ggml_nbytes(y_bct);
        y_bct_out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(y_bct, y_bct_out.data(), 0, nbytes);
    }
    if (cache_in_out) {
        cache_in_out->n_time    = n_timesteps;
        cache_in_out->depth     = depth_;
        cache_in_out->num_heads = num_heads_;
        cache_in_out->head_dim  = head_dim_;
        cache_in_out->cnn_ne = { cache_out.cnn_cache->ne[0], cache_out.cnn_cache->ne[1], cache_out.cnn_cache->ne[2],
                                 cache_out.cnn_cache->ne[3] };
        cache_in_out->att_ne = { cache_out.att_cache->ne[0], cache_out.att_cache->ne[1], cache_out.att_cache->ne[2],
                                 cache_out.att_cache->ne[3] };
        const size_t cnn_bytes     = ggml_nbytes(cnn_out_c);
        cache_in_out->cnn.resize(cnn_bytes / sizeof(float));
        ggml_backend_tensor_get(cnn_out_c, cache_in_out->cnn.data(), 0, cnn_bytes);
        const size_t att_bytes     = ggml_nbytes(att_out_c);
        cache_in_out->att.resize(att_bytes / sizeof(float));
        ggml_backend_tensor_get(att_out_c, cache_in_out->att.data(), 0, att_bytes);
    }
    ggml_free(ctx);
    return true;
}
}  // namespace flow_matching
}  // namespace omni
// 这一层主要做接口封装
// 保持实现细节集中在 common
namespace omni {
namespace flow_matching {
// 构建调制算子计算
ggml_tensor * fmModulateUtils::build_modulate(ggml_context * ctx,
                                              ggml_tensor *  x,
                                              ggml_tensor *  shift,
                                              ggml_tensor *  scale) {
    return build_modulate(ctx, x, shift, scale);
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace flow_matching {
// 初始化 timestep embedder 的尺寸参数
fmTimestepEmbedder::fmTimestepEmbedder(int hidden_size, int frequency_embedding_size) :
    hidden_size_(hidden_size),
    frequency_embedding_size_(frequency_embedding_size) {}
// 绑定 timestep embedder 的参数
void fmTimestepEmbedder::set_parameters(ggml_tensor * mlp1_weight,
                                        ggml_tensor * mlp1_bias,
                                        ggml_tensor * mlp2_weight,
                                        ggml_tensor * mlp2_bias) {
    mlp1_weight_ = mlp1_weight;
    mlp1_bias_   = mlp1_bias;
    mlp2_weight_ = mlp2_weight;
    mlp2_bias_   = mlp2_bias;
}
// 构建时间步嵌入计算图
ggml_tensor * fmTimestepEmbedder::build_forward_graph(ggml_context * ctx, ggml_tensor * t) const {
    if (ctx == nullptr || t == nullptr) {
        return nullptr;
    }
    const int     max_period = 10000;
    ggml_tensor * t_scaled   = ggml_scale(ctx, t, scale_);
    ggml_tensor * emb        = ggml_timestep_embedding(ctx, t_scaled, frequency_embedding_size_, max_period);
    ggml_tensor * h = build_linear(ctx, emb, mlp1_weight_, mlp1_bias_);
    h               = ggml_silu(ctx, h);
    ggml_tensor * out = build_linear(ctx, h, mlp2_weight_, mlp2_bias_);
    return out;
}
}  // namespace flow_matching
}  // namespace omni
// 这一层主要做接口封装
namespace omni {
namespace flow_matching {
// 初始化转置模块的维度参数
fmTransposeModule::fmTransposeModule(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}
// 构建转置计算图
ggml_tensor * fmTransposeModule::build_forward_graph(ggml_context * ctx, ggml_tensor * x) const {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    if (dim0_ < 0 || dim0_ >= GGML_MAX_DIMS || dim1_ < 0 || dim1_ >= GGML_MAX_DIMS || dim0_ == dim1_) {
        return x;
    }
    int perm[GGML_MAX_DIMS] = { 0, 1, 2, 3 };
    std::swap(perm[dim0_], perm[dim1_]);
    return ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
}
}  // namespace flow_matching
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
ueBaseSubsampling::ueBaseSubsampling() {
    right_context    = 0;
    subsampling_rate = 1;
}
// 生成位置编码
ggml_tensor * ueBaseSubsampling::position_encoding(ggml_context * ctx, int32_t offset, int32_t size) const {
    if (!ctx || !pos_enc_) {
        return nullptr;
    }
    return pos_enc_->position_encoding(ctx, offset, size);
}
// 生成位置编码
ggml_tensor * ueBaseSubsampling::position_encoding(ggml_context * ctx, std::nullptr_t, int32_t size) const {
    return position_encoding(ctx, 0, size);
}
ggml_tensor * ueBaseSubsampling::position_encoding(ggml_context * ctx, ggml_tensor *, int32_t size) const {
    return position_encoding(ctx, 0, size);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
// 常用算子封装
namespace omni {
namespace upsample_encoder_v2 {
// 构建线性层计算
ggml_tensor * ue_build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (!ctx || !x || !w) {
        return nullptr;
    }
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}
// 构建 layer norm 计算
ggml_tensor * ue_build_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    if (!ctx || !x) {
        return nullptr;
    }
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (w) {
        y = ggml_mul(ctx, y, w);
    }
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
// conformer 编码层实现
namespace omni {
namespace upsample_encoder_v2 {
ueConformerEncoderLayer::ueConformerEncoderLayer(int32_t                                            size,
                                                 std::shared_ptr<ueRelPositionMultiHeadedAttention> self_attn,
                                                 std::shared_ptr<uePositionwiseFeedForward>         feed_forward,
                                                 float                                              dropout_rate,
                                                 bool                                               normalize_before) :
    size_(size),
    dropout_rate_(dropout_rate),
    normalize_before_(normalize_before),
    self_attn_(std::move(self_attn)),
    feed_forward_(std::move(feed_forward)) {}
// 绑定归一化参数
void ueConformerEncoderLayer::set_parameters(ggml_tensor * norm_ff_weight,
                                             ggml_tensor * norm_ff_bias,
                                             ggml_tensor * norm_mha_weight,
                                             ggml_tensor * norm_mha_bias) {
    norm_ff_weight_  = norm_ff_weight;
    norm_ff_bias_    = norm_ff_bias;
    norm_mha_weight_ = norm_mha_weight;
    norm_mha_bias_   = norm_mha_bias;
}
// 构建编码层前向计算图
ggml_tensor * ueConformerEncoderLayer::build_forward_graph(ggml_context * ctx,
                                                           ggml_tensor *  x_ctb,
                                                           ggml_tensor *  mask,
                                                           ggml_tensor *  pos_emb,
                                                           ggml_tensor *  mask_pad,
                                                           ggml_tensor *  att_cache,
                                                           ggml_tensor *  cnn_cache,
                                                           ggml_tensor ** new_att_cache_out,
                                                           ggml_tensor ** new_cnn_cache_out) const {
    (void) dropout_rate_;
    (void) mask_pad;
    (void) cnn_cache;
    if (new_att_cache_out) {
        *new_att_cache_out = nullptr;
    }
    if (new_cnn_cache_out) {
        *new_cnn_cache_out = nullptr;
    }
    if (!ctx || !x_ctb || !self_attn_ || !feed_forward_) {
        return nullptr;
    }
    ggml_tensor * x = x_ctb;
    ggml_tensor * x_mha_in = x;
    if (normalize_before_) {
        x_mha_in = ue_build_layer_norm(ctx, x, norm_mha_weight_, norm_mha_bias_, 1e-12f);
    }
    ggml_tensor * att_out_cache = nullptr;
    ggml_tensor * x_att =
        self_attn_->build_forward_graph(ctx, x_mha_in, x_mha_in, x_mha_in, mask, pos_emb, att_cache, &att_out_cache);
    if (x_att) {
        x = ggml_add(ctx, x, x_att);
    }
    if (!normalize_before_) {
        x = ue_build_layer_norm(ctx, x, norm_mha_weight_, norm_mha_bias_, 1e-12f);
    }
    ggml_tensor * x_ff_in = x;
    if (normalize_before_) {
        x_ff_in = ue_build_layer_norm(ctx, x, norm_ff_weight_, norm_ff_bias_, 1e-12f);
    }
    ggml_tensor * x_ff = feed_forward_->build_forward_graph(ctx, x_ff_in);
    if (x_ff) {
        x = ggml_add(ctx, x, x_ff);
    }
    if (!normalize_before_) {
        x = ue_build_layer_norm(ctx, x, norm_ff_weight_, norm_ff_bias_, 1e-12f);
    }
    if (new_att_cache_out) {
        *new_att_cache_out = att_out_cache;
    }
    if (new_cnn_cache_out) {
        *new_cnn_cache_out = ggml_new_tensor_3d(ctx, x_ctb->type, 0, 0, 0);
    }
    return x;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
ueEspnetRelPositionalEncoding::ueEspnetRelPositionalEncoding(int32_t d_model, float dropout_rate, int32_t max_len) :
    d_model_(d_model),
    dropout_rate_(dropout_rate),
    max_len_(max_len) {}
// 扩展位置编码缓存
void ueEspnetRelPositionalEncoding::extend_pe(ggml_tensor * x) {
    if (!x) {
        return;
    }
    extend_pe((int32_t) x->ne[1]);
}
void ueEspnetRelPositionalEncoding::extend_pe(int32_t size) {
    if (size <= 0) {
        return;
    }
    const int32_t need = std::max(size, max_len_);
    if (max_len_ >= need && !pe_1td_.empty()) {
        return;
    }
    max_len_        = need;
    const int64_t L = 2LL * max_len_ - 1;
    pe_1td_.assign((size_t) (L * d_model_), 0.0f);
    const float   log_10000 = std::log(10000.0f);
    const int32_t half      = (d_model_ + 1) / 2;
    std::vector<float> div_term((size_t) half, 0.0f);
    for (int32_t k = 0; k < half; ++k) {
        const int32_t dim    = 2 * k;
        div_term[(size_t) k] = std::exp(-(log_10000 / (float) d_model_) * (float) dim);
    }
    for (int64_t t_out = 0; t_out < L; ++t_out) {
        int32_t pos = 0;
        bool    neg = false;
        if (t_out < max_len_) {
            pos = (int32_t) (max_len_ - 1 - t_out);
            neg = false;
        } else {
            pos = (int32_t) (t_out - max_len_ + 1);
            neg = true;
        }
        for (int32_t k = 0; k < half; ++k) {
            const float   angle = (float) pos * div_term[(size_t) k];
            const int32_t d0    = 2 * k;
            const int32_t d1    = d0 + 1;
            const float   s     = std::sin(angle);
            const float   c     = std::cos(angle);
            pe_1td_[(size_t) d0 + (size_t) d_model_ * (size_t) t_out] = neg ? -s : s;
            if (d1 < d_model_) {
                pe_1td_[(size_t) d1 + (size_t) d_model_ * (size_t) t_out] = c;
            }
        }
    }
}
// 生成位置编码
ggml_tensor * ueEspnetRelPositionalEncoding::position_encoding(ggml_context * ctx, int32_t offset, int32_t size) const {
    (void) offset;
    return build_position_encoding_placeholder(ctx, size, "ue_pos_emb");
}
// 生成位置编码
ggml_tensor * ueEspnetRelPositionalEncoding::position_encoding(ggml_context * ctx, std::nullptr_t, int32_t size) const {
    return position_encoding(ctx, 0, size);
    // 生成位置编码
}
ggml_tensor * ueEspnetRelPositionalEncoding::position_encoding(ggml_context * ctx, ggml_tensor *, int32_t size) const {
    return position_encoding(ctx, 0, size);
}
// 生成位置编码数据
void ueEspnetRelPositionalEncoding::position_encoding_host(int32_t size, std::vector<float> & out_1td) const {
    out_1td.clear();
    if (size <= 0) {
        return;
    }
    const_cast<ueEspnetRelPositionalEncoding *>(this)->extend_pe(size);
    const int64_t L      = 2LL * max_len_ - 1;
    const int64_t center = L / 2;
    const int64_t start  = center - size + 1;
    const int64_t end    = center + size;
    const int64_t len    = end - start;
    out_1td.resize((size_t) (len * d_model_), 0.0f);
    for (int64_t t = 0; t < len; ++t) {
        const int64_t src_t = start + t;
        const float * src   = pe_1td_.data() + (size_t) d_model_ * (size_t) src_t;
        float *       dst   = out_1td.data() + (size_t) d_model_ * (size_t) t;
        std::copy(src, src + d_model_, dst);
    }
}
ggml_tensor * ueEspnetRelPositionalEncoding::build_position_encoding_placeholder(ggml_context * ctx,
                                                                                 int32_t        size,
                                                                                 const char *   name) const {
    if (!ctx || size <= 0) {
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model_, 2LL * size - 1, 1);
    if (name) {
        ggml_set_name(t, name);
    }
    ggml_set_input(t);
    return t;
}
ggml_tensor * ueEspnetRelPositionalEncoding::build_forward_graph(ggml_context * ctx,
                                                                 ggml_tensor *  x_ctb,
                                                                 int32_t        offset,
                                                                 ggml_tensor ** pos_emb_placeholder_out) const {
    (void) dropout_rate_;
    (void) offset;
    if (!ctx || !x_ctb) {
        if (pos_emb_placeholder_out) {
            *pos_emb_placeholder_out = nullptr;
        }
        return nullptr;
    }
    const int64_t T   = x_ctb->ne[1];
    ggml_tensor * pos = build_position_encoding_placeholder(ctx, (int32_t) T, "ue_pos_emb");
    if (pos_emb_placeholder_out) {
        *pos_emb_placeholder_out = pos;
    }
    const float   scale    = std::sqrt((float) d_model_);
    ggml_tensor * x_scaled = ggml_scale(ctx, x_ctb, scale);
    return x_scaled;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
// 生成 pad mask
// 生成 pad mask
ggml_tensor * ueMaskUtils::make_pad_mask(ggml_context * ctx, ggml_tensor * lengths_b, int64_t max_len) {
    const int64_t B = lengths_b->ne[0];
    ggml_tensor * seq_range_t = ggml_arange(ctx, 0.0f, (float) max_len, 1.0f);
    ggml_tensor * seq_range_tb = ggml_repeat_4d(ctx, seq_range_t, max_len, B, 1, 1);
    ggml_tensor * lengths_f32_b =
        (lengths_b->type == GGML_TYPE_F32) ? lengths_b : ggml_cast(ctx, lengths_b, GGML_TYPE_F32);
    ggml_tensor * lengths_1b = ggml_reshape_2d(ctx, lengths_f32_b, 1, B);
    ggml_tensor * lengths_tb = ggml_repeat_4d(ctx, lengths_1b, max_len, B, 1, 1);
    ggml_tensor * len_minus_i_tb = ggml_sub(ctx, lengths_tb, seq_range_tb);
    ggml_tensor * valid_f32_tb   = ggml_step(ctx, len_minus_i_tb);
    ggml_tensor * one_f32    = ggml_arange(ctx, 1.0f, 2.0f, 1.0f);
    // 🔧 用 ggml_add + ggml_repeat 替代 ggml_add1，支持 Metal 加速
    ggml_tensor * neg_valid  = ggml_neg(ctx, valid_f32_tb);
    ggml_tensor * one_rep    = ggml_repeat(ctx, one_f32, neg_valid);
    ggml_tensor * pad_f32_tb = ggml_add(ctx, neg_valid, one_rep);
    ggml_tensor * mask_i32_tb = ggml_cast(ctx, pad_f32_tb, GGML_TYPE_I32);
    return ggml_reshape_2d(ctx, mask_i32_tb, max_len, B);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
// 判断 backend 是否在设备侧
bool ue_loader_backend_is_device(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return !ggml_backend_buft_is_host(buft);
}
void backend_tensor_set(ggml_backend_t backend,
                                      ggml_tensor *  tensor,
                                      const void *   data,
                                      size_t         size_bytes) {
    if (!backend || !tensor || !data || size_bytes == 0) {
        return;
    }
    if (ue_loader_backend_is_device(backend)) {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size_bytes);
    } else {
        ggml_backend_tensor_set(tensor, data, 0, size_bytes);
    }
}
// 绑定 gguf 权重
void ue_loader_bind_all_weights(const ueUpsampleEncoderModelLoaderGGUF & loader, ueUpsampleConformerEncoderV2 & m) {
    {
        std::shared_ptr<ueLinearNoSubsampling> e = m.embed();
        e->set_parameters(loader.get_tensor("embed.out.0.weight"), loader.get_tensor("embed.out.0.bias"),
                          loader.get_tensor("embed.out.1.weight"), loader.get_tensor("embed.out.1.bias"));
    }
    {
        std::shared_ptr<uePreLookaheadLayer> pl = m.pre_lookahead_layer();
        pl->set_parameters(
            loader.get_tensor("pre_lookahead_layer.conv1.weight"), loader.get_tensor("pre_lookahead_layer.conv1.bias"),
            loader.get_tensor("pre_lookahead_layer.conv2.weight"), loader.get_tensor("pre_lookahead_layer.conv2.bias"));
    }
    {
        auto & layers = m.encoders();
        for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
            std::shared_ptr<ueConformerEncoderLayer> &         layer = layers[(size_t) i];
            std::shared_ptr<ueRelPositionMultiHeadedAttention> att   = layer->self_attn();
            std::shared_ptr<uePositionwiseFeedForward>         ffn   = layer->feed_forward();
            layer->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".norm_ff.weight"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_ff.bias"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_mha.weight"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_mha.bias"));
            att->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_q.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_q.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_k.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_k.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_v.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_v.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_out.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_out.bias"));
            att->set_relpos_parameters(
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_pos.weight"),
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.pos_bias_u"),
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.pos_bias_v"));
            ffn->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_1.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_1.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_2.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_2.bias"));
        }
    }
    {
        std::shared_ptr<ueUpsample1D> up = m.up_layer();
        up->set_parameters(loader.get_tensor("up_layer.conv.weight"), loader.get_tensor("up_layer.conv.bias"));
    }
    {
        std::shared_ptr<ueLinearNoSubsampling> ue = m.up_embed();
        ue->set_parameters(loader.get_tensor("up_embed.out.0.weight"), loader.get_tensor("up_embed.out.0.bias"),
                           loader.get_tensor("up_embed.out.1.weight"), loader.get_tensor("up_embed.out.1.bias"));
    }
    {
        auto & layers = m.up_encoders();
        for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
            std::shared_ptr<ueConformerEncoderLayer> &         layer = layers[(size_t) i];
            std::shared_ptr<ueRelPositionMultiHeadedAttention> att   = layer->self_attn();
            std::shared_ptr<uePositionwiseFeedForward>         ffn   = layer->feed_forward();
            layer->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_ff.weight"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_ff.bias"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_mha.weight"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_mha.bias"));
            att->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_q.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_q.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_k.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_k.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_v.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_v.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_out.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_out.bias"));
            att->set_relpos_parameters(
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_pos.weight"),
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.pos_bias_u"),
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.pos_bias_v"));
            ffn->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_1.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_1.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_2.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_2.bias"));
        }
    }
    m.set_after_norm_parameters(loader.get_tensor("after_norm.weight"), loader.get_tensor("after_norm.bias"));
}
// 初始化 CPU 后端
ggml_backend_t ue_loader_init_backend_cpu(std::string & backend_name_out) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
// 解析设备字符串中的GPU索引 (e.g., "gpu:1" -> 1, "gpu" -> 0)
int ue_parse_gpu_index(const std::string & device) {
    if (device.find("gpu:") == 0 && device.size() > 4) {
        try {
            return std::stoi(device.substr(4));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}
// 初始化指定索引的 GPU 后端
ggml_backend_t ue_loader_init_backend_gpu_idx(int gpu_idx, std::string & backend_name_out) {
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    backend = ggml_backend_cuda_init(gpu_idx);
#endif
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
// 优先初始化 GPU 后端 (第一个GPU)
ggml_backend_t ue_loader_init_backend_gpu_first(std::string & backend_name_out) {
    return ue_loader_init_backend_gpu_idx(0, backend_name_out);
}
}  // namespace
ueUpsampleEncoderGGUFModelRunner::ueUpsampleEncoderGGUFModelRunner() = default;
ueUpsampleEncoderGGUFModelRunner::~ueUpsampleEncoderGGUFModelRunner() {
    reset();
}
// 重置并释放旧资源
void ueUpsampleEncoderGGUFModelRunner::reset() {
    weights_.reset();
    model_.reset();
    if (galloc_) {
        ggml_gallocr_free(galloc_);
        galloc_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    backend_name_.clear();
    gguf_path_.clear();
    chunk_call_id_ = 0;
}
// 初始化后端
bool ueUpsampleEncoderGGUFModelRunner::init_backend(const std::string & device) {
    if (backend_) {
        return true;
    }
    // Support "gpu", "gpu:0", "gpu:1" etc.
    if (device.find("gpu") == 0) {
        int gpu_idx = ue_parse_gpu_index(device);
        backend_ = ue_loader_init_backend_gpu_idx(gpu_idx, backend_name_);
        std::fprintf(stderr, "ueUpsampleEncoderGGUFModelRunner: init_backend device=%s, gpu_idx=%d, backend=%s\n",
                device.c_str(), gpu_idx, backend_name_.c_str());
    } else {
        backend_ = ue_loader_init_backend_cpu(backend_name_);
    }
    if (!backend_) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner: failed to init backend\n");
        return false;
    }
    return true;
}
// 绑定参数
void ueUpsampleEncoderGGUFModelRunner::set_num_threads(int n_threads) {
    num_threads_ = n_threads > 0 ? n_threads : 1;
}
// 重置流式状态
void ueUpsampleEncoderGGUFModelRunner::reset_stream() {
    chunk_call_id_ = 0;
}
// 绑定权重到模型结构
bool ueUpsampleEncoderGGUFModelRunner::bind_parameters() {
    if (!weights_ || !model_) {
        return false;
    }
    ue_loader_bind_all_weights(*weights_, *model_);
    return true;
}
// 从 gguf 加载权重
bool ueUpsampleEncoderGGUFModelRunner::load_from_gguf(const std::string & gguf_path, const std::string & device) {
    reset();
    gguf_path_ = gguf_path;
    if (!init_backend(device)) {
        return false;
    }
    weights_                        = std::make_unique<ueUpsampleEncoderModelLoaderGGUF>();
    if (!weights_->load_from_file(gguf_path, backend_)) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner: gguf weight load failed: %s\n", gguf_path.c_str());
        reset();
        return false;
    }
    int32_t pre_lookahead_len = 3;
    if (ggml_tensor * pl_w = weights_->get_tensor("pre_lookahead_layer.conv1.weight")) {
        const int64_t K = pl_w->ne[0];
        if (K >= 1) {
            pre_lookahead_len = (int32_t) (K - 1);
        }
    }
    pre_lookahead_len_ = pre_lookahead_len;
    output_size_ = 512;
    up_stride_   = 2;
    model_ = std::make_shared<ueUpsampleConformerEncoderV2>(512, 512, pre_lookahead_len, 6, 4, 2, 2.0f, 8, true, 2048,
                                                            0.0f, 0.0f, 0.0f, true);
    if (!bind_parameters()) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner: parameter binding failed\n");
        reset();
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    galloc_                         = ggml_gallocr_new(buft);
    if (!galloc_) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner: failed to create ggml_gallocr\n");
        reset();
        return false;
    }
    reset_stream();
    return true;
}
bool ueUpsampleEncoderGGUFModelRunner::forward(const float *          xs_btc,
                                               const int32_t *        xs_lens_b,
                                               int64_t                B,
                                               int64_t                T,
                                               int64_t                C,
                                               std::vector<float> &   ys_btc_out,
                                               std::vector<uint8_t> * masks_out) const {
    ys_btc_out.clear();
    if (masks_out) {
        masks_out->clear();
    }
    if (!backend_ || !model_ || !galloc_) {
        return false;
    }
    if (!xs_btc || !xs_lens_b || B <= 0 || T <= 0 || C <= 0) {
        return false;
    }
    if (C != model_->output_size()) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner.forward: expected C=%d, got %lld\n",
                     model_->output_size(), (long long) C);
        return false;
    }
    ggml_init_params params{};
    params.mem_size    = 2048ull * 1024ull * 1024ull;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * xs_ctb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, B);
    ggml_tensor * lens_b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
    ueUpsampleConformerEncoderV2::ueForwardOut out = model_->forward(ctx, xs_ctb, lens_b);
    if (!out.ys_ctb) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * mask_tb = nullptr;
    if (masks_out && out.masks) {
        mask_tb = ggml_cont(ctx, out.masks);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 256, false);
    ggml_build_forward_expand(gf, out.ys_ctb);
    if (mask_tb) {
        ggml_build_forward_expand(gf, mask_tb);
    }
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        ggml_free(ctx);
        return false;
    }
    backend_tensor_set(backend_, xs_ctb, xs_btc, (size_t) B * (size_t) T * (size_t) C * sizeof(float));
    backend_tensor_set(backend_, lens_b, xs_lens_b, (size_t) B * sizeof(int32_t));
    {
        if (std::shared_ptr<ueLinearNoSubsampling> e = model_->embed()) {
            if (std::shared_ptr<ueEspnetRelPositionalEncoding> pe = e->pos_enc()) {
                if (ggml_tensor * pos0 = ggml_get_tensor(ctx, "ue_uce_pos0")) {
                    const int32_t      size0 = (int32_t) ((pos0->ne[1] + 1) / 2);
                    std::vector<float> host;
                    pe->position_encoding_host(size0, host);
                    backend_tensor_set(backend_, pos0, host.data(), host.size() * sizeof(float));
                }
            }
        }
        if (std::shared_ptr<ueLinearNoSubsampling> ue = model_->up_embed()) {
            if (std::shared_ptr<ueEspnetRelPositionalEncoding> pe = ue->pos_enc()) {
                if (ggml_tensor * pos1 = ggml_get_tensor(ctx, "ue_uce_pos1")) {
                    const int32_t      size1 = (int32_t) ((pos1->ne[1] + 1) / 2);
                    std::vector<float> host;
                    pe->position_encoding_host(size1, host);
                    backend_tensor_set(backend_, pos1, host.data(), host.size() * sizeof(float));
                }
            }
        }
    }
    const ggml_status st = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    if (ue_loader_backend_is_device(backend_)) {
        ggml_backend_synchronize(backend_);
    }
    {
        const size_t nbytes = ggml_nbytes(out.ys_ctb);
        ys_btc_out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(out.ys_ctb, ys_btc_out.data(), 0, nbytes);
    }
    if (mask_tb && masks_out) {
        const int64_t      Tb = mask_tb->ne[0];
        const int64_t      Bb = mask_tb->ne[1];
        std::vector<float> mhost((size_t) Tb * (size_t) Bb);
        ggml_backend_tensor_get(mask_tb, mhost.data(), 0, mhost.size() * sizeof(float));
        masks_out->resize((size_t) B * (size_t) Tb);
        for (int64_t b = 0; b < Bb; ++b) {
            for (int64_t t = 0; t < Tb; ++t) {
                const float v                                       = mhost[(size_t) t + (size_t) Tb * (size_t) b];
                (*masks_out)[(size_t) b * (size_t) Tb + (size_t) t] = (uint8_t) (v > 0.5f ? 1 : 0);
            }
        }
    }
    ggml_free(ctx);
    return true;
}
bool ueUpsampleEncoderGGUFModelRunner::forward_chunk(const float *        xs_btc,
                                                     int64_t              B,
                                                     int64_t              dt,
                                                     int64_t              C,
                                                     bool                 last_chunk,
                                                     ueEncoderCacheHost * cache_in_out,
                                                     std::vector<float> & ys_btc_out) {
    ys_btc_out.clear();
    if (!backend_ || !model_ || !galloc_) {
        return false;
    }
    if (!xs_btc || B <= 0 || dt <= 0 || C <= 0) {
        return false;
    }
    if (C != model_->output_size()) {
        LOG_ERROR( "ueUpsampleEncoderGGUFModelRunner.forward_chunk: expected C=%d, got %lld\n",
                     model_->output_size(), (long long) C);
        return false;
    }
    const int call_id = chunk_call_id_;
    chunk_call_id_++;
    (void) call_id;
    ggml_init_params params{};
    params.mem_size    = 2048ull * 1024ull * 1024ull;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * xs_ctb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, dt, B);
    ggml_tensor * cnn_upload_tcb = nullptr;
    ggml_tensor * cnn_ctb        = nullptr;
    ggml_tensor * att            = nullptr;
    if (cache_in_out && cache_in_out->has_cnn_cache()) {
        if (cache_in_out->cnn_B != B || cache_in_out->cnn_C != C) {
            LOG_ERROR( "forward_chunk: cnn_cache shape mismatch\n");
            ggml_free(ctx);
            return false;
        }
        cnn_upload_tcb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cache_in_out->cnn_T, C, B);
        cnn_ctb        = ggml_cont(ctx, ggml_permute(ctx, cnn_upload_tcb, 1, 0, 2, 3));
    }
    if (cache_in_out && cache_in_out->has_att_cache()) {
        if (cache_in_out->att_B != B) {
            LOG_ERROR( "forward_chunk: att_cache shape mismatch (B)\n");
            ggml_free(ctx);
            return false;
        }
        const int64_t BL = cache_in_out->att_B * cache_in_out->att_L;
        att = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cache_in_out->att_E, cache_in_out->att_T, cache_in_out->att_H, BL);
    }
    ggml_tensor * new_cnn_ctb = nullptr;
    ggml_tensor * new_att     = nullptr;
    ggml_tensor * ys_ctb =
        model_->build_forward_chunk_graph(ctx, xs_ctb, last_chunk, cnn_ctb, att, &new_cnn_ctb, &new_att);
    if (!ys_ctb || !new_cnn_ctb || !new_att) {
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * new_cnn_tcb = ggml_cont(ctx, ggml_permute(ctx, new_cnn_ctb, 1, 0, 2, 3));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 256, false);
    ggml_build_forward_expand(gf, ys_ctb);
    ggml_build_forward_expand(gf, new_cnn_tcb);
    ggml_build_forward_expand(gf, new_att);
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        ggml_free(ctx);
        return false;
    }
    backend_tensor_set(backend_, xs_ctb, xs_btc, (size_t) B * (size_t) dt * (size_t) C * sizeof(float));
    if (cnn_upload_tcb && cache_in_out) {
        backend_tensor_set(backend_, cnn_upload_tcb, cache_in_out->cnn_cache_bct.data(),
                                         cache_in_out->cnn_cache_bct.size() * sizeof(float));
    }
    if (att && cache_in_out) {
        backend_tensor_set(backend_, att, cache_in_out->att_cache_lbhte.data(),
                                         cache_in_out->att_cache_lbhte.size() * sizeof(float));
    }
    {
        std::shared_ptr<ueLinearNoSubsampling> e = model_->embed();
        if (e && e->pos_enc()) {
            std::shared_ptr<ueEspnetRelPositionalEncoding> pe = e->pos_enc();
            if (ggml_tensor * pos1 = ggml_get_tensor(ctx, "ue_uce_pos_stream1")) {
                const int32_t      size1 = (int32_t) ((pos1->ne[1] + 1) / 2);
                std::vector<float> host;
                pe->position_encoding_host(size1, host);
                backend_tensor_set(backend_, pos1, host.data(), host.size() * sizeof(float));
            }
            if (ggml_tensor * pos2 = ggml_get_tensor(ctx, "ue_uce_pos_stream2")) {
                const int32_t      size2 = (int32_t) ((pos2->ne[1] + 1) / 2);
                std::vector<float> host;
                pe->position_encoding_host(size2, host);
                backend_tensor_set(backend_, pos2, host.data(), host.size() * sizeof(float));
            }
        }
    }
    const ggml_status st = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    if (ue_loader_backend_is_device(backend_)) {
        ggml_backend_synchronize(backend_);
    }
    {
        const size_t nbytes = ggml_nbytes(ys_ctb);
        ys_btc_out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(ys_ctb, ys_btc_out.data(), 0, nbytes);
    }
    if (cache_in_out) {
        const size_t cnn_bytes = ggml_nbytes(new_cnn_tcb);
        cache_in_out->cnn_cache_bct.resize(cnn_bytes / sizeof(float));
        ggml_backend_tensor_get(new_cnn_tcb, cache_in_out->cnn_cache_bct.data(), 0, cnn_bytes);
        cache_in_out->cnn_B = B;
        cache_in_out->cnn_C = C;
        cache_in_out->cnn_T = new_cnn_tcb->ne[0];
        const size_t att_bytes = ggml_nbytes(new_att);
        cache_in_out->att_cache_lbhte.resize(att_bytes / sizeof(float));
        ggml_backend_tensor_get(new_att, cache_in_out->att_cache_lbhte.data(), 0, att_bytes);
        cache_in_out->att_E = new_att->ne[0];
        cache_in_out->att_T = new_att->ne[1];
        cache_in_out->att_H = new_att->ne[2];
        cache_in_out->att_B = B;
        cache_in_out->att_L = (new_att->ne[3] / B);
    }
    ggml_free(ctx);
    return true;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
ggml_tensor * ue_mha_reshape_heads_4d(ggml_context * ctx,
                               ggml_tensor *  x_ctb,
                               int            head_dim,
                               int            num_heads,
                               int64_t        T,
                               int64_t        B) {
    return ggml_reshape_4d(ctx, x_ctb, head_dim, num_heads, T, B);
}
ggml_tensor * ue_mha_flatten_heads_qk(ggml_context * ctx,
                               ggml_tensor *  heads_dhtb,
                               int            head_dim,
                               int64_t        T,
                               int64_t        B,
                               int            num_heads) {
    if (!heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * permuted   = ggml_permute(ctx, heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, head_dim, T, (int64_t) num_heads * B);
}
ggml_tensor * ue_mha_flatten_heads_v(ggml_context * ctx,
                              ggml_tensor *  heads_dhtb,
                              int            head_dim,
                              int64_t        T,
                              int64_t        B,
                              int            num_heads) {
    if (!heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * flat_qk    = ue_mha_flatten_heads_qk(ctx, heads_dhtb, head_dim, T, B, num_heads);
    ggml_tensor * permuted   = ggml_permute(ctx, flat_qk, 1, 0, 2, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, T, head_dim, (int64_t) num_heads * B);
}
ggml_tensor * ue_mha_merge_heads_to_channels(ggml_context * ctx,
                                      ggml_tensor *  heads_flat_dtbH,
                                      int            head_dim,
                                      int            num_heads,
                                      int64_t        T,
                                      int64_t        B) {
    if (!heads_flat_dtbH) {
        return nullptr;
    }
    ggml_tensor * view4d     = ggml_reshape_4d(ctx, heads_flat_dtbH, head_dim, T, num_heads, B);
    ggml_tensor * permuted   = ggml_permute(ctx, view4d, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, (int64_t) head_dim * num_heads, T, B);
}
// 从张量中切片
ggml_tensor * ue_mha_slice_att_cache(ggml_context * ctx, ggml_tensor * cache, int head_dim, bool value_slice) {
    if (!cache) {
        return nullptr;
    }
    const int64_t t_cache   = cache->ne[1];
    const int64_t num_heads = cache->ne[2];
    const int64_t B         = cache->ne[3];
    const size_t offset = value_slice ? (size_t) head_dim * cache->nb[0] : 0;
    return ggml_view_4d(ctx, cache, head_dim, t_cache, num_heads, B, cache->nb[1], cache->nb[2], cache->nb[3], offset);
}
// 调整张量维度顺序
ggml_tensor * ue_mha_permute_cache_to_heads(ggml_context * ctx, ggml_tensor * cache_slice_dthb) {
    if (!cache_slice_dthb) {
        return nullptr;
    }
    return ggml_permute(ctx, cache_slice_dthb, 0, 2, 1, 3);
}
// 拼接 k v 缓存
ggml_tensor * ue_mha_build_new_att_cache(ggml_context * ctx, ggml_tensor * k_heads_dhtb, ggml_tensor * v_heads_dhtb) {
    if (!k_heads_dhtb || !v_heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * k_perm = ggml_permute(ctx, k_heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx, v_heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * k_cont = ggml_cont(ctx, k_perm);
    ggml_tensor * v_cont = ggml_cont(ctx, v_perm);
    ggml_tensor * kv     = ggml_concat(ctx, k_cont, v_cont, 0);
    return ggml_cont(ctx, kv);
}
//
ggml_tensor * ue_mha_mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, const char * label) {
    const bool ok = (a && b) && (a->ne[0] == b->ne[0]) && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0);
    if (!ok) {
        LOG_ERROR(
                     "[ueMultiHeadedAttention] mul_mat mismatch (%s): "
                     "a.ne=(%lld,%lld,%lld,%lld) b.ne=(%lld,%lld,%lld,%lld)\n",
                     label, (a ? a->ne[0] : -1LL), (a ? a->ne[1] : -1LL), (a ? a->ne[2] : -1LL), (a ? a->ne[3] : -1LL),
                     (b ? b->ne[0] : -1LL), (b ? b->ne[1] : -1LL), (b ? b->ne[2] : -1LL), (b ? b->ne[3] : -1LL));
    }
    return ggml_mul_mat(ctx, a, b);
}
ggml_tensor * ue_mha_prepare_valid_mask_bh(ggml_context * ctx,
                                    ggml_tensor *  mask_valid,
                                    int64_t        T_k,
                                    int64_t        T_q,
                                    int64_t        B,
                                    int            num_heads) {
    if (!mask_valid) {
        return nullptr;
    }
    if (mask_valid->ne[0] == 0 || mask_valid->ne[1] == 0 || mask_valid->ne[2] == 0) {
        return nullptr;
    }
    ggml_tensor * m_f32 = (mask_valid->type == GGML_TYPE_F32) ? mask_valid : ggml_cast(ctx, mask_valid, GGML_TYPE_F32);
    ggml_tensor * m_tqb = nullptr;
    const bool    is_2d = (m_f32->ne[2] == 1 && m_f32->ne[3] == 1);
    const bool    is_3d = (m_f32->ne[3] == 1 && m_f32->ne[2] == B);
    if (is_2d) {
        ggml_tensor * m2 = m_f32;
        if (m2->ne[0] != T_k) {
            m2 = ggml_view_2d(ctx, m2, T_k, B, m2->nb[1], 0);
        }
        ggml_tensor * m_t1b = ggml_reshape_3d(ctx, m2, T_k, 1, B);
        ggml_tensor * tmpl  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_k, T_q, B);
        m_tqb               = ggml_repeat(ctx, m_t1b, tmpl);
    } else if (is_3d) {
        ggml_tensor * m3 = m_f32;
        if (m3->ne[0] != T_k) {
            m3 = ggml_view_3d(ctx, m3, T_k, m3->ne[1], B, m3->nb[1], m3->nb[2], 0);
        }
        if (m3->ne[1] == 1 && T_q > 1) {
            ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_k, T_q, B);
            m_tqb              = ggml_repeat(ctx, m3, tmpl);
        } else {
            m_tqb = m3;
        }
    } else {
    }
    ggml_tensor * m_4d      = ggml_reshape_4d(ctx, m_tqb, T_k, T_q, 1, B);
    ggml_tensor * tmpl_4d   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T_k, T_q, num_heads, B);
    ggml_tensor * m_hb      = ggml_repeat(ctx, m_4d, tmpl_4d);
    ggml_tensor * m_hb_cont = ggml_cont(ctx, m_hb);
    return ggml_reshape_3d(ctx, m_hb_cont, T_k, T_q, B * num_heads);
}
}  // namespace
ueMultiHeadedAttention::ueMultiHeadedAttention(int32_t n_head, int32_t n_feat, float dropout_rate, bool key_bias) :
    n_head_(n_head),
    n_feat_(n_feat),
    dropout_rate_(dropout_rate),
    key_bias_(key_bias) {
    d_k_ = (n_head_ > 0) ? (n_feat_ / n_head_) : 0;
}
void ueMultiHeadedAttention::set_parameters(ggml_tensor * linear_q_weight,
                                            ggml_tensor * linear_q_bias,
                                            ggml_tensor * linear_k_weight,
                                            ggml_tensor * linear_k_bias,
                                            ggml_tensor * linear_v_weight,
                                            ggml_tensor * linear_v_bias,
                                            ggml_tensor * linear_out_weight,
                                            ggml_tensor * linear_out_bias) {
    linear_q_weight_   = linear_q_weight;
    linear_q_bias_     = linear_q_bias;
    linear_k_weight_   = linear_k_weight;
    linear_k_bias_     = linear_k_bias;
    linear_v_weight_   = linear_v_weight;
    linear_v_bias_     = linear_v_bias;
    linear_out_weight_ = linear_out_weight;
    linear_out_bias_   = linear_out_bias;
}
void ueMultiHeadedAttention::build_forward_qkv(ggml_context * ctx,
                                               ggml_tensor *  query_ctb,
                                               ggml_tensor *  key_ctb,
                                               ggml_tensor *  value_ctb,
                                               ggml_tensor ** q_dhtb_out,
                                               ggml_tensor ** k_dhtb_out,
                                               ggml_tensor ** v_dhtb_out) const {
    if (q_dhtb_out) {
        *q_dhtb_out = nullptr;
    }
    if (k_dhtb_out) {
        *k_dhtb_out = nullptr;
    }
    if (v_dhtb_out) {
        *v_dhtb_out = nullptr;
    }
    if (!ctx || !query_ctb || !key_ctb || !value_ctb) {
        return;
    }
    const int64_t Bq = query_ctb->ne[2];
    const int64_t Tq = query_ctb->ne[1];
    const int64_t Cq = query_ctb->ne[0];
    const int64_t Bk = key_ctb->ne[2];
    const int64_t Tk = key_ctb->ne[1];
    const int64_t Ck = key_ctb->ne[0];
    const int64_t Bv = value_ctb->ne[2];
    const int64_t Tv = value_ctb->ne[1];
    const int64_t Cv = value_ctb->ne[0];
    ggml_tensor * q_ctb = ue_build_linear(ctx, query_ctb, linear_q_weight_, linear_q_bias_);
    ggml_tensor * k_ctb = ue_build_linear(ctx, key_ctb, linear_k_weight_, key_bias_ ? linear_k_bias_ : nullptr);
    ggml_tensor * v_ctb = ue_build_linear(ctx, value_ctb, linear_v_weight_, linear_v_bias_);
    ggml_tensor * q_dhtb = ue_mha_reshape_heads_4d(ctx, q_ctb, d_k_, n_head_, Tq, Bq);
    ggml_tensor * k_dhtb = ue_mha_reshape_heads_4d(ctx, k_ctb, d_k_, n_head_, Tk, Bq);
    ggml_tensor * v_dhtb = ue_mha_reshape_heads_4d(ctx, v_ctb, d_k_, n_head_, Tk, Bq);
    if (q_dhtb_out) {
        *q_dhtb_out = q_dhtb;
    }
    if (k_dhtb_out) {
        *k_dhtb_out = k_dhtb;
    }
    if (v_dhtb_out) {
        *v_dhtb_out = v_dhtb;
    }
}
// 构建注意力计算图
ggml_tensor * ueMultiHeadedAttention::build_forward_attention(ggml_context * ctx,
                                                              ggml_tensor *  value_dhtb,
                                                              ggml_tensor *  scores_ktqh_b,
                                                              ggml_tensor *  mask) const {
    if (!ctx || !value_dhtb || !scores_ktqh_b) {
        return nullptr;
    }
    const int64_t T_k = scores_ktqh_b->ne[0];
    const int64_t T_q = scores_ktqh_b->ne[1];
    const int64_t H   = scores_ktqh_b->ne[2];
    const int64_t B   = scores_ktqh_b->ne[3];
    ggml_tensor * scores_cont = ggml_cont(ctx, scores_ktqh_b);
    ggml_tensor * scores_bh   = ggml_reshape_3d(ctx, scores_cont, T_k, T_q, B * H);
    ggml_tensor * probs = nullptr;
    if (mask != nullptr) {
        ggml_tensor * valid_bh      = ue_mha_prepare_valid_mask_bh(ctx, mask, T_k, T_q, B, (int) H);
        ggml_tensor * one_f32       = ggml_arange(ctx, 1.0f, 2.0f, 1.0f);
        // 🔧 用 ggml_add + ggml_repeat 替代 ggml_add1，支持 Metal 加速
        ggml_tensor * neg_valid     = ggml_neg(ctx, valid_bh);
        ggml_tensor * one_rep       = ggml_repeat(ctx, one_f32, neg_valid);
        ggml_tensor * masked_bh     = ggml_add(ctx, neg_valid, one_rep);
        ggml_tensor * mask_add      = ggml_scale(ctx, masked_bh, -1e9f);
        ggml_tensor * scores_masked = ggml_add(ctx, scores_bh, mask_add);
        probs                       = ggml_soft_max(ctx, scores_masked);
        probs                       = ggml_mul(ctx, probs, valid_bh);
    } else {
        probs = ggml_soft_max(ctx, scores_bh);
    }
    (void) dropout_rate_;
    ggml_tensor * v_flat  = ue_mha_flatten_heads_v(ctx, value_dhtb, d_k_, T_k, B, n_head_);
    ggml_tensor * context = ue_mha_mul_mat_checked(ctx, v_flat, probs, "v@p");
    ggml_tensor * merged  = ue_mha_merge_heads_to_channels(ctx, context, d_k_, n_head_, T_q, B);
    return ue_build_linear(ctx, merged, linear_out_weight_, linear_out_bias_);
}
// 构建计算图
ggml_tensor * ueMultiHeadedAttention::build_forward_graph(ggml_context * ctx,
                                                          ggml_tensor *  query_ctb,
                                                          ggml_tensor *  key_ctb,
                                                          ggml_tensor *  value_ctb,
                                                          ggml_tensor *  mask,
                                                          ggml_tensor *  pos_emb,
                                                          ggml_tensor *  cache,
                                                          ggml_tensor ** new_cache_out) const {
    if (new_cache_out) {
        *new_cache_out = nullptr;
    }
    (void) pos_emb;
    if (!ctx || !query_ctb || !key_ctb || !value_ctb) {
        return nullptr;
    }
    const int64_t B  = query_ctb->ne[2];
    const int64_t Tq = query_ctb->ne[1];
    const int64_t C  = query_ctb->ne[0];
    ggml_tensor * q_heads = nullptr;
    ggml_tensor * k_heads = nullptr;
    ggml_tensor * v_heads = nullptr;
    build_forward_qkv(ctx, query_ctb, key_ctb, value_ctb, &q_heads, &k_heads, &v_heads);
    ggml_tensor * k_total = k_heads;
    ggml_tensor * v_total = v_heads;
    int64_t       t_cache = 0;
    if (cache != nullptr && cache->ne[0] > 0) {
        t_cache = cache->ne[1];
        ggml_tensor * k_cache_slice = ue_mha_slice_att_cache(ctx, cache, d_k_, false);
        ggml_tensor * v_cache_slice = ue_mha_slice_att_cache(ctx, cache, d_k_, true);
        ggml_tensor * k_cache_heads = ue_mha_permute_cache_to_heads(ctx, k_cache_slice);
        ggml_tensor * v_cache_heads = ue_mha_permute_cache_to_heads(ctx, v_cache_slice);
        k_total = ggml_concat(ctx, k_cache_heads, k_heads, 2);
        v_total = ggml_concat(ctx, v_cache_heads, v_heads, 2);
    }
    const int64_t T_k = k_total->ne[2];
    if (new_cache_out) {
        *new_cache_out = ue_mha_build_new_att_cache(ctx, k_total, v_total);
    }
    ggml_tensor * q_flat = ue_mha_flatten_heads_qk(ctx, q_heads, d_k_, Tq, B, n_head_);
    ggml_tensor * k_flat = ue_mha_flatten_heads_qk(ctx, k_total, d_k_, T_k, B, n_head_);
    ggml_tensor * scores_bh = ue_mha_mul_mat_checked(ctx, k_flat, q_flat, "k@q");
    scores_bh               = ggml_scale(ctx, scores_bh, 1.0f / std::sqrt((float) d_k_));
    ggml_tensor * scores_4d = ggml_reshape_4d(ctx, scores_bh, T_k, Tq, n_head_, B);
    ggml_tensor * mask_use = mask;
    if (mask_use && (mask_use->ne[0] == 0 || mask_use->ne[1] == 0 || mask_use->ne[2] == 0)) {
        mask_use = nullptr;
    }
    return build_forward_attention(ctx, v_total, scores_4d, mask_use);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
// 初始化位置感知前馈网络
uePositionwiseFeedForward::uePositionwiseFeedForward(int32_t      idim,
                                                     int32_t      hidden_units,
                                                     float        dropout_rate,
                                                     const char * activation) :
    idim_(idim),
    hidden_units_(hidden_units),
    dropout_rate_(dropout_rate),
    activation_(activation ? activation : "relu") {}
void uePositionwiseFeedForward::set_parameters(ggml_tensor * w1_weight,
                                               ggml_tensor * w1_bias,
                                               ggml_tensor * w2_weight,
                                               ggml_tensor * w2_bias) {
    w1_weight_ = w1_weight;
    w1_bias_   = w1_bias;
    w2_weight_ = w2_weight;
    w2_bias_   = w2_bias;
}
// 构建前向计算图
ggml_tensor * uePositionwiseFeedForward::build_forward_graph(ggml_context * ctx, ggml_tensor * xs_ctb) const {
    if (ctx == nullptr || xs_ctb == nullptr) {
        return nullptr;
    }
    const int64_t C = xs_ctb->ne[0];
    ggml_tensor * h = ue_build_linear(ctx, xs_ctb, w1_weight_, w1_bias_);
    if (activation_ != nullptr) {
        const std::string act(activation_);
        if (act == "identity" || act == "none") {
        } else if (act == "swish" || act == "silu" || act == "SiLU") {
            h = ggml_silu(ctx, h);
        } else if (act == "gelu" || act == "GELU") {
            h = ggml_gelu(ctx, h);
        } else {
            h = ggml_relu(ctx, h);
        }
    }
    (void) dropout_rate_;
    ggml_tensor * y = ue_build_linear(ctx, h, w2_weight_, w2_bias_);
    return y;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
// 构建 1D 卷积计算图
static ggml_tensor * ue_prelook_conv1d_im2col_f32_n1(ggml_context * ctx,
                                          ggml_tensor *  w_kic_oc,
                                          ggml_tensor *  x_tcb,
                                          int            stride,
                                          int            padding,
                                          int            dilation) {
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cin  = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    ggml_tensor * im2col = ggml_im2col(ctx, w_kic_oc, x_tcb, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, K * Cin, Cout);
    ggml_tensor * mm = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, im2col->ne[1], Cout, im2col->ne[2]);
    return y_tcb;
}
static ggml_tensor * ue_prelook_build_pad_zeros_ctb(ggml_context * ctx, int64_t C, int64_t Tpad, int64_t B) {
    ggml_tensor * zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);
    ggml_tensor * pad_4d      = ggml_repeat_4d(ctx, zero_scalar, C, Tpad, B, 1);
    ggml_tensor * pad_3d      = ggml_reshape_3d(ctx, pad_4d, C, Tpad, B);
    return ggml_cont(ctx, pad_3d);
}
static ggml_tensor * ue_prelook_build_conv1d_f32_ctb(ggml_context * ctx,
                                          ggml_tensor *  w_kic_oc,
                                          ggml_tensor *  b_oc,
                                          ggml_tensor *  x_ctb) {
    const int64_t Cin  = x_ctb->ne[0];
    const int64_t B    = x_ctb->ne[2];
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cinw = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    ggml_tensor * x_tcb = ggml_permute(ctx, x_ctb, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    ggml_tensor * y_tcb = nullptr;
    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
        const size_t  offset = x_tcb->nb[2] * static_cast<size_t>(b_idx);
        ggml_tensor * x_b = ggml_view_3d(ctx, x_tcb, x_tcb->ne[0], x_tcb->ne[1], 1, x_tcb->nb[1], x_tcb->nb[2], offset);
        ggml_tensor * y_b = ue_prelook_conv1d_im2col_f32_n1(ctx, w_kic_oc, x_b, 1, 0, 1);
        if (y_tcb == nullptr) {
            y_tcb = y_b;
        } else {
            y_tcb = ggml_concat(ctx, y_tcb, y_b, 2);
        }
    }
    ggml_tensor * y_ctb = ggml_permute(ctx, y_tcb, 1, 0, 2, 3);
    y_ctb               = ggml_cont(ctx, y_ctb);
    if (b_oc != nullptr) {
        ggml_tensor * b_bcast = ggml_reshape_3d(ctx, b_oc, Cout, 1, 1);
        y_ctb                 = ggml_add(ctx, y_ctb, b_bcast);
    }
    return y_ctb;
}
static ggml_tensor * ue_prelook_slice_time_ctb(ggml_context * ctx, ggml_tensor * x_ctb, int64_t t0, int64_t tlen) {
    const size_t  off = x_ctb->nb[1] * static_cast<size_t>(t0);
    ggml_tensor * v   = ggml_view_3d(ctx, x_ctb, x_ctb->ne[0], tlen, x_ctb->ne[2], x_ctb->nb[1], x_ctb->nb[2], off);
    return ggml_cont(ctx, v);
}
}  // namespace
// 初始化前向prelook
uePreLookaheadLayer::uePreLookaheadLayer(int32_t channels, int32_t pre_lookahead_len) :
    channels_(channels),
    pre_lookahead_len_(pre_lookahead_len) {}
void uePreLookaheadLayer::set_parameters(ggml_tensor * conv1_weight,
                                         ggml_tensor * conv1_bias,
                                         ggml_tensor * conv2_weight,
                                         ggml_tensor * conv2_bias) {
    conv1_weight_ = conv1_weight;
    conv1_bias_   = conv1_bias;
    conv2_weight_ = conv2_weight;
    conv2_bias_   = conv2_bias;
}
// 构建前向计算图
ggml_tensor * uePreLookaheadLayer::build_forward_graph(ggml_context * ctx, ggml_tensor * inputs_ctb) const {
    if (ctx == nullptr || inputs_ctb == nullptr) {
        return nullptr;
    }
    const int64_t C = inputs_ctb->ne[0];
    const int64_t T = inputs_ctb->ne[1];
    const int64_t B = inputs_ctb->ne[2];
    const int64_t K1 = conv1_weight_->ne[0];
    ggml_tensor * pad_r = ue_prelook_build_pad_zeros_ctb(ctx, C, pre_lookahead_len_, B);
    ggml_tensor * x_cat = ggml_concat(ctx, inputs_ctb, pad_r, 1);
    x_cat               = ggml_cont(ctx, x_cat);
    ggml_tensor * y1 = ue_prelook_build_conv1d_f32_ctb(ctx, conv1_weight_, conv1_bias_, x_cat);
    y1 = ggml_leaky_relu(ctx, y1, 0.01f, false);
    y1 = ggml_cont(ctx, y1);
    const int64_t K2 = conv2_weight_->ne[0];
    ggml_tensor * pad_l2 = ue_prelook_build_pad_zeros_ctb(ctx, C, 2, B);
    ggml_tensor * y1_cat = ggml_concat(ctx, pad_l2, y1, 1);
    y1_cat               = ggml_cont(ctx, y1_cat);
    ggml_tensor * y2 = ue_prelook_build_conv1d_f32_ctb(ctx, conv2_weight_, conv2_bias_, y1_cat);
    y2               = ggml_cont(ctx, y2);
    ggml_tensor * out = ggml_add(ctx, y2, inputs_ctb);
    return ggml_cont(ctx, out);
}
ggml_tensor * uePreLookaheadLayer::build_forward_chunk_graph(ggml_context * ctx,
                                                             ggml_tensor *  inputs_ctb,
                                                             ggml_tensor *  cache_ctb,
                                                             ggml_tensor ** new_cache_ctb_out) const {
    if (new_cache_ctb_out) {
        *new_cache_ctb_out = nullptr;
    }
    if (ctx == nullptr || inputs_ctb == nullptr) {
        return nullptr;
    }
    const int64_t C  = inputs_ctb->ne[0];
    const int64_t dt = inputs_ctb->ne[1];
    const int64_t B  = inputs_ctb->ne[2];
    const int64_t K1 = conv1_weight_->ne[0];
    ggml_tensor * y1 = ue_prelook_build_conv1d_f32_ctb(ctx, conv1_weight_, conv1_bias_, inputs_ctb);
    y1               = ggml_leaky_relu(ctx, y1, 0.01f, false);
    y1               = ggml_cont(ctx, y1);
    const int64_t t1 = y1->ne[1];
    ggml_tensor * new_cache = ue_prelook_slice_time_ctb(ctx, y1, t1 - cache_t(), cache_t());
    if (new_cache_ctb_out) {
        *new_cache_ctb_out = new_cache;
    }
    ggml_tensor * cache = cache_ctb;
    if (cache == nullptr) {
        cache = ue_prelook_build_pad_zeros_ctb(ctx, C, cache_t(), B);
    } else {
    }
    ggml_tensor * y1_cat = ggml_concat(ctx, cache, y1, 1);
    y1_cat               = ggml_cont(ctx, y1_cat);
    ggml_tensor * y2 = ue_prelook_build_conv1d_f32_ctb(ctx, conv2_weight_, conv2_bias_, y1_cat);
    y2               = ggml_cont(ctx, y2);
    ggml_tensor * in_trunc = ue_prelook_slice_time_ctb(ctx, inputs_ctb, 0, t1);
    ggml_tensor * out      = ggml_add(ctx, y2, in_trunc);
    return ggml_cont(ctx, out);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
ggml_tensor * ue_rel_mha_reshape_heads_4d(ggml_context * ctx,
                               ggml_tensor *  x_ctb,
                               int            head_dim,
                               int            num_heads,
                               int64_t        T,
                               int64_t        B) {
    return ggml_reshape_4d(ctx, x_ctb, head_dim, num_heads, T, B);
}
ggml_tensor * ue_rel_mha_flatten_heads_qk(ggml_context * ctx,
                               ggml_tensor *  heads_dhtb,
                               int            head_dim,
                               int64_t        T,
                               int64_t        B,
                               int            num_heads) {
    if (!heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * permuted   = ggml_permute(ctx, heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, head_dim, T, (int64_t) num_heads * B);
}
// 把通道维 reshape 成 head 维度
ggml_tensor * ue_rel_mha_flatten_heads_v(ggml_context * ctx,
                              ggml_tensor *  heads_dhtb,
                              int            head_dim,
                              int64_t        T,
                              int64_t        B,
                              int            num_heads) {
    if (!heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * flat_qk    = ue_rel_mha_flatten_heads_qk(ctx, heads_dhtb, head_dim, T, B, num_heads);
    ggml_tensor * permuted   = ggml_permute(ctx, flat_qk, 1, 0, 2, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, T, head_dim, (int64_t) num_heads * B);
}
ggml_tensor * ue_rel_mha_merge_heads_to_channels(ggml_context * ctx,
                                      ggml_tensor *  heads_flat_dtbH,
                                      int            head_dim,
                                      int            num_heads,
                                      int64_t        T,
                                      int64_t        B) {
    if (!heads_flat_dtbH) {
        return nullptr;
    }
    ggml_tensor * view4d     = ggml_reshape_4d(ctx, heads_flat_dtbH, head_dim, T, num_heads, B);
    ggml_tensor * permuted   = ggml_permute(ctx, view4d, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, (int64_t) head_dim * num_heads, T, B);
}
// 从张量中切片
ggml_tensor * ue_rel_mha_slice_att_cache(ggml_context * ctx, ggml_tensor * cache, int head_dim, bool value_slice) {
    if (!cache) {
        return nullptr;
    }
    const int64_t t_cache   = cache->ne[1];
    const int64_t num_heads = cache->ne[2];
    const int64_t B         = cache->ne[3];
    const size_t offset = value_slice ? (size_t) head_dim * cache->nb[0] : 0;
    return ggml_view_4d(ctx, cache, head_dim, t_cache, num_heads, B, cache->nb[1], cache->nb[2], cache->nb[3], offset);
}
// 调整张量维度顺序
ggml_tensor * ue_rel_mha_permute_cache_to_heads(ggml_context * ctx, ggml_tensor * cache_slice_dthb) {
    if (!cache_slice_dthb) {
        return nullptr;
    }
    return ggml_permute(ctx, cache_slice_dthb, 0, 2, 1, 3);
}
// 构建计算图
ggml_tensor * ue_rel_mha_build_new_att_cache(ggml_context * ctx, ggml_tensor * k_heads_dhtb, ggml_tensor * v_heads_dhtb) {
    if (!k_heads_dhtb || !v_heads_dhtb) {
        return nullptr;
    }
    ggml_tensor * k_perm = ggml_permute(ctx, k_heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx, v_heads_dhtb, 0, 2, 1, 3);
    ggml_tensor * k_cont = ggml_cont(ctx, k_perm);
    ggml_tensor * v_cont = ggml_cont(ctx, v_perm);
    ggml_tensor * kv     = ggml_concat(ctx, k_cont, v_cont, 0);
    return ggml_cont(ctx, kv);
}
ggml_tensor * ue_rel_mha_mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, const char * label) {
    const bool ok = (a && b) && (a->ne[0] == b->ne[0]) && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0);
    if (!ok) {
        LOG_ERROR(
                     "[ueRelPositionMultiHeadedAttention] mul_mat mismatch (%s): "
                     "a.ne=(%lld,%lld,%lld,%lld) b.ne=(%lld,%lld,%lld,%lld)\n",
                     label, (a ? a->ne[0] : -1LL), (a ? a->ne[1] : -1LL), (a ? a->ne[2] : -1LL), (a ? a->ne[3] : -1LL),
                     (b ? b->ne[0] : -1LL), (b ? b->ne[1] : -1LL), (b ? b->ne[2] : -1LL), (b ? b->ne[3] : -1LL));
    }
    return ggml_mul_mat(ctx, a, b);
}
}  // namespace
ueRelPositionMultiHeadedAttention::ueRelPositionMultiHeadedAttention(int32_t n_head,
                                                                     int32_t n_feat,
                                                                     float   dropout_rate,
                                                                     bool    key_bias) :
    ueMultiHeadedAttention(n_head, n_feat, dropout_rate, key_bias) {}
void ueRelPositionMultiHeadedAttention::set_relpos_parameters(ggml_tensor * linear_pos_weight,
                                                              ggml_tensor * pos_bias_u,
                                                              ggml_tensor * pos_bias_v) {
    linear_pos_weight_ = linear_pos_weight;
    pos_bias_u_        = pos_bias_u;
    pos_bias_v_        = pos_bias_v;
}
// 构建计算图
ggml_tensor * ueRelPositionMultiHeadedAttention::build_rel_shift(ggml_context * ctx, ggml_tensor * x) const {
    if (!ctx || !x) {
        return nullptr;
    }
    const int64_t Tp = x->ne[0];
    const int64_t Tq = x->ne[1];
    const int64_t H  = x->ne[2];
    const int64_t B  = x->ne[3];
    ggml_tensor * zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);
    ggml_tensor * zero_pad    = ggml_repeat_4d(ctx, zero_scalar, 1, Tq, H, B);
    ggml_tensor * x_padded = ggml_concat(ctx, zero_pad, x, 0);
    x_padded               = ggml_cont(ctx, x_padded);
    ggml_tensor * x_view = ggml_reshape_4d(ctx, x_padded, Tq, Tp + 1, H, B);
    ggml_tensor * x_sliced =
        ggml_view_4d(ctx, x_view, Tq, Tp, H, B, x_view->nb[1], x_view->nb[2], x_view->nb[3], x_view->nb[1]);
    ggml_tensor * x_sliced_cont = ggml_cont(ctx, x_sliced);
    ggml_tensor * x_view_as     = ggml_reshape_4d(ctx, x_sliced_cont, Tp, Tq, H, B);
    const int64_t Tp2 = Tp / 2 + 1;
    return ggml_view_4d(ctx, x_view_as, Tp2, Tq, H, B, x_view_as->nb[1], x_view_as->nb[2], x_view_as->nb[3], 0);
}
ggml_tensor * ueRelPositionMultiHeadedAttention::build_forward_graph(ggml_context * ctx,
                                                                     ggml_tensor *  query_ctb,
                                                                     ggml_tensor *  key_ctb,
                                                                     ggml_tensor *  value_ctb,
                                                                     ggml_tensor *  mask,
                                                                     ggml_tensor *  pos_emb,
                                                                     ggml_tensor *  cache,
                                                                     ggml_tensor ** new_cache_out) const {
    if (new_cache_out) {
        *new_cache_out = nullptr;
    }
    if (!ctx || !query_ctb || !key_ctb || !value_ctb || !pos_emb) {
        return nullptr;
    }
    const int64_t B  = query_ctb->ne[2];
    const int64_t Tq = query_ctb->ne[1];
    const int64_t C  = query_ctb->ne[0];
    ggml_tensor * q_heads = nullptr;
    ggml_tensor * k_heads = nullptr;
    ggml_tensor * v_heads = nullptr;
    build_forward_qkv(ctx, query_ctb, key_ctb, value_ctb, &q_heads, &k_heads, &v_heads);
    ggml_tensor * k_total = k_heads;
    ggml_tensor * v_total = v_heads;
    int64_t       t_cache = 0;
    if (cache != nullptr && cache->ne[0] > 0) {
        t_cache = cache->ne[1];
        ggml_tensor * k_cache_slice = ue_rel_mha_slice_att_cache(ctx, cache, d_k_, false);
        ggml_tensor * v_cache_slice = ue_rel_mha_slice_att_cache(ctx, cache, d_k_, true);
        ggml_tensor * k_cache_heads = ue_rel_mha_permute_cache_to_heads(ctx, k_cache_slice);
        ggml_tensor * v_cache_heads = ue_rel_mha_permute_cache_to_heads(ctx, v_cache_slice);
        k_total = ggml_concat(ctx, k_cache_heads, k_heads, 2);
        v_total = ggml_concat(ctx, v_cache_heads, v_heads, 2);
    }
    const int64_t Tk = k_total->ne[2];
    if (new_cache_out) {
        *new_cache_out = ue_rel_mha_build_new_att_cache(ctx, k_total, v_total);
    }
    const int64_t Bp = pos_emb->ne[2];
    ggml_tensor * pos_use = pos_emb;
    if (Bp == 1 && B > 1) {
        ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, pos_emb->ne[1], B);
        pos_use            = ggml_repeat(ctx, pos_emb, tmpl);
    }
    ggml_tensor * p_ctb   = ue_build_linear(ctx, pos_use, linear_pos_weight_, nullptr);
    const int64_t Tp      = p_ctb->ne[1];
    ggml_tensor * p_heads = ue_rel_mha_reshape_heads_4d(ctx, p_ctb, d_k_, n_head_, Tp, B);
    ggml_tensor * u_4d  = ggml_reshape_4d(ctx, pos_bias_u_, d_k_, n_head_, 1, 1);
    ggml_tensor * v_4d  = ggml_reshape_4d(ctx, pos_bias_v_, d_k_, n_head_, 1, 1);
    ggml_tensor * u_rep = ggml_repeat_4d(ctx, u_4d, d_k_, n_head_, Tq, B);
    ggml_tensor * v_rep = ggml_repeat_4d(ctx, v_4d, d_k_, n_head_, Tq, B);
    ggml_tensor * q_u = ggml_add(ctx, q_heads, u_rep);
    ggml_tensor * q_v = ggml_add(ctx, q_heads, v_rep);
    ggml_tensor * q_u_flat  = ue_rel_mha_flatten_heads_qk(ctx, q_u, d_k_, Tq, B, n_head_);
    ggml_tensor * k_flat    = ue_rel_mha_flatten_heads_qk(ctx, k_total, d_k_, Tk, B, n_head_);
    ggml_tensor * matrix_ac = ue_rel_mha_mul_mat_checked(ctx, k_flat, q_u_flat, "relpos/ac");
    ggml_tensor * q_v_flat  = ue_rel_mha_flatten_heads_qk(ctx, q_v, d_k_, Tq, B, n_head_);
    ggml_tensor * p_flat    = ue_rel_mha_flatten_heads_qk(ctx, p_heads, d_k_, Tp, B, n_head_);
    ggml_tensor * matrix_bd = ue_rel_mha_mul_mat_checked(ctx, p_flat, q_v_flat, "relpos/bd");
    if (matrix_bd->ne[0] != matrix_ac->ne[0]) {
        ggml_tensor * bd_4d         = ggml_reshape_4d(ctx, matrix_bd, Tp, Tq, n_head_, B);
        ggml_tensor * bd_shift      = build_rel_shift(ctx, bd_4d);
        ggml_tensor * bd_shift_cont = ggml_cont(ctx, bd_shift);
        matrix_bd =
            ggml_reshape_3d(ctx, bd_shift_cont, bd_shift->ne[0], bd_shift->ne[1], bd_shift->ne[2] * bd_shift->ne[3]);
    }
    ggml_tensor * scores    = ggml_add(ctx, matrix_ac, matrix_bd);
    scores                  = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) d_k_));
    ggml_tensor * scores_4d = ggml_reshape_4d(ctx, scores, Tk, Tq, n_head_, B);
    ggml_tensor * mask_use = mask;
    if (mask_use && (mask_use->ne[0] == 0 || mask_use->ne[1] == 0 || mask_use->ne[2] == 0)) {
        mask_use = nullptr;
    }
    return build_forward_attention(ctx, v_total, scores_4d, mask_use);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
// 线性无下采样输入层
namespace omni {
namespace upsample_encoder_v2 {
ueLinearNoSubsampling::ueLinearNoSubsampling(int32_t                                        idim,
                                             int32_t                                        odim,
                                             float                                          dropout_rate,
                                             std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc) :
    idim_(idim),
    odim_(odim),
    dropout_rate_(dropout_rate),
    pos_enc_(std::move(pos_enc)) {
    set_pos_enc(pos_enc_);
}
void ueLinearNoSubsampling::set_parameters(ggml_tensor * linear_weight,
                                           ggml_tensor * linear_bias,
                                           ggml_tensor * ln_weight,
                                           ggml_tensor * ln_bias) {
    linear_weight_ = linear_weight;
    linear_bias_   = linear_bias;
    ln_weight_     = ln_weight;
    ln_bias_       = ln_bias;
}
// 构建输入层前向计算图
ggml_tensor * ueLinearNoSubsampling::build_forward_graph(ggml_context * ctx,
                                                         ggml_tensor *  x_ctb,
                                                         ggml_tensor *  x_mask,
                                                         int32_t        offset,
                                                         ggml_tensor ** pos_emb_placeholder_out,
                                                         ggml_tensor ** out_mask_out) const {
    (void) dropout_rate_;
    if (!ctx || !x_ctb || !pos_enc_) {
        if (pos_emb_placeholder_out) {
            *pos_emb_placeholder_out = nullptr;
        }
        if (out_mask_out) {
            *out_mask_out = x_mask;
        }
        return nullptr;
    }
    ggml_tensor * y = ue_build_linear(ctx, x_ctb, linear_weight_, linear_bias_);
    y               = ue_build_layer_norm(ctx, y, ln_weight_, ln_bias_, 1e-5f);
    ggml_tensor * pos      = nullptr;
    ggml_tensor * y_scaled = pos_enc_->build_forward_graph(ctx, y, offset, &pos);
    if (pos_emb_placeholder_out) {
        *pos_emb_placeholder_out = pos;
    }
    if (out_mask_out) {
        *out_mask_out = x_mask;
    }
    return y_scaled;
}
// 构建输入层前向计算图
ggml_tensor * ueLinearNoSubsampling::build_forward_graph(ggml_context * ctx,
                                                         ggml_tensor *  x_ctb,
                                                         std::nullptr_t,
                                                         int32_t        offset,
                                                         ggml_tensor ** pos_emb_placeholder_out) const {
    ggml_tensor * out_mask = nullptr;
    return build_forward_graph(ctx, x_ctb, nullptr, offset, pos_emb_placeholder_out, &out_mask);
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
static ggml_tensor * ue_up1d_conv1d_im2col_f32_n1(ggml_context * ctx,
                                          ggml_tensor *  w_kic_oc,
                                          ggml_tensor *  x_tcb,
                                          int            stride,
                                          int            padding,
                                          int            dilation) {
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cin  = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    ggml_tensor * im2col = ggml_im2col(ctx, w_kic_oc, x_tcb, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, K * Cin, Cout);
    ggml_tensor * mm = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, im2col->ne[1], Cout, im2col->ne[2]);
    return y_tcb;
}
// 构建计算图
static ggml_tensor * ue_up1d_repeat_upsample_ctb(ggml_context * ctx, ggml_tensor * x_ctb, int32_t up_factor) {
    const int64_t C = x_ctb->ne[0];
    const int64_t T = x_ctb->ne[1];
    const int64_t B = x_ctb->ne[2];
    ggml_tensor * x_4d = ggml_reshape_4d(ctx, x_ctb, C, 1, T, B);
    ggml_tensor * rep_4d = ggml_repeat_4d(ctx, x_4d, C, up_factor, T, B);
    ggml_tensor * y_ctb = ggml_reshape_3d(ctx, rep_4d, C, up_factor * T, B);
    return ggml_cont(ctx, y_ctb);
}
// 构建计算图
// 构建左侧补零
static ggml_tensor * ue_up1d_build_left_pad_zeros_ctb(ggml_context * ctx, int64_t C, int64_t Tpad, int64_t B) {
    ggml_tensor * zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);
    ggml_tensor * pad_4d      = ggml_repeat_4d(ctx, zero_scalar, C, Tpad, B, 1);
    ggml_tensor * pad_3d      = ggml_reshape_3d(ctx, pad_4d, C, Tpad, B);
    return ggml_cont(ctx, pad_3d);
}
static ggml_tensor * ue_up1d_build_conv1d_f32_ctb(ggml_context * ctx,
                                          ggml_tensor *  w_kic_oc,
                                          ggml_tensor *  b_oc,
                                          ggml_tensor *  x_ctb) {
    const int64_t Cin  = x_ctb->ne[0];
    const int64_t B    = x_ctb->ne[2];
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cinw = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    ggml_tensor * x_tcb = ggml_permute(ctx, x_ctb, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    ggml_tensor * y_tcb = nullptr;
    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
        const size_t  offset = x_tcb->nb[2] * static_cast<size_t>(b_idx);
        ggml_tensor * x_b = ggml_view_3d(ctx, x_tcb, x_tcb->ne[0], x_tcb->ne[1], 1, x_tcb->nb[1], x_tcb->nb[2], offset);
        ggml_tensor * y_b = ue_up1d_conv1d_im2col_f32_n1(ctx, w_kic_oc, x_b, 1, 0, 1);
        if (y_tcb == nullptr) {
            y_tcb = y_b;
        } else {
            y_tcb = ggml_concat(ctx, y_tcb, y_b, 2);
        }
    }
    ggml_tensor * y_ctb = ggml_permute(ctx, y_tcb, 1, 0, 2, 3);
    y_ctb               = ggml_cont(ctx, y_ctb);
    if (b_oc != nullptr) {
        ggml_tensor * b_bcast = ggml_reshape_3d(ctx, b_oc, Cout, 1, 1);
        y_ctb                 = ggml_add(ctx, y_ctb, b_bcast);
    }
    return y_ctb;
}
}  // namespace
ueUpsample1D::ueUpsample1D(int32_t channels, int32_t out_channels, int32_t stride, float scale_factor) :
    channels_(channels),
    out_channels_(out_channels),
    stride_(stride),
    scale_factor_(scale_factor) {}
// 绑定参数
void ueUpsample1D::set_parameters(ggml_tensor * conv_weight, ggml_tensor * conv_bias) {
    conv_weight_ = conv_weight;
    conv_bias_   = conv_bias;
}
// 构建前向计算图
// 构建上采样前向计算图
ggml_tensor * ueUpsample1D::build_forward_graph(ggml_context * ctx, ggml_tensor * inputs_ctb) const {
    if (ctx == nullptr || inputs_ctb == nullptr) {
        return nullptr;
    }
    const int64_t C = inputs_ctb->ne[0];
    const int64_t B = inputs_ctb->ne[2];
    const int32_t up_factor = static_cast<int32_t>(std::llround(static_cast<double>(scale_factor_)));
    const int     pad_left = stride_ * 2;
    const int64_t K        = conv_weight_->ne[0];
    ggml_tensor * up_ctb  = ue_up1d_repeat_upsample_ctb(ctx, inputs_ctb, up_factor);
    ggml_tensor * pad_ctb = ue_up1d_build_left_pad_zeros_ctb(ctx, C, pad_left, B);
    ggml_tensor * x_cat   = ggml_concat(ctx, pad_ctb, up_ctb, 1);
    x_cat                 = ggml_cont(ctx, x_cat);
    return ue_up1d_build_conv1d_f32_ctb(ctx, conv_weight_, conv_bias_, x_cat);
}
ggml_tensor * ueUpsample1D::build_forward_chunk_graph(ggml_context * ctx,
                                                      ggml_tensor *  inputs_ctb,
                                                      ggml_tensor *  cache_ctb,
                                                      ggml_tensor ** new_cache_ctb_out) const {
    if (new_cache_ctb_out) {
        *new_cache_ctb_out = nullptr;
    }
    if (ctx == nullptr || inputs_ctb == nullptr) {
        return nullptr;
    }
    const int64_t C = inputs_ctb->ne[0];
    const int64_t B = inputs_ctb->ne[2];
    const int32_t up_factor = static_cast<int32_t>(std::llround(static_cast<double>(scale_factor_)));
    const int     cache_t = stride_ * 2;
    const int64_t K       = conv_weight_->ne[0];
    ggml_tensor * up_ctb = ue_up1d_repeat_upsample_ctb(ctx, inputs_ctb, up_factor);
    ggml_tensor * cache_use = cache_ctb;
    if (cache_use == nullptr) {
        cache_use = ue_up1d_build_left_pad_zeros_ctb(ctx, C, cache_t, B);
    } else {
    }
    ggml_tensor * x_cat = ggml_concat(ctx, cache_use, up_ctb, 1);
    x_cat               = ggml_cont(ctx, x_cat);
    if (new_cache_ctb_out) {
        const int64_t Tcat = x_cat->ne[1];
        const size_t  offset         = static_cast<size_t>(Tcat - cache_t) * x_cat->nb[1];
        ggml_tensor * new_cache_view = ggml_view_3d(ctx, x_cat, C, cache_t, B, x_cat->nb[1], x_cat->nb[2], offset);
        *new_cache_ctb_out           = ggml_cont(ctx, new_cache_view);
    }
    return ue_up1d_build_conv1d_f32_ctb(ctx, conv_weight_, conv_bias_, x_cat);
}
ueUpsample1D::ueForwardOut ueUpsample1D::forward(ggml_context * ctx,
                                                 ggml_tensor *  inputs_ctb,
                                                 ggml_tensor *  input_lengths_b) const {
    (void) input_lengths_b;
    ueForwardOut out;
    out.outputs_ctb   = build_forward_graph(ctx, inputs_ctb);
    out.out_lengths_b = nullptr;
    return out;
}
ueUpsample1D::ueForwardChunkOut ueUpsample1D::forward_chunk(ggml_context * ctx,
                                                            ggml_tensor *  inputs_ctb,
                                                            ggml_tensor *  input_lengths_b,
                                                            ggml_tensor *  cache_ctb) const {
    (void) input_lengths_b;
    ueForwardChunkOut out;
    out.outputs_ctb   = build_forward_chunk_graph(ctx, inputs_ctb, cache_ctb, &out.new_cache_ctb);
    out.out_lengths_b = nullptr;
    return out;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
// 构建计算图
ggml_tensor * ue_uce_make_valid_mask(ggml_context * ctx, ggml_tensor * lengths_b, int64_t T) {
    ggml_tensor * pad_i32_tb = ueMaskUtils::make_pad_mask(ctx, lengths_b, T);
    ggml_tensor * pad_f32_tb =
        (pad_i32_tb->type == GGML_TYPE_F32) ? pad_i32_tb : ggml_cast(ctx, pad_i32_tb, GGML_TYPE_F32);
    ggml_tensor * one_f32      = ggml_arange(ctx, 1.0f, 2.0f, 1.0f);
    // 🔧 用 ggml_add + ggml_repeat 替代 ggml_add1，支持 Metal 加速
    ggml_tensor * neg_pad      = ggml_neg(ctx, pad_f32_tb);
    ggml_tensor * one_rep      = ggml_repeat(ctx, one_f32, neg_pad);
    ggml_tensor * valid_f32_tb = ggml_add(ctx, neg_pad, one_rep);
    return ggml_cont(ctx, valid_f32_tb);
}
// 从张量中切片
ggml_tensor * ue_uce_slice_time_ctb(ggml_context * ctx, ggml_tensor * x_ctb, int64_t t0, int64_t tlen) {
    const size_t  off = x_ctb->nb[1] * static_cast<size_t>(t0);
    ggml_tensor * v   = ggml_view_3d(ctx, x_ctb, x_ctb->ne[0], tlen, x_ctb->ne[2], x_ctb->nb[1], x_ctb->nb[2], off);
    return ggml_cont(ctx, v);
}
// 构建计算图
ggml_tensor * ue_uce_build_right_pad_zeros_ctb(ggml_context * ctx, ggml_tensor * x_ctb, int64_t Tpad) {
    if (Tpad == 0) {
        return x_ctb;
    }
    const int64_t C           = x_ctb->ne[0];
    const int64_t B           = x_ctb->ne[2];
    ggml_tensor * zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);
    ggml_tensor * pad_4d      = ggml_repeat_4d(ctx, zero_scalar, C, Tpad, B, 1);
    ggml_tensor * pad_ctb     = ggml_reshape_3d(ctx, pad_4d, C, Tpad, B);
    pad_ctb                   = ggml_cont(ctx, pad_ctb);
    ggml_tensor * out         = ggml_concat(ctx, x_ctb, pad_ctb, 1);
    return ggml_cont(ctx, out);
}
// 从张量中切片
ggml_tensor * ue_uce_slice_att_cache_layer(ggml_context * ctx, ggml_tensor * att_cache_packed, int64_t layer_idx, int64_t B) {
    const int64_t E   = att_cache_packed->ne[0];
    const int64_t T   = att_cache_packed->ne[1];
    const int64_t H   = att_cache_packed->ne[2];
    const size_t  off = att_cache_packed->nb[3] * static_cast<size_t>(layer_idx * B);
    ggml_tensor * v = ggml_view_4d(ctx, att_cache_packed, E, T, H, B, att_cache_packed->nb[1], att_cache_packed->nb[2],
                                   att_cache_packed->nb[3], off);
    return ggml_cont(ctx, v);
}
// 从张量中切片
ggml_tensor * ue_uce_slice_att_cache_prefix(ggml_context * ctx, ggml_tensor * cache_full_ethb, int64_t offset1) {
    if (offset1 == 0) {
        return nullptr;
    }
    ggml_tensor * v =
        ggml_view_4d(ctx, cache_full_ethb, cache_full_ethb->ne[0], offset1, cache_full_ethb->ne[2],
                     cache_full_ethb->ne[3], cache_full_ethb->nb[1], cache_full_ethb->nb[2], cache_full_ethb->nb[3], 0);
    return ggml_cont(ctx, v);
}
}  // namespace
ueUpsampleConformerEncoderV2::ueUpsampleConformerEncoderV2(int32_t input_size,
                                                           int32_t output_size,
                                                           int32_t pre_lookahead_len,
                                                           int32_t num_blocks,
                                                           int32_t num_up_blocks,
                                                           int32_t up_stride,
                                                           float   up_scale_factor,
                                                           int32_t attention_heads,
                                                           bool    key_bias,
                                                           int32_t linear_units,
                                                           float   dropout_rate,
                                                           float   positional_dropout_rate,
                                                           float   attention_dropout_rate,
                                                           bool    normalize_before) :
    input_size_(input_size),
    output_size_(output_size),
    pre_lookahead_len_(pre_lookahead_len),
    num_blocks_(num_blocks),
    num_up_blocks_(num_up_blocks),
    up_stride_(up_stride),
    up_scale_factor_(up_scale_factor),
    attention_heads_(attention_heads),
    key_bias_(key_bias),
    linear_units_(linear_units),
    dropout_rate_(dropout_rate),
    positional_dropout_rate_(positional_dropout_rate),
    attention_dropout_rate_(attention_dropout_rate),
    normalize_before_(normalize_before) {
    auto pos_enc = std::make_shared<ueEspnetRelPositionalEncoding>(output_size_, positional_dropout_rate_, 5000);
    embed_       = std::make_shared<ueLinearNoSubsampling>(input_size_, output_size_, dropout_rate_, pos_enc);
    pre_lookahead_layer_ = std::make_shared<uePreLookaheadLayer>(output_size_, pre_lookahead_len_);
    encoders_.reserve((size_t) num_blocks_);
    for (int32_t i = 0; i < num_blocks_; ++i) {
        auto attn = std::make_shared<ueRelPositionMultiHeadedAttention>(attention_heads_, output_size_,
                                                                        attention_dropout_rate_, key_bias_);
        auto ffn = std::make_shared<uePositionwiseFeedForward>(output_size_, linear_units_, dropout_rate_, "swish");
        encoders_.push_back(
            std::make_shared<ueConformerEncoderLayer>(output_size_, attn, ffn, dropout_rate_, normalize_before_));
    }
    up_layer_ = std::make_shared<ueUpsample1D>(output_size_, output_size_, up_stride_, up_scale_factor_);
    auto up_pos_enc = std::make_shared<ueEspnetRelPositionalEncoding>(output_size_, positional_dropout_rate_, 5000);
    up_embed_       = std::make_shared<ueLinearNoSubsampling>(input_size_, output_size_, dropout_rate_, up_pos_enc);
    up_encoders_.reserve((size_t) num_up_blocks_);
    for (int32_t i = 0; i < num_up_blocks_; ++i) {
        auto attn = std::make_shared<ueRelPositionMultiHeadedAttention>(attention_heads_, output_size_,
                                                                        attention_dropout_rate_, key_bias_);
        auto ffn  = std::make_shared<uePositionwiseFeedForward>(output_size_, linear_units_, dropout_rate_, "swish");
        up_encoders_.push_back(
            std::make_shared<ueConformerEncoderLayer>(output_size_, attn, ffn, dropout_rate_, normalize_before_));
    }
}
// 绑定参数
void ueUpsampleConformerEncoderV2::set_after_norm_parameters(ggml_tensor * after_norm_weight,
                                                             ggml_tensor * after_norm_bias) {
    after_norm_weight_ = after_norm_weight;
    after_norm_bias_   = after_norm_bias;
}
ueUpsampleConformerEncoderV2::ueForwardOut
// 执行前向推理
ueUpsampleConformerEncoderV2::forward(ggml_context * ctx, ggml_tensor * xs_ctb, ggml_tensor * xs_lens_b) const {
    ueForwardOut out;
    if (!ctx || !xs_ctb || !xs_lens_b) {
        return out;
    }
    out.ys_ctb = build_forward_graph(ctx, xs_ctb, xs_lens_b, &out.masks);
    return out;
}
// 构建编码器前向计算图
ggml_tensor * ueUpsampleConformerEncoderV2::build_forward_graph(ggml_context * ctx,
                                                                ggml_tensor *  xs_ctb,
                                                                ggml_tensor *  masks,
                                                                ggml_tensor ** out_masks_out) const {
    if (out_masks_out) {
        *out_masks_out = nullptr;
    }
    if (!ctx || !xs_ctb || !embed_ || !pre_lookahead_layer_ || !up_layer_ || !up_embed_) {
        return nullptr;
    }
    const int64_t B  = xs_ctb->ne[2];
    const int64_t T0 = xs_ctb->ne[1];
    ggml_tensor * lengths_b = nullptr;
    ggml_tensor * mask0_in  = nullptr;
    if (masks != nullptr && masks->ne[0] == B && masks->ne[1] == 1 && masks->ne[2] == 1) {
        lengths_b = masks;
    } else {
        mask0_in = masks;
    }
    ggml_tensor * valid0_tb = nullptr;
    if (lengths_b != nullptr) {
        valid0_tb = ue_uce_make_valid_mask(ctx, lengths_b, T0);
        mask0_in  = valid0_tb;
    }
    ggml_tensor * pos_emb0 = nullptr;
    ggml_tensor * mask0    = nullptr;
    ggml_tensor * x        = embed_->build_forward_graph(ctx, xs_ctb, mask0_in, 0, &pos_emb0, &mask0);
    if (pos_emb0) {
        ggml_set_name(pos_emb0, "ue_uce_pos0");
    }
    x = ggml_cont(ctx, x);
    x = pre_lookahead_layer_->build_forward_graph(ctx, x);
    x = ggml_cont(ctx, x);
    for (int64_t li = 0; li < (int64_t) encoders_.size(); ++li) {
        const auto & layer = encoders_[(size_t) li];
        if (!layer) {
            continue;
        }
        x = layer->build_forward_graph(ctx, x, mask0, pos_emb0, nullptr, nullptr, nullptr, nullptr, nullptr);
        x = ggml_cont(ctx, x);
    }
    ggml_tensor * x_up = up_layer_->build_forward_graph(ctx, x);
    x_up               = ggml_cont(ctx, x_up);
    ggml_tensor * pos_emb1       = nullptr;
    ggml_tensor * mask1_valid_tb = nullptr;
    if (lengths_b != nullptr) {
        ggml_tensor * lengths_f32_b =
            (lengths_b->type == GGML_TYPE_F32) ? lengths_b : ggml_cast(ctx, lengths_b, GGML_TYPE_F32);
        ggml_tensor * lengths_up_f32_b = ggml_scale(ctx, lengths_f32_b, (float) up_stride_);
        mask1_valid_tb                 = ue_uce_make_valid_mask(ctx, lengths_up_f32_b, x_up->ne[1]);
    }
    ggml_tensor * tmp_mask_out = nullptr;
    ggml_tensor * x2           = up_embed_->build_forward_graph(ctx, x_up, mask1_valid_tb, 0, &pos_emb1, &tmp_mask_out);
    if (pos_emb1) {
        ggml_set_name(pos_emb1, "ue_uce_pos1");
    }
    x2 = ggml_cont(ctx, x2);
    for (int64_t li = 0; li < (int64_t) up_encoders_.size(); ++li) {
        const auto & layer = up_encoders_[(size_t) li];
        if (!layer) {
            continue;
        }
        x2 = layer->build_forward_graph(ctx, x2, tmp_mask_out, pos_emb1, nullptr, nullptr, nullptr, nullptr, nullptr);
        x2 = ggml_cont(ctx, x2);
    }
    if (normalize_before_) {
        x2 = ue_build_layer_norm(ctx, x2, after_norm_weight_, after_norm_bias_, 1e-5f);
        x2 = ggml_cont(ctx, x2);
    }
    if (out_masks_out) {
        *out_masks_out = tmp_mask_out;
    }
    return x2;
}
ggml_tensor * ueUpsampleConformerEncoderV2::build_forward_chunk_graph(ggml_context * ctx,
                                                                      ggml_tensor *  xs_ctb,
                                                                      bool           last_chunk,
                                                                      ggml_tensor *  cnn_cache_ctb,
                                                                      ggml_tensor *  att_cache,
                                                                      ggml_tensor ** new_cnn_cache_ctb_out,
                                                                      ggml_tensor ** new_att_cache_out) const {
    if (new_cnn_cache_ctb_out) {
        *new_cnn_cache_ctb_out = nullptr;
    }
    if (new_att_cache_out) {
        *new_att_cache_out = nullptr;
    }
    if (!ctx || !xs_ctb || !embed_ || !pre_lookahead_layer_ || !up_layer_ || !up_embed_) {
        return nullptr;
    }
    const int64_t B = xs_ctb->ne[2];
    ggml_tensor * cnn_cache1 = nullptr;
    ggml_tensor * cnn_cache2 = nullptr;
    if (cnn_cache_ctb != nullptr) {
        cnn_cache1 = ue_uce_slice_time_ctb(ctx, cnn_cache_ctb, 0, uePreLookaheadLayer::cache_t());
        cnn_cache2 = ue_uce_slice_time_ctb(ctx, cnn_cache_ctb, uePreLookaheadLayer::cache_t(), up_layer_->cache_t());
    }
    int64_t offset1 = 0;
    if (att_cache != nullptr) {
        offset1 = att_cache->ne[1] / 2;
    }
    ggml_tensor * tmp_pos = nullptr;
    ggml_tensor * x       = embed_->build_forward_graph(ctx, xs_ctb, nullptr, 0, &tmp_pos, nullptr);
    x                     = ggml_cont(ctx, x);
    if (last_chunk) {
        x = ue_uce_build_right_pad_zeros_ctb(ctx, x, pre_lookahead_len_);
    }
    ggml_tensor * new_cnn_cache1 = nullptr;
    x                            = pre_lookahead_layer_->build_forward_chunk_graph(ctx, x, cnn_cache1, &new_cnn_cache1);
    x                            = ggml_cont(ctx, x);
    const int64_t T1   = x->ne[1];
    ggml_tensor * pos1 = embed_->position_encoding(ctx, nullptr, (int32_t) (offset1 + T1));
    if (pos1) {
        ggml_set_name(pos1, "ue_uce_pos_stream1");
    }
    ggml_tensor * chunk_masks = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 0, 0, 0);
    ggml_tensor * packed_new_att_cache1 = nullptr;
    int64_t       Tk1                   = 0;
    for (int64_t li = 0; li < (int64_t) encoders_.size(); ++li) {
        const auto & layer = encoders_[(size_t) li];
        if (!layer) {
            continue;
        }
        ggml_tensor * cache_in = nullptr;
        if (att_cache != nullptr && offset1 > 0) {
            ggml_tensor * full = ue_uce_slice_att_cache_layer(ctx, att_cache, li, B);
            cache_in           = ue_uce_slice_att_cache_prefix(ctx, full, offset1);
        }
        ggml_tensor * new_cache     = nullptr;
        ggml_tensor * new_cnn_dummy = nullptr;
        x = layer->build_forward_graph(ctx, x, chunk_masks, pos1, nullptr, cache_in, nullptr, &new_cache,
                                       &new_cnn_dummy);
        x = ggml_cont(ctx, x);
        ggml_tensor * new_cache_rep = ggml_concat(ctx, new_cache, new_cache, 1);
        new_cache_rep               = ggml_cont(ctx, new_cache_rep);
        if (li == 0) {
            Tk1                   = new_cache->ne[1];
            packed_new_att_cache1 = new_cache_rep;
        } else {
            packed_new_att_cache1 = ggml_concat(ctx, packed_new_att_cache1, new_cache_rep, 3);
            packed_new_att_cache1 = ggml_cont(ctx, packed_new_att_cache1);
        }
    }
    ggml_tensor * new_cnn_cache2 = nullptr;
    ggml_tensor * x_up           = up_layer_->build_forward_chunk_graph(ctx, x, cnn_cache2, &new_cnn_cache2);
    x_up                         = ggml_cont(ctx, x_up);
    ggml_tensor * tmp_pos2 = nullptr;
    ggml_tensor * x2       = up_embed_->build_forward_graph(ctx, x_up, nullptr, 0, &tmp_pos2, nullptr);
    x2                     = ggml_cont(ctx, x2);
    const int64_t T2   = x2->ne[1];
    ggml_tensor * pos2 = embed_->position_encoding(ctx, nullptr, (int32_t) (offset1 * up_stride_ + T2));
    if (pos2) {
        ggml_set_name(pos2, "ue_uce_pos_stream2");
    }
    ggml_tensor * packed_new_att_cache2 = nullptr;
    int64_t       Tk2                   = 0;
    for (int64_t li = 0; li < (int64_t) up_encoders_.size(); ++li) {
        const auto & layer = up_encoders_[(size_t) li];
        if (!layer) {
            continue;
        }
        ggml_tensor * cache_in = nullptr;
        if (att_cache != nullptr && att_cache->ne[1] > 0) {
            const int64_t layer_idx = (int64_t) encoders_.size() + li;
            cache_in                = ue_uce_slice_att_cache_layer(ctx, att_cache, layer_idx, B);
        }
        ggml_tensor * new_cache     = nullptr;
        ggml_tensor * new_cnn_dummy = nullptr;
        x2 = layer->build_forward_graph(ctx, x2, chunk_masks, pos2, nullptr, cache_in, nullptr, &new_cache,
                                        &new_cnn_dummy);
        x2 = ggml_cont(ctx, x2);
        if (li == 0) {
            Tk2                   = new_cache->ne[1];
            packed_new_att_cache2 = new_cache;
        } else {
            packed_new_att_cache2 = ggml_concat(ctx, packed_new_att_cache2, new_cache, 3);
            packed_new_att_cache2 = ggml_cont(ctx, packed_new_att_cache2);
        }
    }
    if (normalize_before_) {
        x2 = ue_build_layer_norm(ctx, x2, after_norm_weight_, after_norm_bias_, 1e-5f);
        x2 = ggml_cont(ctx, x2);
    }
    ggml_tensor * new_att_packed = nullptr;
    if (packed_new_att_cache1 != nullptr && packed_new_att_cache2 != nullptr) {
        new_att_packed = ggml_concat(ctx, packed_new_att_cache1, packed_new_att_cache2, 3);
        new_att_packed = ggml_cont(ctx, new_att_packed);
    } else if (packed_new_att_cache1 != nullptr) {
        new_att_packed = packed_new_att_cache1;
    } else {
        new_att_packed = packed_new_att_cache2;
    }
    ggml_tensor * new_cnn_packed = ggml_concat(ctx, new_cnn_cache1, new_cnn_cache2, 1);
    new_cnn_packed               = ggml_cont(ctx, new_cnn_packed);
    if (new_att_cache_out) {
        *new_att_cache_out = new_att_packed;
    }
    if (new_cnn_cache_ctb_out) {
        *new_cnn_cache_ctb_out = new_cnn_packed;
    }
    return x2;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
namespace {
// 释放 gguf 资源
void ue_loader_gguf_free_gguf_context(gguf_context * ctx) {
    if (ctx) {
        gguf_free(ctx);
    }
}
// 释放 ggml 上下文
void ue_loader_gguf_free_ggml_context(ggml_context * ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}
// 释放后端 buffer
void ue_loader_gguf_free_backend_buffer(ggml_backend_buffer_t buf) {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
}
}  // namespace
ueUpsampleEncoderModelLoaderGGUF::ueUpsampleEncoderModelLoaderGGUF() = default;
ueUpsampleEncoderModelLoaderGGUF::~ueUpsampleEncoderModelLoaderGGUF() {
    reset();
}
// 重置并释放旧资源
void ueUpsampleEncoderModelLoaderGGUF::reset() {
    tensors_.clear();
    ue_loader_gguf_free_backend_buffer(buf_weights_);
    buf_weights_ = nullptr;
    ue_loader_gguf_free_ggml_context(ctx_data_);
    ctx_data_ = nullptr;
    ue_loader_gguf_free_ggml_context(ctx_meta_);
    ctx_meta_ = nullptr;
    ue_loader_gguf_free_gguf_context(ctx_gguf_);
    ctx_gguf_ = nullptr;
    backend_ = nullptr;
    path_.clear();
}
// 从 gguf 加载权重
bool ueUpsampleEncoderModelLoaderGGUF::load_from_file(const std::string & gguf_path, ggml_backend_t backend) {
    reset();
    if (!backend) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: backend is null\n");
        return false;
    }
    backend_ = backend;
    path_    = gguf_path;
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;
    ctx_gguf_     = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf_) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: failed to open gguf: %s\n", gguf_path.c_str());
        return false;
    }
    ctx_meta_ = meta;
    if (!ctx_meta_) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: gguf meta ctx is null: %s\n", gguf_path.c_str());
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_);
    if (n_tensors <= 0) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: no tensors in gguf: %s\n", gguf_path.c_str());
        return false;
    }
    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        const size_t off = gguf_get_data_offset(ctx_gguf_) + gguf_get_tensor_offset(ctx_gguf_, i);
        offsets.emplace(std::string(name), off);
    }
    ggml_init_params data_params{};
    data_params.mem_size   = (size_t) (n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_data_              = ggml_init(data_params);
    if (!ctx_data_) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: failed to init ctx_data\n");
        return false;
    }
    tensors_.reserve((size_t) n_tensors);
    std::vector<ggml_tensor *> to_load;
    to_load.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta_, name);
        if (!meta_tensor) {
            LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: missing meta tensor: %s\n", name);
            return false;
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data_, meta_tensor);
        ggml_set_name(data_tensor, name);
        tensors_.emplace(std::string(name), data_tensor);
        to_load.push_back(data_tensor);
    }
    ggml_backend_buffer_type_t buft     = ggml_backend_get_default_buffer_type(backend_);
    buf_weights_                        = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data_, buft);
    if (!buf_weights_) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: backend weight alloc failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buf_weights_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: failed to open file stream: %s\n", gguf_path.c_str());
        return false;
    }
    std::vector<uint8_t> staging;
    for (ggml_tensor * t : to_load) {
        if (!t || t->name[0] == '\0') {
            continue;
        }
        const auto it = offsets.find(std::string(t->name));
        if (it == offsets.end()) {
            LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: missing offset for tensor: %s\n", t->name);
            return false;
        }
        const size_t off = it->second;
        fin.seekg((std::streamoff) off, std::ios::beg);
        if (!fin) {
            LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: seek failed for tensor: %s\n", t->name);
            return false;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(t->data), (std::streamsize) nbytes);
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), (std::streamsize) nbytes);
            backend_tensor_set(backend_, t, staging.data(), nbytes);
        }
        if (!fin) {
            LOG_ERROR( "ueUpsampleEncoderModelLoaderGGUF: read failed for tensor: %s\n", t->name);
            return false;
        }
    }
    if (!ggml_backend_buft_is_host(buft)) {
        ggml_backend_synchronize(backend_);
    }
    return true;
}
// 按名称获取张量
ggml_tensor * ueUpsampleEncoderModelLoaderGGUF::get_tensor(const std::string & name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}
}  // namespace upsample_encoder_v2
}  // namespace omni
// 主要负责这一块的前向构图
// hifigan2 f0_predictor 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static ggml_tensor * hg_f0_predictor_conv1d_k3_p1_f32(ggml_context * ctx,
                                                       ggml_tensor *  x_tcb,
                                                       ggml_tensor *  w_kic_oc,
                                                       ggml_tensor *  b_oc) {
    if (!ctx || !x_tcb || !w_kic_oc || !b_oc) {
        return nullptr;
    }
    if (x_tcb->type != GGML_TYPE_F32 || w_kic_oc->type != GGML_TYPE_F32 || b_oc->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_f0_predictor_conv1d_k3_p1_f32: expected F32 tensors\n");
        return nullptr;
    }
    const int64_t T   = x_tcb->ne[0];
    const int64_t Cin = x_tcb->ne[1];
    const int64_t B   = x_tcb->ne[2];
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cinw = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    if (K != 3) {
        LOG_ERROR( "hg2_f0_predictor_conv1d_k3_p1_f32: expected K=3, got %lld\n", (long long) K);
        return nullptr;
    }
    if (Cinw != Cin) {
        LOG_ERROR( "hg2_f0_predictor_conv1d_k3_p1_f32: Cin mismatch, x Cin=%lld w Cin=%lld\n",
                     (long long) Cin, (long long) Cinw);
        return nullptr;
    }
    if (b_oc->ne[0] != Cout) {
        LOG_ERROR( "hg2_f0_predictor_conv1d_k3_p1_f32: bias shape mismatch, expected %lld got %lld\n",
                     (long long) Cout, (long long) b_oc->ne[0]);
        return nullptr;
    }
    ggml_tensor * im2col = ggml_im2col(ctx, w_kic_oc, x_tcb, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    im2col_2d               = ggml_cont(ctx, im2col_2d);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, K * Cin, Cout);
    w_2d               = ggml_cont(ctx, w_2d);
    ggml_tensor * mm = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, T, Cout, B);
    y_tcb               = ggml_cont(ctx, y_tcb);
    ggml_tensor * b_1c1  = ggml_reshape_3d(ctx, b_oc, 1, Cout, 1);
    b_1c1                = ggml_cont(ctx, b_1c1);
    ggml_tensor * b_tcb  = ggml_repeat(ctx, b_1c1, y_tcb);
    ggml_tensor * y_bias = ggml_add(ctx, y_tcb, b_tcb);
    return y_bias;
}
// 构建 f0 预测计算
ggml_tensor * hg2_f0_predictor::hg_f0_predictor_build_graph(ggml_context * ctx, ggml_tensor * x_c80_t_b) const {
    if (!ctx || !x_c80_t_b) {
        return nullptr;
    }
    if (!conv0_weight || !conv1_weight || !conv2_weight || !conv3_weight || !conv4_weight || !linear_weight ||
        !linear_bias) {
        LOG_ERROR( "hg2_f0_predictor_build_graph: missing weights\n");
        return nullptr;
    }
    if (x_c80_t_b->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_f0_predictor_build_graph: expected x F32\n");
        return nullptr;
    }
    const int64_t Cin = x_c80_t_b->ne[0];
    const int64_t T   = x_c80_t_b->ne[1];
    const int64_t B   = x_c80_t_b->ne[2];
    if (Cin != 80) {
        LOG_ERROR( "hg2_f0_predictor_build_graph: expected Cin=80, got %lld\n", (long long) Cin);
        return nullptr;
    }
    ggml_tensor * x_tcb = ggml_permute(ctx, x_c80_t_b, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    ggml_tensor * h = hg_f0_predictor_conv1d_k3_p1_f32(ctx, x_tcb, conv0_weight, conv0_bias);
    if (!h) {
        return nullptr;
    }
    h = ggml_elu(ctx, h);
    h = hg_f0_predictor_conv1d_k3_p1_f32(ctx, h, conv1_weight, conv1_bias);
    if (!h) {
        return nullptr;
    }
    h = ggml_elu(ctx, h);
    h = hg_f0_predictor_conv1d_k3_p1_f32(ctx, h, conv2_weight, conv2_bias);
    if (!h) {
        return nullptr;
    }
    h = ggml_elu(ctx, h);
    h = hg_f0_predictor_conv1d_k3_p1_f32(ctx, h, conv3_weight, conv3_bias);
    if (!h) {
        return nullptr;
    }
    h = ggml_elu(ctx, h);
    h = hg_f0_predictor_conv1d_k3_p1_f32(ctx, h, conv4_weight, conv4_bias);
    if (!h) {
        return nullptr;
    }
    h = ggml_elu(ctx, h);
    h = ggml_cont(ctx, h);
    ggml_tensor * h_ctb = ggml_permute(ctx, h, 1, 0, 2, 3);
    h_ctb               = ggml_cont(ctx, h_ctb);
    const int64_t C     = h_ctb->ne[0];
    ggml_tensor * h_2d  = ggml_reshape_2d(ctx, h_ctb, C, T * B);
    h_2d                = ggml_cont(ctx, h_2d);
    if (linear_weight->type != GGML_TYPE_F32 || linear_weight->ne[0] != C || linear_weight->ne[1] != 1) {
        LOG_ERROR( "hg2_f0_predictor_build_graph: linear_weight must be [C,1] (got [%lld,%lld])\n",
                     (long long) linear_weight->ne[0], (long long) linear_weight->ne[1]);
        return nullptr;
    }
    ggml_tensor * mm   = ggml_mul_mat(ctx, h_2d, linear_weight);
    ggml_tensor * y_tb = ggml_reshape_2d(ctx, mm, T, B);
    if (linear_bias->ne[0] != 1) {
        LOG_ERROR( "hg2_f0_predictor_build_graph: linear_bias must have 1 element\n");
        return nullptr;
    }
    ggml_tensor * bias_s = ggml_reshape_4d(ctx, linear_bias, 1, 1, 1, 1);
    // 🔧 用 ggml_add + ggml_repeat 替代 ggml_add1，支持 Metal 加速
    ggml_tensor * bias_rep = ggml_repeat(ctx, bias_s, y_tb);
    y_tb                 = ggml_add(ctx, y_tb, bias_rep);
    y_tb                 = ggml_abs(ctx, y_tb);
    y_tb                 = ggml_cont(ctx, y_tb);
    ggml_set_name(y_tb, "hg2.f0_predictor.f0_tb");
    return y_tb;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 hifigan2_model_loader_gguf 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static void hg_free_gguf_context(gguf_context * ctx) {
    if (ctx) {
        gguf_free(ctx);
    }
}
static void hg_free_ggml_context(ggml_context * ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}
static void hg_free_backend_buffer(ggml_backend_buffer_t buf) {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
}
// 重置并释放旧资源
void hg2_gguf_model_loader::hg_gguf_model_loader_reset() {
    tensors.clear();
    hg_free_backend_buffer(buf_weights);
    buf_weights = nullptr;
    hg_free_ggml_context(ctx_data);
    ctx_data = nullptr;
    hg_free_ggml_context(ctx_meta);
    ctx_meta = nullptr;
    hg_free_gguf_context(ctx_gguf);
    ctx_gguf = nullptr;
    backend = nullptr;
    path.clear();
}
// 加载模型权重
bool hg2_gguf_model_loader::hg_gguf_model_loader_load_from_file(const std::string & gguf_path,
                                                                 ggml_backend_t      backend_in) {
    hg_gguf_model_loader_reset();
    if (!backend_in) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: backend is null\n");
        return false;
    }
    backend = backend_in;
    path    = gguf_path;
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;
    ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: failed to open gguf: %s\n", gguf_path.c_str());
        return false;
    }
    ctx_meta = meta;
    if (!ctx_meta) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: gguf meta ctx is null\n");
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors <= 0) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: no tensors in gguf: %s\n", gguf_path.c_str());
        return false;
    }
    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (!name) {
            continue;
        }
        const size_t off = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        offsets.emplace(std::string(name), off);
    }
    ggml_init_params data_params{};
    data_params.mem_size   = (size_t) (n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_data               = ggml_init(data_params);
    if (!ctx_data) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: failed to init ctx_data\n");
        return false;
    }
    std::vector<ggml_tensor *> to_load;
    tensors.reserve((size_t) n_tensors);
    to_load.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (!name) {
            continue;
        }
        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta, name);
        if (!meta_tensor) {
            LOG_ERROR( "hg2_gguf_model_loader_load_from_file: missing meta tensor: %s\n", name);
            return false;
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data, meta_tensor);
        ggml_set_name(data_tensor, name);
        tensors.emplace(std::string(name), data_tensor);
        to_load.push_back(data_tensor);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    buf_weights                     = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
    if (!buf_weights) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: backend weight alloc failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR( "hg2_gguf_model_loader_load_from_file: failed to open file stream: %s\n",
                     gguf_path.c_str());
        return false;
    }
    std::vector<uint8_t> staging;
    for (ggml_tensor * t : to_load) {
        if (!t || !t->name) {
            continue;
        }
        const auto it = offsets.find(std::string(t->name));
        if (it == offsets.end()) {
            LOG_ERROR( "hg2_gguf_model_loader_load_from_file: missing offset for tensor: %s\n", t->name);
            return false;
        }
        const size_t off = it->second;
        fin.seekg((std::streamoff) off, std::ios::beg);
        if (!fin) {
            LOG_ERROR( "hg2_gguf_model_loader_load_from_file: seek failed for tensor: %s\n", t->name);
            return false;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(t->data), (std::streamsize) nbytes);
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), (std::streamsize) nbytes);
            ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        }
        if (!fin) {
            LOG_ERROR( "hg2_gguf_model_loader_load_from_file: read failed for tensor: %s\n", t->name);
            return false;
        }
    }
    // 按名称获取张量
    return true;
}
ggml_tensor * hg2_gguf_model_loader::hg_gguf_model_loader_get_tensor(const std::string & name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        return nullptr;
    }
    return it->second;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 构建 hifigan 生成器计算
// 主要负责这一块的前向构图
// hifigan2 hift_generator 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static ggml_tensor * hg_hift_conv1d_f32(ggml_context * ctx,
                                        ggml_tensor *  x_tcb,
                                        ggml_tensor *  w_kic_oc,
                                        ggml_tensor *  b_oc,
                                        int            stride,
                                        int            padding,
                                        int            dilation) {
    if (!ctx || !x_tcb || !w_kic_oc || !b_oc) {
        return nullptr;
    }
    if (x_tcb->type != GGML_TYPE_F32 || w_kic_oc->type != GGML_TYPE_F32 || b_oc->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_hift_conv1d_f32: expected F32 tensors\n");
        return nullptr;
    }
    const int64_t T   = x_tcb->ne[0];
    const int64_t Cin = x_tcb->ne[1];
    const int64_t B   = x_tcb->ne[2];
    (void) T;
    const int64_t K    = w_kic_oc->ne[0];
    const int64_t Cinw = w_kic_oc->ne[1];
    const int64_t Cout = w_kic_oc->ne[2];
    if (Cinw != Cin) {
        LOG_ERROR( "hg2_hift_conv1d_f32: Cin mismatch, x Cin=%lld w Cin=%lld\n", (long long) Cin,
                     (long long) Cinw);
        return nullptr;
    }
    if (b_oc->ne[0] != Cout) {
        LOG_ERROR( "hg2_hift_conv1d_f32: bias mismatch, expected %lld got %lld\n", (long long) Cout,
                     (long long) b_oc->ne[0]);
        return nullptr;
    }
    ggml_tensor * im2col = ggml_im2col(ctx, w_kic_oc, x_tcb, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    im2col_2d               = ggml_cont(ctx, im2col_2d);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_kic_oc, K * Cin, Cout);
    w_2d               = ggml_cont(ctx, w_2d);
    ggml_tensor * mm    = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, im2col->ne[1], Cout, B);
    y_tcb               = ggml_cont(ctx, y_tcb);
    ggml_tensor * b_1c1 = ggml_reshape_3d(ctx, b_oc, 1, Cout, 1);
    b_1c1               = ggml_cont(ctx, b_1c1);
    ggml_tensor * b_tcb = ggml_repeat(ctx, b_1c1, y_tcb);
    ggml_tensor * y     = ggml_add(ctx, y_tcb, b_tcb);
    return y;
}
static ggml_tensor * hg_hift_deconv1d_pad_f32_b1(ggml_context * ctx,
                                                            ggml_tensor *  x_tcb_b1,
                                                            ggml_tensor *  w_koc_ic_b1,
                                                            ggml_tensor *  b_oc,
                                                            int            stride,
                                                            int            padding) {
    if (!ctx || !x_tcb_b1 || !w_koc_ic_b1 || !b_oc) {
        return nullptr;
    }
    if (x_tcb_b1->type != GGML_TYPE_F32 || w_koc_ic_b1->type != GGML_TYPE_F32 || b_oc->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_hift_deconv1d_pad_f32_b1: expected F32 tensors\n");
        return nullptr;
    }
    if (x_tcb_b1->ne[2] != 1) {
        LOG_ERROR( "hg2_hift_deconv1d_pad_f32_b1: only B==1 supported\n");
        return nullptr;
    }
    const int64_t Tin  = x_tcb_b1->ne[0];
    const int64_t Cin  = x_tcb_b1->ne[1];
    const int64_t Cout = w_koc_ic_b1->ne[1];
    const int64_t Cinw = w_koc_ic_b1->ne[2];
    if (Cinw != Cin) {
        LOG_ERROR( "hg2_hift_deconv1d_pad_f32_b1: Cin mismatch\n");
        return nullptr;
    }
    if (b_oc->ne[0] != Cout) {
        LOG_ERROR( "hg2_hift_deconv1d_pad_f32_b1: bias mismatch\n");
        return nullptr;
    }
    ggml_tensor * x_mat = ggml_reshape_2d(ctx, x_tcb_b1, Tin, Cin);
    x_mat               = ggml_cont(ctx, x_mat);
    ggml_tensor * y_full = ggml_conv_transpose_1d(ctx, w_koc_ic_b1, x_mat, stride, 0, 1);
    y_full               = ggml_cont(ctx, y_full);
    const int64_t L_full = y_full->ne[0];
    const int64_t L_out  = L_full - 2 * (int64_t) padding;
    if (L_out <= 0) {
        LOG_ERROR( "hg2_hift_deconv1d_pad_f32_b1: invalid output length\n");
        return nullptr;
    }
    ggml_tensor * y2d = ggml_reshape_2d(ctx, y_full, L_full, Cout);
    y2d               = ggml_cont(ctx, y2d);
    ggml_tensor * y_slice = ggml_view_2d(ctx, y2d, L_out, Cout, y2d->nb[1], (size_t) padding * y2d->nb[0]);
    y_slice               = ggml_cont(ctx, y_slice);
    ggml_tensor * b_1c   = ggml_reshape_2d(ctx, b_oc, 1, Cout);
    b_1c                 = ggml_cont(ctx, b_1c);
    ggml_tensor * b_tc   = ggml_repeat(ctx, b_1c, y_slice);
    ggml_tensor * y_bias = ggml_add(ctx, y_slice, b_tc);
    y_bias               = ggml_cont(ctx, y_bias);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, y_bias, L_out, Cout, 1);
    y_tcb               = ggml_cont(ctx, y_tcb);
    return y_tcb;
}
static ggml_tensor * hg_hift_cache_overwrite_prefix(ggml_context * ctx,
                                                            ggml_tensor *  s_t1_b,
                                                            ggml_tensor *  cache_t1_b) {
    if (!ctx || !s_t1_b || !cache_t1_b) {
        return nullptr;
    }
    const int64_t Ts = s_t1_b->ne[0];
    const int64_t Tc = cache_t1_b->ne[0];
    if (cache_t1_b->ne[1] != 1 || s_t1_b->ne[1] != 1 || cache_t1_b->ne[2] != s_t1_b->ne[2]) {
        LOG_ERROR( "hg2_hift_cache_overwrite_prefix: shape mismatch\n");
        return nullptr;
    }
    if (Tc == 0) {
        return s_t1_b;
    }
    if (Tc > Ts) {
        LOG_ERROR( "hg2_hift_cache_overwrite_prefix: Tc > Ts\n");
        return nullptr;
    }
    ggml_tensor * tail =
        ggml_view_3d(ctx, s_t1_b, Ts - Tc, 1, s_t1_b->ne[2], s_t1_b->nb[1], s_t1_b->nb[2], (size_t) Tc * s_t1_b->nb[0]);
    tail              = ggml_cont(ctx, tail);
    ggml_tensor * out = ggml_concat(ctx, cache_t1_b, tail, 0);
    out               = ggml_cont(ctx, out);
    return out;
}
bool hg2_hift_generator::build_graph_forward(ggml_context * ctx,
                                            ggml_tensor *  speech_feat_c80_t_b,
                                            ggml_tensor *  cache_source_t1_b,
                                            ggml_tensor ** out_wave_t_b,
                                            ggml_tensor ** out_source_t1_b) {
    if (!ctx || !speech_feat_c80_t_b || !cache_source_t1_b || !out_wave_t_b || !out_source_t1_b) {
        return false;
    }
    if (speech_feat_c80_t_b->type != GGML_TYPE_F32 || cache_source_t1_b->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_hift.build_graph_forward: expected F32 inputs\n");
        return false;
    }
    const int64_t B = speech_feat_c80_t_b->ne[2];
    if (B != 1) {
        LOG_ERROR( "hg2_hift.build_graph_forward: only B==1 supported for now\n");
        return false;
    }
    ggml_tensor * f0_tb = f0_pred.hg_f0_predictor_build_graph(ctx, speech_feat_c80_t_b);
    if (!f0_tb) {
        LOG_ERROR( "hg2_hift.build_graph_forward: f0_pred failed\n");
        return false;
    }
    f0_tb = ggml_cont(ctx, f0_tb);
    const int64_t Tm       = f0_tb->ne[0];
    const int64_t T_audio  = Tm * HG2_SAMPLES_PER_MEL;
    // NEAREST 上采样：每个 f0 值重复 HG2_SAMPLES_PER_MEL 次
    // f0_tb: [Tm, B] -> [Tm, 1, B] -> repeat ne[1] -> [Tm, scale, B]
    // -> permute(1,0,2,3) -> [scale, Tm, B] -> reshape -> [T_audio, 1, B]
    const int64_t scale = HG2_SAMPLES_PER_MEL;
    ggml_tensor * f0_t1b = ggml_reshape_3d(ctx, f0_tb, Tm, 1, B);
    f0_t1b = ggml_cont(ctx, f0_t1b);
    ggml_tensor * f0_tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Tm, scale, B);
    ggml_tensor * f0_rep  = ggml_repeat(ctx, f0_t1b, f0_tmpl);
    f0_rep = ggml_cont(ctx, f0_rep);
    ggml_tensor * f0_perm = ggml_permute(ctx, f0_rep, 1, 0, 2, 3);
    f0_perm = ggml_cont(ctx, f0_perm);
    ggml_tensor * f0_t1_b = ggml_reshape_3d(ctx, f0_perm, T_audio, 1, B);
    f0_t1_b = ggml_cont(ctx, f0_t1_b);
    ggml_tensor * s_t1_b     = nullptr;
    ggml_tensor * noise_t1_b = nullptr;
    ggml_tensor * uv_t1_b    = nullptr;
    if (!source_nsf.hg_source_nsf2_build_graph(ctx, f0_t1_b, &s_t1_b, &noise_t1_b, &uv_t1_b)) {
        LOG_ERROR( "hg2_hift.build_graph_forward: source_nsf failed\n");
        return false;
    }
    (void) noise_t1_b;
    (void) uv_t1_b;
    s_t1_b = ggml_cont(ctx, s_t1_b);
    ggml_tensor * s_over = hg_hift_cache_overwrite_prefix(ctx, s_t1_b, cache_source_t1_b);
    if (!s_over) {
        LOG_ERROR( "hg2_hift.build_graph_forward: cache overwrite failed\n");
        return false;
    }
    s_over = ggml_cont(ctx, s_over);
    ggml_tensor * wave_t_b = nullptr;
    if (!build_graph_decode(ctx, speech_feat_c80_t_b, s_over, &wave_t_b)) {
        LOG_ERROR( "hg2_hift.build_graph_forward: decode failed\n");
        return false;
    }
    *out_wave_t_b    = wave_t_b;
    *out_source_t1_b = s_over;
    return true;
}
bool hg2_hift_generator::build_graph_decode(ggml_context * ctx,
                                            ggml_tensor *  speech_feat_c80_t_b,
                                            ggml_tensor *  source_t1_b,
                                            ggml_tensor ** out_wave_t_b) {
    if (!ctx || !speech_feat_c80_t_b || !source_t1_b || !out_wave_t_b) {
        return false;
    }
    if (speech_feat_c80_t_b->type != GGML_TYPE_F32 || source_t1_b->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_hift.build_graph_decode: expected F32 inputs\n");
        return false;
    }
    const int64_t B = speech_feat_c80_t_b->ne[2];
    if (B != 1 || source_t1_b->ne[2] != 1) {
        LOG_ERROR( "hg2_hift.build_graph_decode: only B==1 supported for now\n");
        return false;
    }
    if (!dsp.window || !dsp.dft_cos_t || !dsp.window_sq || !dsp.istft_ola_kernel) {
        LOG_ERROR( "hg2_hift.build_graph_decode: dsp params not initialized\n");
        return false;
    }
    if (!conv_pre_weight || !conv_pre_bias || !conv_post_weight || !conv_post_bias || !up0_weight || !up0_bias ||
        !up1_weight || !up1_bias || !up2_weight || !up2_bias || !source_down0_weight || !source_down0_bias ||
        !source_down1_weight || !source_down1_bias || !source_down2_weight || !source_down2_bias) {
        LOG_ERROR( "hg2_hift.build_graph_decode: missing weights\n");
        return false;
    }
    if (resblocks.size() != 9) {
        LOG_ERROR( "hg2_hift.build_graph_decode: expected 9 resblocks, got %zu\n",
                     resblocks.size());
        return false;
    }
    const int64_t Ts            = source_t1_b->ne[0];
    ggml_tensor * s_tb          = ggml_reshape_2d(ctx, source_t1_b, Ts, B);
    ggml_tensor * stft_real_ftb = nullptr;
    ggml_tensor * stft_imag_ftb = nullptr;
    if (!hg2_stft16::hg_stft16_build_graph(ctx, dsp, s_tb, &stft_real_ftb, &stft_imag_ftb)) {
        LOG_ERROR( "hg2_hift.build_graph_decode: stft failed\n");
        return false;
    }
    ggml_tensor * stft_real_tfb = ggml_permute(ctx, stft_real_ftb, 1, 0, 2, 3);
    ggml_tensor * stft_imag_tfb = ggml_permute(ctx, stft_imag_ftb, 1, 0, 2, 3);
    stft_real_tfb               = ggml_cont(ctx, stft_real_tfb);
    stft_imag_tfb               = ggml_cont(ctx, stft_imag_tfb);
    ggml_tensor * s_stft_tcb    = ggml_concat(ctx, stft_real_tfb, stft_imag_tfb, 1);
    s_stft_tcb                  = ggml_cont(ctx, s_stft_tcb);
    ggml_tensor * x_tcb = ggml_permute(ctx, speech_feat_c80_t_b, 1, 0, 2, 3);
    x_tcb               = ggml_cont(ctx, x_tcb);
    x_tcb               = hg_hift_conv1d_f32(ctx, x_tcb, conv_pre_weight, conv_pre_bias, 1, 3, 1);
    if (!x_tcb) {
        LOG_ERROR( "hg2_hift.build_graph_decode: conv_pre failed\n");
        return false;
    }
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = ggml_leaky_relu(ctx, x_tcb, lrelu_slope, false);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = hg_hift_deconv1d_pad_f32_b1(ctx, x_tcb, up0_weight, up0_bias, 8, 4);
    if (!x_tcb) {
        return false;
    }
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * si0 =
        hg_hift_conv1d_f32(ctx, s_stft_tcb, source_down0_weight, source_down0_bias, 15, 7, 1);
    if (!si0) {
        return false;
    }
    si0 = ggml_cont(ctx, si0);
    si0 = source_rb0.hg_resblock_build_graph(ctx, si0);
    if (!si0) {
        return false;
    }
    si0   = ggml_cont(ctx, si0);
    x_tcb = ggml_add(ctx, x_tcb, si0);
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * xs0 = nullptr;
    for (int j = 0; j < 3; ++j) {
        ggml_tensor * y = resblocks[(size_t) (0 * 3 + j)].hg_resblock_build_graph(ctx, x_tcb);
        if (!y) {
            return false;
        }
        y   = ggml_cont(ctx, y);
        xs0 = (xs0 == nullptr) ? y : ggml_add(ctx, xs0, y);
        xs0 = ggml_cont(ctx, xs0);
    }
    x_tcb = ggml_scale(ctx, xs0, 1.0f / 3.0f);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = ggml_leaky_relu(ctx, x_tcb, lrelu_slope, false);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = hg_hift_deconv1d_pad_f32_b1(ctx, x_tcb, up1_weight, up1_bias, 5, 3);
    if (!x_tcb) {
        return false;
    }
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * si1 = hg_hift_conv1d_f32(ctx, s_stft_tcb, source_down1_weight, source_down1_bias, 3, 1, 1);
    if (!si1) {
        return false;
    }
    si1 = ggml_cont(ctx, si1);
    si1 = source_rb1.hg_resblock_build_graph(ctx, si1);
    if (!si1) {
        return false;
    }
    si1   = ggml_cont(ctx, si1);
    x_tcb = ggml_add(ctx, x_tcb, si1);
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * xs1 = nullptr;
    for (int j = 0; j < 3; ++j) {
        ggml_tensor * y = resblocks[(size_t) (1 * 3 + j)].hg_resblock_build_graph(ctx, x_tcb);
        if (!y) {
            return false;
        }
        y   = ggml_cont(ctx, y);
        xs1 = (xs1 == nullptr) ? y : ggml_add(ctx, xs1, y);
        xs1 = ggml_cont(ctx, xs1);
    }
    x_tcb = ggml_scale(ctx, xs1, 1.0f / 3.0f);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = ggml_leaky_relu(ctx, x_tcb, lrelu_slope, false);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = hg_hift_deconv1d_pad_f32_b1(ctx, x_tcb, up2_weight, up2_bias, 3, 2);
    if (!x_tcb) {
        return false;
    }
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb = hg2_ops::hg_ops_reflect_pad_left_1(ctx, x_tcb);
    if (!x_tcb) {
        return false;
    }
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * si2 = hg_hift_conv1d_f32(ctx, s_stft_tcb, source_down2_weight, source_down2_bias, 1, 0, 1);
    if (!si2) {
        return false;
    }
    si2 = ggml_cont(ctx, si2);
    si2 = source_rb2.hg_resblock_build_graph(ctx, si2);
    if (!si2) {
        return false;
    }
    si2   = ggml_cont(ctx, si2);
    x_tcb = ggml_add(ctx, x_tcb, si2);
    x_tcb = ggml_cont(ctx, x_tcb);
    ggml_tensor * xs2 = nullptr;
    for (int j = 0; j < 3; ++j) {
        ggml_tensor * y = resblocks[(size_t) (2 * 3 + j)].hg_resblock_build_graph(ctx, x_tcb);
        if (!y) {
            return false;
        }
        y   = ggml_cont(ctx, y);
        xs2 = (xs2 == nullptr) ? y : ggml_add(ctx, xs2, y);
        xs2 = ggml_cont(ctx, xs2);
    }
    x_tcb = ggml_scale(ctx, xs2, 1.0f / 3.0f);
    x_tcb = ggml_cont(ctx, x_tcb);
    x_tcb                  = ggml_leaky_relu(ctx, x_tcb, 0.01f, false);
    x_tcb                  = ggml_cont(ctx, x_tcb);
    ggml_tensor * post_tcb = hg_hift_conv1d_f32(ctx, x_tcb, conv_post_weight, conv_post_bias, 1, 3, 1);
    if (!post_tcb) {
        return false;
    }
    post_tcb = ggml_cont(ctx, post_tcb);
    const int64_t TT    = post_tcb->ne[0];
    const int64_t Cpost = post_tcb->ne[1];
    if (Cpost != 18) {
        LOG_ERROR( "hg2_hift.build_graph_decode: conv_post expected 18 channels, got %lld\n",
                     (long long) Cpost);
        return false;
    }
    ggml_tensor * mag_tfb = ggml_view_3d(ctx, post_tcb, TT, HG2_F, B, post_tcb->nb[1], post_tcb->nb[2], 0);
    ggml_tensor * raw_phase_tfb =
        ggml_view_3d(ctx, post_tcb, TT, HG2_F, B, post_tcb->nb[1], post_tcb->nb[2], post_tcb->nb[1] * (size_t) HG2_F);
    mag_tfb       = ggml_cont(ctx, mag_tfb);
    raw_phase_tfb = ggml_cont(ctx, raw_phase_tfb);
    ggml_tensor * magnitude = ggml_exp(ctx, mag_tfb);
    magnitude               = ggml_clamp(ctx, magnitude, -1e30f, 1e2f);
    magnitude               = ggml_cont(ctx, magnitude);
    ggml_tensor * phase = ggml_sin(ctx, raw_phase_tfb);
    phase               = ggml_cont(ctx, phase);
    ggml_tensor * cos_p         = ggml_cos(ctx, phase);
    ggml_tensor * sin_p         = ggml_sin(ctx, phase);
    ggml_tensor * ifft_real_tfb = ggml_mul(ctx, magnitude, cos_p);
    ggml_tensor * ifft_imag_tfb = ggml_mul(ctx, magnitude, sin_p);
    ifft_real_tfb               = ggml_cont(ctx, ifft_real_tfb);
    ifft_imag_tfb               = ggml_cont(ctx, ifft_imag_tfb);
    ggml_tensor * real_ftb = ggml_permute(ctx, ifft_real_tfb, 1, 0, 2, 3);
    ggml_tensor * imag_ftb = ggml_permute(ctx, ifft_imag_tfb, 1, 0, 2, 3);
    real_ftb               = ggml_cont(ctx, real_ftb);
    imag_ftb               = ggml_cont(ctx, imag_ftb);
    ggml_tensor * wave_tb = nullptr;
    if (!hg2_istft16::hg_istft16_build_graph(ctx, dsp, real_ftb, imag_ftb, &wave_tb)) {
        LOG_ERROR( "hg2_hift.build_graph_decode: istft failed\n");
        return false;
    }
    wave_tb = ggml_cont(ctx, wave_tb);
    wave_tb = ggml_clamp(ctx, wave_tb, -audio_limit, audio_limit);
    wave_tb = ggml_cont(ctx, wave_tb);
    *out_wave_t_b = wave_tb;
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 构建 istft 计算
// hifigan2 istft16 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
// 检查 repeat 维度是否可广播
static ggml_tensor * hg_istft16_repeat_checked(ggml_context * ctx,
                                                ggml_tensor *  a,
                                                ggml_tensor *  like,
                                                const char *   what) {
    if (!ggml_can_repeat(a, like)) {
        LOG_ERROR(
            "hg2_istft16_repeat_checked(%s): cannot repeat a[%lld,%lld,%lld,%lld] to like[%lld,%lld,%lld,%lld]\n", what,
            (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
            (long long) like->ne[0], (long long) like->ne[1], (long long) like->ne[2], (long long) like->ne[3]);
        return nullptr;
    }
    return ggml_repeat(ctx, a, like);
}
// 构建 istft 计算
bool hg2_istft16::hg_istft16_build_graph(ggml_context *            ctx,
                                          const hg2_stft16_params & params,
                                          ggml_tensor *             real_ftb,
                                          ggml_tensor *             imag_ftb,
                                          ggml_tensor **            out_y_tb) {
    if (!ctx || !real_ftb || !imag_ftb || !out_y_tb) {
        return false;
    }
    if (real_ftb->type != GGML_TYPE_F32 || imag_ftb->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_istft16_build_graph: expected F32 inputs\n");
        return false;
    }
    if (!params.window || !params.window_sq || !params.dft_cos_t || !params.dft_sin_t || !params.nyq_sign) {
        LOG_ERROR( "hg2_istft16_build_graph: params not initialized\n");
        return false;
    }
    if (!params.istft_ola_kernel) {
        LOG_ERROR( "hg2_istft16_build_graph: missing istft_ola_kernel const\n");
        return false;
    }
    const int64_t F  = real_ftb->ne[0];
    const int64_t TT = real_ftb->ne[1];
    const int64_t B  = real_ftb->ne[2];
    if (imag_ftb->ne[0] != F || imag_ftb->ne[1] != TT || imag_ftb->ne[2] != B) {
        LOG_ERROR( "hg2_istft16_build_graph: real/imag shape mismatch\n");
        return false;
    }
    if (F != hg2_stft16_params::HG2_F) {
        LOG_ERROR( "hg2_istft16_build_graph: expected F=%d, got %lld\n", hg2_stft16_params::HG2_F,
                     (long long) F);
        return false;
    }
    if (B != 1) {
        LOG_ERROR( "hg2_istft16_build_graph: only B==1 is supported for now, got B=%lld\n", (long long) B);
        return false;
    }
    ggml_tensor * xr_ft = real_ftb;
    ggml_tensor * xi_ft = imag_ftb;
    // 转成 TF 布局，方便按频段切片
    ggml_tensor * xr_tf = ggml_transpose(ctx, xr_ft);
    ggml_tensor * xi_tf = ggml_transpose(ctx, xi_ft);
    xr_tf               = ggml_cont(ctx, xr_tf);
    xi_tf               = ggml_cont(ctx, xi_tf);
    // 拆出 DC、Nyquist 和中间频段
    ggml_tensor * x0_t1 = ggml_view_2d(ctx, xr_tf, TT, 1, xr_tf->nb[1], 0);
    ggml_tensor * xnyq_t1 =
        ggml_view_2d(ctx, xr_tf, TT, 1, xr_tf->nb[1], xr_tf->nb[1] * (size_t) (hg2_stft16_params::HG2_F - 1));
    ggml_tensor * xr_mid_t7 =
        ggml_view_2d(ctx, xr_tf, TT, hg2_stft16_params::HG2_F - 2, xr_tf->nb[1], xr_tf->nb[1] * (size_t) 1);
    ggml_tensor * xi_mid_t7 =
        ggml_view_2d(ctx, xi_tf, TT, hg2_stft16_params::HG2_F - 2, xi_tf->nb[1], xi_tf->nb[1] * (size_t) 1);
    xr_mid_t7 = ggml_cont(ctx, xr_mid_t7);
    xi_mid_t7 = ggml_cont(ctx, xi_mid_t7);
    // 取 DFT 表的中频部分
    ggml_tensor * cos_mid_n7 =
        ggml_view_2d(ctx, params.dft_cos_t, hg2_stft16_params::HG2_N_FFT, hg2_stft16_params::HG2_F - 2,
                     params.dft_cos_t->nb[1], params.dft_cos_t->nb[1] * (size_t) 1);
    ggml_tensor * sin_mid_n7 =
        ggml_view_2d(ctx, params.dft_sin_t, hg2_stft16_params::HG2_N_FFT, hg2_stft16_params::HG2_F - 2,
                     params.dft_sin_t->nb[1], params.dft_sin_t->nb[1] * (size_t) 1);
    ggml_tensor * cos_mid_7n = ggml_transpose(ctx, cos_mid_n7);
    ggml_tensor * sin_mid_7n = ggml_transpose(ctx, sin_mid_n7);
    cos_mid_7n               = ggml_cont(ctx, cos_mid_7n);
    sin_mid_7n               = ggml_cont(ctx, sin_mid_7n);
    ggml_tensor * xr_mid_7t = ggml_transpose(ctx, xr_mid_t7);
    ggml_tensor * xi_mid_7t = ggml_transpose(ctx, xi_mid_t7);
    xr_mid_7t               = ggml_cont(ctx, xr_mid_7t);
    xi_mid_7t               = ggml_cont(ctx, xi_mid_7t);
    // 中频按 cos/sin 重建到时域并求和
    ggml_tensor * mm_cos_Tn = ggml_mul_mat(ctx, xr_mid_7t, cos_mid_7n);
    ggml_tensor * mm_sin_Tn = ggml_mul_mat(ctx, xi_mid_7t, sin_mid_7n);
    ggml_tensor * sum_k_Tn  = ggml_sub(ctx, mm_cos_Tn, mm_sin_Tn);
    sum_k_Tn                = ggml_cont(ctx, sum_k_Tn);
    ggml_tensor * x0_Tn   = hg_istft16_repeat_checked(ctx, x0_t1, sum_k_Tn, "x0");
    ggml_tensor * xnyq_Tn = hg_istft16_repeat_checked(ctx, xnyq_t1, sum_k_Tn, "xnyq");
    if (!x0_Tn || !xnyq_Tn) {
        return false;
    }
    // 带符号翻转
    ggml_tensor * nyq_n1 = ggml_reshape_2d(ctx, params.nyq_sign, hg2_stft16_params::HG2_N_FFT, 1);
    ggml_tensor * nyq_1n = ggml_transpose(ctx, nyq_n1);
    nyq_1n               = ggml_cont(ctx, nyq_1n);
    ggml_tensor * nyq_Tn = hg_istft16_repeat_checked(ctx, nyq_1n, sum_k_Tn, "nyq_sign");
    if (!nyq_Tn) {
        return false;
    }
    ggml_tensor * win_n1 = ggml_reshape_2d(ctx, params.window, hg2_stft16_params::HG2_N_FFT, 1);
    ggml_tensor * win_1n = ggml_transpose(ctx, win_n1);
    win_1n               = ggml_cont(ctx, win_1n);
    ggml_tensor * win_Tn = hg_istft16_repeat_checked(ctx, win_1n, sum_k_Tn, "window");
    if (!win_Tn) {
        return false;
    }
    // 合成时域帧：DC + Nyq + 2 * 中频
    ggml_tensor * term_nyq = ggml_mul(ctx, xnyq_Tn, nyq_Tn);
    ggml_tensor * term_k   = ggml_scale(ctx, sum_k_Tn, 2.0f);
    ggml_tensor * time_Tn  = ggml_add(ctx, x0_Tn, term_nyq);
    time_Tn                = ggml_add(ctx, time_Tn, term_k);
    time_Tn                = ggml_scale(ctx, time_Tn, 1.0f / float(hg2_stft16_params::HG2_N_FFT));
    // 乘窗后做 overlap-add
    time_Tn = ggml_mul(ctx, time_Tn, win_Tn);
    time_Tn = ggml_cont(ctx, time_Tn);
    ggml_tensor * y_t1b1 =
        ggml_conv_transpose_1d(ctx, params.istft_ola_kernel, time_Tn, hg2_stft16_params::HG2_HOP, 0, 1);
    y_t1b1 = ggml_cont(ctx, y_t1b1);
    // 用窗平方的 overlap-add 做归一化
    ggml_tensor * wsq_n1   = ggml_reshape_2d(ctx, params.window_sq, hg2_stft16_params::HG2_N_FFT, 1);
    ggml_tensor * dummy_nT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hg2_stft16_params::HG2_N_FFT, TT);
    ggml_tensor * wsq_nT   = hg_istft16_repeat_checked(ctx, wsq_n1, dummy_nT, "window_sq_repeat");
    if (!wsq_nT) {
        return false;
    }
    ggml_tensor * wsq_Tn = ggml_transpose(ctx, wsq_nT);
    wsq_Tn               = ggml_cont(ctx, wsq_Tn);
    ggml_tensor * wsum_t1b1 =
        ggml_conv_transpose_1d(ctx, params.istft_ola_kernel, wsq_Tn, hg2_stft16_params::HG2_HOP, 0, 1);
    wsum_t1b1 = ggml_cont(ctx, wsum_t1b1);
    ggml_tensor * wsum_clamped = ggml_clamp(ctx, wsum_t1b1, 1e-8f, 1e30f);
    ggml_tensor * y_norm       = ggml_div(ctx, y_t1b1, wsum_clamped);
    y_norm                     = ggml_cont(ctx, y_norm);
    // 去掉 stft pad 对应的首尾样本
    const int64_t out_len = y_norm->ne[0];
    const int64_t trim    = 2 * (int64_t) hg2_stft16_params::HG2_PAD;
    if (out_len <= trim) {
        LOG_ERROR( "hg2_istft16_build_graph: output too short to trim\n");
        return false;
    }
    const int64_t T_trim = out_len - trim;
    ggml_tensor * y_outlen_1 = ggml_reshape_2d(ctx, y_norm, out_len, 1);
    ggml_tensor * y_trim     = ggml_view_2d(ctx, y_outlen_1, T_trim, 1, y_outlen_1->nb[1],
                                            (size_t) hg2_stft16_params::HG2_PAD * y_outlen_1->nb[0]);
    y_trim                   = ggml_cont(ctx, y_trim);
    ggml_set_name(y_trim, "hg2.istft16.y_tb");
    *out_y_tb = y_trim;
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 model_loader 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static ggml_tensor * hg_model_req_tensor(const hg2_gguf_model_loader & loader, const std::string & name) {
    ggml_tensor * t = loader.hg_gguf_model_loader_get_tensor(name);
    if (!t) {
        LOG_ERROR( "hg2_model_bind_from_loader: missing tensor: %s\n", name.c_str());
    }
    return t;
}
static void hg_model_init_resblock_struct(hg2_resblock & rb,
                                           int64_t        channels,
                                           int32_t        kernel_size,
                                           const int32_t  dils_135[3]) {
    rb.convs1.resize(3);
    rb.convs2.resize(3);
    rb.activations1_alpha.resize(3);
    rb.activations2_alpha.resize(3);
    for (int i = 0; i < 3; ++i) {
        rb.convs1[(size_t) i].kernel_size   = kernel_size;
        rb.convs1[(size_t) i].dilation      = dils_135[i];
        rb.convs1[(size_t) i].padding       = hg2_ops::hg_ops_get_padding(kernel_size, dils_135[i]);
        rb.convs1[(size_t) i].weight_kic_oc = nullptr;
        rb.convs1[(size_t) i].bias_oc       = nullptr;
        rb.convs2[(size_t) i].kernel_size   = kernel_size;
        rb.convs2[(size_t) i].dilation      = 1;
        rb.convs2[(size_t) i].padding       = hg2_ops::hg_ops_get_padding(kernel_size, 1);
        rb.convs2[(size_t) i].weight_kic_oc = nullptr;
        rb.convs2[(size_t) i].bias_oc       = nullptr;
        rb.activations1_alpha[(size_t) i] = nullptr;
        rb.activations2_alpha[(size_t) i] = nullptr;
    }
    (void) channels;
}
// 绑定权重到模型结构
static bool hg_bind_resblock(hg2_resblock &                rb,
                                                const hg2_gguf_model_loader & loader,
                                                const std::string &           prefix) {
    for (int i = 0; i < 3; ++i) {
        rb.convs1[(size_t) i].weight_kic_oc =
            hg_model_req_tensor(loader, prefix + ".convs1." + std::to_string(i) + ".weight");
        rb.convs1[(size_t) i].bias_oc = hg_model_req_tensor(loader, prefix + ".convs1." + std::to_string(i) + ".bias");
        rb.convs2[(size_t) i].weight_kic_oc =
            hg_model_req_tensor(loader, prefix + ".convs2." + std::to_string(i) + ".weight");
        rb.convs2[(size_t) i].bias_oc = hg_model_req_tensor(loader, prefix + ".convs2." + std::to_string(i) + ".bias");
        rb.activations1_alpha[(size_t) i] =
            hg_model_req_tensor(loader, prefix + ".activations1." + std::to_string(i) + ".alpha");
        rb.activations2_alpha[(size_t) i] =
            hg_model_req_tensor(loader, prefix + ".activations2." + std::to_string(i) + ".alpha");
    }
    for (int i = 0; i < 3; ++i) {
        if (!rb.convs1[(size_t) i].weight_kic_oc || !rb.convs1[(size_t) i].bias_oc ||
            !rb.convs2[(size_t) i].weight_kic_oc || !rb.convs2[(size_t) i].bias_oc ||
            !rb.activations1_alpha[(size_t) i] || !rb.activations2_alpha[(size_t) i]) {
            return false;
        }
    }
    return true;
}
// 绑定权重到模型结构
bool hg2_model::hg_model_bind_from_loader(const hg2_gguf_model_loader & loader) {
    gen.f0_pred.conv0_weight  = hg_model_req_tensor(loader, "f0_predictor.condnet.0.weight");
    gen.f0_pred.conv0_bias    = hg_model_req_tensor(loader, "f0_predictor.condnet.0.bias");
    gen.f0_pred.conv1_weight  = hg_model_req_tensor(loader, "f0_predictor.condnet.2.weight");
    gen.f0_pred.conv1_bias    = hg_model_req_tensor(loader, "f0_predictor.condnet.2.bias");
    gen.f0_pred.conv2_weight  = hg_model_req_tensor(loader, "f0_predictor.condnet.4.weight");
    gen.f0_pred.conv2_bias    = hg_model_req_tensor(loader, "f0_predictor.condnet.4.bias");
    gen.f0_pred.conv3_weight  = hg_model_req_tensor(loader, "f0_predictor.condnet.6.weight");
    gen.f0_pred.conv3_bias    = hg_model_req_tensor(loader, "f0_predictor.condnet.6.bias");
    gen.f0_pred.conv4_weight  = hg_model_req_tensor(loader, "f0_predictor.condnet.8.weight");
    gen.f0_pred.conv4_bias    = hg_model_req_tensor(loader, "f0_predictor.condnet.8.bias");
    gen.f0_pred.linear_weight = hg_model_req_tensor(loader, "f0_predictor.classifier.weight");
    gen.f0_pred.linear_bias   = hg_model_req_tensor(loader, "f0_predictor.classifier.bias");
    gen.source_nsf.linear_weight = hg_model_req_tensor(loader, "m_source.l_linear.weight");
    gen.source_nsf.linear_bias   = hg_model_req_tensor(loader, "m_source.l_linear.bias");
    gen.conv_pre_weight  = hg_model_req_tensor(loader, "conv_pre.weight");
    gen.conv_pre_bias    = hg_model_req_tensor(loader, "conv_pre.bias");
    gen.conv_post_weight = hg_model_req_tensor(loader, "conv_post.weight");
    gen.conv_post_bias   = hg_model_req_tensor(loader, "conv_post.bias");
    gen.up0_weight = hg_model_req_tensor(loader, "ups.0.weight");
    gen.up0_bias   = hg_model_req_tensor(loader, "ups.0.bias");
    gen.up1_weight = hg_model_req_tensor(loader, "ups.1.weight");
    gen.up1_bias   = hg_model_req_tensor(loader, "ups.1.bias");
    gen.up2_weight = hg_model_req_tensor(loader, "ups.2.weight");
    gen.up2_bias   = hg_model_req_tensor(loader, "ups.2.bias");
    gen.source_down0_weight = hg_model_req_tensor(loader, "source_downs.0.weight");
    gen.source_down0_bias   = hg_model_req_tensor(loader, "source_downs.0.bias");
    gen.source_down1_weight = hg_model_req_tensor(loader, "source_downs.1.weight");
    gen.source_down1_bias   = hg_model_req_tensor(loader, "source_downs.1.bias");
    gen.source_down2_weight = hg_model_req_tensor(loader, "source_downs.2.weight");
    gen.source_down2_bias   = hg_model_req_tensor(loader, "source_downs.2.bias");
    const int32_t dils_135[3] = { 1, 3, 5 };
    hg_model_init_resblock_struct(gen.source_rb0, 256, 7, dils_135);
    hg_model_init_resblock_struct(gen.source_rb1, 128, 7, dils_135);
    hg_model_init_resblock_struct(gen.source_rb2, 64, 11, dils_135);
    if (!hg_bind_resblock(gen.source_rb0, loader, "source_resblocks.0") ||
        !hg_bind_resblock(gen.source_rb1, loader, "source_resblocks.1") ||
        !hg_bind_resblock(gen.source_rb2, loader, "source_resblocks.2")) {
        LOG_ERROR( "hg2_model_bind_from_loader: missing source_resblocks tensors\n");
        return false;
    }
    gen.resblocks.resize(9);
    const int32_t klist[3] = { 3, 7, 11 };
    for (int stage = 0; stage < 3; ++stage) {
        const int64_t ch = (stage == 0) ? 256 : (stage == 1) ? 128 : 64;
        (void) ch;
        for (int j = 0; j < 3; ++j) {
            const int idx = stage * 3 + j;
            hg_model_init_resblock_struct(gen.resblocks[(size_t) idx], ch, klist[j], dils_135);
            if (!hg_bind_resblock(gen.resblocks[(size_t) idx], loader,
                                                     "resblocks." + std::to_string(idx))) {
                LOG_ERROR( "hg2_model_bind_from_loader: missing resblocks tensors at %d\n", idx);
                return false;
            }
        }
    }
    if (!gen.f0_pred.conv0_weight || !gen.source_nsf.linear_weight || !gen.conv_pre_weight || !gen.up0_weight ||
        !gen.conv_post_weight) {
        LOG_ERROR( "hg2_model_bind_from_loader: missing required tensors\n");
        return false;
    }
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 ops 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
// 计算卷积 padding
int32_t hg2_ops::hg_ops_get_padding(int32_t kernel_size, int32_t dilation) {
    return int32_t((kernel_size * dilation - dilation) / 2);
}
void hg2_ops::
    // 构建反射 padding
    hg_ops_init_weights_for_debug() {}
ggml_tensor * hg2_ops::hg_ops_reflect_pad_left_1(ggml_context * ctx, ggml_tensor * x_tcb) {
    if (!ctx || !x_tcb) {
        return nullptr;
    }
    if (x_tcb->ne[0] < 2) {
        LOG_ERROR( "hg2_ops_reflect_pad_left_1: requires T>=2, got T=%lld\n", (long long) x_tcb->ne[0]);
        return nullptr;
    }
    return ggml_pad_reflect_1d(ctx, x_tcb, 1, 0);
}
int32_t hg2_ops::hg_ops_conv_transpose1d_out_len(int32_t in_len,
                                                  int32_t stride,
                                                  int32_t padding,
                                                  int32_t kernel_size,
                                                  int32_t dilation,
                                                  int32_t out_pad) {
    return (in_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_pad + 1;
}
ggml_tensor * hg2_ops::hg_ops_sample_phase_zeros(ggml_context * ctx,
                                                  int64_t        ne0,
                                                  int64_t        ne1,
                                                  int64_t        ne2,
                                                  int64_t        ne3) {
    if (!ctx) {
        return nullptr;
    }
    ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_name(z, "hg2.phase_zeros");
    if (z->data) {
        std::memset(z->data, 0, ggml_nbytes(z));
    }
    return z;
}
ggml_tensor * hg2_ops::hg_ops_sample_noise_zeros_like(ggml_context * ctx, ggml_tensor * ref) {
    if (!ctx || !ref) {
        return nullptr;
    }
    const int     nd = ggml_n_dims(ref);
    ggml_tensor * z  = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ref->ne);
    ggml_set_name(z, "hg2.noise_zeros_like");
    if (z->data) {
        std::memset(z->data, 0, ggml_nbytes(z));
    }
    return z;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 主要负责这一块的前向构图
// hifigan2 resblock 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static ggml_tensor * hg_resblock_conv1d_f32(ggml_context *                          ctx,
                                             ggml_tensor *                           x_tcb,
                                             const hg2_resblock::hg2_resblock_conv & c) {
    if (!ctx || !x_tcb || !c.weight_kic_oc || !c.bias_oc) {
        return nullptr;
    }
    if (x_tcb->type != GGML_TYPE_F32 || c.weight_kic_oc->type != GGML_TYPE_F32 || c.bias_oc->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_resblock_conv1d_f32: expected F32 tensors\n");
        return nullptr;
    }
    const int64_t T   = x_tcb->ne[0];
    const int64_t Cin = x_tcb->ne[1];
    const int64_t B   = x_tcb->ne[2];
    const int64_t K    = c.weight_kic_oc->ne[0];
    const int64_t Cinw = c.weight_kic_oc->ne[1];
    const int64_t Cout = c.weight_kic_oc->ne[2];
    if (Cinw != Cin) {
        LOG_ERROR( "hg2_resblock_conv1d_f32: Cin mismatch, x Cin=%lld w Cin=%lld\n", (long long) Cin,
                     (long long) Cinw);
        return nullptr;
    }
    if (c.bias_oc->ne[0] != Cout) {
        LOG_ERROR( "hg2_resblock_conv1d_f32: bias mismatch, expected %lld got %lld\n", (long long) Cout,
                     (long long) c.bias_oc->ne[0]);
        return nullptr;
    }
    if (c.kernel_size != (int32_t) K) {
        LOG_ERROR( "hg2_resblock_conv1d_f32: kernel_size mismatch, c.kernel_size=%d, w.K=%lld\n",
                     c.kernel_size, (long long) K);
        return nullptr;
    }
    ggml_tensor * im2col =
        ggml_im2col(ctx, c.weight_kic_oc, x_tcb, 1, 0, c.padding, 0, c.dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    im2col_2d               = ggml_cont(ctx, im2col_2d);
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, c.weight_kic_oc, K * Cin, Cout);
    w_2d               = ggml_cont(ctx, w_2d);
    ggml_tensor * mm    = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_tensor * y_tcb = ggml_reshape_3d(ctx, mm, T, Cout, B);
    y_tcb               = ggml_cont(ctx, y_tcb);
    ggml_tensor * b_1c1 = ggml_reshape_3d(ctx, c.bias_oc, 1, Cout, 1);
    b_1c1               = ggml_cont(ctx, b_1c1);
    ggml_tensor * b_tcb = ggml_repeat(ctx, b_1c1, y_tcb);
    ggml_tensor * y     = ggml_add(ctx, y_tcb, b_tcb);
    return y;
}
// 构建残差块计算
ggml_tensor * hg2_resblock::hg_resblock_build_graph(ggml_context * ctx, ggml_tensor * x_tcb) const {
    if (!ctx || !x_tcb) {
        return nullptr;
    }
    if (x_tcb->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_resblock_build_graph: expected F32 input\n");
        return nullptr;
    }
    if (convs1.size() != convs2.size()) {
        LOG_ERROR( "hg2_resblock_build_graph: convs1/convs2 size mismatch\n");
        return nullptr;
    }
    if (activations1_alpha.size() != convs1.size() || activations2_alpha.size() != convs2.size()) {
        LOG_ERROR( "hg2_resblock_build_graph: activations alpha size mismatch\n");
        return nullptr;
    }
    ggml_tensor * x = x_tcb;
    for (size_t i = 0; i < convs1.size(); ++i) {
        ggml_tensor * a1 = activations1_alpha[i];
        ggml_tensor * a2 = activations2_alpha[i];
        if (!a1 || !a2) {
            LOG_ERROR( "hg2_resblock_build_graph: missing Snake alpha at idx=%zu\n", i);
            return nullptr;
        }
        ggml_tensor * xt = hg2_snake::hg_snake_build_graph(ctx, x, a1);
        if (!xt) {
            return nullptr;
        }
        xt = hg_resblock_conv1d_f32(ctx, xt, convs1[i]);
        if (!xt) {
            return nullptr;
        }
        xt = ggml_cont(ctx, xt);
        xt = hg2_snake::hg_snake_build_graph(ctx, xt, a2);
        if (!xt) {
            return nullptr;
        }
        xt = hg_resblock_conv1d_f32(ctx, xt, convs2[i]);
        if (!xt) {
            return nullptr;
        }
        xt = ggml_cont(ctx, xt);
        x = ggml_add(ctx, x, xt);
        x = ggml_cont(ctx, x);
    }
    ggml_set_name(x, "hg2.resblock.y_tcb");
    return x;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 sine_gen2 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
// 初始化正弦源参数
bool hg2_sine_gen2::hg_sine_gen2_init(ggml_context * ctx, int32_t harmonic_num_in) {
    if (!ctx) {
        return false;
    }
    harmonic_num = harmonic_num_in;
    dim          = harmonic_num + 1;
    harmonic_mul = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    ggml_set_name(harmonic_mul, "hg2.const.harmonic_mul");
    host_harmonic_mul.resize((size_t) dim);
    for (int32_t i = 0; i < dim; ++i) {
        host_harmonic_mul[(size_t) i] = float(i + 1);
    }
    if (harmonic_mul->data) {
        std::memcpy(harmonic_mul->data, host_harmonic_mul.data(), host_harmonic_mul.size() * sizeof(float));
    }
    return true;
}
// 上传常量到后端
bool hg2_sine_gen2::hg_sine_gen2_upload_consts(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    if (!harmonic_mul || host_harmonic_mul.empty()) {
        return false;
    }
    ggml_backend_tensor_set(harmonic_mul, host_harmonic_mul.data(), 0, host_harmonic_mul.size() * sizeof(float));
    return true;
}
bool hg2_sine_gen2::hg_sine_gen2_build_graph(ggml_context * ctx,
                                              ggml_tensor *  f0_t1_b,
                                              ggml_tensor ** out_sine_tdb,
                                              ggml_tensor ** out_uv_t1_b,
                                              ggml_tensor ** out_noise_tdb) {
    if (!ctx || !f0_t1_b || !out_sine_tdb || !out_uv_t1_b || !out_noise_tdb) {
        return false;
    }
    if (flag_for_pulse) {
        LOG_ERROR( "hg2_sine_gen2_build_graph: flag_for_pulse=true is not supported\n");
        return false;
    }
    if (!harmonic_mul) {
        LOG_ERROR( "hg2_sine_gen2_build_graph: not initialized\n");
        return false;
    }
    if (f0_t1_b->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_sine_gen2_build_graph: expected f0 F32\n");
        return false;
    }
    const int64_t T = f0_t1_b->ne[0];
    const int64_t B = f0_t1_b->ne[2];
    if (f0_t1_b->ne[1] != 1) {
        LOG_ERROR( "hg2_sine_gen2_build_graph: expected f0 ne1==1\n");
        return false;
    }
    if ((T % HG2_UPSAMPLE_SCALE) != 0) {
        LOG_ERROR( "hg2_sine_gen2_build_graph: expected T divisible by %d, got T=%lld\n", HG2_UPSAMPLE_SCALE,
                     (long long) T);
        return false;
    }
    const int64_t Tm = T / HG2_UPSAMPLE_SCALE;
    ggml_tensor * f0_shift = ggml_scale_bias(ctx, f0_t1_b, 1.0f, -voiced_threshold);
    ggml_tensor * uv_t1_b  = ggml_step(ctx, f0_shift);
    uv_t1_b                = ggml_cont(ctx, uv_t1_b);
    ggml_tensor * dummy_tdb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, dim, B);
    ggml_tensor * f0_tdb    = ggml_repeat(ctx, f0_t1_b, dummy_tdb);
    ggml_tensor * hm_1d1 = ggml_reshape_3d(ctx, harmonic_mul, 1, dim, 1);
    hm_1d1               = ggml_cont(ctx, hm_1d1);
    ggml_tensor * hm_tdb = ggml_repeat(ctx, hm_1d1, dummy_tdb);
    ggml_tensor * fn_tdb = ggml_mul(ctx, f0_tdb, hm_tdb);
    ggml_tensor * rad_tdb = ggml_scale(ctx, fn_tdb, 1.0f / float(HG2_SAMPLING_RATE));
    rad_tdb               = ggml_cont(ctx, rad_tdb);
    // 🔧 用 reshape + sum + scale 替代 ggml_interpolate 下采样，支持 Metal 加速
    // 下采样 [T, dim, B] -> [Tm, dim, B]，对每 scale 个值取平均
    const int64_t scale_ds = HG2_UPSAMPLE_SCALE;
    // reshape [T, dim, B] -> [scale, Tm, dim, B]
    ggml_tensor * rad_4d = ggml_reshape_4d(ctx, rad_tdb, scale_ds, Tm, dim, B);
    rad_4d = ggml_cont(ctx, rad_4d);
    // 在 dim=0 (scale 维度) 上求和，得到 [1, Tm, dim, B]
    ggml_tensor * rad_sum_4d = ggml_sum_rows(ctx, rad_4d);
    rad_sum_4d = ggml_cont(ctx, rad_sum_4d);
    // 除以 scale 得到平均值
    ggml_tensor * rad_dn_4d = ggml_scale(ctx, rad_sum_4d, 1.0f / float(scale_ds));
    rad_dn_4d = ggml_cont(ctx, rad_dn_4d);
    ggml_tensor * rad_dn_t1db = ggml_reshape_4d(ctx, rad_dn_4d, Tm, 1, dim, B);
    rad_dn_t1db               = ggml_cont(ctx, rad_dn_t1db);
    ggml_tensor * rad_dn_tdb = ggml_reshape_3d(ctx, rad_dn_t1db, Tm, dim, B);
    rad_dn_tdb               = ggml_cont(ctx, rad_dn_tdb);
    dbg_rad_dn_tdb           = rad_dn_tdb;
    ggml_tensor * running   = ggml_view_3d(ctx, rad_dn_tdb, 1, dim, B, rad_dn_tdb->nb[1], rad_dn_tdb->nb[2], 0);
    running                 = ggml_cont(ctx, running);
    ggml_tensor * phase_tdb = running;
    for (int64_t t = 1; t < Tm; ++t) {
        ggml_tensor * rt = ggml_view_3d(ctx, rad_dn_tdb, 1, dim, B, rad_dn_tdb->nb[1], rad_dn_tdb->nb[2],
                                        (size_t) t * rad_dn_tdb->nb[0]);
        rt               = ggml_cont(ctx, rt);
        running          = ggml_add(ctx, running, rt);
        running          = ggml_cont(ctx, running);
        phase_tdb        = ggml_concat(ctx, phase_tdb, running, 0);
        phase_tdb        = ggml_cont(ctx, phase_tdb);
    }
    phase_tdb = ggml_scale(ctx, phase_tdb, 2.0f * float(M_PI));
    phase_tdb = ggml_cont(ctx, phase_tdb);
    ggml_tensor * phase_t1db    = ggml_reshape_4d(ctx, phase_tdb, Tm, 1, dim, B);
    // 🔧 用线性插值替代 ggml_interpolate BILINEAR，支持 Metal 加速
    // 关键：相位需要平滑过渡，不能用 repeat（会产生电音）
    // 方法：对每个 frame 的相位差进行线性展开
    // phase[i] 到 phase[i+1] 之间的 scale 个点应该是线性递增的
    const int64_t scale_up = HG2_UPSAMPLE_SCALE;
    // 1. 计算相邻 frame 的相位差 delta_phase[i] = phase[i+1] - phase[i]
    //    对最后一个 frame，使用前一个差值
    ggml_tensor * phase_shift = ggml_view_4d(ctx, phase_t1db, Tm-1, 1, dim, B,
                                             phase_t1db->nb[0], phase_t1db->nb[1], phase_t1db->nb[2],
                                             phase_t1db->nb[0]);  // phase[1:Tm]
    phase_shift = ggml_cont(ctx, phase_shift);
    ggml_tensor * phase_base = ggml_view_4d(ctx, phase_t1db, Tm-1, 1, dim, B,
                                            phase_t1db->nb[0], phase_t1db->nb[1], phase_t1db->nb[2], 0);  // phase[0:Tm-1]
    phase_base = ggml_cont(ctx, phase_base);
    ggml_tensor * delta = ggml_sub(ctx, phase_shift, phase_base);  // [Tm-1, 1, dim, B]
    delta = ggml_cont(ctx, delta);
    // 获取最后一个差值并拼接，得到完整的 delta [Tm, 1, dim, B]
    ggml_tensor * last_delta = ggml_view_4d(ctx, delta, 1, 1, dim, B,
                                            delta->nb[0], delta->nb[1], delta->nb[2],
                                            (Tm-2) * delta->nb[0]);  // delta[-1]
    last_delta = ggml_cont(ctx, last_delta);
    ggml_tensor * delta_full = ggml_concat(ctx, delta, last_delta, 0);  // [Tm, 1, dim, B]
    delta_full = ggml_cont(ctx, delta_full);
    // 2. 创建 [0, 1, 2, ..., scale-1] / scale 的斜坡，用于线性插值
    ggml_tensor * ramp = ggml_arange(ctx, 0.0f, (float)scale_up, 1.0f);  // [scale]
    ramp = ggml_scale(ctx, ramp, 1.0f / (float)scale_up);  // [0, 1/scale, 2/scale, ...]
    ramp = ggml_cont(ctx, ramp);
    // 3. phase_up[t*scale + k] = phase[t] + delta[t] * (k / scale)
    //    repeat phase_t1db [Tm,1,dim,B] -> [Tm,scale,dim,B] (element repeat in ne[1])
    //    repeat delta_full [Tm,1,dim,B] -> [Tm,scale,dim,B]
    //    ramp [1,scale,1,1] -> [Tm,scale,dim,B]
    ggml_tensor * phase_tmpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Tm, scale_up, dim, B);
    ggml_tensor * phase_rep  = ggml_repeat(ctx, phase_t1db, phase_tmpl);
    phase_rep = ggml_cont(ctx, phase_rep);
    ggml_tensor * delta_rep = ggml_repeat(ctx, delta_full, phase_tmpl);
    delta_rep = ggml_cont(ctx, delta_rep);
    ggml_tensor * ramp_4d = ggml_reshape_4d(ctx, ramp, 1, scale_up, 1, 1);
    ramp_4d = ggml_cont(ctx, ramp_4d);
    ggml_tensor * ramp_rep = ggml_repeat(ctx, ramp_4d, phase_tmpl);
    ramp_rep = ggml_cont(ctx, ramp_rep);
    // 线性插值：phase_up = phase + delta * ramp  (all [Tm, scale, dim, B])
    ggml_tensor * interp_add = ggml_mul(ctx, delta_rep, ramp_rep);
    interp_add = ggml_cont(ctx, interp_add);
    ggml_tensor * phase_interp = ggml_add(ctx, phase_rep, interp_add);
    phase_interp = ggml_cont(ctx, phase_interp);
    // 4. permute(1,0,2,3) -> [scale,Tm,dim,B], then reshape -> [T,dim,B]
    ggml_tensor * phase_perm = ggml_permute(ctx, phase_interp, 1, 0, 2, 3);
    phase_perm = ggml_cont(ctx, phase_perm);
    // 5. 最后乘以 scale（因为原来 phase_scaled = phase * scale）
    ggml_tensor * phase_scaled_perm = ggml_scale(ctx, phase_perm, float(HG2_UPSAMPLE_SCALE));
    phase_scaled_perm = ggml_cont(ctx, phase_scaled_perm);
    ggml_tensor * phase_up_t1db = ggml_reshape_4d(ctx, phase_scaled_perm, T, 1, dim, B);
    phase_up_t1db               = ggml_cont(ctx, phase_up_t1db);
    ggml_tensor * phase_up_tdb  = ggml_reshape_3d(ctx, phase_up_t1db, T, dim, B);
    phase_up_tdb                = ggml_cont(ctx, phase_up_tdb);
    dbg_phase_up_tdb            = phase_up_tdb;
    ggml_tensor * sines_tdb = ggml_sin(ctx, phase_up_tdb);
    ggml_tensor * sine_tdb  = ggml_scale(ctx, sines_tdb, sine_amp);
    ggml_tensor * uv_1db     = ggml_reshape_3d(ctx, uv_t1_b, T, 1, B);
    uv_1db                   = ggml_cont(ctx, uv_1db);
    ggml_tensor * uv_tdb     = ggml_repeat(ctx, uv_1db, sine_tdb);
    ggml_tensor * sine_gated = ggml_mul(ctx, sine_tdb, uv_tdb);
    sine_gated               = ggml_cont(ctx, sine_gated);
    ggml_tensor * noise_tdb = ggml_scale(ctx, sine_gated, 0.0f);
    noise_tdb               = ggml_cont(ctx, noise_tdb);
    *out_sine_tdb  = sine_gated;
    *out_uv_t1_b   = uv_t1_b;
    *out_noise_tdb = noise_tdb;
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 主要负责这一块的前向构图
// hifigan2 snake 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
// 构建 snake 激活计算
ggml_tensor * hg2_snake::hg_snake_build_graph(ggml_context * ctx, ggml_tensor * x_tcb, ggml_tensor * alpha_c) {
    if (!ctx || !x_tcb || !alpha_c) {
        return nullptr;
    }
    if (x_tcb->type != GGML_TYPE_F32 || alpha_c->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_snake_build_graph: expected F32 tensors\n");
        return nullptr;
    }
    const int64_t C = x_tcb->ne[1];
    if (alpha_c->ne[0] != C) {
        LOG_ERROR( "hg2_snake_build_graph: alpha shape mismatch (expected C=%lld, got %lld)\n", (long long) C,
                     (long long) alpha_c->ne[0]);
        return nullptr;
    }
    ggml_tensor * alpha_1cb = ggml_reshape_3d(ctx, alpha_c, 1, C, 1);
    ggml_tensor * alpha_tcb = ggml_repeat(ctx, alpha_1cb, x_tcb);
    ggml_tensor * xa = ggml_mul(ctx, x_tcb, alpha_tcb);
    ggml_tensor * s = ggml_sin(ctx, xa);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * denom = ggml_scale_bias(ctx, alpha_tcb, 1.0f, 1e-9f);
    ggml_tensor * term = ggml_div(ctx, s2, denom);
    ggml_tensor * y = ggml_add(ctx, x_tcb, term);
    ggml_set_name(y, "hg2.snake.y_tcb");
    return y;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 构建 NSF 源模块计算
// 主要负责这一块的前向构图
// hifigan2 source_module_nsf2 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
bool hg2_source_nsf2::hg_source_nsf2_build_graph(ggml_context * ctx,
                                                  ggml_tensor *  f0_t1_b,
                                                  ggml_tensor ** out_sine_merge_t1_b,
                                                  ggml_tensor ** out_noise_t1_b,
                                                  ggml_tensor ** out_uv_t1_b) {
    if (!ctx || !f0_t1_b || !out_sine_merge_t1_b || !out_noise_t1_b || !out_uv_t1_b) {
        return false;
    }
    if (!linear_weight || !linear_bias) {
        LOG_ERROR( "hg2_source_nsf2_build_graph: missing linear weights\n");
        return false;
    }
    ggml_tensor * sine_tdb  = nullptr;
    ggml_tensor * uv_t1_b   = nullptr;
    ggml_tensor * noise_tdb = nullptr;
    if (!sine_gen.hg_sine_gen2_build_graph(ctx, f0_t1_b, &sine_tdb, &uv_t1_b, &noise_tdb)) {
        LOG_ERROR( "hg2_source_nsf2_build_graph: sine_gen build_graph failed\n");
        return false;
    }
    sine_tdb = ggml_cont(ctx, sine_tdb);
    uv_t1_b  = ggml_cont(ctx, uv_t1_b);
    const int64_t T   = sine_tdb->ne[0];
    const int64_t dim = sine_tdb->ne[1];
    const int64_t B   = sine_tdb->ne[2];
    ggml_tensor * w = linear_weight;
    if (w->type != GGML_TYPE_F32 || linear_bias->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_source_nsf2_build_graph: expected F32 weights\n");
        return false;
    }
    if (!(w->ne[0] == dim && w->ne[1] == 1)) {
        LOG_ERROR( "hg2_source_nsf2_build_graph: linear_weight must be [dim,1]\n");
        return false;
    }
    if (linear_bias->ne[0] != 1) {
        LOG_ERROR( "hg2_source_nsf2_build_graph: linear_bias must have 1 element\n");
        return false;
    }
    ggml_tensor * sine_dtb = ggml_permute(ctx, sine_tdb, 1, 0, 2, 3);
    sine_dtb               = ggml_cont(ctx, sine_dtb);
    ggml_tensor * sine_2d  = ggml_reshape_2d(ctx, sine_dtb, dim, T * B);
    sine_2d                = ggml_cont(ctx, sine_2d);
    ggml_tensor * mm   = ggml_mul_mat(ctx, sine_2d, w);
    ggml_tensor * y_tb = ggml_reshape_2d(ctx, mm, T, B);
    ggml_tensor * b_s = ggml_reshape_4d(ctx, linear_bias, 1, 1, 1, 1);
    // 🔧 用 ggml_add + ggml_repeat 替代 ggml_add1，支持 Metal 加速
    ggml_tensor * b_rep = ggml_repeat(ctx, b_s, y_tb);
    y_tb              = ggml_add(ctx, y_tb, b_rep);
    y_tb              = ggml_tanh(ctx, y_tb);
    y_tb              = ggml_cont(ctx, y_tb);
    ggml_tensor * sine_merge_t1_b = ggml_reshape_3d(ctx, y_tb, T, 1, B);
    sine_merge_t1_b               = ggml_cont(ctx, sine_merge_t1_b);
    ggml_set_name(sine_merge_t1_b, "hg2.source_nsf2.sine_merge_t1_b");
    ggml_tensor * noise_t1_b = ggml_scale(ctx, uv_t1_b, 0.0f);
    noise_t1_b               = ggml_cont(ctx, noise_t1_b);
    ggml_set_name(noise_t1_b, "hg2.source_nsf2.noise_t1_b");
    *out_sine_merge_t1_b = sine_merge_t1_b;
    *out_noise_t1_b      = noise_t1_b;
    *out_uv_t1_b         = uv_t1_b;
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// 构建 stft 计算
// hifigan2 stft16 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
// 构建 stft 计算
bool hg2_stft16::hg_stft16_build_graph(ggml_context *            ctx,
                                        const hg2_stft16_params & params,
                                        ggml_tensor *             x_tb,
                                        ggml_tensor **            out_real_ftb,
                                        ggml_tensor **            out_imag_ftb) {
    if (!ctx || !x_tb || !out_real_ftb || !out_imag_ftb) {
        return false;
    }
    if (x_tb->type != GGML_TYPE_F32) {
        LOG_ERROR( "hg2_stft16_build_graph: expected F32 input\n");
        return false;
    }
    if (!params.window || !params.dft_cos_t || !params.dft_sin_t) {
        LOG_ERROR( "hg2_stft16_build_graph: params not initialized\n");
        return false;
    }
    const int64_t T = x_tb->ne[0];
    const int64_t B = x_tb->ne[1];
    (void) T;
    // 对输入做 reflect pad，方便按窗口取帧
    ggml_tensor * x_pad_tb = ggml_pad_reflect_1d(ctx, x_tb, hg2_stft16_params::HG2_PAD, hg2_stft16_params::HG2_PAD);
    x_pad_tb               = ggml_cont(ctx, x_pad_tb);
    ggml_tensor * x_pad_t1b = ggml_reshape_3d(ctx, x_pad_tb, x_pad_tb->ne[0], 1, B);
    x_pad_t1b               = ggml_cont(ctx, x_pad_t1b);
    ggml_tensor * dummy_kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hg2_stft16_params::HG2_N_FFT, 1, 1);
    ggml_set_name(dummy_kernel, "hg2.stft16.dummy_kernel");
    // 按 hop 分帧，得到每帧长度为 N_FFT 的时间片
    ggml_tensor * im2col =
        ggml_im2col(ctx, dummy_kernel, x_pad_t1b, hg2_stft16_params::HG2_HOP, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * frames_ntb = ggml_reshape_3d(ctx, im2col, im2col->ne[0], im2col->ne[1], im2col->ne[2]);
    frames_ntb               = ggml_cont(ctx, frames_ntb);
    // 逐帧乘窗
    ggml_tensor * window_n11 = ggml_reshape_3d(ctx, params.window, hg2_stft16_params::HG2_N_FFT, 1, 1);
    ggml_tensor * window_ntb = ggml_repeat(ctx, window_n11, frames_ntb);
    ggml_tensor * frames_w   = ggml_mul(ctx, frames_ntb, window_ntb);
    frames_w                 = ggml_cont(ctx, frames_w);
    ggml_tensor * frames_2d =
        ggml_reshape_2d(ctx, frames_w, hg2_stft16_params::HG2_N_FFT, frames_w->ne[1] * frames_w->ne[2]);
    frames_2d = ggml_cont(ctx, frames_2d);
    // 用预计算的 DFT 矩阵做投影，得到实部和虚部
    ggml_tensor * mm_cos = ggml_mul_mat(ctx, frames_2d, params.dft_cos_t);
    ggml_tensor * mm_sin = ggml_mul_mat(ctx, frames_2d, params.dft_sin_t);
    const int64_t TT       = frames_w->ne[1];
    ggml_tensor * real_tfb = ggml_reshape_3d(ctx, mm_cos, TT, hg2_stft16_params::HG2_F, B);
    ggml_tensor * imag_tfb = ggml_reshape_3d(ctx, mm_sin, TT, hg2_stft16_params::HG2_F, B);
    imag_tfb               = ggml_neg(ctx, imag_tfb);
    // 输出布局转成 FTB
    ggml_tensor * real_ftb = ggml_permute(ctx, real_tfb, 1, 0, 2, 3);
    ggml_tensor * imag_ftb = ggml_permute(ctx, imag_tfb, 1, 0, 2, 3);
    real_ftb               = ggml_cont(ctx, real_ftb);
    imag_ftb               = ggml_cont(ctx, imag_ftb);
    ggml_set_name(real_ftb, "hg2.stft16.real_ftb");
    ggml_set_name(imag_ftb, "hg2.stft16.imag_ftb");
    *out_real_ftb = real_ftb;
    *out_imag_ftb = imag_ftb;
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 stft_istft_params 模块
namespace omni {
namespace vocoder {
namespace hifigan2 {
static void hg_stft16_fill_periodic_hann_f32(float * w, int32_t N) {
    for (int32_t n = 0; n < N; ++n) {
        w[n] = 0.5f - 0.5f * std::cos(2.0f * float(M_PI) * float(n) / float(N));
    }
}
// 初始化 stft 相关参数
bool hg2_stft16_params::hg_stft16_params_init(ggml_context * ctx) {
    return hg_stft16_params_build_consts(ctx);
}
// 构建 stft 或 istft 计算
bool hg2_stft16_params::hg_stft16_params_build_consts(ggml_context * ctx) {
    if (!ctx) {
        return false;
    }
    // 初始化 window、DFT 表、Nyquist 符号与 OLA kernel
    window    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HG2_N_FFT);
    window_sq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HG2_N_FFT);
    nyq_sign  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HG2_N_FFT);
    ggml_set_name(window, "hg2.const.window");
    ggml_set_name(window_sq, "hg2.const.window_sq");
    ggml_set_name(nyq_sign, "hg2.const.nyq_sign");
    host_window.resize((size_t) HG2_N_FFT);
    host_window_sq.resize((size_t) HG2_N_FFT);
    host_nyq_sign.resize((size_t) HG2_N_FFT);
    // 周期 Hann 窗与窗平方
    hg_stft16_fill_periodic_hann_f32(host_window.data(), HG2_N_FFT);
    for (int32_t n = 0; n < HG2_N_FFT; ++n) {
        host_window_sq[(size_t) n] = host_window[(size_t) n] * host_window[(size_t) n];
        host_nyq_sign[(size_t) n]  = std::cos(float(M_PI) * float(n));
    }
    // DFT 的 cos/sin 查表，stft/istft 都复用
    dft_cos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HG2_N_FFT, HG2_F);
    dft_sin_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HG2_N_FFT, HG2_F);
    ggml_set_name(dft_cos_t, "hg2.const.dft_cos_t");
    ggml_set_name(dft_sin_t, "hg2.const.dft_sin_t");
    host_dft_cos_t.resize((size_t) HG2_N_FFT * (size_t) HG2_F);
    host_dft_sin_t.resize((size_t) HG2_N_FFT * (size_t) HG2_F);
    for (int32_t n = 0; n < HG2_N_FFT; ++n) {
        for (int32_t k = 0; k < HG2_F; ++k) {
            const float ang = (2.0f * float(M_PI) / float(HG2_N_FFT)) * float(k) * float(n);
            host_dft_cos_t[(size_t) n + (size_t) HG2_N_FFT * (size_t) k] = std::cos(ang);
            host_dft_sin_t[(size_t) n + (size_t) HG2_N_FFT * (size_t) k] = std::sin(ang);
        }
    }
    // 用反卷积做 overlap-add，这里 kernel 是单位矩阵形式
    istft_ola_kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HG2_N_FFT, 1, HG2_N_FFT, 1);
    ggml_set_name(istft_ola_kernel, "hg2.const.istft_ola_kernel");
    host_istft_ola_kernel.assign((size_t) HG2_N_FFT * (size_t) HG2_N_FFT, 0.0f);
    for (int32_t ic = 0; ic < HG2_N_FFT; ++ic) {
        for (int32_t k = 0; k < HG2_N_FFT; ++k) {
            const size_t idx           = (size_t) k + (size_t) HG2_N_FFT * (size_t) ic;
            host_istft_ola_kernel[idx] = (k == ic) ? 1.0f : 0.0f;
        }
    }
    // 把 host 常量写进 ggml tensor（CPU 模式）
    if (window->data) {
        std::memcpy(window->data, host_window.data(), host_window.size() * sizeof(float));
    }
    if (window_sq->data) {
        std::memcpy(window_sq->data, host_window_sq.data(), host_window_sq.size() * sizeof(float));
    }
    if (nyq_sign->data) {
        std::memcpy(nyq_sign->data, host_nyq_sign.data(), host_nyq_sign.size() * sizeof(float));
    }
    if (dft_cos_t->data) {
        std::memcpy(dft_cos_t->data, host_dft_cos_t.data(), host_dft_cos_t.size() * sizeof(float));
    }
    if (dft_sin_t->data) {
        std::memcpy(dft_sin_t->data, host_dft_sin_t.data(), host_dft_sin_t.size() * sizeof(float));
    }
    if (istft_ola_kernel->data) {
        std::memcpy(istft_ola_kernel->data, host_istft_ola_kernel.data(), host_istft_ola_kernel.size() * sizeof(float));
    }
    return true;
}
// 上传常量到后端
bool hg2_stft16_params::hg_stft16_params_upload_consts(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    if (!window || !window_sq || !dft_cos_t || !dft_sin_t || !nyq_sign || !istft_ola_kernel) {
        return false;
    }
    if (host_window.empty() || host_window_sq.empty() || host_dft_cos_t.empty() || host_dft_sin_t.empty() ||
        host_nyq_sign.empty() || host_istft_ola_kernel.empty()) {
        return false;
    }
    // 把常量拷到 backend（GPU/CPU）
    ggml_backend_tensor_set(window, host_window.data(), 0, host_window.size() * sizeof(float));
    ggml_backend_tensor_set(window_sq, host_window_sq.data(), 0, host_window_sq.size() * sizeof(float));
    ggml_backend_tensor_set(nyq_sign, host_nyq_sign.data(), 0, host_nyq_sign.size() * sizeof(float));
    ggml_backend_tensor_set(dft_cos_t, host_dft_cos_t.data(), 0, host_dft_cos_t.size() * sizeof(float));
    ggml_backend_tensor_set(dft_sin_t, host_dft_sin_t.data(), 0, host_dft_sin_t.size() * sizeof(float));
    ggml_backend_tensor_set(istft_ola_kernel, host_istft_ola_kernel.data(), 0,
                            host_istft_ola_kernel.size() * sizeof(float));
    return true;
}
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
// hifigan2 vocoder_hifigan2 模块
namespace omni {
namespace vocoder {
namespace {
// 判断 backend 是否在设备侧
inline bool hg_backend_is_device(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return !ggml_backend_buft_is_host(buft);
}
inline void hg_backend_tensor_set(ggml_backend_t backend,
                                                     ggml_tensor *  tensor,
                                                     const void *   data,
                                                     size_t         nbytes) {
    if (!backend || !tensor || !data || nbytes == 0) {
        return;
    }
    if (hg_backend_is_device(backend)) {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, nbytes);
    } else {
        ggml_backend_tensor_set(tensor, data, 0, nbytes);
    }
}
// 读取张量到 host
bool hg_read_tensor_2d_tb_f32(ggml_backend_t backend, ggml_tensor * t, std::vector<float> & out_tb) {
    if (!backend || !t || t->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t T      = t->ne[0];
    const int64_t B      = t->ne[1];
    const size_t  nb0    = t->nb[0];
    const size_t  nb1    = t->nb[1];
    const size_t  nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (nbytes > 0) {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }
    out_tb.resize((size_t) T * (size_t) B);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t tt = 0; tt < T; ++tt) {
            const size_t  off                             = (size_t) tt * nb0 + (size_t) b * nb1;
            const float * p                               = reinterpret_cast<const float *>(raw.data() + off);
            out_tb[(size_t) tt + (size_t) T * (size_t) b] = *p;
        }
    }
    return true;
    // 读取张量到 host
}
bool hg_read_tensor_3d_tcb_f32(ggml_backend_t backend, ggml_tensor * t, std::vector<float> & out_tcb) {
    if (!backend || !t || t->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t T      = t->ne[0];
    const int64_t C      = t->ne[1];
    const int64_t B      = t->ne[2];
    const size_t  nb0    = t->nb[0];
    const size_t  nb1    = t->nb[1];
    const size_t  nb2    = t->nb[2];
    const size_t  nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (nbytes > 0) {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }
    out_tcb.resize((size_t) T * (size_t) C * (size_t) B);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t tt = 0; tt < T; ++tt) {
                const size_t  off = (size_t) tt * nb0 + (size_t) c * nb1 + (size_t) b * nb2;
                const float * p   = reinterpret_cast<const float *>(raw.data() + off);
                out_tcb[(size_t) tt + (size_t) T * ((size_t) c + (size_t) C * (size_t) b)] = *p;
            }
        }
    }
    // 做维度转换
    return true;
}
void hg_tb_to_bt(const std::vector<float> & tb, int64_t T, int64_t B, std::vector<float> & bt_out) {
    bt_out.resize((size_t) B * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            bt_out[(size_t) b * (size_t) T + (size_t) t] = tb[(size_t) t + (size_t) T * (size_t) b];
            // 做维度转换
        }
    }
}
void hg_t1b_to_bt1(const std::vector<float> & t1b, int64_t T, int64_t B, std::vector<float> & bt1_out) {
    bt1_out.resize((size_t) B * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            bt1_out[(size_t) b * (size_t) T + (size_t) t] = t1b[(size_t) t + (size_t) T * (size_t) b];
        }
    }
}
}  // namespace
bool voc_hg2_model::voc_hg2_model_init_from_gguf(const std::string & gguf_path_in,
                                                 const std::string & device,
                                                 int32_t             num_threads_in) {
    voc_hg2_model_free();
    gguf_path   = gguf_path_in;
    num_threads = num_threads_in > 0 ? num_threads_in : 1;
    ggml_backend_load_all();
    // Support "gpu", "gpu:0", "gpu:1" etc.
    if (device.find("gpu") == 0) {
        int gpu_idx = 0;
        if (device.find("gpu:") == 0 && device.size() > 4) {
            try {
                gpu_idx = std::stoi(device.substr(4));
            } catch (...) {
                gpu_idx = 0;
            }
        }
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(gpu_idx);
#endif
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
        std::fprintf(stderr, "voc_hg2_model: init_backend device=%s, gpu_idx=%d, backend=%s\n",
                device.c_str(), gpu_idx, backend ? ggml_backend_name(backend) : "null");
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        LOG_ERROR( "voc_hg2_model_init_from_gguf: failed to init backend\n");
        voc_hg2_model_free();
        return false;
    }

    auto loader = std::make_shared<hifigan2::hg2_gguf_model_loader>();
    if (!loader->hg_gguf_model_loader_load_from_file(gguf_path, backend)) {
        LOG_ERROR( "voc_hg2_model_init_from_gguf: failed to load gguf: %s\n", gguf_path.c_str());
        voc_hg2_model_free();
        return false;
    }

    hg2 = std::make_shared<hifigan2::hg2_model>();
    if (!hg2->hg_model_bind_from_loader(*loader)) {
        LOG_ERROR( "voc_hg2_model_init_from_gguf: failed to bind weights\n");
        voc_hg2_model_free();
        return false;
    }

    weights_owner = loader;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    galloc                          = ggml_gallocr_new(buft);
    if (!galloc) {
        LOG_ERROR( "voc_hg2_model_init_from_gguf: failed to create gallocr\n");
        voc_hg2_model_free();
        return false;
    }

    // 释放模型资源
    return true;
}
void voc_hg2_model::voc_hg2_model_free() {
    weights_owner.reset();
    hg2.reset();
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    if (backend) {
        ggml_backend_free(backend);
        backend = nullptr;
    }
    gguf_path.clear();
    num_threads = 1;
}
bool voc_hg2_runner::voc_hg2_runner_build_graph(ggml_context * ctx,
                                                ggml_cgraph *  gf,
                                                ggml_tensor *  speech_feat_c80_t_b,
                                                ggml_tensor *  cache_source_t1_b,
                                                ggml_tensor ** out_wave_t_b,
                                                ggml_tensor ** out_source_t1_b) const {
    if (!model || !model->hg2 || !ctx || !gf || !speech_feat_c80_t_b || !cache_source_t1_b || !out_wave_t_b ||
        !out_source_t1_b) {
        return false;
    }
    model->hg2->gen.dsp.hg_stft16_params_init(ctx);
    model->hg2->gen.source_nsf.sine_gen.sine_amp         = 0.1f;
    model->hg2->gen.source_nsf.sine_gen.noise_std        = 0.003f;
    model->hg2->gen.source_nsf.sine_gen.voiced_threshold = 10.0f;
    model->hg2->gen.source_nsf.sine_gen.flag_for_pulse   = false;
    model->hg2->gen.source_nsf.sine_gen.hg_sine_gen2_init(ctx, 8);
    ggml_tensor * wave_t_b    = nullptr;
    ggml_tensor * source_t1_b = nullptr;
    if (!model->hg2->gen.build_graph_forward(ctx, speech_feat_c80_t_b, cache_source_t1_b, &wave_t_b, &source_t1_b)) {
        return false;
    }
    wave_t_b    = ggml_cont(ctx, wave_t_b);
    source_t1_b = ggml_cont(ctx, source_t1_b);
    ggml_build_forward_expand(gf, wave_t_b);
    ggml_build_forward_expand(gf, source_t1_b);
    *out_wave_t_b    = wave_t_b;
    *out_source_t1_b = source_t1_b;
    return true;
}
bool voc_hg2_runner::voc_hg2_runner_eval(const std::vector<float> & speech_feat_bct,
                                         int64_t                    T_mel,
                                         std::vector<float> &       out_wave_bt,
                                         int64_t &                  out_T_audio) const {
    std::vector<float> out_source_dummy;
    int64_t            out_T_source_dummy = 0;
    std::vector<float> empty_cache;
    return voc_hg2_runner_eval_stream(speech_feat_bct, T_mel, empty_cache, 0, out_wave_bt, out_T_audio,
                                      out_source_dummy, out_T_source_dummy);
}
bool voc_hg2_runner::voc_hg2_runner_eval_stream(const std::vector<float> & speech_feat_bct,
                                                int64_t                    T_mel,
                                                const std::vector<float> & cache_source_bt1,
                                                int64_t                    Tc,
                                                std::vector<float> &       out_wave_bt,
                                                int64_t &                  out_T_audio,
                                                std::vector<float> &       out_source_bt1,
                                                int64_t &                  out_T_source) const {
    if (!model || !model->hg2 || !model->backend || !model->galloc) {
        return false;
    }
    const int64_t B = 1;
    const int64_t C = 80;
    if (T_mel <= 0) {
        return false;
    }
    if (speech_feat_bct.size() != (size_t) (B * C * T_mel)) {
        LOG_ERROR( "voc_hg2_runner_eval_stream: invalid speech_feat_bct size\n");
        return false;
    }
    if (Tc < 0) {
        return false;
    }
    if (!(Tc == 0 && cache_source_bt1.empty()) && (int64_t) cache_source_bt1.size() != Tc * B) {
        LOG_ERROR( "voc_hg2_runner_eval_stream: invalid cache_source_bt1 size\n");
        return false;
    }
    ggml_init_params params{};
    params.mem_size    = 2048ull * 1024ull * 1024ull;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * speech_upload_tcb   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_mel, C, B);
    ggml_tensor * speech_feat_c80_t_b = ggml_cont(ctx, ggml_permute(ctx, speech_upload_tcb, 1, 0, 2, 3));
    ggml_tensor * cache_source_t1_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Tc, 1, B);
    ggml_tensor * wave_t_b    = nullptr;
    ggml_tensor * source_t1_b = nullptr;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 256, false);
    if (!voc_hg2_runner_build_graph(ctx, gf, speech_feat_c80_t_b, cache_source_t1_b, &wave_t_b, &source_t1_b)) {
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(model->galloc, gf)) {
        LOG_ERROR( "voc_hg2_runner_eval_stream: ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    model->hg2->gen.dsp.hg_stft16_params_upload_consts(model->backend);
    model->hg2->gen.source_nsf.sine_gen.hg_sine_gen2_upload_consts(model->backend);
    hg_backend_tensor_set(model->backend, speech_upload_tcb, speech_feat_bct.data(),
                                             speech_feat_bct.size() * sizeof(float));
    if (Tc > 0) {
        hg_backend_tensor_set(model->backend, cache_source_t1_b, cache_source_bt1.data(),
                                                 cache_source_bt1.size() * sizeof(float));
    }
    const ggml_status st = ggml_backend_graph_compute(model->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        LOG_ERROR( "voc_hg2_runner_eval_stream: ggml_backend_graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }
    std::vector<float> wave_tb;
    if (!hg_read_tensor_2d_tb_f32(model->backend, wave_t_b, wave_tb)) {
        ggml_free(ctx);
        return false;
    }
    out_T_audio = wave_t_b->ne[0];
    hg_tb_to_bt(wave_tb, out_T_audio, B, out_wave_bt);
    std::vector<float> source_tcb;
    if (!hg_read_tensor_3d_tcb_f32(model->backend, source_t1_b, source_tcb)) {
        ggml_free(ctx);
        return false;
    }
    out_T_source = source_t1_b->ne[0];
    std::vector<float> source_t1b((size_t) out_T_source);
    for (int64_t t = 0; t < out_T_source; ++t) {
        source_t1b[(size_t) t] = source_tcb[(size_t) t];
    }
    hg_t1b_to_bt1(source_t1b, out_T_source, B, out_source_bt1);
    ggml_free(ctx);
    return true;
}
}  // namespace vocoder
}  // namespace omni




namespace omni {
namespace flow {
// 用于将GGUF张量绑定到运行时模块参数
bool bind_flow_extra_weights(const flowExtraModelLoaderGGUF & loader, flowCausalMaskedDiffWithXvec & flow) {
    ggml_tensor * emb_w = loader.get_tensor("input_embedding.weight");
    ggml_tensor * spk_w = loader.get_tensor("spk_embed_affine_layer.weight");
    ggml_tensor * spk_b = loader.get_tensor("spk_embed_affine_layer.bias");
    ggml_tensor * prj_w = loader.get_tensor("encoder_proj.weight");
    ggml_tensor * prj_b = loader.get_tensor("encoder_proj.bias");
    if (!emb_w || !spk_w || !spk_b || !prj_w || !prj_b) {
        LOG_ERROR( "bind_flow_extra_weights: missing required tensors\n");
        LOG_ERROR( "  input_embedding.weight=%p\n", (void *) emb_w);
        LOG_ERROR( "  spk_embed_affine_layer.weight=%p bias=%p\n", (void *) spk_w, (void *) spk_b);
        LOG_ERROR( "  encoder_proj.weight=%p bias=%p\n", (void *) prj_w, (void *) prj_b);
        return false;
    }
    flow.set_parameters(emb_w, spk_w, spk_b, prj_w, prj_b);
    return true;
}
// 用于绑定flow_matching(DiT/CFM)权重
bool bind_flow_matching_weights(const flow_matching::fmFlowMatchingModelLoaderGGUF & loader,
                                flow_matching::fmDiT &                               dit) {
    flow_matching::fmTimestepEmbedder * te = dit.timestep_embedder();
    if (!te) {
        LOG_ERROR( "bind_flow_matching_weights: timestep_embedder is null\n");
        return false;
    }
    te->set_parameters(
        loader.get_tensor("estimator.t_embedder.mlp.0.weight"), loader.get_tensor("estimator.t_embedder.mlp.0.bias"),
        loader.get_tensor("estimator.t_embedder.mlp.2.weight"), loader.get_tensor("estimator.t_embedder.mlp.2.bias"));
    dit.set_parameters(loader.get_tensor("estimator.in_proj.weight"), loader.get_tensor("estimator.in_proj.bias"));
    auto & blocks = dit.blocks();
    // 按block索引绑定每层注意力/卷积/MLP参数
    for (int i = 0; i < (int) blocks.size(); ++i) {
        flow_matching::fmDiTBlock * blk = blocks[(size_t) i];
        if (!blk) {
            LOG_ERROR( "bind_flow_matching_weights: null block at %d\n", i);
            return false;
        }
        blk->set_parameters(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".adaLN_modulation.1.weight"),
                            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".adaLN_modulation.1.bias"));
        blk->set_attention_parameters(
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_q.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_q.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_k.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_k.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_v.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.to_v.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.q_norm.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.q_norm.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.k_norm.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.k_norm.bias"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.proj.weight"),
            loader.get_tensor("estimator.blocks." + std::to_string(i) + ".attn.proj.bias"));
        blk->set_conv_parameters(loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.1.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.1.bias"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.6.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.6.bias"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.3.weight"),
                                 loader.get_tensor("estimator.blocks." + std::to_string(i) + ".conv.block.3.bias"));
        blk->set_mlp_parameters(loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc1.weight"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc1.bias"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc2.weight"),
                                loader.get_tensor("estimator.blocks." + std::to_string(i) + ".mlp.fc2.bias"));
    }
    flow_matching::fmFinalLayer * fl = dit.final_layer();
    if (!fl) {
        LOG_ERROR( "bind_flow_matching_weights: final_layer is null\n");
        return false;
    }
    fl->set_parameters(loader.get_tensor("estimator.final_layer.adaLN_modulation.1.weight"),
                       loader.get_tensor("estimator.final_layer.adaLN_modulation.1.bias"), nullptr, nullptr,
                       loader.get_tensor("estimator.final_layer.linear.weight"),
                       loader.get_tensor("estimator.final_layer.linear.bias"));
    return true;
}
// 用于绑定upsample_encoder_v2权重
bool bind_upsample_encoder_weights(const upsample_encoder_v2::ueUpsampleEncoderModelLoaderGGUF & loader,
                                   upsample_encoder_v2::ueUpsampleConformerEncoderV2 &           encoder) {
    using namespace upsample_encoder_v2;
    {
        std::shared_ptr<ueLinearNoSubsampling> e = encoder.embed();
        if (!e) {
            LOG_ERROR( "bind_upsample_encoder_weights: embed is null\n");
            return false;
        }
        e->set_parameters(loader.get_tensor("embed.out.0.weight"), loader.get_tensor("embed.out.0.bias"),
                          loader.get_tensor("embed.out.1.weight"), loader.get_tensor("embed.out.1.bias"));
    }
    {
        std::shared_ptr<uePreLookaheadLayer> pl = encoder.pre_lookahead_layer();
        if (!pl) {
            LOG_ERROR( "bind_upsample_encoder_weights: pre_lookahead_layer is null\n");
            return false;
        }
        pl->set_parameters(
            loader.get_tensor("pre_lookahead_layer.conv1.weight"), loader.get_tensor("pre_lookahead_layer.conv1.bias"),
            loader.get_tensor("pre_lookahead_layer.conv2.weight"), loader.get_tensor("pre_lookahead_layer.conv2.bias"));
    }
    {
        auto & layers = encoder.encoders();
        // encoder主干层参数绑定
        for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
            std::shared_ptr<ueConformerEncoderLayer> & layer = layers[(size_t) i];
            if (!layer) {
                LOG_ERROR( "bind_upsample_encoder_weights: null encoder layer %d\n", (int) i);
                return false;
            }
            std::shared_ptr<ueRelPositionMultiHeadedAttention> att = layer->self_attn();
            std::shared_ptr<uePositionwiseFeedForward>         ffn = layer->feed_forward();
            if (!att || !ffn) {
                LOG_ERROR( "bind_upsample_encoder_weights: null submodule in encoders.%d\n", (int) i);
                return false;
            }
            layer->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".norm_ff.weight"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_ff.bias"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_mha.weight"),
                                  loader.get_tensor("encoders." + std::to_string(i) + ".norm_mha.bias"));
            att->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_q.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_q.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_k.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_k.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_v.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_v.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_out.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_out.bias"));
            att->set_relpos_parameters(
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.linear_pos.weight"),
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.pos_bias_u"),
                loader.get_tensor("encoders." + std::to_string(i) + ".self_attn.pos_bias_v"));
            ffn->set_parameters(loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_1.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_1.bias"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_2.weight"),
                                loader.get_tensor("encoders." + std::to_string(i) + ".feed_forward.w_2.bias"));
        }
    }
    {
        std::shared_ptr<ueUpsample1D> up = encoder.up_layer();
        if (!up) {
            LOG_ERROR( "bind_upsample_encoder_weights: up_layer is null\n");
            return false;
        }
        up->set_parameters(loader.get_tensor("up_layer.conv.weight"), loader.get_tensor("up_layer.conv.bias"));
    }
    {
        std::shared_ptr<ueLinearNoSubsampling> ue = encoder.up_embed();
        if (!ue) {
            LOG_ERROR( "bind_upsample_encoder_weights: up_embed is null\n");
            return false;
        }
        ue->set_parameters(loader.get_tensor("up_embed.out.0.weight"), loader.get_tensor("up_embed.out.0.bias"),
                           loader.get_tensor("up_embed.out.1.weight"), loader.get_tensor("up_embed.out.1.bias"));
    }
    {
        auto & layers = encoder.up_encoders();
        // up_encoder层参数绑定
        for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
            std::shared_ptr<ueConformerEncoderLayer> & layer = layers[(size_t) i];
            if (!layer) {
                LOG_ERROR( "bind_upsample_encoder_weights: null up_encoder layer %d\n", (int) i);
                return false;
            }
            std::shared_ptr<ueRelPositionMultiHeadedAttention> att = layer->self_attn();
            std::shared_ptr<uePositionwiseFeedForward>         ffn = layer->feed_forward();
            if (!att || !ffn) {
                LOG_ERROR( "bind_upsample_encoder_weights: null submodule in up_encoders.%d\n", (int) i);
                return false;
            }
            layer->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_ff.weight"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_ff.bias"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_mha.weight"),
                                  loader.get_tensor("up_encoders." + std::to_string(i) + ".norm_mha.bias"));
            att->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_q.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_q.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_k.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_k.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_v.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_v.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_out.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_out.bias"));
            att->set_relpos_parameters(
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.linear_pos.weight"),
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.pos_bias_u"),
                loader.get_tensor("up_encoders." + std::to_string(i) + ".self_attn.pos_bias_v"));
            ffn->set_parameters(loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_1.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_1.bias"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_2.weight"),
                                loader.get_tensor("up_encoders." + std::to_string(i) + ".feed_forward.w_2.bias"));
        }
    }
    encoder.set_after_norm_parameters(loader.get_tensor("after_norm.weight"), loader.get_tensor("after_norm.bias"));
    return true;
}
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow {
namespace {
static void flow_extra_gguf_free_gguf_context(gguf_context * ctx) {
    if (ctx) {
        gguf_free(ctx);
    }
}
static void flow_extra_gguf_free_ggml_context(ggml_context * ctx) {
    if (ctx) {
        ggml_free(ctx);
    }
}
static void flow_extra_gguf_free_backend_buffer(ggml_backend_buffer_t buf) {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
}
}  // namespace
flowExtraModelLoaderGGUF::flowExtraModelLoaderGGUF() = default;
flowExtraModelLoaderGGUF::~flowExtraModelLoaderGGUF() {
    reset();
}
void flowExtraModelLoaderGGUF::reset() {
    tensors_.clear();
    flow_extra_gguf_free_backend_buffer(buf_weights_);
    buf_weights_ = nullptr;
    flow_extra_gguf_free_ggml_context(ctx_data_);
    ctx_data_ = nullptr;
    flow_extra_gguf_free_ggml_context(ctx_meta_);
    ctx_meta_ = nullptr;
    flow_extra_gguf_free_gguf_context(ctx_gguf_);
    ctx_gguf_ = nullptr;
    backend_ = nullptr;
    path_.clear();
}
bool flowExtraModelLoaderGGUF::load_from_file(const std::string & gguf_path, ggml_backend_t backend) {
    reset();
    if (!backend) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: backend is null\n");
        return false;
    }
    backend_ = backend;
    path_    = gguf_path;
    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;
    ctx_gguf_ = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf_) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: failed to open gguf: %s\n", gguf_path.c_str());
        return false;
    }
    ctx_meta_ = meta;
    if (!ctx_meta_) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: gguf meta ctx is null: %s\n", gguf_path.c_str());
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_);
    if (n_tensors <= 0) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: no tensors in gguf: %s\n", gguf_path.c_str());
        return false;
    }
    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        const size_t off = gguf_get_data_offset(ctx_gguf_) + gguf_get_tensor_offset(ctx_gguf_, i);
        offsets.emplace(std::string(name), off);
    }
    ggml_init_params data_params{};
    data_params.mem_size   = (size_t) (n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_data_              = ggml_init(data_params);
    if (!ctx_data_) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: failed to init ctx_data\n");
        return false;
    }
    tensors_.reserve((size_t) n_tensors);
    std::vector<ggml_tensor *> to_load;
    to_load.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_, i);
        if (!name) {
            continue;
        }
        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta_, name);
        if (!meta_tensor) {
            LOG_ERROR( "flowExtraModelLoaderGGUF: missing meta tensor: %s\n", name);
            return false;
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data_, meta_tensor);
        ggml_set_name(data_tensor, name);
        tensors_.emplace(std::string(name), data_tensor);
        to_load.push_back(data_tensor);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    buf_weights_                    = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data_, buft);
    if (!buf_weights_) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: backend weight alloc failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buf_weights_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR( "flowExtraModelLoaderGGUF: failed to open file stream: %s\n", gguf_path.c_str());
        return false;
    }
    std::vector<uint8_t> staging;
    for (ggml_tensor * t : to_load) {
        if (!t || !t->name) {
            continue;
        }
        const auto it = offsets.find(std::string(t->name));
        if (it == offsets.end()) {
            LOG_ERROR( "flowExtraModelLoaderGGUF: missing offset for tensor: %s\n", t->name);
            return false;
        }
        const size_t off = it->second;
        fin.seekg((std::streamoff) off, std::ios::beg);
        if (!fin) {
            LOG_ERROR( "flowExtraModelLoaderGGUF: seek failed for tensor: %s\n", t->name);
            return false;
        }
        const size_t nbytes = ggml_nbytes(t);
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(t->data), (std::streamsize) nbytes);
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), (std::streamsize) nbytes);
            ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        }
        if (!fin) {
            LOG_ERROR( "flowExtraModelLoaderGGUF: read failed for tensor: %s\n", t->name);
            return false;
        }
    }
    return true;
}
ggml_tensor * flowExtraModelLoaderGGUF::get_tensor(const std::string & name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow {
namespace {
ggml_backend_t flow_loader_init_backend_cpu(std::string & backend_name_out) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
ggml_backend_t flow_loader_init_backend_gpu_first(std::string & backend_name_out) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend) {
        backend_name_out = ggml_backend_name(backend);
    }
    return backend;
}
}  // namespace
flowGGUFModelLoader::flowGGUFModelLoader() = default;
flowGGUFModelLoader::~flowGGUFModelLoader() {
    reset();
}
void flowGGUFModelLoader::reset() {
    flow_.reset();
    decoder_.reset();
    estimator_.reset();
    encoder_model_.reset();
    encoder_weights_.reset();
    flow_matching_.reset();
    flow_extra_.reset();
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    backend_name_.clear();
    encoder_pre_lookahead_len_ = 3;
}
bool flowGGUFModelLoader::init_backend(const std::string & device) {
    if (backend_) {
        return true;
    }
    // Support "gpu", "gpu:0", "gpu:1", "cuda", "cuda:0", etc.
    if (device.find("gpu") == 0 || device.find("cuda") == 0 || device == "auto") {
        int gpu_idx = 0;
        // Parse "gpu:N" or "cuda:N" format
        auto colon_pos = device.find(':');
        if (colon_pos != std::string::npos && colon_pos + 1 < device.size()) {
            try {
                gpu_idx = std::stoi(device.substr(colon_pos + 1));
            } catch (...) {
                gpu_idx = 0;
            }
        }
#ifdef GGML_USE_CUDA
        backend_ = ggml_backend_cuda_init(gpu_idx);
#endif
        if (!backend_) {
            backend_ = flow_loader_init_backend_gpu_first(backend_name_);
        }
        if (backend_) {
            backend_name_ = ggml_backend_name(backend_);
        }
        std::fprintf(stderr, "flowGGUFModelLoader: init_backend device=%s, gpu_idx=%d, backend=%s\n",
                device.c_str(), gpu_idx, backend_name_.c_str());
    } else {
        backend_ = flow_loader_init_backend_cpu(backend_name_);
    }
    if (!backend_) {
        LOG_ERROR( "flowGGUFModelLoader: failed to init backend\n");
        return false;
    }
    return true;
}
bool flowGGUFModelLoader::bind_all() {
    if (!flow_extra_ || !flow_matching_ || !encoder_weights_) {
        return false;
    }
    if (!encoder_model_ || !estimator_ || !decoder_ || !flow_) {
        return false;
    }
    if (!bind_upsample_encoder_weights(*encoder_weights_, *encoder_model_)) {
        LOG_ERROR( "flowGGUFModelLoader: bind_upsample_encoder_weights failed\n");
        return false;
    }
    if (!bind_flow_matching_weights(*flow_matching_, *estimator_)) {
        LOG_ERROR( "flowGGUFModelLoader: bind_flow_matching_weights failed\n");
        return false;
    }
    if (!bind_flow_extra_weights(*flow_extra_, *flow_)) {
        LOG_ERROR( "flowGGUFModelLoader: bind_flow_extra_weights failed\n");
        return false;
    }
    return true;
}
bool flowGGUFModelLoader::load_from_gguf(const std::string & encoder_gguf_path,
                                         const std::string & flow_matching_gguf_path,
                                         const std::string & flow_extra_gguf_path,
                                         const std::string & device) {
    reset();
    if (!init_backend(device)) {
        return false;
    }
    encoder_weights_ = std::make_unique<upsample_encoder_v2::ueUpsampleEncoderModelLoaderGGUF>();
    if (!encoder_weights_->load_from_file(encoder_gguf_path, backend_)) {
        LOG_ERROR( "flowGGUFModelLoader: failed to load encoder gguf: %s\n", encoder_gguf_path.c_str());
        reset();
        return false;
    }
    flow_matching_ = std::make_unique<flow_matching::fmFlowMatchingModelLoaderGGUF>();
    if (!flow_matching_->load_from_file(flow_matching_gguf_path, backend_)) {
        LOG_ERROR( "flowGGUFModelLoader: failed to load flow_matching gguf: %s\n",
                     flow_matching_gguf_path.c_str());
        reset();
        return false;
    }
    flow_extra_ = std::make_unique<flowExtraModelLoaderGGUF>();
    if (!flow_extra_->load_from_file(flow_extra_gguf_path, backend_)) {
        LOG_ERROR( "flowGGUFModelLoader: failed to load flow_extra gguf: %s\n", flow_extra_gguf_path.c_str());
        reset();
        return false;
    }
    encoder_pre_lookahead_len_ = 3;
    if (ggml_tensor * pl_w = encoder_weights_->get_tensor("pre_lookahead_layer.conv1.weight")) {
        const int64_t K = pl_w->ne[0];
        if (K >= 1) {
            encoder_pre_lookahead_len_ = (int32_t) (K - 1);
        }
    }
    encoder_model_ = std::make_shared<upsample_encoder_v2::ueUpsampleConformerEncoderV2>(
        512, 512, encoder_pre_lookahead_len_, 6, 4, 2, 2.0f, 8, true, 2048, 0.0f, 0.0f, 0.0f, true);
    estimator_ = std::make_shared<flow_matching::fmDiT>(320, 80, 4.0f, 16, 8, 64, 512);
    decoder_ = std::make_shared<flow_matching::fmCausalConditionalCFM>(estimator_, 0.7f);
    flow_ = std::make_shared<flowCausalMaskedDiffWithXvec>(512, 80, 192, 6561, encoder_model_, decoder_);
    if (!bind_all()) {
        LOG_ERROR( "flowGGUFModelLoader: bind_all failed\n");
        reset();
        return false;
    }
    decoder_->reset_stream_state();
    return true;
}
}  // namespace flow
}  // namespace omni


namespace omni {
namespace flow {
namespace {
// 用于统一后端张量写入策略(CPU/GPU)
bool runner_backend_is_device(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return !ggml_backend_buft_is_host(buft);
}
void backend_tensor_set(ggml_backend_t backend,
                                      ggml_tensor *  tensor,
                                      const void *   data,
                                      size_t         size_bytes) {
    if (!backend || !tensor || !data) {
        return;
    }
    if (runner_backend_is_device(backend)) {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size_bytes);
    } else {
        ggml_backend_tensor_set(tensor, data, 0, size_bytes);
    }
}
bool runner_read_tensor_bytes(ggml_backend_t backend, ggml_tensor * t, std::vector<uint8_t> & out) {
    out.clear();
    if (!backend || !t) {
        return false;
    }
    const size_t n = ggml_nbytes(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n);
    return true;
}
void runner_bc_to_cb(const float * bc, int64_t B, int64_t C, std::vector<float> & cb_out) {
    cb_out.resize((size_t) B * (size_t) C);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            cb_out[(size_t) c + (size_t) C * (size_t) b] = bc[(size_t) b * (size_t) C + (size_t) c];
        }
    }
}
void runner_bt_to_tb(const int32_t * bt, int64_t B, int64_t T, std::vector<int32_t> & tb_out) {
    tb_out.resize((size_t) B * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            tb_out[(size_t) t + (size_t) T * (size_t) b] = bt[(size_t) b * (size_t) T + (size_t) t];
        }
    }
}
void runner_btc_to_ctb(const float * btc, int64_t B, int64_t T, int64_t C, std::vector<float> & ctb_out) {
    ctb_out.resize((size_t) B * (size_t) T * (size_t) C);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                const size_t idx_btc = (size_t) b * (size_t) T * (size_t) C + (size_t) t * (size_t) C + (size_t) c;
                const size_t idx_ctb = (size_t) c + (size_t) C * ((size_t) t + (size_t) T * (size_t) b);
                ctb_out[idx_ctb]     = btc[idx_btc];
            }
        }
    }
}
void runner_ctb_to_bct(const std::vector<float> & ctb, int64_t C, int64_t T, int64_t B, std::vector<float> & bct_out) {
    bct_out.resize((size_t) B * (size_t) C * (size_t) T);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t = 0; t < T; ++t) {
                const size_t idx_ctb = (size_t) c + (size_t) C * ((size_t) t + (size_t) T * (size_t) b);
                const size_t idx_bct = (size_t) b * (size_t) C * (size_t) T + (size_t) c * (size_t) T + (size_t) t;
                bct_out[idx_bct]     = ctb[idx_ctb];
            }
        }
    }
}
void runner_build_cosine_t_span(int n_timesteps, std::vector<float> & t_span_out) {
    const int steps = n_timesteps > 0 ? n_timesteps : 1;
    t_span_out.resize((size_t) steps + 1);
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 0; i <= steps; ++i) {
        const float u          = (float) i / (float) steps;
        t_span_out[(size_t) i] = 1.0f - std::cos(u * 0.5f * kPi);
    }
}
void runner_fill_noise_ctb(std::vector<float> & noise_ctb,
                    int64_t              C,
                    int64_t              T,
                    int64_t              B,
                    float                temperature,
                    int64_t              offset_ct) {
    noise_ctb.resize((size_t) C * (size_t) T * (size_t) B);
    const int64_t total = C * T * B;
    // 🔧 使用真随机数替代周期性伪噪声，避免音频伪影
    static std::mt19937 gen(42);  // 固定种子保证可复现
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < total; ++i) {
        noise_ctb[(size_t) i] = temperature * dist(gen);
    }
}
void runner_fill_timestep_1d(std::vector<float> & t_host, int64_t B_total, float t_value) {
    t_host.resize((size_t) B_total);
    for (int64_t i = 0; i < B_total; ++i) {
        t_host[(size_t) i] = t_value;
    }
}
// 用于填充encoder流式位置编码
void runner_feed_enc_stream_pos(ggml_backend_t                                                             backend,
                                    ggml_context *                                                             ctx,
                                    const std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> & enc) {
    if (!backend || !ctx || !enc) {
        return;
    }
    std::shared_ptr<upsample_encoder_v2::ueLinearNoSubsampling> embed = enc->embed();
    if (!embed || !embed->pos_enc()) {
        return;
    }
    std::shared_ptr<upsample_encoder_v2::ueEspnetRelPositionalEncoding> pe = embed->pos_enc();
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        const char * name = t->name;
        if (!name) {
            continue;
        }
        if (std::strcmp(name, "ue_uce_pos_stream1") != 0 && std::strcmp(name, "ue_uce_pos_stream2") != 0) {
            continue;
        }
        const int32_t      size = (int32_t) ((t->ne[1] + 1) / 2);
        std::vector<float> host;
        pe->position_encoding_host(size, host);
        backend_tensor_set(backend, t, host.data(), host.size() * sizeof(float));
    }
}
// 用于填充CFM噪声与时间步输入
void runner_feed_cfm_noise_ts(ggml_backend_t backend,
                                  ggml_context * ctx,
                                  int            call_id,
                                  int            n_timesteps,
                                  float          temperature,
                                  int64_t        last_att_len,
                                  int64_t        C,
                                  int64_t        T,
                                  int64_t        B) {
    if (!backend || !ctx) {
        return;
    }
    (void) T;
    const int64_t offset_ct = last_att_len * C * B;
    {
        char noise_name[64];
        std::snprintf(noise_name, sizeof(noise_name), "fm_cfm_noise_chunk%d", call_id);
        if (ggml_tensor * t_noise = ggml_get_tensor(ctx, noise_name)) {
            std::vector<float> noise_ctb;
            runner_fill_noise_ctb(noise_ctb, t_noise->ne[0], t_noise->ne[1], t_noise->ne[2], temperature, offset_ct);
            backend_tensor_set(backend, t_noise, noise_ctb.data(), noise_ctb.size() * sizeof(float));
        }
    }
    {
        std::vector<float> t_span;
        runner_build_cosine_t_span(n_timesteps, t_span);
        const int64_t B_total = 2 * B;
        for (int step = 0; step < n_timesteps; ++step) {
            char name_buf[80];
            std::snprintf(name_buf, sizeof(name_buf), "fm_cfm_t_in_chunk%d_step%d", call_id, step);
            if (ggml_tensor * tt = ggml_get_tensor(ctx, name_buf)) {
                std::vector<float> t_host;
                runner_fill_timestep_1d(t_host, B_total, t_span[(size_t) step]);
                backend_tensor_set(backend, tt, t_host.data(), t_host.size() * sizeof(float));
            }
        }
    }
}
int64_t runner_n_elements_approx(const ggml_tensor * t) {
    if (!t) {
        return 0;
    }
    const int nd = ggml_n_dims(t);
    int64_t   n  = t->ne[0];
    if (nd > 1) {
        n *= t->ne[1];
    }
    if (nd > 2) {
        n *= t->ne[2];
    }
    if (nd > 3) {
        n *= t->ne[3];
    }
    return n;
}
ggml_tensor * runner_slice_time_dim1_4d(ggml_context * ctx, ggml_tensor * x, int64_t t0, int64_t tlen) {
    const size_t  off = x->nb[1] * (size_t) t0;
    ggml_tensor * v   = ggml_view_4d(ctx, x, x->ne[0], tlen, x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], off);
    return ggml_cont(ctx, v);
}
}  // namespace
// 用于上下文/计算图与流式缓存张量
struct flowGGUFModelRunner::streamSession {
    ggml_context *        ctx = nullptr;
    ggml_gallocr_t        galloc = nullptr;
    int64_t B              = 0;
    int64_t T_prompt_token = 0;
    int64_t T_prompt_mel   = 0;
    int64_t T_chunk_token  = 0;
    int   n_timesteps = 0;
    float temperature = 1.0f;
    ggml_tensor * prompt_token_ids_tb = nullptr;
    ggml_tensor * prompt_mel_ctb      = nullptr;
    ggml_tensor * spk_cb              = nullptr;
    ggml_tensor * chunk_token_ids_tb  = nullptr;
    ggml_tensor * conf_cnn_cache = nullptr;
    ggml_tensor * conf_att_cache = nullptr;
    ggml_tensor * est_cnn_cache  = nullptr;
    ggml_tensor * est_att_cache  = nullptr;
    ggml_tensor * out_feat_nonlast_ctb = nullptr;
    ggml_tensor * out_feat_last_ctb    = nullptr;
    ggml_cgraph * gf_setup   = nullptr;
    ggml_cgraph * gf_nonlast = nullptr;
    ggml_cgraph * gf_last    = nullptr;
    int call_id_setup   = -1;
    int call_id_nonlast = -1;
    int call_id_last    = -1;
    void clear() {
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        *this = streamSession();
    }
};
flowGGUFModelRunner::flowGGUFModelRunner() = default;
flowGGUFModelRunner::~flowGGUFModelRunner() {
    reset();
}
void flowGGUFModelRunner::reset() {
    if (sess_) {
        sess_->clear();
        sess_.reset();
    }
    loader_.~flowGGUFModelLoader();
    new (&loader_) flowGGUFModelLoader();
    num_threads_           = 1;
    export_caches_to_host_ = true;
}
bool flowGGUFModelRunner::load_from_gguf(const std::string & encoder_gguf_path,
                                         const std::string & flow_matching_gguf_path,
                                         const std::string & flow_extra_gguf_path,
                                         const std::string & device) {
    reset();
    if (!loader_.load_from_gguf(encoder_gguf_path, flow_matching_gguf_path, flow_extra_gguf_path, device)) {
        return false;
    }
    set_num_threads(num_threads_);
    reset_stream();
    return true;
}
void flowGGUFModelRunner::set_num_threads(int n_threads) {
    num_threads_ = n_threads > 0 ? n_threads : 1;
    if (loader_.backend() && !runner_backend_is_device(loader_.backend())) {
        ggml_backend_cpu_set_n_threads(loader_.backend(), num_threads_);
    }
}
void flowGGUFModelRunner::reset_stream() {
    if (sess_) {
        sess_->clear();
        sess_.reset();
    }
    if (std::shared_ptr<flow_matching::fmCausalConditionalCFM> dec = loader_.decoder()) {
        dec->reset_stream_state();
    }
}
bool flowGGUFModelRunner::setup_cache(const int32_t *       token_bt,
                                      int64_t               B,
                                      int64_t               T_token,
                                      const float *         mel_btc,
                                      int64_t               T_mel,
                                      int64_t               C_mel,
                                      const float *         spk_bc,
                                      int64_t               C_spk,
                                      int                   n_timesteps,
                                      float                 temperature,
                                      flowStreamCacheHost & cache_out) {
    cache_out.clear();
    if (!token_bt || !mel_btc || !spk_bc || B <= 0 || T_token <= 0 || T_mel <= 0) {
        return false;
    }
    if (C_mel != 80) {
        LOG_ERROR( "flowGGUFModelRunner.setup_cache: expected C_mel=80, got %lld\n", (long long) C_mel);
        return false;
    }
    if (C_spk != 192) {
        LOG_ERROR( "flowGGUFModelRunner.setup_cache: expected C_spk=192, got %lld\n", (long long) C_spk);
        return false;
    }
    if (!loader_.backend() || !loader_.model()) {
        return false;
    }
    const int64_t T_chunk_token = 28;
    if (!sess_) {
        sess_ = std::make_unique<streamSession>();
    }
    const bool need_rebuild = sess_->ctx == nullptr || sess_->B != B || sess_->T_prompt_token != T_token ||
                              sess_->T_prompt_mel != T_mel || sess_->T_chunk_token != T_chunk_token ||
                              sess_->n_timesteps != n_timesteps;
    // 需要时重建图与持久化cache张量
    if (need_rebuild) {
        sess_->clear();
        sess_->B              = B;
        sess_->T_prompt_token = T_token;
        sess_->T_prompt_mel   = T_mel;
        sess_->T_chunk_token  = T_chunk_token;
        sess_->n_timesteps    = n_timesteps;
        sess_->temperature    = temperature;
        if (std::shared_ptr<flow_matching::fmCausalConditionalCFM> dec = loader_.decoder()) {
            dec->reset_stream_state();
        }
        ggml_init_params p{};
        p.mem_size   = 2048ull * 1024ull * 1024ull;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        sess_->ctx   = ggml_init(p);
        if (!sess_->ctx) {
            sess_->clear();
            return false;
        }
        sess_->prompt_token_ids_tb = ggml_new_tensor_2d(sess_->ctx, GGML_TYPE_I32, T_token, B);
        ggml_set_input(sess_->prompt_token_ids_tb);
        sess_->prompt_mel_ctb = ggml_new_tensor_3d(sess_->ctx, GGML_TYPE_F32, C_mel, T_mel, B);
        ggml_set_input(sess_->prompt_mel_ctb);
        sess_->spk_cb = ggml_new_tensor_2d(sess_->ctx, GGML_TYPE_F32, C_spk, B);
        ggml_set_input(sess_->spk_cb);
        sess_->chunk_token_ids_tb = ggml_new_tensor_2d(sess_->ctx, GGML_TYPE_I32, T_chunk_token, B);
        ggml_set_input(sess_->chunk_token_ids_tb);
        flow_matching::fmCFMCache est_cache_setup_out;
        est_cache_setup_out.clear();
        auto out0 =
            loader_.model()->build_setup_cache_graph(sess_->ctx, sess_->prompt_token_ids_tb, sess_->prompt_mel_ctb,
                                                     sess_->spk_cb, n_timesteps, temperature, &est_cache_setup_out);
        if (!out0.conformer_cnn_cache || !out0.conformer_att_cache || !est_cache_setup_out.cnn_cache ||
            !est_cache_setup_out.att_cache) {
            sess_->clear();
            return false;
        }
        sess_->call_id_setup = 0;
        sess_->conf_cnn_cache = ggml_dup_tensor(sess_->ctx, out0.conformer_cnn_cache);
        sess_->conf_att_cache = ggml_dup_tensor(sess_->ctx, out0.conformer_att_cache);
        sess_->est_cnn_cache  = ggml_dup_tensor(sess_->ctx, est_cache_setup_out.cnn_cache);
        sess_->est_att_cache  = ggml_dup_tensor(sess_->ctx, est_cache_setup_out.att_cache);
        if (!sess_->conf_cnn_cache || !sess_->conf_att_cache || !sess_->est_cnn_cache || !sess_->est_att_cache) {
            sess_->clear();
            return false;
        }
        // Persistent state: do not let the graph allocator overwrite these buffers.
        ggml_set_output(sess_->conf_cnn_cache);
        ggml_set_output(sess_->conf_att_cache);
        ggml_set_output(sess_->est_cnn_cache);
        ggml_set_output(sess_->est_att_cache);
        if (runner_n_elements_approx(out0.conformer_cnn_cache) != runner_n_elements_approx(sess_->conf_cnn_cache) ||
            runner_n_elements_approx(out0.conformer_att_cache) != runner_n_elements_approx(sess_->conf_att_cache) ||
            runner_n_elements_approx(est_cache_setup_out.cnn_cache) != runner_n_elements_approx(sess_->est_cnn_cache) ||
            runner_n_elements_approx(est_cache_setup_out.att_cache) != runner_n_elements_approx(sess_->est_att_cache)) {
            LOG_ERROR(
                "flowGGUFModelRunner.setup_cache: cache tensor shape mismatch when initializing persistent session\n");
            sess_->clear();
            return false;
        }
        ggml_tensor * cpy_conf_cnn = ggml_cpy(sess_->ctx, out0.conformer_cnn_cache, sess_->conf_cnn_cache);
        ggml_tensor * cpy_conf_att = ggml_cpy(sess_->ctx, out0.conformer_att_cache, sess_->conf_att_cache);
        ggml_tensor * cpy_est_cnn  = ggml_cpy(sess_->ctx, est_cache_setup_out.cnn_cache, sess_->est_cnn_cache);
        ggml_tensor * cpy_est_att  = ggml_cpy(sess_->ctx, est_cache_setup_out.att_cache, sess_->est_att_cache);
        sess_->gf_setup = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(sess_->gf_setup, cpy_conf_cnn);
        ggml_build_forward_expand(sess_->gf_setup, cpy_conf_att);
        ggml_build_forward_expand(sess_->gf_setup, cpy_est_cnn);
        ggml_build_forward_expand(sess_->gf_setup, cpy_est_att);
        flow_matching::fmCFMCache est_in;
        est_in.clear();
        est_in.cnn_cache = sess_->est_cnn_cache;
        est_in.att_cache = sess_->est_att_cache;
        flow_matching::fmCFMCache est_out_nonlast;
        est_out_nonlast.clear();
        auto out1 = loader_.model()->build_inference_chunk_graph(sess_->ctx, sess_->chunk_token_ids_tb, sess_->spk_cb,
                                                                 false, sess_->conf_cnn_cache, sess_->conf_att_cache,
                                                                 &est_in, n_timesteps, temperature, &est_out_nonlast);
        if (!out1.feat_ctb || !out1.conformer_cnn_cache || !out1.conformer_att_cache || !est_out_nonlast.cnn_cache ||
            !est_out_nonlast.att_cache) {
            sess_->clear();
            return false;
        }
        sess_->out_feat_nonlast_ctb = out1.feat_ctb;
        sess_->call_id_nonlast      = 1;
        const int64_t L_conf_att = sess_->conf_att_cache->ne[1];
        const int64_t L_est_att  = sess_->est_att_cache->ne[1];
        const int64_t delta      = out1.feat_ctb->ne[1];
        ggml_tensor * out1_conf_att_trim = out1.conformer_att_cache;
        if (out1.conformer_att_cache->ne[1] > L_conf_att) {
            out1_conf_att_trim = runner_slice_time_dim1_4d(sess_->ctx, out1.conformer_att_cache, delta, L_conf_att);
        }
        ggml_tensor * out1_est_att_trim = est_out_nonlast.att_cache;
        if (est_out_nonlast.att_cache->ne[1] > L_est_att) {
            out1_est_att_trim = runner_slice_time_dim1_4d(sess_->ctx, est_out_nonlast.att_cache, delta, L_est_att);
        }
        ggml_tensor * cpy_conf_cnn_1 = ggml_cpy(sess_->ctx, out1.conformer_cnn_cache, sess_->conf_cnn_cache);
        ggml_tensor * cpy_conf_att_1 = ggml_cpy(sess_->ctx, out1_conf_att_trim, sess_->conf_att_cache);
        ggml_tensor * cpy_est_cnn_1  = ggml_cpy(sess_->ctx, est_out_nonlast.cnn_cache, sess_->est_cnn_cache);
        ggml_tensor * cpy_est_att_1  = ggml_cpy(sess_->ctx, out1_est_att_trim, sess_->est_att_cache);
        sess_->gf_nonlast = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(sess_->gf_nonlast, sess_->out_feat_nonlast_ctb);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_conf_cnn_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_conf_att_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_est_cnn_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_est_att_1);
        flow_matching::fmCFMCache est_out_last;
        est_out_last.clear();
        auto out2 = loader_.model()->build_inference_chunk_graph(sess_->ctx, sess_->chunk_token_ids_tb, sess_->spk_cb,
                                                                 true, sess_->conf_cnn_cache, sess_->conf_att_cache,
                                                                 &est_in, n_timesteps, temperature, &est_out_last);
        if (!out2.feat_ctb || !out2.conformer_cnn_cache || !out2.conformer_att_cache || !est_out_last.cnn_cache ||
            !est_out_last.att_cache) {
            sess_->clear();
            return false;
        }
        sess_->out_feat_last_ctb = out2.feat_ctb;
        sess_->call_id_last      = 2;
        const int64_t delta2             = out2.feat_ctb->ne[1];
        ggml_tensor * out2_conf_att_trim = out2.conformer_att_cache;
        if (out2.conformer_att_cache->ne[1] > L_conf_att) {
            out2_conf_att_trim = runner_slice_time_dim1_4d(sess_->ctx, out2.conformer_att_cache, delta2, L_conf_att);
        }
        ggml_tensor * out2_est_att_trim = est_out_last.att_cache;
        if (est_out_last.att_cache->ne[1] > L_est_att) {
            out2_est_att_trim = runner_slice_time_dim1_4d(sess_->ctx, est_out_last.att_cache, delta2, L_est_att);
        }
        ggml_tensor * cpy_conf_cnn_2 = ggml_cpy(sess_->ctx, out2.conformer_cnn_cache, sess_->conf_cnn_cache);
        ggml_tensor * cpy_conf_att_2 = ggml_cpy(sess_->ctx, out2_conf_att_trim, sess_->conf_att_cache);
        ggml_tensor * cpy_est_cnn_2  = ggml_cpy(sess_->ctx, est_out_last.cnn_cache, sess_->est_cnn_cache);
        ggml_tensor * cpy_est_att_2  = ggml_cpy(sess_->ctx, out2_est_att_trim, sess_->est_att_cache);
        sess_->gf_last = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(sess_->gf_last, sess_->out_feat_last_ctb);
        ggml_build_forward_expand(sess_->gf_last, cpy_conf_cnn_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_conf_att_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_est_cnn_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_est_att_2);

        
        ggml_cgraph * gf_alloc = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(gf_alloc, cpy_conf_cnn);
        ggml_build_forward_expand(gf_alloc, cpy_conf_att);
        ggml_build_forward_expand(gf_alloc, cpy_est_cnn);
        ggml_build_forward_expand(gf_alloc, cpy_est_att);
        ggml_build_forward_expand(gf_alloc, sess_->out_feat_nonlast_ctb);
        ggml_build_forward_expand(gf_alloc, cpy_conf_cnn_1);
        ggml_build_forward_expand(gf_alloc, cpy_conf_att_1);
        ggml_build_forward_expand(gf_alloc, cpy_est_cnn_1);
        ggml_build_forward_expand(gf_alloc, cpy_est_att_1);
        ggml_build_forward_expand(gf_alloc, sess_->out_feat_last_ctb);
        ggml_build_forward_expand(gf_alloc, cpy_conf_cnn_2);
        ggml_build_forward_expand(gf_alloc, cpy_conf_att_2);
        ggml_build_forward_expand(gf_alloc, cpy_est_cnn_2);
        ggml_build_forward_expand(gf_alloc, cpy_est_att_2);

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(loader_.backend());
        sess_->galloc = ggml_gallocr_new(buft);
        if (!sess_->galloc) {
            sess_->clear();
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sess_->galloc, gf_alloc)) {
            sess_->clear();
            return false;
        }
    }
    std::vector<int32_t> token_tb;
    runner_bt_to_tb(token_bt, B, T_token, token_tb);
    std::vector<float> spk_cb;
    runner_bc_to_cb(spk_bc, B, C_spk, spk_cb);
    std::vector<float> mel_ctb;
    runner_btc_to_ctb(mel_btc, B, T_mel, C_mel, mel_ctb);
    backend_tensor_set(loader_.backend(), sess_->prompt_token_ids_tb, token_tb.data(),
                                     token_tb.size() * sizeof(int32_t));
    backend_tensor_set(loader_.backend(), sess_->spk_cb, spk_cb.data(), spk_cb.size() * sizeof(float));
    backend_tensor_set(loader_.backend(), sess_->prompt_mel_ctb, mel_ctb.data(),
                                     mel_ctb.size() * sizeof(float));
    runner_feed_enc_stream_pos(loader_.backend(), sess_->ctx, loader_.encoder());
    runner_feed_cfm_noise_ts(loader_.backend(), sess_->ctx, sess_->call_id_setup, n_timesteps, temperature, 0,
                                 C_mel, T_mel, B);
    const ggml_status st = ggml_backend_graph_compute(loader_.backend(), sess_->gf_setup);
    if (st != GGML_STATUS_SUCCESS) {
        return false;
    }
    cache_out.n_timesteps = n_timesteps;
    if (export_caches_to_host_) {
        if (runner_backend_is_device(loader_.backend())) {
            ggml_backend_synchronize(loader_.backend());
        }
        cache_out.conformer_cnn_ne = { sess_->conf_cnn_cache->ne[0], sess_->conf_cnn_cache->ne[1],
                                       sess_->conf_cnn_cache->ne[2] };
        cache_out.conformer_att_ne = { sess_->conf_att_cache->ne[0], sess_->conf_att_cache->ne[1],
                                       sess_->conf_att_cache->ne[2], sess_->conf_att_cache->ne[3] };
        cache_out.estimator_cnn_ne = { sess_->est_cnn_cache->ne[0], sess_->est_cnn_cache->ne[1],
                                       sess_->est_cnn_cache->ne[2], sess_->est_cnn_cache->ne[3] };
        cache_out.estimator_att_ne = { sess_->est_att_cache->ne[0], sess_->est_att_cache->ne[1],
                                       sess_->est_att_cache->ne[2], sess_->est_att_cache->ne[3] };
        runner_read_tensor_bytes(loader_.backend(), sess_->conf_cnn_cache, cache_out.conformer_cnn_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->conf_att_cache, cache_out.conformer_att_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->est_cnn_cache, cache_out.estimator_cnn_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->est_att_cache, cache_out.estimator_att_cache);
    }
    return true;
}
bool flowGGUFModelRunner::inference_chunk(const int32_t *             token_bt,
                                          int64_t                     B,
                                          int64_t                     T_token,
                                          const float *               spk_bc,
                                          int64_t                     C_spk,
                                          bool                        last_chunk,
                                          const flowStreamCacheHost & cache_in,
                                          int                         n_timesteps,
                                          float                       temperature,
                                          std::vector<float> &        mel_bct_out,
                                          flowStreamCacheHost &       cache_out) {
    mel_bct_out.clear();
    cache_out.clear();
    if (!token_bt || !spk_bc || B <= 0 || T_token <= 0) {
        return false;
    }
    if (C_spk != 192) {
        LOG_ERROR( "flowGGUFModelRunner.inference_chunk: expected C_spk=192, got %lld\n", (long long) C_spk);
        return false;
    }
    if (!loader_.backend() || !loader_.model()) {
        return false;
    }
    (void) cache_in;
    if (!sess_ || !sess_->ctx || !sess_->gf_nonlast || !sess_->gf_last) {
        LOG_ERROR(
                     "flowGGUFModelRunner.inference_chunk: stream session not initialized (call setup_cache first)\n");
        return false;
    }
    if (sess_->B != B) {
        LOG_ERROR( "flowGGUFModelRunner.inference_chunk: B mismatch (sess=%lld, got=%lld)\n",
                     (long long) sess_->B, (long long) B);
        return false;
    }
    if (sess_->T_chunk_token != T_token) {
        LOG_ERROR( "flowGGUFModelRunner.inference_chunk: expected T_token=%lld, got=%lld\n",
                     (long long) sess_->T_chunk_token, (long long) T_token);
        return false;
    }
    if (sess_->n_timesteps != n_timesteps) {
        LOG_ERROR( "flowGGUFModelRunner.inference_chunk: n_timesteps mismatch (sess=%d, got=%d)\n",
                     sess_->n_timesteps, n_timesteps);
        return false;
    }
    std::vector<int32_t> token_tb;
    runner_bt_to_tb(token_bt, B, T_token, token_tb);
    std::vector<float> spk_cb;
    runner_bc_to_cb(spk_bc, B, C_spk, spk_cb);
    backend_tensor_set(loader_.backend(), sess_->chunk_token_ids_tb, token_tb.data(),
                                     token_tb.size() * sizeof(int32_t));
    backend_tensor_set(loader_.backend(), sess_->spk_cb, spk_cb.data(), spk_cb.size() * sizeof(float));
    runner_feed_enc_stream_pos(loader_.backend(), sess_->ctx, loader_.encoder());
    const int     call_id      = last_chunk ? sess_->call_id_last : sess_->call_id_nonlast;
    const int64_t last_att_len = sess_->est_att_cache ? sess_->est_att_cache->ne[1] : 0;
    ggml_tensor * feat         = last_chunk ? sess_->out_feat_last_ctb : sess_->out_feat_nonlast_ctb;
    const int64_t C            = feat->ne[0];
    const int64_t T            = feat->ne[1];
    runner_feed_cfm_noise_ts(loader_.backend(), sess_->ctx, call_id, n_timesteps, temperature, last_att_len, C, T,
                                 B);
    // 根据last_chunk选择图并执行推理
    ggml_cgraph *     gf = last_chunk ? sess_->gf_last : sess_->gf_nonlast;
    const ggml_status st = ggml_backend_graph_compute(loader_.backend(), gf);
    if (st != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (runner_backend_is_device(loader_.backend())) {
        ggml_backend_synchronize(loader_.backend());
    }
    {
        const int64_t      Bb = feat->ne[2];
        std::vector<float> feat_ctb((size_t) C * (size_t) T * (size_t) Bb);
        ggml_backend_tensor_get(feat, feat_ctb.data(), 0, feat_ctb.size() * sizeof(float));
        runner_ctb_to_bct(feat_ctb, C, T, Bb, mel_bct_out);
    }
    cache_out.n_timesteps = n_timesteps;
    if (export_caches_to_host_) {
        cache_out.conformer_cnn_ne = { sess_->conf_cnn_cache->ne[0], sess_->conf_cnn_cache->ne[1],
                                       sess_->conf_cnn_cache->ne[2] };
        cache_out.conformer_att_ne = { sess_->conf_att_cache->ne[0], sess_->conf_att_cache->ne[1],
                                       sess_->conf_att_cache->ne[2], sess_->conf_att_cache->ne[3] };
        cache_out.estimator_cnn_ne = { sess_->est_cnn_cache->ne[0], sess_->est_cnn_cache->ne[1],
                                       sess_->est_cnn_cache->ne[2], sess_->est_cnn_cache->ne[3] };
        cache_out.estimator_att_ne = { sess_->est_att_cache->ne[0], sess_->est_att_cache->ne[1],
                                       sess_->est_att_cache->ne[2], sess_->est_att_cache->ne[3] };
        runner_read_tensor_bytes(loader_.backend(), sess_->conf_cnn_cache, cache_out.conformer_cnn_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->conf_att_cache, cache_out.conformer_att_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->est_cnn_cache, cache_out.estimator_cnn_cache);
        runner_read_tensor_bytes(loader_.backend(), sess_->est_att_cache, cache_out.estimator_att_cache);
    }
    return true;
}
bool flowGGUFModelRunner::init_from_host_caches(const flowStreamCacheHost & cache_host,
                                                const float *               spk_bc,
                                                int64_t                     B,
                                                int                         n_timesteps,
                                                float                       temperature) {
    if (!loader_.backend() || !loader_.model()) {
        return false;
    }
    if (!spk_bc || B <= 0) {
        return false;
    }
    if (cache_host.empty()) {
        LOG_ERROR( "flowGGUFModelRunner.init_from_host_caches: cache_host is empty\n");
        return false;
    }
    if (cache_host.n_timesteps != 0 && cache_host.n_timesteps != n_timesteps) {
        LOG_ERROR( "flowGGUFModelRunner.init_from_host_caches: n_timesteps mismatch (cache=%d, got=%d)\n",
                     cache_host.n_timesteps, n_timesteps);
        return false;
    }
    if (cache_host.conformer_cnn_ne.size() != 3 || cache_host.conformer_att_ne.size() != 4 ||
        cache_host.estimator_cnn_ne.size() != 4 || cache_host.estimator_att_ne.size() != 4) {
        LOG_ERROR( "flowGGUFModelRunner.init_from_host_caches: bad cache ne vectors\n");
        return false;
    }
    if ((int64_t) cache_host.estimator_cnn_ne[3] != (2 * B) || (int64_t) cache_host.estimator_att_ne[3] != (2 * B)) {
        LOG_ERROR(
                     "flowGGUFModelRunner.init_from_host_caches: estimator cache batch mismatch "
                     "(est_cnn_ne3=%lld est_att_ne3=%lld expected=%lld)\n",
                     (long long) cache_host.estimator_cnn_ne[3], (long long) cache_host.estimator_att_ne[3],
                     (long long) (2 * B));
        return false;
    }
    const int64_t T_chunk_token = 28;
    if (!sess_) {
        sess_ = std::make_unique<streamSession>();
    }
    const bool need_rebuild = sess_->ctx == nullptr || sess_->B != B || sess_->T_chunk_token != T_chunk_token ||
                              sess_->n_timesteps != n_timesteps;
    if (need_rebuild) {
        sess_->clear();
        sess_->B              = B;
        sess_->T_prompt_token = 0;
        sess_->T_prompt_mel   = 0;
        sess_->T_chunk_token  = T_chunk_token;
        sess_->n_timesteps    = n_timesteps;
        sess_->temperature    = temperature;
        if (std::shared_ptr<flow_matching::fmCausalConditionalCFM> dec = loader_.decoder()) {
            dec->reset_stream_state();
        }
        ggml_init_params p{};
        p.mem_size   = 2048ull * 1024ull * 1024ull;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        sess_->ctx   = ggml_init(p);
        if (!sess_->ctx) {
            sess_->clear();
            return false;
        }
        sess_->spk_cb = ggml_new_tensor_2d(sess_->ctx, GGML_TYPE_F32, 192, B);
        ggml_set_input(sess_->spk_cb);
        sess_->chunk_token_ids_tb = ggml_new_tensor_2d(sess_->ctx, GGML_TYPE_I32, T_chunk_token, B);
        ggml_set_input(sess_->chunk_token_ids_tb);
        sess_->conf_cnn_cache = ggml_new_tensor_3d(sess_->ctx, GGML_TYPE_F32, cache_host.conformer_cnn_ne[0],
                                                   cache_host.conformer_cnn_ne[1], cache_host.conformer_cnn_ne[2]);
        sess_->conf_att_cache = ggml_new_tensor_4d(sess_->ctx, GGML_TYPE_F32, cache_host.conformer_att_ne[0],
                                                   cache_host.conformer_att_ne[1], cache_host.conformer_att_ne[2],
                                                   cache_host.conformer_att_ne[3]);
        sess_->est_cnn_cache  = ggml_new_tensor_4d(sess_->ctx, GGML_TYPE_F32, cache_host.estimator_cnn_ne[0],
                                                   cache_host.estimator_cnn_ne[1], cache_host.estimator_cnn_ne[2],
                                                   cache_host.estimator_cnn_ne[3]);
        sess_->est_att_cache  = ggml_new_tensor_4d(sess_->ctx, GGML_TYPE_F32, cache_host.estimator_att_ne[0],
                                                   cache_host.estimator_att_ne[1], cache_host.estimator_att_ne[2],
                                                   cache_host.estimator_att_ne[3]);
        if (!sess_->conf_cnn_cache || !sess_->conf_att_cache || !sess_->est_cnn_cache || !sess_->est_att_cache) {
            sess_->clear();
            return false;
        }
        // Persistent state: do not let the graph allocator overwrite these buffers.
        ggml_set_output(sess_->conf_cnn_cache);
        ggml_set_output(sess_->conf_att_cache);
        ggml_set_output(sess_->est_cnn_cache);
        ggml_set_output(sess_->est_att_cache);
        flow_matching::fmCFMCache est_in;
        est_in.clear();
        est_in.cnn_cache = sess_->est_cnn_cache;
        est_in.att_cache = sess_->est_att_cache;
        flow_matching::fmCFMCache est_out_nonlast;
        est_out_nonlast.clear();
        auto out1 = loader_.model()->build_inference_chunk_graph(sess_->ctx, sess_->chunk_token_ids_tb, sess_->spk_cb,
                                                                 false, sess_->conf_cnn_cache, sess_->conf_att_cache,
                                                                 &est_in, n_timesteps, temperature, &est_out_nonlast);
        if (!out1.feat_ctb || !out1.conformer_cnn_cache || !out1.conformer_att_cache || !est_out_nonlast.cnn_cache ||
            !est_out_nonlast.att_cache) {
            sess_->clear();
            return false;
        }
        sess_->out_feat_nonlast_ctb = out1.feat_ctb;
        sess_->call_id_nonlast      = 0;
        const int64_t L_conf_att = sess_->conf_att_cache->ne[1];
        const int64_t L_est_att  = sess_->est_att_cache->ne[1];
        const int64_t delta      = out1.feat_ctb->ne[1];
        ggml_tensor * out1_conf_att_trim = out1.conformer_att_cache;
        if (out1.conformer_att_cache->ne[1] > L_conf_att) {
            out1_conf_att_trim = runner_slice_time_dim1_4d(sess_->ctx, out1.conformer_att_cache, delta, L_conf_att);
        }
        ggml_tensor * out1_est_att_trim = est_out_nonlast.att_cache;
        if (est_out_nonlast.att_cache->ne[1] > L_est_att) {
            out1_est_att_trim = runner_slice_time_dim1_4d(sess_->ctx, est_out_nonlast.att_cache, delta, L_est_att);
        }
        ggml_tensor * cpy_conf_cnn_1 = ggml_cpy(sess_->ctx, out1.conformer_cnn_cache, sess_->conf_cnn_cache);
        ggml_tensor * cpy_conf_att_1 = ggml_cpy(sess_->ctx, out1_conf_att_trim, sess_->conf_att_cache);
        ggml_tensor * cpy_est_cnn_1  = ggml_cpy(sess_->ctx, est_out_nonlast.cnn_cache, sess_->est_cnn_cache);
        ggml_tensor * cpy_est_att_1  = ggml_cpy(sess_->ctx, out1_est_att_trim, sess_->est_att_cache);
        sess_->gf_nonlast = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(sess_->gf_nonlast, sess_->out_feat_nonlast_ctb);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_conf_cnn_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_conf_att_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_est_cnn_1);
        ggml_build_forward_expand(sess_->gf_nonlast, cpy_est_att_1);
        flow_matching::fmCFMCache est_out_last;
        est_out_last.clear();
        auto out2 = loader_.model()->build_inference_chunk_graph(sess_->ctx, sess_->chunk_token_ids_tb, sess_->spk_cb,
                                                                 true, sess_->conf_cnn_cache, sess_->conf_att_cache,
                                                                 &est_in, n_timesteps, temperature, &est_out_last);
        if (!out2.feat_ctb || !out2.conformer_cnn_cache || !out2.conformer_att_cache || !est_out_last.cnn_cache ||
            !est_out_last.att_cache) {
            sess_->clear();
            return false;
        }
        sess_->out_feat_last_ctb = out2.feat_ctb;
        sess_->call_id_last      = 1;
        const int64_t delta2             = out2.feat_ctb->ne[1];
        ggml_tensor * out2_conf_att_trim = out2.conformer_att_cache;
        if (out2.conformer_att_cache->ne[1] > L_conf_att) {
            out2_conf_att_trim = runner_slice_time_dim1_4d(sess_->ctx, out2.conformer_att_cache, delta2, L_conf_att);
        }
        ggml_tensor * out2_est_att_trim = est_out_last.att_cache;
        if (est_out_last.att_cache->ne[1] > L_est_att) {
            out2_est_att_trim = runner_slice_time_dim1_4d(sess_->ctx, est_out_last.att_cache, delta2, L_est_att);
        }
        ggml_tensor * cpy_conf_cnn_2 = ggml_cpy(sess_->ctx, out2.conformer_cnn_cache, sess_->conf_cnn_cache);
        ggml_tensor * cpy_conf_att_2 = ggml_cpy(sess_->ctx, out2_conf_att_trim, sess_->conf_att_cache);
        ggml_tensor * cpy_est_cnn_2  = ggml_cpy(sess_->ctx, est_out_last.cnn_cache, sess_->est_cnn_cache);
        ggml_tensor * cpy_est_att_2  = ggml_cpy(sess_->ctx, out2_est_att_trim, sess_->est_att_cache);
        sess_->gf_last = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(sess_->gf_last, sess_->out_feat_last_ctb);
        ggml_build_forward_expand(sess_->gf_last, cpy_conf_cnn_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_conf_att_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_est_cnn_2);
        ggml_build_forward_expand(sess_->gf_last, cpy_est_att_2);

        ggml_cgraph * gf_alloc = ggml_new_graph_custom(sess_->ctx, GGML_DEFAULT_GRAPH_SIZE * 1024, false);
        ggml_build_forward_expand(gf_alloc, sess_->out_feat_nonlast_ctb);
        ggml_build_forward_expand(gf_alloc, cpy_conf_cnn_1);
        ggml_build_forward_expand(gf_alloc, cpy_conf_att_1);
        ggml_build_forward_expand(gf_alloc, cpy_est_cnn_1);
        ggml_build_forward_expand(gf_alloc, cpy_est_att_1);
        ggml_build_forward_expand(gf_alloc, sess_->out_feat_last_ctb);
        ggml_build_forward_expand(gf_alloc, cpy_conf_cnn_2);
        ggml_build_forward_expand(gf_alloc, cpy_conf_att_2);
        ggml_build_forward_expand(gf_alloc, cpy_est_cnn_2);
        ggml_build_forward_expand(gf_alloc, cpy_est_att_2);

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(loader_.backend());
        sess_->galloc = ggml_gallocr_new(buft);
        if (!sess_->galloc) {
            sess_->clear();
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sess_->galloc, gf_alloc)) {
            sess_->clear();
            return false;
        }
    }
    std::vector<float> spk_cb;
    runner_bc_to_cb(spk_bc, B, 192, spk_cb);
    backend_tensor_set(loader_.backend(), sess_->spk_cb, spk_cb.data(), spk_cb.size() * sizeof(float));
    backend_tensor_set(loader_.backend(), sess_->conf_cnn_cache, cache_host.conformer_cnn_cache.data(),
                                     cache_host.conformer_cnn_cache.size());
    backend_tensor_set(loader_.backend(), sess_->conf_att_cache, cache_host.conformer_att_cache.data(),
                                     cache_host.conformer_att_cache.size());
    backend_tensor_set(loader_.backend(), sess_->est_cnn_cache, cache_host.estimator_cnn_cache.data(),
                                     cache_host.estimator_cnn_cache.size());
    backend_tensor_set(loader_.backend(), sess_->est_att_cache, cache_host.estimator_att_cache.data(),
                                     cache_host.estimator_att_cache.size());
    if (runner_backend_is_device(loader_.backend())) {
        ggml_backend_synchronize(loader_.backend());
    }
    if (runner_backend_is_device(loader_.backend())) {
        const int32_t        pad_token = 4218;
        std::vector<int32_t> token_tb((size_t) T_chunk_token * (size_t) B, pad_token);
        backend_tensor_set(loader_.backend(), sess_->chunk_token_ids_tb, token_tb.data(),
                                         token_tb.size() * sizeof(int32_t));
        runner_feed_enc_stream_pos(loader_.backend(), sess_->ctx, loader_.encoder());
        {
            const int     call_id      = sess_->call_id_nonlast;
            const int64_t last_att_len = sess_->est_att_cache ? sess_->est_att_cache->ne[1] : 0;
            ggml_tensor * feat         = sess_->out_feat_nonlast_ctb;
            const int64_t C            = feat ? feat->ne[0] : 80;
            const int64_t T            = feat ? feat->ne[1] : 1;
            runner_feed_cfm_noise_ts(loader_.backend(), sess_->ctx, call_id, n_timesteps, temperature, last_att_len,
                                         C, T, B);
        }
        (void) ggml_backend_graph_compute(loader_.backend(), sess_->gf_nonlast);
        ggml_backend_synchronize(loader_.backend());
        backend_tensor_set(loader_.backend(), sess_->conf_cnn_cache,
                                         cache_host.conformer_cnn_cache.data(), cache_host.conformer_cnn_cache.size());
        backend_tensor_set(loader_.backend(), sess_->conf_att_cache,
                                         cache_host.conformer_att_cache.data(), cache_host.conformer_att_cache.size());
        backend_tensor_set(loader_.backend(), sess_->est_cnn_cache, cache_host.estimator_cnn_cache.data(),
                                         cache_host.estimator_cnn_cache.size());
        backend_tensor_set(loader_.backend(), sess_->est_att_cache, cache_host.estimator_att_cache.data(),
                                         cache_host.estimator_att_cache.size());
        ggml_backend_synchronize(loader_.backend());
    }
    return true;
}

}  // namespace flow
}  // namespace omni

namespace omni {
namespace flow {

namespace {

bool t2m_read_bin(const std::string & path, std::vector<uint8_t> & out) {
    out.clear();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff n = ifs.tellg();
    if (n < 0) {
        return false;
    }
    out.resize((size_t) n);
    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(reinterpret_cast<char *>(out.data()), n)) {
        return false;
    }
    return true;
}

bool t2m_read_gguf_tensor_bytes(const std::string &                             gguf_path,
                            const std::unordered_map<std::string, size_t> & offsets,
                            const std::string &                             name,
                            size_t                                          nbytes,
                            std::vector<uint8_t> &                          out) {
    out.clear();
    const auto it = offsets.find(name);
    if (it == offsets.end()) {
        return false;
    }
    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        return false;
    }
    out.resize(nbytes);
    fin.seekg((std::streamoff) it->second, std::ios::beg);
    if (!fin.read(reinterpret_cast<char *>(out.data()), (std::streamsize) nbytes)) {
        out.clear();
        return false;
    }
    return true;
}

bool t2m_load_prompt_cache_gguf(const std::string &               gguf_path,
                            omni::flow::flowStreamCacheHost & cache_out,
                            std::vector<float> &              spk_bc_out,
                            int64_t &                         B_out,
                            int &                             n_timesteps_out,
                            float &                           temperature_out) {
    // 用于从prompt_cache.gguf加载spk与流式缓存张量
    cache_out.clear();
    spk_bc_out.clear();
    B_out           = 0;
    n_timesteps_out = 0;
    temperature_out = 1.0f;

    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf || !meta) {
        if (ctx_gguf) {
            gguf_free(ctx_gguf);
        }
        if (meta) {
            ggml_free(meta);
        }
        return false;
    }

    {  // 读取gguf中的n_timesteps/temperature元信息
        const int64_t k_nt = gguf_find_key(ctx_gguf, "mtmd.prompt_cache.n_timesteps");
        if (k_nt >= 0) {
            n_timesteps_out = (int) gguf_get_val_i32(ctx_gguf, k_nt);
        }
        const int64_t k_tmp = gguf_find_key(ctx_gguf, "mtmd.prompt_cache.temperature");
        if (k_tmp >= 0) {
            temperature_out = gguf_get_val_f32(ctx_gguf, k_tmp);
        }
    }

    // 构建张量名到文件偏移的索引表
    const int64_t                           n_tensors = gguf_get_n_tensors(ctx_gguf);
    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve((size_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (!name) {
            continue;
        }
        const size_t off = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        offsets.emplace(std::string(name), off);
    }

    auto load_tensor = [&](const char * name, std::vector<uint8_t> & bytes_out, std::vector<int64_t> & ne_out) -> bool {
        ggml_tensor * meta_t = ggml_get_tensor(meta, name);
        if (!meta_t) {
            return false;
        }
        const size_t nbytes = ggml_nbytes(meta_t);
        if (!t2m_read_gguf_tensor_bytes(gguf_path, offsets, std::string(name), nbytes, bytes_out)) {
            return false;
        }
        ne_out.clear();

        for (int i = 0; i < 4; ++i) {
            ne_out.push_back(meta_t->ne[i]);
        }
        return true;
    };

    {  // 加载spk向量(gguf内为CB布局)
        std::vector<uint8_t> spk_bytes;
        std::vector<int64_t> spk_ne;
        if (!load_tensor("prompt_cache.spk_cb", spk_bytes, spk_ne)) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        if (spk_ne.size() < 2 || spk_ne[0] != 192) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        const int64_t B = spk_ne[1];
        B_out           = B;
        spk_bc_out.resize((size_t) B * (size_t) 192);
        std::memcpy(spk_bc_out.data(), spk_bytes.data(), spk_bytes.size());
    }

    {  // 加载conformer/estimator缓存张量
        std::vector<uint8_t> bytes;
        std::vector<int64_t> ne;

        if (!load_tensor("prompt_cache.conformer_cnn_cache", cache_out.conformer_cnn_cache, ne)) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        cache_out.conformer_cnn_ne = { ne[0], ne[1], ne[2] };

        if (!load_tensor("prompt_cache.conformer_att_cache", cache_out.conformer_att_cache, ne)) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        cache_out.conformer_att_ne = { ne[0], ne[1], ne[2], ne[3] };

        if (!load_tensor("prompt_cache.estimator_cnn_cache", cache_out.estimator_cnn_cache, ne)) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        cache_out.estimator_cnn_ne = { ne[0], ne[1], ne[2], ne[3] };

        if (!load_tensor("prompt_cache.estimator_att_cache", cache_out.estimator_att_cache, ne)) {
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }
        cache_out.estimator_att_ne = { ne[0], ne[1], ne[2], ne[3] };
    }

    cache_out.n_timesteps = n_timesteps_out;

    gguf_free(ctx_gguf);
    ggml_free(meta);
    return true;
}

}  // namespace

bool Token2Mel::load_model(const std::string & encoder_gguf,
                           const std::string & flow_matching_gguf,
                           const std::string & flow_extra_gguf,
                           const std::string & device,
                           int                 threads) {
    // 用于加载三段GGUF并初始化runner(失败时回退到cpu)
    reset_stream();
    runner_.set_num_threads(threads);
    if (!runner_.load_from_gguf(encoder_gguf, flow_matching_gguf, flow_extra_gguf, device)) {
        if (device != "cpu") {
            LOG_ERROR( "Token2Mel.load_model: load_from_gguf(%s) failed, fallback to cpu\n", device.c_str());
            if (!runner_.load_from_gguf(encoder_gguf, flow_matching_gguf, flow_extra_gguf, "cpu")) {
                LOG_ERROR( "Token2Mel.load_model: fallback cpu failed\n");
                model_loaded_ = false;
                return false;
            }
        } else {
            model_loaded_ = false;
            return false;
        }
    }

    runner_.set_export_caches_to_host(false);

    model_loaded_   = true;
    stream_started_ = false;
    return true;
}

bool Token2Mel::load_prompt_bundle_dir(const std::string & dir, PromptBundle & out) {
    out   = PromptBundle();
    out.B = 1;

    std::vector<uint8_t> buf;

    if (!t2m_read_bin(dir + "/spk_f32.bin", buf) || buf.size() != (size_t) kSpkDim * sizeof(float)) {
        LOG_ERROR( "Token2Mel.load_prompt_bundle_dir: failed to read spk_f32.bin from %s\n", dir.c_str());
        return false;
    }
    out.spk_bc.resize((size_t) kSpkDim);
    std::memcpy(out.spk_bc.data(), buf.data(), buf.size());

    if (!t2m_read_bin(dir + "/prompt_tokens_i32.bin", buf) || buf.size() % sizeof(int32_t) != 0) {
        LOG_ERROR( "Token2Mel.load_prompt_bundle_dir: failed to read prompt_tokens_i32.bin from %s\n",
                     dir.c_str());
        return false;
    }
    out.prompt_tokens_bt.resize(buf.size() / sizeof(int32_t));
    std::memcpy(out.prompt_tokens_bt.data(), buf.data(), buf.size());
    out.T_prompt_token = (int64_t) out.prompt_tokens_bt.size();

    if (!t2m_read_bin(dir + "/prompt_mel_btc_f32.bin", buf) || buf.size() % sizeof(float) != 0) {
        LOG_ERROR( "Token2Mel.load_prompt_bundle_dir: failed to read prompt_mel_btc_f32.bin from %s\n",
                     dir.c_str());
        return false;
    }
    const size_t n_f32 = buf.size() / sizeof(float);
    if (n_f32 % (size_t) kMelChannels != 0) {
        LOG_ERROR( "Token2Mel.load_prompt_bundle_dir: prompt mel size not divisible by 80\n");
        return false;
    }
    out.prompt_mel_btc.resize(n_f32);
    std::memcpy(out.prompt_mel_btc.data(), buf.data(), buf.size());
    out.T_prompt_mel = (int64_t) (n_f32 / (size_t) kMelChannels);

    constexpr int32_t up_rate = 2;
    if (out.T_prompt_token <= kPreLookahead) {
        LOG_ERROR( "Token2Mel.load_prompt_bundle_dir: prompt token length too small\n");
        return false;
    }
    const int64_t expect_T_mel = (out.T_prompt_token - (int64_t) kPreLookahead) * (int64_t) up_rate;
    if (out.T_prompt_mel != expect_T_mel) {
        LOG_ERROR(
                     "Token2Mel.load_prompt_bundle_dir: prompt bundle shape mismatch: "
                     "T_token=%lld T_mel=%lld expect_T_mel=%lld\n",
                     (long long) out.T_prompt_token, (long long) out.T_prompt_mel, (long long) expect_T_mel);
        return false;
    }

    return true;
}

bool Token2Mel::start_stream_with_prompt(const PromptBundle & prompt, int n_timesteps, float temperature) {
    reset_stream();

    // 用于执行setup_cache并开始流式推理
    if (!model_loaded_) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: model not loaded\n");
        return false;
    }
    if (prompt.B != 1) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: only B=1 is supported (got %lld)\n",
                     (long long) prompt.B);
        return false;
    }
    if ((int64_t) prompt.spk_bc.size() != (int64_t) kSpkDim) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: bad spk size\n");
        return false;
    }
    if (prompt.T_prompt_token <= 0 || (int64_t) prompt.prompt_tokens_bt.size() != prompt.T_prompt_token) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: bad prompt token size\n");
        return false;
    }
    if (prompt.T_prompt_mel <= 0 ||
        (int64_t) prompt.prompt_mel_btc.size() != prompt.T_prompt_mel * (int64_t) kMelChannels) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: bad prompt mel size\n");
        return false;
    }

    n_timesteps_ = n_timesteps;
    temperature_ = temperature;
    spk_bc_      = prompt.spk_bc;

    // 临时开启 cache 导出以获取 host 数据
    runner_.set_export_caches_to_host(true);
    flowStreamCacheHost cache0;
    if (!runner_.setup_cache(prompt.prompt_tokens_bt.data(), 1, prompt.T_prompt_token, prompt.prompt_mel_btc.data(),
                             prompt.T_prompt_mel, kMelChannels, spk_bc_.data(), kSpkDim, n_timesteps_, temperature_,
                             cache0)) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt: runner.setup_cache failed\n");
        runner_.set_export_caches_to_host(false);
        return false;
    }
    runner_.set_export_caches_to_host(false);

    cache_in_       = cache0;
    stream_started_ = true;

    // 自动导出 prompt_cache.gguf (如果 cache 有效且同目录没有新版 GGUF)
    // 这样下次启动可以直接用 init_from_host_caches 加载，省去 setup_cache 实时计算
    if (!cache0.empty()) {
        // 导出 prompt_cache.gguf 到指定目录
        // 设置环境变量 T2W_EXPORT_CACHE_DIR 触发导出
        const char * export_dir = ::getenv("T2W_EXPORT_CACHE_DIR");
        if (export_dir && export_dir[0] != '\0') {
            std::string out_path = std::string(export_dir) + "/prompt_cache.gguf";
            std::ifstream probe(out_path);
            if (!probe.good()) {
                // 写 GGUF
                gguf_context * guf = gguf_init_empty();
                if (guf) {
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.version", 2);
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.n_timesteps", n_timesteps);
                    gguf_set_val_f32(guf, "mtmd.prompt_cache.temperature", temperature);
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.pre_lookahead", kPreLookahead);
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.chunk_main", kChunkMain);
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.chunk_total", kDt);
                    gguf_set_val_i32(guf, "mtmd.prompt_cache.up_rate", 2);

                    // spk_cb: 直接用 BC layout (就是 spk_bc_ 的数据)
                    ggml_init_params p = { 256 * 1024, nullptr, true };
                    ggml_context * tmp = ggml_init(p);
                    if (tmp) {
                        ggml_tensor * spk = ggml_new_tensor_1d(tmp, GGML_TYPE_F32, kSpkDim);
                        ggml_set_name(spk, "prompt_cache.spk_cb");
                        spk->data = (void *) spk_bc_.data();
                        gguf_add_tensor(guf, spk);

                        auto add_cache = [&](const char * name, const std::vector<uint8_t> & bytes,
                                             const std::vector<int64_t> & ne) {
                            int ndims = (int) ne.size();
                            ggml_tensor * t = nullptr;
                            if (ndims == 3)
                                t = ggml_new_tensor_3d(tmp, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
                            else if (ndims == 4)
                                t = ggml_new_tensor_4d(tmp, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
                            else
                                return;
                            ggml_set_name(t, name);
                            t->data = (void *) bytes.data();
                            gguf_add_tensor(guf, t);
                        };
                        add_cache("prompt_cache.conformer_cnn_cache", cache0.conformer_cnn_cache, cache0.conformer_cnn_ne);
                        add_cache("prompt_cache.conformer_att_cache", cache0.conformer_att_cache, cache0.conformer_att_ne);
                        add_cache("prompt_cache.estimator_cnn_cache", cache0.estimator_cnn_cache, cache0.estimator_cnn_ne);
                        add_cache("prompt_cache.estimator_att_cache", cache0.estimator_att_cache, cache0.estimator_att_ne);

                        if (gguf_write_to_file(guf, out_path.c_str(), false)) {
                            std::fprintf(stderr, "[Token2Mel] exported prompt_cache.gguf: %s\n", out_path.c_str());
                        }
                        ggml_free(tmp);
                    }
                    gguf_free(guf);
                }
            }
        }
    }

    return true;
}

bool Token2Mel::start_stream_with_prompt_cache_gguf(const std::string & prompt_cache_gguf_path,
                                                    int                 n_timesteps,
                                                    float               temperature) {
    // 用于从prompt_cache.gguf恢复缓存并初始化runner状态
    reset_stream();
    if (!model_loaded_) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt_cache_gguf: model not loaded\n");
        return false;
    }

    flowStreamCacheHost cache_host;
    std::vector<float>  spk_bc;
    int64_t             B         = 0;
    int                 n_ts_file = 0;
    float               temp_file = 1.0f;
    if (!t2m_load_prompt_cache_gguf(prompt_cache_gguf_path, cache_host, spk_bc, B, n_ts_file, temp_file)) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt_cache_gguf: failed to load cache gguf: %s\n",
                     prompt_cache_gguf_path.c_str());
        return false;
    }
    if (B != 1) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt_cache_gguf: only B=1 is supported (got %lld)\n",
                     (long long) B);
        return false;
    }

    const int   use_nt   = (n_timesteps > 0) ? n_timesteps : n_ts_file;
    const float use_temp = (temperature > 0.0f) ? temperature : temp_file;

    if (!runner_.init_from_host_caches(cache_host, spk_bc.data(), B, use_nt, use_temp)) {
        LOG_ERROR( "Token2Mel.start_stream_with_prompt_cache_gguf: runner.init_from_host_caches failed\n");
        return false;
    }

    n_timesteps_ = use_nt;
    temperature_ = use_temp;
    spk_bc_      = std::move(spk_bc);
    cache_in_.clear();
    stream_started_ = true;
    return true;
}

bool Token2Mel::ensure_ready_for_infer() const {
    // 用于检查模型与stream状态是否就绪
    if (!model_loaded_) {
        LOG_ERROR( "Token2Mel: model not loaded\n");
        return false;
    }
    if (!stream_started_) {
        LOG_ERROR( "Token2Mel: stream not started (call start_stream_with_prompt)\n");
        return false;
    }
    if ((int64_t) spk_bc_.size() != (int64_t) kSpkDim) {
        LOG_ERROR( "Token2Mel: missing speaker embedding\n");
        return false;
    }
    return true;
}

bool Token2Mel::infer_one_chunk(const std::vector<int32_t> & chunk_bt, bool last_chunk, std::vector<float> & mel_bct) {
    // 用于推理一个固定窗口(dt=28)并更新内部cache
    mel_bct.clear();
    if (!ensure_ready_for_infer()) {
        return false;
    }
    if ((int64_t) chunk_bt.size() != (int64_t) kDt) {
        LOG_ERROR( "Token2Mel.infer_one_chunk: expected dt=%d tokens, got %lld\n", (int) kDt,
                     (long long) chunk_bt.size());
        return false;
    }

    flowStreamCacheHost cache_out;
    if (!runner_.inference_chunk(chunk_bt.data(), 1, kDt, spk_bc_.data(), kSpkDim, last_chunk, cache_in_, n_timesteps_,
                                 temperature_, mel_bct, cache_out)) {
        LOG_ERROR( "Token2Mel.infer_one_chunk: runner.inference_chunk failed\n");
        return false;
    }

    cache_in_ = cache_out;
    return true;
}

void Token2Mel::append_bct_along_time(const std::vector<float> & src_bct,
                                      int64_t                    B,
                                      int64_t                    C,
                                      std::vector<float> &       dst_bct_inout) {
    if (B <= 0 || C <= 0) {
        return;
    }
    if (src_bct.empty()) {
        return;
    }

    const int64_t src_T = (int64_t) src_bct.size() / (B * C);
    if (src_T <= 0) {
        return;
    }

    const int64_t dst_T = dst_bct_inout.empty() ? 0 : (int64_t) dst_bct_inout.size() / (B * C);

    std::vector<float> next((size_t) (B * C * (dst_T + src_T)));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            float * dst = next.data() + (size_t) b * (size_t) C * (size_t) (dst_T + src_T) +
                          (size_t) c * (size_t) (dst_T + src_T);
            if (!dst_bct_inout.empty()) {
                const float * src0 =
                    dst_bct_inout.data() + (size_t) b * (size_t) C * (size_t) dst_T + (size_t) c * (size_t) dst_T;
                std::memcpy(dst, src0, (size_t) dst_T * sizeof(float));
            }
            const float * src1 =
                src_bct.data() + (size_t) b * (size_t) C * (size_t) src_T + (size_t) c * (size_t) src_T;
            std::memcpy(dst + (size_t) dst_T, src1, (size_t) src_T * sizeof(float));
        }
    }

    dst_bct_inout.swap(next);
}

bool Token2Mel::push_tokens(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & mel_bct_out) {
    // 用于将tokens补齐并推理一次(必要时按n_tokens裁剪输出T)
    mel_bct_out.clear();
    if (!ensure_ready_for_infer()) {
        return false;
    }

    if (n_tokens < 0 || n_tokens > kDt) {
        LOG_ERROR( "Token2Mel.push_tokens: expected 0 < n_tokens <= %d, got %lld\n", (int) kDt,
                     (long long) n_tokens);
        return false;
    }
    if (n_tokens == 0) {
        return true;
    }

    std::vector<int32_t> chunk_bt((size_t) kDt, kPadToken);
    if (tokens && n_tokens > 0) {
        std::memcpy(chunk_bt.data(), tokens, (size_t) n_tokens * sizeof(int32_t));
    }

    std::vector<float> mel_chunk_bct;
    if (!infer_one_chunk(chunk_bt, is_final, mel_chunk_bct)) {
        return false;
    }

    if (n_tokens > 0) {
        const int64_t B           = 1;
        const int64_t C           = kMelChannels;
        const int64_t T           = (int64_t) mel_chunk_bct.size() / (B * C);
        const int64_t max_valid_T = n_tokens * 2;
        const int64_t valid_T     = std::min<int64_t>(T, max_valid_T);
        if (valid_T < T && valid_T > 0) {
            std::vector<float> cropped((size_t) B * (size_t) C * (size_t) valid_T);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t c = 0; c < C; ++c) {
                    const float * src =
                        mel_chunk_bct.data() + (size_t) b * (size_t) C * (size_t) T + (size_t) c * (size_t) T;
                    float * dst =
                        cropped.data() + (size_t) b * (size_t) C * (size_t) valid_T + (size_t) c * (size_t) valid_T;
                    std::memcpy(dst, src, (size_t) valid_T * sizeof(float));
                }
            }
            mel_chunk_bct.swap(cropped);
        }
    }

    append_bct_along_time(mel_chunk_bct, 1, kMelChannels, mel_bct_out);

    return true;
}

void Token2Mel::reset_stream() {
    runner_.reset_stream();
    stream_started_ = false;
    spk_bc_.clear();
    cache_in_.clear();
}

bool Token2MelSession::init_from_prompt_bundle(const std::string & encoder_gguf,
                                               const std::string & flow_matching_gguf,
                                               const std::string & flow_extra_gguf,
                                               const std::string & device,
                                               int                 threads,
                                               const std::string & prompt_bundle_dir,
                                               int                 n_timesteps,
                                               float               temperature) {
    // 初始化会话方式1：读取prompt_bundle并执行setup_cache，用于后续feed_*流式推理。
    reset();
    if (!t2m.load_model(encoder_gguf, flow_matching_gguf, flow_extra_gguf, device, threads)) {
        return false;
    }

    Token2Mel::PromptBundle pb;
    if (!Token2Mel::load_prompt_bundle_dir(prompt_bundle_dir, pb)) {
        return false;
    }

    if (!t2m.start_stream_with_prompt(pb, n_timesteps, temperature)) {
        return false;
    }

    return true;
}

bool Token2MelSession::init_from_prompt_cache_gguf(const std::string & encoder_gguf,
                                                   const std::string & flow_matching_gguf,
                                                   const std::string & flow_extra_gguf,
                                                   const std::string & device,
                                                   int                 threads,
                                                   const std::string & prompt_cache_gguf_path,
                                                   int                 n_timesteps,
                                                   float               temperature) {
    // 初始化会话方式2（从方式1或方式2选自其中一个即可，推荐方式2直接加载gguf）：从prompt_cache.gguf恢复缓存并开始流式推理，跳过prompt_bundle与setup_cache计算。
    reset();
    if (!t2m.load_model(encoder_gguf, flow_matching_gguf, flow_extra_gguf, device, threads)) {
        return false;
    }
    if (!t2m.start_stream_with_prompt_cache_gguf(prompt_cache_gguf_path, n_timesteps, temperature)) {
        return false;
    }
    return true;
}

bool Token2MelSession::feed_tokens(const int32_t *      tokens,
                                   int64_t              n_tokens,
                                   bool                 is_final,
                                   std::vector<float> & mel_bct_out) {
    // 流式接口（如果难以确定输入）：追加tokens并按25外加prelook-token窗口推理，输出mel为BCT(f32)且累积多段。
    mel_bct_out.clear();
    if (tokens && n_tokens > 0) {
        pending_.insert(pending_.end(), tokens, tokens + n_tokens);
    }

    while ((int64_t) pending_.size() >= Token2Mel::kDt) {
        std::vector<int32_t> window(pending_.begin(), pending_.begin() + Token2Mel::kDt);
        std::vector<float>   mel_call;
        if (!t2m.push_tokens(window, false, mel_call)) {
            return false;
        }
        Token2Mel::append_bct_along_time(mel_call, 1, Token2Mel::kMelChannels, mel_bct_out);

        pending_.erase(pending_.begin(), pending_.begin() + Token2Mel::kChunkMain);
    }

    if (is_final) {
        if (!pending_.empty()) {
            std::vector<float> mel_call;
            if (!t2m.push_tokens(pending_.data(), (int64_t) pending_.size(), true, mel_call)) {
                return false;
            }
            Token2Mel::append_bct_along_time(mel_call, 1, Token2Mel::kMelChannels, mel_bct_out);
            pending_.clear();
        } else {
        }
    }

    return true;
}

bool Token2MelSession::feed_window(const int32_t *      tokens,
                                   int64_t              n_tokens,
                                   bool                 is_final,
                                   std::vector<float> & mel_bct_out) {
    // 窗口接口（预期是外部输入和内部消费逻辑一致，外面传几个，内部使用几个）：外部提供<=25 tokens + prelook，每次调用只推理一次并返回该次mel（已完成读入python前转置）(BCT,f32)
    mel_bct_out.clear();

    return t2m.push_tokens(tokens, n_tokens, is_final, mel_bct_out);
}

void Token2MelSession::reset() {
    pending_.clear();
    t2m.reset_stream();
}

namespace token2wav_utils {

void append_bt_along_time_b1(const std::vector<float> & src_bt, std::vector<float> & dst_bt_inout) {
    if (src_bt.empty()) {
        return;
    }
    const size_t dst_n = dst_bt_inout.size();
    dst_bt_inout.resize(dst_n + src_bt.size());
    std::memcpy(dst_bt_inout.data() + dst_n, src_bt.data(), src_bt.size() * sizeof(float));
}

void crop_bct_tail_b1(const std::vector<float> & in_bct,
                      int64_t                    C,
                      int64_t                    T,
                      int64_t                    keep_T,
                      std::vector<float> &       out_bct) {
    out_bct.clear();
    if (C <= 0 || T <= 0 || keep_T <= 0) {
        return;
    }
    keep_T = std::min<int64_t>(keep_T, T);
    if ((int64_t) in_bct.size() != C * T) {
        return;
    }
    out_bct.resize((size_t) C * (size_t) keep_T);
    for (int64_t c = 0; c < C; ++c) {
        const float * src = in_bct.data() + (size_t) c * (size_t) T + (size_t) (T - keep_T);
        float *       dst = out_bct.data() + (size_t) c * (size_t) keep_T;
        std::memcpy(dst, src, (size_t) keep_T * sizeof(float));
    }
}

void crop_t_tail_b1(const std::vector<float> & in_bt, int64_t keep_T, std::vector<float> & out_bt) {
    out_bt.clear();
    if (keep_T <= 0 || in_bt.empty()) {
        return;
    }
    const int64_t T = (int64_t) in_bt.size();
    keep_T          = std::min<int64_t>(keep_T, T);
    out_bt.resize((size_t) keep_T);
    const float * src = in_bt.data() + (size_t) (T - keep_T);
    std::memcpy(out_bt.data(), src, (size_t) keep_T * sizeof(float));
}

void ensure_hamming_window_2n(int64_t n, std::vector<float> & window_out) {
    // Hamming window for overlap-add crossfade.
    window_out.clear();
    if (n <= 0) {
        return;
    }
    const int64_t N     = 2 * n;
    const double  denom = (N > 1) ? (double) (N - 1) : 1.0;
    const double  pi    = std::acos(-1.0);

    window_out.resize((size_t) N);
    for (int64_t i = 0; i < N; ++i) {
        const double a         = 2.0 * pi * (double) i / denom;
        window_out[(size_t) i] = (float) (0.54 - 0.46 * std::cos(a));
    }
}

void fade_in_out_b1(std::vector<float> &       wave_inout,
                    const std::vector<float> & prev_tail,
                    const std::vector<float> & window_2n,
                    int64_t                    n) {
    if (n <= 0) {
        return;
    }
    if ((int64_t) prev_tail.size() < n || (int64_t) wave_inout.size() < n || (int64_t) window_2n.size() < 2 * n) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        wave_inout[(size_t) i] =
            wave_inout[(size_t) i] * window_2n[(size_t) i] + prev_tail[(size_t) i] * window_2n[(size_t) (i + n)];
    }
}

}  // namespace token2wav_utils

bool Token2Wav::load_models(const std::string & encoder_gguf,
                            const std::string & flow_matching_gguf,
                            const std::string & flow_extra_gguf,
                            const std::string & vocoder_gguf,
                            const std::string & device_token2mel,
                            const std::string & device_vocoder) {
    reset_stream();

    constexpr int kDefaultThreads = 8;
    if (!t2m_.load_model(encoder_gguf, flow_matching_gguf, flow_extra_gguf, device_token2mel, kDefaultThreads)) {
        LOG_ERROR( "Token2Wav.load_models: Token2Mel.load_model failed\n");
        models_loaded_ = false;
        return false;
    }
    if (!voc_model_.voc_hg2_model_init_from_gguf(vocoder_gguf, device_vocoder, kDefaultThreads)) {
        LOG_ERROR( "Token2Wav.load_models: voc_hg2_model_init_from_gguf failed\n");
        models_loaded_ = false;
        return false;
    }

    voc_runner_.model = &voc_model_;
    models_loaded_    = true;
    return true;
}

bool Token2Wav::start_stream_with_prompt_cache_gguf(const std::string & prompt_cache_gguf_path,
                                                    int                 n_timesteps,
                                                    float               temperature) {
    voc_mel_cache_bct_.clear();
    voc_cache_source_bt1_.clear();
    voc_Tc_ = 0;
    voc_speech_cache_bt_.clear();
    token2wav_utils::ensure_hamming_window_2n((int64_t) kSourceCacheLen, voc_speech_window_);

    if (!models_loaded_) {
        LOG_ERROR( "Token2Wav.start_stream_with_prompt_cache_gguf: models not loaded\n");
        return false;
    }

    if (!t2m_.start_stream_with_prompt_cache_gguf(prompt_cache_gguf_path, n_timesteps, temperature)) {
        LOG_ERROR(
            "Token2Wav.start_stream_with_prompt_cache_gguf: Token2Mel.start_stream_with_prompt_cache_gguf failed\n");
        return false;
    }
    return true;
}

bool Token2Wav::start_stream_with_prompt(const Token2Mel::PromptBundle & prompt, int n_timesteps, float temperature) {
    voc_mel_cache_bct_.clear();
    voc_cache_source_bt1_.clear();
    voc_Tc_ = 0;
    voc_speech_cache_bt_.clear();
    token2wav_utils::ensure_hamming_window_2n((int64_t) kSourceCacheLen, voc_speech_window_);

    if (!models_loaded_) {
        LOG_ERROR( "Token2Wav.start_stream_with_prompt: models not loaded\n");
        return false;
    }

    if (!t2m_.start_stream_with_prompt(prompt, n_timesteps, temperature)) {
        LOG_ERROR( "Token2Wav.start_stream_with_prompt: Token2Mel.start_stream_with_prompt failed\n");
        return false;
    }
    return true;
}

bool Token2Wav::push_tokens_window(const int32_t *      tokens,
                                   int64_t              n_tokens,
                                   bool                 is_final,
                                   std::vector<float> & wave_bt_out,
                                   int64_t &            out_T_audio) {
    wave_bt_out.clear();
    out_T_audio = 0;

    if (!models_loaded_) {
        LOG_ERROR( "Token2Wav.push_tokens_window: models not loaded\n");
        return false;
    }

    if (n_tokens < 0 || n_tokens > Token2Mel::kDt) {
        LOG_ERROR( "Token2Wav.push_tokens_window: expected 0 <= n_tokens <= %d, got %lld\n",
                     (int) Token2Mel::kDt, (long long) n_tokens);
        return false;
    }

    using clock = std::chrono::steady_clock;
    const auto t_total0 = clock::now();

    std::vector<float> mel_bct;
    const auto t_t2m0 = clock::now();
    if (!t2m_.push_tokens(tokens, n_tokens, is_final, mel_bct)) {
        LOG_ERROR( "Token2Wav.push_tokens_window: Token2Mel.push_tokens failed\n");
        return false;
    }
    const auto t_t2m1 = clock::now();

    if (mel_bct.empty()) {
        const double t2m_ms =
            std::chrono::duration<double, std::milli>(t_t2m1 - t_t2m0).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(clock::now() - t_total0).count();
        static thread_local int64_t call_id = 0;
        const int64_t cid = call_id++;
        std::fprintf(stderr,
                     "[timing] call=%lld tokens=%lld final=%d token2mel=%.3fms vocoder=%.3fms total=%.3fms\n",
                     (long long) cid, (long long) n_tokens, (int) is_final, t2m_ms, 0.0, total_ms);
        return true;
    }
    if (mel_bct.size() % (size_t) Token2Mel::kMelChannels != 0) {
        LOG_ERROR( "Token2Wav.push_tokens_window: invalid mel size (not divisible by 80)\n");
        return false;
    }

    std::vector<float> mel_in_bct = voc_mel_cache_bct_;
    Token2Mel::append_bct_along_time(mel_bct, 1, Token2Mel::kMelChannels, mel_in_bct);

    const int64_t T_mel = (int64_t) mel_in_bct.size() / (int64_t) Token2Mel::kMelChannels;

    std::vector<float> out_source_bt1;
    int64_t            out_T_source = 0;
    const auto t_voc0 = clock::now();
    if (!voc_runner_.voc_hg2_runner_eval_stream(mel_in_bct, T_mel, voc_cache_source_bt1_, voc_Tc_, wave_bt_out,
                                                out_T_audio, out_source_bt1, out_T_source)) {
        LOG_ERROR( "Token2Wav.push_tokens_window: voc_hg2_runner_eval_stream failed\n");
        return false;
    }
    const auto t_voc1 = clock::now();

    if (!voc_speech_cache_bt_.empty()) {
        token2wav_utils::fade_in_out_b1(wave_bt_out, voc_speech_cache_bt_, voc_speech_window_, (int64_t) kSourceCacheLen);
    }

    {
        const int64_t      C       = Token2Mel::kMelChannels;
        const int64_t      T_total = (int64_t) mel_in_bct.size() / C;
        std::vector<float> next_mel_cache;
        token2wav_utils::crop_bct_tail_b1(mel_in_bct, C, T_total, (int64_t) kMelCacheLen, next_mel_cache);
        voc_mel_cache_bct_.swap(next_mel_cache);
    }

    {
        std::vector<float> next_source_cache;
        token2wav_utils::crop_t_tail_b1(out_source_bt1, (int64_t) kSourceCacheLen, next_source_cache);
        voc_cache_source_bt1_.swap(next_source_cache);
        voc_Tc_ = (int64_t) voc_cache_source_bt1_.size();
    }

    {
        std::vector<float> next_speech_cache;
        token2wav_utils::crop_t_tail_b1(wave_bt_out, (int64_t) kSourceCacheLen, next_speech_cache);
        voc_speech_cache_bt_.swap(next_speech_cache);
    }

    if (!is_final && (int64_t) wave_bt_out.size() > (int64_t) kSourceCacheLen) {
        wave_bt_out.resize(wave_bt_out.size() - (size_t) kSourceCacheLen);
        out_T_audio = (int64_t) wave_bt_out.size();
    }

    const double t2m_ms =
        std::chrono::duration<double, std::milli>(t_t2m1 - t_t2m0).count();
    const double voc_ms =
        std::chrono::duration<double, std::milli>(t_voc1 - t_voc0).count();
    const double total_ms =
        std::chrono::duration<double, std::milli>(clock::now() - t_total0).count();
    static thread_local int64_t call_id = 0;
    const int64_t cid = call_id++;
    std::fprintf(stderr,
                 "[timing] call=%lld tokens=%lld final=%d token2mel=%.3fms vocoder=%.3fms total=%.3fms audio=%lld\n",
                 (long long) cid, (long long) n_tokens, (int) is_final, t2m_ms, voc_ms, total_ms,
                 (long long) out_T_audio);

    return true;
}

void Token2Wav::reset_stream() {
    t2m_.reset_stream();
    voc_mel_cache_bct_.clear();
    voc_cache_source_bt1_.clear();
    voc_Tc_ = 0;
    voc_speech_cache_bt_.clear();
    voc_speech_window_.clear();
}

}  // namespace flow
}  // namespace omni
