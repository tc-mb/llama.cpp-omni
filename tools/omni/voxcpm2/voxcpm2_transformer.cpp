#include "voxcpm2_transformer.h"

#include "log.h"

#define GGML_KQ_MASK_PAD 256

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr float kCfgPairMaskNeg = -1.0e9f;

static bool starts_with_any(const char * name, const std::vector<std::string> & prefixes) {
    if (!name) {
        return false;
    }
    for (const std::string & prefix : prefixes) {
        if (std::strncmp(name, prefix.c_str(), prefix.size()) == 0) {
            return true;
        }
    }
    return false;
}

static ggml_tensor * reshape_bias_for(ggml_context * ctx, ggml_tensor * bias, ggml_tensor * output) {
    const int n_dims = ggml_n_dims(output);
    if (n_dims <= 1) {
        return bias;
    }
    if (n_dims == 2) {
        return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    }
    if (n_dims == 3) {
        return ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1);
    }
    return ggml_reshape_4d(ctx, bias, bias->ne[0], 1, 1, 1);
}

static ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    if (!weight) {
        return normed;
    }
    return ggml_mul(ctx, normed, weight);
}

static ggml_tensor * apply_rope(ggml_context *                    ctx,
                                const VoxCPM2TransformerConfig &  cfg,
                                const VoxCPM2TransformerWeights & weights,
                                ggml_tensor *                     x,
                                ggml_tensor *                     positions,
                                int                               seq_len) {
    if (cfg.no_rope) {
        return x;
    }

    float attn_factor = 1.0f;
    if (seq_len > cfg.rope_original_max && cfg.rope_original_max > 1) {
        const float scale = static_cast<float>(seq_len) / static_cast<float>(cfg.rope_original_max);
        attn_factor       = std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(cfg.rope_original_max)));
    }

    return ggml_rope_ext(ctx, x, positions, weights.freq_factors, cfg.head_dim, GGML_ROPE_TYPE_NEOX,
                         cfg.rope_original_max, cfg.rope_freq_base, 1.0f, 0.0f, attn_factor, 32.0f, 1.0f);
}

static ggml_tensor * attention_forward(ggml_context *                         ctx,
                                       const VoxCPM2TransformerConfig &       cfg,
                                       const VoxCPM2TransformerWeights &      weights,
                                       ggml_tensor *                          hidden,
                                       ggml_tensor *                          positions,
                                       ggml_tensor *                          attention_mask,
                                       const VoxCPM2TransformerLayerWeights & lw,
                                       int                                    n_tokens) {
    ggml_tensor * q = ggml_mul_mat(ctx, lw.wq, hidden);
    ggml_tensor * k = ggml_mul_mat(ctx, lw.wk, hidden);
    ggml_tensor * v = ggml_mul_mat(ctx, lw.wv, hidden);

    q = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, n_tokens);

    q = apply_rope(ctx, cfg, weights, q, positions, n_tokens);
    k = apply_rope(ctx, cfg, weights, k, positions, n_tokens);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    ggml_tensor * attn = nullptr;
    if (cfg.use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q, k, v, attention_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, attn, cfg.n_heads * cfg.head_dim, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, attention_mask, attn_scale, 0.0f);

        ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor * kqv = ggml_mul_mat(ctx, v_t, kq);
        attn              = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        attn              = ggml_cont_2d(ctx, attn, cfg.n_heads * cfg.head_dim, n_tokens);
    }

    return ggml_mul_mat(ctx, lw.wo, attn);
}

static ggml_tensor * mlp_forward(ggml_context * ctx, ggml_tensor * hidden, const VoxCPM2TransformerLayerWeights & lw) {
    ggml_tensor * gate  = ggml_mul_mat(ctx, lw.ffn_gate, hidden);
    ggml_tensor * up    = ggml_mul_mat(ctx, lw.ffn_up, hidden);
    gate                = ggml_silu(ctx, gate);
    ggml_tensor * fused = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, lw.ffn_down, fused);
}

static bool bind_tensor(const std::unordered_map<std::string, ggml_tensor *> & tensors,
                        const std::string &                                    name,
                        ggml_tensor **                                         dst) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        LOG_ERR("VoxCPM2Transformer: missing tensor %s\n", name.c_str());
        return false;
    }
    *dst = it->second;
    return true;
}

}  // namespace

VoxCPM2GGUFWeightStore::~VoxCPM2GGUFWeightStore() {
    free();
}

void VoxCPM2GGUFWeightStore::free() {
    if (weight_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        weight_buffer = nullptr;
    }
    if (ctx_weights) {
        ggml_free(ctx_weights);
        ctx_weights = nullptr;
    }
    if (ctx_meta) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (ctx_gguf) {
        gguf_free(ctx_gguf);
        ctx_gguf = nullptr;
    }
    tensors.clear();
    backend = nullptr;
}

bool VoxCPM2GGUFWeightStore::load(const std::string &              path,
                                  ggml_backend_t                   backend_in,
                                  const std::vector<std::string> & prefixes) {
    free();

    if (!backend_in) {
        LOG_ERR("VoxCPM2GGUFWeightStore: backend is null\n");
        return false;
    }
    backend = backend_in;

    ggml_context *   meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("VoxCPM2GGUFWeightStore: failed to open GGUF: %s\n", path.c_str());
        free();
        return false;
    }
    ctx_meta = meta;
    if (!ctx_meta) {
        LOG_ERR("VoxCPM2GGUFWeightStore: GGUF metadata context is null: %s\n", path.c_str());
        free();
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors <= 0) {
        LOG_ERR("VoxCPM2GGUFWeightStore: no tensors in GGUF: %s\n", path.c_str());
        free();
        return false;
    }

    ggml_init_params data_params{};
    data_params.mem_size   = static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_weights            = ggml_init(data_params);
    if (!ctx_weights) {
        LOG_ERR("VoxCPM2GGUFWeightStore: failed to initialize weight context\n");
        free();
        return false;
    }

    std::vector<std::pair<ggml_tensor *, size_t>> tensors_to_load;
    tensors_to_load.reserve(static_cast<size_t>(n_tensors));

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (!starts_with_any(name, prefixes)) {
            continue;
        }

        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta, name);
        if (!meta_tensor) {
            LOG_ERR("VoxCPM2GGUFWeightStore: missing tensor metadata for %s\n", name ? name : "<null>");
            free();
            return false;
        }

        ggml_tensor * tensor = ggml_dup_tensor(ctx_weights, meta_tensor);
        ggml_set_name(tensor, name);
        tensors.emplace(std::string(name), tensor);
        tensors_to_load.emplace_back(tensor, gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i));
    }

    if (tensors_to_load.empty()) {
        LOG_ERR("VoxCPM2GGUFWeightStore: no tensors matched requested prefixes in GGUF: %s\n", path.c_str());
        free();
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    weight_buffer                   = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, buft);
    if (!weight_buffer) {
        LOG_ERR("VoxCPM2GGUFWeightStore: failed to allocate weight buffer\n");
        free();
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        LOG_ERR("VoxCPM2GGUFWeightStore: failed to open GGUF for reading: %s\n", path.c_str());
        free();
        return false;
    }

    std::vector<uint8_t> staging;
    for (const auto & item : tensors_to_load) {
        ggml_tensor * tensor = item.first;
        const size_t  offset = item.second;
        const size_t  nbytes = ggml_nbytes(tensor);

        fin.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!fin) {
            LOG_ERR("VoxCPM2GGUFWeightStore: seek failed for tensor %s\n", tensor->name);
            free();
            return false;
        }

        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(tensor->data), static_cast<std::streamsize>(nbytes));
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(nbytes));
            if (!fin) {
                LOG_ERR("VoxCPM2GGUFWeightStore: read failed for tensor %s\n", tensor->name);
                free();
                return false;
            }
            ggml_backend_tensor_set(tensor, staging.data(), 0, nbytes);
        }

        if (!fin) {
            LOG_ERR("VoxCPM2GGUFWeightStore: read failed for tensor %s\n", tensor->name);
            free();
            return false;
        }
    }

    return true;
}

ggml_tensor * VoxCPM2GGUFWeightStore::get(const std::string & name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

bool VoxCPM2GGUFWeightStore::get_u32(const char * key, int & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0) {
        return false;
    }
    dst = static_cast<int>(gguf_get_val_u32(ctx_gguf, id));
    return true;
}

bool VoxCPM2GGUFWeightStore::get_f32(const char * key, float & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0) {
        return false;
    }
    dst = gguf_get_val_f32(ctx_gguf, id);
    return true;
}

bool VoxCPM2GGUFWeightStore::get_bool(const char * key, bool & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0) {
        return false;
    }
    dst = gguf_get_val_bool(ctx_gguf, id);
    return true;
}

bool VoxCPM2GGUFWeightStore::get_string(const char * key, std::string & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_STRING) {
        return false;
    }
    const char * value = gguf_get_val_str(ctx_gguf, id);
    if (!value) {
        return false;
    }
    dst = value;
    return true;
}

bool VoxCPM2GGUFWeightStore::get_i32_array(const char * key, std::vector<int> & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_ARRAY) {
        return false;
    }

    const enum gguf_type type = gguf_get_arr_type(ctx_gguf, id);
    const size_t         n    = gguf_get_arr_n(ctx_gguf, id);
    const void *         data = gguf_get_arr_data(ctx_gguf, id);
    if (!data && n > 0) {
        return false;
    }

    dst.resize(n);
    if (type == GGUF_TYPE_INT32) {
        const int32_t * values = static_cast<const int32_t *>(data);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<int>(values[i]);
        }
        return true;
    }
    if (type == GGUF_TYPE_UINT32) {
        const uint32_t * values = static_cast<const uint32_t *>(data);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<int>(values[i]);
        }
        return true;
    }

    dst.clear();
    return false;
}

bool VoxCPM2GGUFWeightStore::get_f32_array(const char * key, std::vector<float> & dst) const {
    if (!ctx_gguf) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx_gguf, key);
    if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_ARRAY) {
        return false;
    }

    const enum gguf_type type = gguf_get_arr_type(ctx_gguf, id);
    const size_t         n    = gguf_get_arr_n(ctx_gguf, id);
    const void *         data = gguf_get_arr_data(ctx_gguf, id);
    if (!data && n > 0) {
        return false;
    }

    dst.resize(n);
    if (type == GGUF_TYPE_FLOAT32) {
        const float * values = static_cast<const float *>(data);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = values[i];
        }
        return true;
    }

    dst.clear();
    return false;
}

int voxcpm2_infer_layer_count(const std::unordered_map<std::string, ggml_tensor *> & tensors,
                              const std::string &                                    prefix) {
    int n_layer = 0;
    while (true) {
        const std::string name = prefix + ".blk." + std::to_string(n_layer) + ".attn_norm.weight";
        if (tensors.find(name) == tensors.end()) {
            break;
        }
        ++n_layer;
    }
    return n_layer;
}

bool voxcpm2_bind_transformer_weights(const std::unordered_map<std::string, ggml_tensor *> & tensors,
                                      const std::string &                                    prefix,
                                      VoxCPM2TransformerConfig &                             cfg,
                                      VoxCPM2TransformerWeights &                            weights) {
    if (cfg.n_layer <= 0) {
        cfg.n_layer = voxcpm2_infer_layer_count(tensors, prefix);
    }
    if (cfg.n_layer <= 0) {
        LOG_ERR("VoxCPM2Transformer: no %s.blk.* layers found\n", prefix.c_str());
        return false;
    }

    weights.layers.assign(static_cast<size_t>(cfg.n_layer), {});

    {
        const std::string norm_name        = prefix + ".norm.weight";
        const std::string output_norm_name = prefix + ".output_norm.weight";
        const auto        norm_it          = tensors.find(norm_name);
        const auto        output_norm_it   = tensors.find(output_norm_name);
        if (norm_it != tensors.end()) {
            weights.norm = norm_it->second;
        } else if (output_norm_it != tensors.end()) {
            weights.norm = output_norm_it->second;
        } else {
            LOG_ERR("VoxCPM2Transformer: missing tensor %s or %s\n", norm_name.c_str(), output_norm_name.c_str());
            return false;
        }
    }

    bool ok = true;
    for (int i = 0; i < cfg.n_layer && ok; ++i) {
        VoxCPM2TransformerLayerWeights & lw = weights.layers[static_cast<size_t>(i)];
        const std::string                p  = prefix + ".blk." + std::to_string(i);
        ok &= bind_tensor(tensors, p + ".attn_norm.weight", &lw.attn_norm);
        ok &= bind_tensor(tensors, p + ".attn_q.weight", &lw.wq);
        ok &= bind_tensor(tensors, p + ".attn_k.weight", &lw.wk);
        ok &= bind_tensor(tensors, p + ".attn_v.weight", &lw.wv);
        ok &= bind_tensor(tensors, p + ".attn_output.weight", &lw.wo);
        ok &= bind_tensor(tensors, p + ".ffn_norm.weight", &lw.ffn_norm);
        ok &= bind_tensor(tensors, p + ".ffn_gate.weight", &lw.ffn_gate);
        ok &= bind_tensor(tensors, p + ".ffn_up.weight", &lw.ffn_up);
        ok &= bind_tensor(tensors, p + ".ffn_down.weight", &lw.ffn_down);
    }
    if (!ok) {
        return false;
    }

    const VoxCPM2TransformerLayerWeights & first = weights.layers.front();
    if (!first.wq || !first.wk || !first.wv || !first.ffn_gate) {
        return false;
    }

    cfg.hidden_size       = static_cast<int>(first.wq->ne[0]);
    cfg.head_dim          = static_cast<int>(first.wq->ne[1] / cfg.n_heads);
    cfg.intermediate_size = static_cast<int>(first.ffn_gate->ne[1]);

    if (cfg.n_heads <= 0 || cfg.n_kv_heads <= 0 || first.wq->ne[1] % cfg.n_heads != 0 ||
        first.wk->ne[1] % cfg.n_kv_heads != 0 || first.wv->ne[1] % cfg.n_kv_heads != 0) {
        LOG_ERR("VoxCPM2Transformer: invalid attention projection shapes for %s\n", prefix.c_str());
        return false;
    }

    const int q_head_dim = static_cast<int>(first.wq->ne[1] / cfg.n_heads);
    const int k_head_dim = static_cast<int>(first.wk->ne[1] / cfg.n_kv_heads);
    const int v_head_dim = static_cast<int>(first.wv->ne[1] / cfg.n_kv_heads);
    if (q_head_dim != k_head_dim || q_head_dim != v_head_dim) {
        LOG_ERR("VoxCPM2Transformer: Q/K/V head_dim mismatch for %s: q=%d k=%d v=%d\n", prefix.c_str(), q_head_dim,
                k_head_dim, v_head_dim);
        return false;
    }
    cfg.head_dim = q_head_dim;

    const int q_rows = cfg.n_heads * cfg.head_dim;
    if (first.wo->ne[0] != q_rows || first.wo->ne[1] != cfg.hidden_size) {
        LOG_ERR("VoxCPM2Transformer: attention output projection shape mismatch for %s: got [%" PRId64 ", %" PRId64
                "], expected [%d, %d]\n",
                prefix.c_str(), first.wo->ne[0], first.wo->ne[1], q_rows, cfg.hidden_size);
        return false;
    }
    if (first.ffn_down->ne[0] != cfg.intermediate_size || first.ffn_down->ne[1] != cfg.hidden_size) {
        LOG_ERR("VoxCPM2Transformer: FFN down projection shape mismatch for %s\n", prefix.c_str());
        return false;
    }
    if (weights.norm->ne[0] != cfg.hidden_size) {
        LOG_ERR("VoxCPM2Transformer: norm shape mismatch for %s\n", prefix.c_str());
        return false;
    }

    return true;
}

ggml_tensor * voxcpm2_add_bias(ggml_context * ctx, ggml_tensor * output, ggml_tensor * bias) {
    if (!bias) {
        return output;
    }
    ggml_tensor * bias_view = reshape_bias_for(ctx, bias, output);
    if (bias_view->type != output->type) {
        bias_view = ggml_cast(ctx, bias_view, output->type);
    }
    return ggml_add(ctx, output, bias_view);
}

ggml_tensor * voxcpm2_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * input) {
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    return voxcpm2_add_bias(ctx, output, bias);
}

ggml_tensor * voxcpm2_build_positions(ggml_context * ctx, int n_tokens) {
    return ggml_cast(ctx, ggml_arange(ctx, 0.0f, static_cast<float>(n_tokens), 1.0f), GGML_TYPE_I32);
}

ggml_tensor * voxcpm2_build_cfg_pair_positions(ggml_context * ctx, int branch_len) {
    GGML_ASSERT(branch_len > 0);

    const int     total_len = branch_len * 2;
    ggml_tensor * ids       = ggml_arange(ctx, 0.0f, static_cast<float>(total_len), 1.0f);

    ggml_tensor * second_branch = ggml_add(
        ctx, ids, ggml_arange(ctx, 0.5f - static_cast<float>(branch_len), 1.5f - static_cast<float>(branch_len), 1.0f));
    second_branch = ggml_scale(ctx, ggml_step(ctx, second_branch), static_cast<float>(branch_len));
    return ggml_cast(ctx, ggml_sub(ctx, ids, second_branch), GGML_TYPE_I32);
}

ggml_tensor * voxcpm2_build_cfg_pair_attention_mask(ggml_context * ctx, int branch_len) {
    GGML_ASSERT(branch_len > 0);

    const int   total_len        = branch_len * 2;
    const int   padded_total_len = GGML_PAD(total_len, GGML_KQ_MASK_PAD);
    const float branch_boundary  = 0.5f - static_cast<float>(branch_len);

    ggml_tensor * key_token_ids   = ggml_arange(ctx, 0.0f, static_cast<float>(total_len), 1.0f);
    ggml_tensor * query_token_ids = ggml_arange(ctx, 0.0f, static_cast<float>(padded_total_len), 1.0f);
    query_token_ids               = ggml_scale_bias(ctx, query_token_ids, 1.0f, -static_cast<float>(total_len));
    query_token_ids               = ggml_scale(ctx, ggml_step(ctx, query_token_ids), static_cast<float>(total_len));
    query_token_ids =
        ggml_sub(ctx, ggml_arange(ctx, 0.0f, static_cast<float>(padded_total_len), 1.0f), query_token_ids);

    ggml_tensor * key_branch_ids =
        ggml_add(ctx, key_token_ids, ggml_arange(ctx, branch_boundary, branch_boundary + 1.0f, 1.0f));
    key_branch_ids = ggml_step(ctx, key_branch_ids);

    ggml_tensor * branch_ids =
        ggml_add(ctx, query_token_ids, ggml_arange(ctx, branch_boundary, branch_boundary + 1.0f, 1.0f));
    branch_ids = ggml_step(ctx, branch_ids);

    ggml_tensor * key_ids   = ggml_reshape_2d(ctx, key_branch_ids, total_len, 1);
    ggml_tensor * query_ids = ggml_reshape_2d(ctx, branch_ids, 1, padded_total_len);
    ggml_tensor * target    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_len, padded_total_len);

    ggml_tensor * key_grid   = ggml_repeat(ctx, key_ids, target);
    ggml_tensor * query_grid = ggml_repeat(ctx, query_ids, target);
    ggml_tensor * mask       = ggml_abs(ctx, ggml_sub(ctx, key_grid, query_grid));
    mask                     = ggml_scale(ctx, mask, kCfgPairMaskNeg);

    return ggml_cont(ctx, ggml_cast(ctx, mask, GGML_TYPE_F16));
}

std::vector<float> voxcpm2_load_rope_factors(const VoxCPM2GGUFWeightStore & store) {
    std::vector<float> factors;
    if (store.get_f32_array("voxcpm.rope.long_factor", factors) && !factors.empty()) {
        return factors;
    }
    LOG_WRN("VoxCPM2Transformer: using hardcoded rope factors (not found in GGUF)\n");
    return voxcpm2_get_rope_factors();
}

const std::vector<float> & voxcpm2_get_rope_factors() {
    // VoxCPM2 LongRoPE short_factor / long_factor (both identical in VoxCPM2).
    // 64 elements for head_dim=128 (one per dimension pair).
    static const std::vector<float> factors = {
        0.9977997200264581f, 1.014658295992452f,  1.0349680404997148f, 1.059429246056193f,  1.0888815016813513f,
        1.1243301355211495f, 1.166977103606075f,  1.2182568066927284f, 1.2798772354275727f, 1.3538666751582975f,
        1.4426259039919596f, 1.5489853358570191f, 1.6762658237220625f, 1.8283407612492941f, 2.0096956085876183f,
        2.225478927469756f,  2.481536379650452f,  2.784415934557119f,  3.1413289096347365f, 3.560047844772632f,
        4.048719380066383f,  4.615569542115128f,  5.2684819496549835f, 6.014438591970396f,  6.858830049237097f,
        7.804668263503327f,  8.851768731513417f,  9.99600492938444f,   11.228766118181639f, 12.536757560834843f,
        13.902257701387796f, 15.303885189125953f, 16.717837610115794f, 18.119465097853947f, 19.484965238406907f,
        20.792956681060105f, 22.02571786985731f,  23.16995406772833f,  24.217054535738416f, 25.16289275000465f,
        26.007284207271347f, 26.753240849586767f, 27.40615325712662f,  27.973003419175363f, 28.461674954469114f,
        28.880393889607006f, 29.237306864684626f, 29.540186419591297f, 29.79624387177199f,  30.01202719065413f,
        30.193382037992453f, 30.34545697551969f,  30.47273746338473f,  30.579096895249787f, 30.66785612408345f,
        30.741845563814174f, 30.80346599254902f,  30.85474569563567f,  30.897392663720595f, 30.932841297560394f,
        30.962293553185553f, 30.986754758742034f, 31.007064503249293f, 31.02392307921529f,
    };
    return factors;
}

ggml_tensor * voxcpm2_build_freq_factors(const std::vector<float> & rope_factors,
                                         ggml_backend_t             backend,
                                         ggml_context **            out_ctx,
                                         ggml_backend_buffer_t *    out_buf) {
    if (out_ctx) {
        *out_ctx = nullptr;
    }
    if (out_buf) {
        *out_buf = nullptr;
    }
    if (rope_factors.empty()) {
        return nullptr;
    }

    const int64_t n         = static_cast<int64_t>(rope_factors.size());
    const size_t  data_size = static_cast<size_t>(n) * sizeof(float);

    // Create context in no_alloc mode (for backend buffer allocation)
    ggml_init_params params{};
    params.mem_size   = ggml_tensor_overhead() + data_size + 256;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    GGML_ASSERT(t != nullptr);

    // Allocate backend buffer for the tensor
    if (backend) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            ggml_free(ctx);
            return nullptr;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (out_buf) {
            *out_buf = buf;
        }
    }

    // Set the tensor data through the backend
    if (backend) {
        ggml_backend_tensor_set(t, rope_factors.data(), 0, data_size);
    } else {
        // CPU fallback: write directly to data pointer
        std::memcpy(t->data, rope_factors.data(), data_size);
    }

    if (out_ctx) {
        *out_ctx = ctx;
    }
    return t;
}

ggml_tensor * voxcpm2_transformer_forward(ggml_context *                    ctx,
                                          const VoxCPM2TransformerConfig &  cfg,
                                          const VoxCPM2TransformerWeights & weights,
                                          ggml_tensor *                     input,
                                          ggml_tensor *                     positions,
                                          ggml_tensor *                     attention_mask) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(input->ne[0] == cfg.hidden_size);

    const int n_tokens = static_cast<int>(input->ne[1] > 0 ? input->ne[1] : 1);
    if (!positions) {
        positions = voxcpm2_build_positions(ctx, n_tokens);
    }

    ggml_tensor * hidden = input;
    for (int i = 0; i < cfg.n_layer; ++i) {
        const VoxCPM2TransformerLayerWeights & lw = weights.layers[static_cast<size_t>(i)];

        ggml_tensor * residual = hidden;
        ggml_tensor * normed   = rms_norm(ctx, hidden, lw.attn_norm, cfg.rms_norm_eps);
        ggml_tensor * attn_out = attention_forward(ctx, cfg, weights, normed, positions, attention_mask, lw, n_tokens);
        hidden                 = ggml_add(ctx, residual, attn_out);

        residual              = hidden;
        normed                = rms_norm(ctx, hidden, lw.ffn_norm, cfg.rms_norm_eps);
        ggml_tensor * mlp_out = mlp_forward(ctx, normed, lw);
        hidden                = ggml_add(ctx, residual, mlp_out);
    }

    return rms_norm(ctx, hidden, weights.norm, cfg.rms_norm_eps);
}

// ---------------------------------------------------------------------------
// VoxCPM2KVCache
// ---------------------------------------------------------------------------

VoxCPM2KVCache::~VoxCPM2KVCache() {
    free();
}

bool VoxCPM2KVCache::init(int n_layer_in, int n_kv_heads_in, int max_length_in, int head_dim_in,
                           ggml_backend_t backend) {
    free();
    if (n_layer_in <= 0 || n_kv_heads_in <= 0 || max_length_in <= 0 || head_dim_in <= 0 || !backend) {
        return false;
    }

    n_layer    = n_layer_in;
    n_kv_heads = n_kv_heads_in;
    max_length = max_length_in;
    head_dim   = head_dim_in;

    const size_t ctx_size = ggml_tensor_overhead() * static_cast<size_t>(n_layer) * 2 + 1024;
    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ctx_kv = ggml_init(params);
    if (!ctx_kv) {
        return false;
    }

    k_caches.reserve(static_cast<size_t>(n_layer));
    v_caches.reserve(static_cast<size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        // Layout: [head_dim, n_kv_heads, max_length] — aligns with K/V projection output
        // so get_k_write uses standard strides (no custom stride swap).
        k_caches.push_back(ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, head_dim, n_kv_heads, max_length));
        v_caches.push_back(ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, head_dim, n_kv_heads, max_length));
    }

    buf_kv = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
    if (!buf_kv) {
        free();
        return false;
    }

    clear();
    return true;
}

void VoxCPM2KVCache::clear() {
    if (buf_kv) {
        ggml_backend_buffer_clear(buf_kv, 0);
    }
}

void VoxCPM2KVCache::free() {
    if (buf_kv) {
        ggml_backend_buffer_free(buf_kv);
        buf_kv = nullptr;
    }
    if (ctx_kv) {
        ggml_free(ctx_kv);
        ctx_kv = nullptr;
    }
    k_caches.clear();
    v_caches.clear();
    n_layer = n_kv_heads = max_length = head_dim = 0;
}

ggml_tensor * VoxCPM2KVCache::get_k(ggml_context * ctx, int layer, int seq_len) const {
    GGML_ASSERT(layer >= 0 && layer < n_layer);
    GGML_ASSERT(seq_len >= 0 && seq_len <= max_length);
    ggml_tensor * k = k_caches[static_cast<size_t>(layer)];
    // View [head_dim, n_kv_heads, seq_len] with standard strides, then permute
    // to [head_dim, seq_len, n_kv_heads] for attention matmul.
    ggml_tensor * v = ggml_view_3d(ctx, k, head_dim, n_kv_heads, seq_len, k->nb[1], k->nb[2], 0);
    return ggml_permute(ctx, v, 0, 2, 1, 3);
}

ggml_tensor * VoxCPM2KVCache::get_v(ggml_context * ctx, int layer, int seq_len) const {
    GGML_ASSERT(layer >= 0 && layer < n_layer);
    GGML_ASSERT(seq_len >= 0 && seq_len <= max_length);
    ggml_tensor * v = v_caches[static_cast<size_t>(layer)];
    ggml_tensor * r = ggml_view_3d(ctx, v, head_dim, n_kv_heads, seq_len, v->nb[1], v->nb[2], 0);
    return ggml_permute(ctx, r, 0, 2, 1, 3);
}

ggml_tensor * VoxCPM2KVCache::get_k_write(ggml_context * ctx, int layer, int start, int n_tokens) const {
    GGML_ASSERT(layer >= 0 && layer < n_layer);
    GGML_ASSERT(start >= 0 && start + n_tokens <= max_length);
    ggml_tensor * k      = k_caches[static_cast<size_t>(layer)];
    // Storage layout [head_dim, n_kv_heads, max_length] matches write source exactly.
    // offset in token dim (ne[2]), standard strides — no custom stride swap.
    const size_t offset = static_cast<size_t>(start) * k->nb[2];
    return ggml_view_3d(ctx, k, head_dim, n_kv_heads, n_tokens, k->nb[1], k->nb[2], offset);
}

ggml_tensor * VoxCPM2KVCache::get_v_write(ggml_context * ctx, int layer, int start, int n_tokens) const {
    GGML_ASSERT(layer >= 0 && layer < n_layer);
    GGML_ASSERT(start >= 0 && start + n_tokens <= max_length);
    ggml_tensor * v      = v_caches[static_cast<size_t>(layer)];
    const size_t  offset = static_cast<size_t>(start) * v->nb[2];
    return ggml_view_3d(ctx, v, head_dim, n_kv_heads, n_tokens, v->nb[1], v->nb[2], offset);
}

// ---------------------------------------------------------------------------
// Attention with KV cache (for forward_step and forward_prefill)
// ---------------------------------------------------------------------------

namespace {

static ggml_tensor * attention_forward_kv(ggml_context *                         ctx,
                                          const VoxCPM2TransformerConfig &       cfg,
                                          const VoxCPM2TransformerWeights &      weights,
                                          ggml_tensor *                          hidden,
                                          ggml_tensor *                          positions,
                                          const VoxCPM2TransformerLayerWeights & lw,
                                          int                                    layer_idx,
                                          int                                    n_tokens,
                                          int                                    n_past,
                                          VoxCPM2KVCache &                       kv_cache) {
    const int total_len = n_past + n_tokens;

    ggml_tensor * q = ggml_mul_mat(ctx, lw.wq, hidden);
    ggml_tensor * k = ggml_mul_mat(ctx, lw.wk, hidden);
    ggml_tensor * v = ggml_mul_mat(ctx, lw.wv, hidden);

    q = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, n_tokens);

    q = apply_rope(ctx, cfg, weights, q, positions, total_len);
    k = apply_rope(ctx, cfg, weights, k, positions, total_len);

    // Write current K/V to cache for future decode steps
    ggml_tensor * k_write_target = kv_cache.get_k_write(ctx, layer_idx, n_past, n_tokens);
    ggml_tensor * v_write_target = kv_cache.get_v_write(ctx, layer_idx, n_past, n_tokens);
    ggml_tensor * k_write = ggml_cpy(ctx, k, k_write_target);
    ggml_tensor * v_write = ggml_cpy(ctx, v, v_write_target);

    // Create a zero-valued sync tensor that depends on the writes.
    // This ensures KV cache write ops are included in the graph and forces
    // the graph scheduler to complete writes before reads in decode path.
    ggml_tensor * kv_sync = ggml_add(ctx, ggml_sum(ctx, ggml_cont(ctx, k_write)),
                                          ggml_sum(ctx, ggml_cont(ctx, v_write)));
    kv_sync = ggml_scale(ctx, kv_sync, 0.0f);

    // Permute K/V for attention: [head_dim, n_kv_heads, n_tokens] → [head_dim, n_tokens, n_kv_heads]
    ggml_tensor * k_cur = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * v_cur = ggml_permute(ctx, v, 0, 2, 1, 3);

    // For prefill (n_past == 0): use fresh k/v directly — matches Python/VoxCPM.cpp behavior.
    // For decode (n_past > 0): concat past KV from cache with current K/V.
    ggml_tensor * k_all = k_cur;
    ggml_tensor * v_all = v_cur;
    if (n_past > 0) {
        ggml_tensor * k_past = kv_cache.get_k(ctx, layer_idx, n_past);
        ggml_tensor * v_past = kv_cache.get_v(ctx, layer_idx, n_past);
        k_all = ggml_concat(ctx, k_past, k_cur, 1);  // concat on token dim (dim 1)
        v_all = ggml_concat(ctx, v_past, v_cur, 1);
    }

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [head_dim, n_heads, n_tokens] → [head_dim, n_tokens, n_heads]

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    ggml_tensor * kq = ggml_mul_mat(ctx, k_all, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    // Causal mask: for prefill, need causal mask. For decode (n_tokens==1), no mask.
    ggml_tensor * mask = nullptr;
    if (n_tokens > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_len, n_tokens);
        ggml_set_input(mask);
    }

    kq = ggml_soft_max_ext(ctx, kq, mask, attn_scale, 0.0f);

    ggml_tensor * v_t  = ggml_cont(ctx, ggml_transpose(ctx, v_all));
    ggml_tensor * kqv  = ggml_mul_mat(ctx, v_t, kq);
    ggml_tensor * attn = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    attn               = ggml_cont_2d(ctx, attn, cfg.n_heads * cfg.head_dim, n_tokens);

    // Add kv_sync to output to enforce write-before-read ordering AND to
    // ensure KV cache write ops are included in the computation graph.
    // kv_sync is [1] with value 0.0f; ggml_add broadcasts it to attn's shape.
    attn = ggml_add(ctx, attn, kv_sync);

    return ggml_mul_mat(ctx, lw.wo, attn);
}

}  // namespace

ggml_tensor * voxcpm2_transformer_forward_step(ggml_context *                    ctx,
                                               const VoxCPM2TransformerConfig &  cfg,
                                               const VoxCPM2TransformerWeights & weights,
                                               ggml_tensor *                     input,
                                               int                               position,
                                               VoxCPM2KVCache &                  kv_cache) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);

    ggml_tensor * hidden = input;
    if (ggml_n_dims(hidden) == 1) {
        hidden = ggml_reshape_2d(ctx, hidden, hidden->ne[0], 1);
    }

    ggml_tensor * positions = ggml_view_1d(ctx, voxcpm2_build_positions(ctx, position + 1),
                                            1, static_cast<size_t>(position) * sizeof(int32_t));

    for (int i = 0; i < cfg.n_layer; ++i) {
        const VoxCPM2TransformerLayerWeights & lw = weights.layers[static_cast<size_t>(i)];

        ggml_tensor * residual = hidden;
        ggml_tensor * normed   = rms_norm(ctx, hidden, lw.attn_norm, cfg.rms_norm_eps);
        ggml_tensor * attn_out = attention_forward_kv(ctx, cfg, weights, normed, positions, lw, i, 1, position, kv_cache);
        hidden                 = ggml_add(ctx, residual, attn_out);

        residual              = hidden;
        normed                = rms_norm(ctx, hidden, lw.ffn_norm, cfg.rms_norm_eps);
        ggml_tensor * mlp_out = mlp_forward(ctx, normed, lw);
        hidden                = ggml_add(ctx, residual, mlp_out);
    }

    hidden = rms_norm(ctx, hidden, weights.norm, cfg.rms_norm_eps);
    return ggml_reshape_1d(ctx, hidden, hidden->ne[0]);
}

ggml_tensor * voxcpm2_transformer_forward_prefill(ggml_context *                    ctx,
                                                  const VoxCPM2TransformerConfig &  cfg,
                                                  const VoxCPM2TransformerWeights & weights,
                                                  ggml_tensor *                     input,
                                                  int                               n_tokens,
                                                  VoxCPM2KVCache &                  kv_cache) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(input->ne[0] == cfg.hidden_size);

    ggml_tensor * positions = voxcpm2_build_positions(ctx, n_tokens);

    ggml_tensor * hidden = input;
    for (int i = 0; i < cfg.n_layer; ++i) {
        const VoxCPM2TransformerLayerWeights & lw = weights.layers[static_cast<size_t>(i)];

        ggml_tensor * residual = hidden;
        ggml_tensor * normed   = rms_norm(ctx, hidden, lw.attn_norm, cfg.rms_norm_eps);
        ggml_tensor * attn_out = attention_forward_kv(ctx, cfg, weights, normed, positions, lw, i, n_tokens, 0, kv_cache);
        hidden                 = ggml_add(ctx, residual, attn_out);

        residual              = hidden;
        normed                = rms_norm(ctx, hidden, lw.ffn_norm, cfg.rms_norm_eps);
        ggml_tensor * mlp_out = mlp_forward(ctx, normed, lw);
        hidden                = ggml_add(ctx, residual, mlp_out);
    }

    return rms_norm(ctx, hidden, weights.norm, cfg.rms_norm_eps);
}
