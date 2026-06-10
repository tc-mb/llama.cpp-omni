#include "voxcpm2_locdit.h"

#include "ggml-cpp.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

LocDiTModel::~LocDiTModel() {
    free();
}

void LocDiTModel::free() {
    store.reset();
    weights = {};
    config  = {};
    backend = nullptr;
}

bool LocDiTModel::validate_weights() const {
    const int hidden_size = config.transformer.hidden_size;
    const int feat_dim    = config.feat_dim;

    if (!weights.in_proj_weight || !weights.in_proj_bias || !weights.cond_proj_weight || !weights.cond_proj_bias ||
        !weights.out_proj_weight || !weights.out_proj_bias || !weights.time_mlp_linear1_weight ||
        !weights.time_mlp_linear1_bias || !weights.time_mlp_linear2_weight || !weights.time_mlp_linear2_bias ||
        !weights.delta_time_mlp_linear1_weight || !weights.delta_time_mlp_linear1_bias ||
        !weights.delta_time_mlp_linear2_weight || !weights.delta_time_mlp_linear2_bias) {
        LOG_ERR("LocDiTModel: missing projection or time MLP tensors\n");
        return false;
    }
    if (!weights.transformer.norm ||
        weights.transformer.layers.size() != static_cast<size_t>(config.transformer.n_layer)) {
        LOG_ERR("LocDiTModel: missing transformer weights\n");
        return false;
    }
    if (weights.in_proj_weight->ne[0] != feat_dim || weights.in_proj_weight->ne[1] != hidden_size) {
        LOG_ERR("LocDiTModel: in_proj.weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                weights.in_proj_weight->ne[0], weights.in_proj_weight->ne[1], feat_dim, hidden_size);
        return false;
    }
    if (weights.cond_proj_weight->ne[0] != feat_dim || weights.cond_proj_weight->ne[1] != hidden_size) {
        LOG_ERR("LocDiTModel: cond_proj.weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                weights.cond_proj_weight->ne[0], weights.cond_proj_weight->ne[1], feat_dim, hidden_size);
        return false;
    }
    if (weights.out_proj_weight->ne[0] != hidden_size || weights.out_proj_weight->ne[1] != feat_dim) {
        LOG_ERR("LocDiTModel: out_proj.weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                weights.out_proj_weight->ne[0], weights.out_proj_weight->ne[1], hidden_size, feat_dim);
        return false;
    }
    if (weights.in_proj_bias->ne[0] != hidden_size || weights.cond_proj_bias->ne[0] != hidden_size ||
        weights.out_proj_bias->ne[0] != feat_dim) {
        LOG_ERR("LocDiTModel: projection bias shape mismatch\n");
        return false;
    }
    if (weights.time_mlp_linear1_weight->ne[0] != hidden_size ||
        weights.time_mlp_linear2_weight->ne[1] != hidden_size ||
        weights.delta_time_mlp_linear1_weight->ne[0] != hidden_size ||
        weights.delta_time_mlp_linear2_weight->ne[1] != hidden_size) {
        LOG_ERR("LocDiTModel: time MLP shape mismatch\n");
        return false;
    }
    return true;
}

bool LocDiTModel::bind_from_store() {
    if (!store) {
        return false;
    }

    store->get_u32("voxcpm.patch_size", config.patch_size);
    store->get_u32("voxcpm.feat_dim", config.feat_dim);
    store->get_u32("voxcpm.locdit.n_layer", config.transformer.n_layer);
    store->get_u32("voxcpm.locdit.n_embd", config.transformer.hidden_size);
    store->get_f32("voxcpm.cfm.cfg_rate", config.cfg_rate);
    float model_version = 2.0f;
    store->get_f32("voxcpm.model_version", model_version);
    config.combine_mu_time = model_version < 2.0f;
    config.transformer.max_length = 4096;
    config.transformer.no_rope    = false;

    weights.in_proj_weight   = store->get("locdit.in_proj.weight");
    weights.in_proj_bias     = store->get("locdit.in_proj.bias");
    weights.cond_proj_weight = store->get("locdit.cond_proj.weight");
    weights.cond_proj_bias   = store->get("locdit.cond_proj.bias");
    weights.out_proj_weight  = store->get("locdit.out_proj.weight");
    weights.out_proj_bias    = store->get("locdit.out_proj.bias");

    weights.time_mlp_linear1_weight = store->get("locdit.time_mlp.linear_1.weight");
    weights.time_mlp_linear1_bias   = store->get("locdit.time_mlp.linear_1.bias");
    weights.time_mlp_linear2_weight = store->get("locdit.time_mlp.linear_2.weight");
    weights.time_mlp_linear2_bias   = store->get("locdit.time_mlp.linear_2.bias");

    weights.delta_time_mlp_linear1_weight = store->get("locdit.delta_time_mlp.linear_1.weight");
    weights.delta_time_mlp_linear1_bias   = store->get("locdit.delta_time_mlp.linear_1.bias");
    weights.delta_time_mlp_linear2_weight = store->get("locdit.delta_time_mlp.linear_2.weight");
    weights.delta_time_mlp_linear2_bias   = store->get("locdit.delta_time_mlp.linear_2.bias");

    if (!voxcpm2_bind_transformer_weights(store->tensors, "locdit", config.transformer, weights.transformer)) {
        return false;
    }

    // Build freq_factors for LongRoPE
    if (!config.transformer.no_rope) {
        weights.transformer.freq_factors = voxcpm2_build_freq_factors(
            voxcpm2_load_rope_factors(*store), backend, &weights.transformer.aux_ctx, &weights.transformer.aux_buf);
    }

    config.feat_dim = static_cast<int>(weights.in_proj_weight ? weights.in_proj_weight->ne[0] : config.feat_dim);
    return validate_weights();
}

bool LocDiTModel::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("LocDiTModel: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "locdit." })) {
        free();
        return false;
    }

    if (!bind_from_store()) {
        free();
        return false;
    }

    LOG_INF("LocDiTModel: loaded layers=%d hidden=%d feat_dim=%d patch_size=%d cfg=%.3f mode=%s\n",
            config.transformer.n_layer, config.transformer.hidden_size, config.feat_dim, config.patch_size,
            config.cfg_rate, config.combine_mu_time ? "v1-mu-plus-time" : "v2-mu-time-tokens");
    return true;
}

bool LocDiTModel::init_manual(const VoxCPM2LocDiTConfig & cfg, const VoxCPM2LocDiTWeights & manual_weights) {
    free();

    config  = cfg;
    weights = manual_weights;

    if (!validate_weights()) {
        weights = {};
        return false;
    }
    return true;
}

ggml_tensor * LocDiTModel::sinusoidal_embedding(ggml_context * ctx, ggml_tensor * scalar, int dim, float scale) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(scalar != nullptr);
    GGML_ASSERT(dim % 2 == 0);

    const int   half_dim = dim / 2;
    const float emb_val  = std::log(10000.0f) / static_cast<float>(half_dim - 1);

    ggml_tensor * arange = ggml_arange(ctx, 0.0f, static_cast<float>(half_dim), 1.0f);
    ggml_tensor * emb    = ggml_scale(ctx, arange, -emb_val);
    emb                  = ggml_exp(ctx, emb);
    emb                  = ggml_scale(ctx, emb, scale);

    ggml_tensor * scalar_view      = ggml_reshape_1d(ctx, scalar, 1);
    ggml_tensor * scalar_broadcast = ggml_repeat(ctx, scalar_view, emb);
    emb                            = ggml_mul(ctx, emb, scalar_broadcast);

    ggml_tensor * sin_emb = ggml_sin(ctx, emb);
    ggml_tensor * cos_emb = ggml_cos(ctx, emb);
    return ggml_concat(ctx, sin_emb, cos_emb, 0);
}

ggml_tensor * LocDiTModel::timestep_mlp(ggml_context * ctx,
                                        ggml_tensor *  input,
                                        ggml_tensor *  linear1_w,
                                        ggml_tensor *  linear1_b,
                                        ggml_tensor *  linear2_w,
                                        ggml_tensor *  linear2_b) const {
    ggml_tensor * x = voxcpm2_linear(ctx, linear1_w, linear1_b, input);
    x               = ggml_silu(ctx, x);
    return voxcpm2_linear(ctx, linear2_w, linear2_b, x);
}

ggml_tensor * LocDiTModel::build_time_embedding(ggml_context * ctx, ggml_tensor * t_scalar) const {
    ggml_tensor * sinusoidal = sinusoidal_embedding(ctx, t_scalar, config.transformer.hidden_size, 1000.0f);
    return timestep_mlp(ctx, sinusoidal, weights.time_mlp_linear1_weight, weights.time_mlp_linear1_bias,
                        weights.time_mlp_linear2_weight, weights.time_mlp_linear2_bias);
}

ggml_tensor * LocDiTModel::build_delta_time_embedding(ggml_context * ctx, ggml_tensor * dt_scalar) const {
    ggml_tensor * sinusoidal = sinusoidal_embedding(ctx, dt_scalar, config.transformer.hidden_size, 1000.0f);
    return timestep_mlp(ctx, sinusoidal, weights.delta_time_mlp_linear1_weight, weights.delta_time_mlp_linear1_bias,
                        weights.delta_time_mlp_linear2_weight, weights.delta_time_mlp_linear2_bias);
}

ggml_tensor * LocDiTModel::build_time_embedding(ggml_context * ctx, float t) const {
    ggml_tensor * scalar = ggml_arange(ctx, t, t + 1.0f, 1.0f);
    return build_time_embedding(ctx, scalar);
}

ggml_tensor * LocDiTModel::build_cfg_mask(ggml_context * ctx, int branch_len) const {
    return voxcpm2_build_cfg_pair_attention_mask(ctx, branch_len);
}

ggml_tensor * LocDiTModel::project_input(ggml_context * ctx, ggml_tensor * x) const {
    return voxcpm2_linear(ctx, weights.in_proj_weight, weights.in_proj_bias, x);
}

ggml_tensor * LocDiTModel::project_condition(ggml_context * ctx, ggml_tensor * cond) const {
    return voxcpm2_linear(ctx, weights.cond_proj_weight, weights.cond_proj_bias, cond);
}

ggml_tensor * LocDiTModel::build_time_token(ggml_context * ctx, ggml_tensor * t_scalar, ggml_tensor * dt_scalar) const {
    ggml_tensor * time_token = build_time_embedding(ctx, t_scalar);
    time_token               = ggml_add(ctx, time_token, build_delta_time_embedding(ctx, dt_scalar));
    return ggml_reshape_2d(ctx, time_token, config.transformer.hidden_size, 1);
}

ggml_tensor * LocDiTModel::reshape_mu_tokens(ggml_context * ctx, ggml_tensor * mu) const {
    GGML_ASSERT(mu != nullptr);
    GGML_ASSERT(mu->ne[0] % config.transformer.hidden_size == 0);
    return ggml_reshape_2d(ctx, mu, config.transformer.hidden_size, mu->ne[0] / config.transformer.hidden_size);
}

ggml_tensor * LocDiTModel::forward_projected(ggml_context * ctx,
                                             ggml_tensor *  x_proj,
                                             ggml_tensor *  prefix_tokens,
                                             ggml_tensor *  cond_proj,
                                             int            generated_prefix_tokens,
                                             int            prefix_len,
                                             int            seq_len) const {
    GGML_ASSERT(x_proj != nullptr);
    GGML_ASSERT(prefix_tokens != nullptr);

    ggml_tensor * seq = prefix_tokens;
    if (cond_proj && prefix_len > 0) {
        seq = ggml_concat(ctx, seq, cond_proj, 1);
    }
    seq = ggml_cont(ctx, ggml_concat(ctx, seq, x_proj, 1));

    ggml_tensor * hidden =
        voxcpm2_transformer_forward(ctx, config.transformer, weights.transformer, seq, nullptr, nullptr);
    ggml_tensor * hidden_out = ggml_view_2d(ctx, hidden, config.transformer.hidden_size, seq_len, hidden->nb[1],
                                            static_cast<size_t>(prefix_len + generated_prefix_tokens) * hidden->nb[1]);
    hidden_out               = ggml_cont(ctx, hidden_out);
    return voxcpm2_linear(ctx, weights.out_proj_weight, weights.out_proj_bias, hidden_out);
}

ggml_tensor * LocDiTModel::forward_single(ggml_context * ctx,
                                          ggml_tensor *  x,
                                          ggml_tensor *  mu,
                                          ggml_tensor *  t_scalar,
                                          ggml_tensor *  cond,
                                          ggml_tensor *  dt_scalar) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(mu != nullptr);
    GGML_ASSERT(t_scalar != nullptr);
    GGML_ASSERT(dt_scalar != nullptr);
    GGML_ASSERT(x->ne[0] == config.feat_dim);
    GGML_ASSERT(mu->ne[0] % config.transformer.hidden_size == 0);

    const int64_t seq_len    = x->ne[1];
    const int64_t prefix_len = cond ? cond->ne[1] : 0;
    const int64_t mu_tokens  = mu->ne[0] / config.transformer.hidden_size;
    GGML_ASSERT(!cond || cond->ne[0] == config.feat_dim);

    ggml_tensor * x_proj     = project_input(ctx, x);
    ggml_tensor * cond_proj  = (cond && prefix_len > 0) ? project_condition(ctx, cond) : nullptr;
    ggml_tensor * time_token = build_time_token(ctx, t_scalar, dt_scalar);

    if (config.combine_mu_time) {
        GGML_ASSERT(mu_tokens == 1);
        ggml_tensor * mu_2d    = ggml_reshape_2d(ctx, mu, config.transformer.hidden_size, 1);
        ggml_tensor * combined = ggml_add(ctx, mu_2d, time_token);
        return forward_projected(ctx, x_proj, combined, cond_proj, 1, static_cast<int>(prefix_len),
                                 static_cast<int>(seq_len));
    }

    ggml_tensor * prefix_tokens = ggml_concat(ctx, reshape_mu_tokens(ctx, mu), time_token, 1);
    return forward_projected(ctx, x_proj, prefix_tokens, cond_proj, static_cast<int>(mu_tokens + 1),
                             static_cast<int>(prefix_len), static_cast<int>(seq_len));
}

void LocDiTModel::forward_cfg_pair_projected(ggml_context * ctx,
                                             ggml_tensor *  x_proj,
                                             ggml_tensor *  mu,
                                             ggml_tensor *  time_token,
                                             ggml_tensor *  cond_proj,
                                             int            prefix_len,
                                             ggml_tensor ** conditioned,
                                             ggml_tensor ** unconditioned) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(conditioned != nullptr);
    GGML_ASSERT(unconditioned != nullptr);
    GGML_ASSERT(x_proj != nullptr);
    GGML_ASSERT(mu != nullptr);
    GGML_ASSERT(time_token != nullptr);

    const int seq_len   = static_cast<int>(x_proj->ne[1]);
    const int mu_tokens = static_cast<int>(mu->ne[0] / config.transformer.hidden_size);
    GGML_ASSERT(x_proj->ne[0] == config.transformer.hidden_size);
    GGML_ASSERT(mu->ne[0] % config.transformer.hidden_size == 0);

    ggml_tensor * conditioned_prefix   = nullptr;
    ggml_tensor * unconditioned_prefix = nullptr;
    if (config.combine_mu_time) {
        GGML_ASSERT(mu_tokens == 1);
        ggml_tensor * mu_2d   = ggml_reshape_2d(ctx, mu, config.transformer.hidden_size, 1);
        conditioned_prefix    = ggml_add(ctx, mu_2d, time_token);
        ggml_tensor * zero_mu = ggml_scale(ctx, mu_2d, 0.0f);
        unconditioned_prefix  = ggml_add(ctx, zero_mu, time_token);
    } else {
        ggml_tensor * mu_prefix      = reshape_mu_tokens(ctx, mu);
        ggml_tensor * zero_mu_prefix = ggml_scale(ctx, mu_prefix, 0.0f);
        conditioned_prefix           = ggml_concat(ctx, mu_prefix, time_token, 1);
        unconditioned_prefix         = ggml_concat(ctx, zero_mu_prefix, time_token, 1);
    }

    const int generated_prefix_tokens = config.combine_mu_time ? 1 : mu_tokens + 1;
    const int branch_len              = prefix_len + seq_len + generated_prefix_tokens;
    const int total_len               = branch_len * 2;

    if (total_len > config.transformer.max_length) {
        *conditioned =
            forward_projected(ctx, x_proj, conditioned_prefix, cond_proj, generated_prefix_tokens, prefix_len, seq_len);
        *unconditioned = forward_projected(ctx, x_proj, unconditioned_prefix, cond_proj, generated_prefix_tokens,
                                           prefix_len, seq_len);
        return;
    }

    ggml_tensor * conditioned_seq   = conditioned_prefix;
    ggml_tensor * unconditioned_seq = unconditioned_prefix;
    if (cond_proj && prefix_len > 0) {
        conditioned_seq   = ggml_concat(ctx, conditioned_seq, cond_proj, 1);
        unconditioned_seq = ggml_concat(ctx, unconditioned_seq, cond_proj, 1);
    }
    conditioned_seq   = ggml_concat(ctx, conditioned_seq, x_proj, 1);
    unconditioned_seq = ggml_concat(ctx, unconditioned_seq, x_proj, 1);

    ggml_tensor * paired_seq = ggml_concat(ctx, conditioned_seq, unconditioned_seq, 1);
    ggml_tensor * positions  = voxcpm2_build_cfg_pair_positions(ctx, branch_len);
    ggml_tensor * mask       = voxcpm2_build_cfg_pair_attention_mask(ctx, branch_len);

    ggml_tensor * paired_hidden =
        voxcpm2_transformer_forward(ctx, config.transformer, weights.transformer, paired_seq, positions, mask);

    ggml_tensor * conditioned_hidden =
        ggml_view_2d(ctx, paired_hidden, config.transformer.hidden_size, seq_len, paired_hidden->nb[1],
                     static_cast<size_t>(prefix_len + generated_prefix_tokens) * paired_hidden->nb[1]);
    ggml_tensor * unconditioned_hidden =
        ggml_view_2d(ctx, paired_hidden, config.transformer.hidden_size, seq_len, paired_hidden->nb[1],
                     static_cast<size_t>(branch_len + prefix_len + generated_prefix_tokens) * paired_hidden->nb[1]);

    ggml_tensor * cond_hidden_cont   = ggml_cont(ctx, conditioned_hidden);
    ggml_tensor * uncond_hidden_cont = ggml_cont(ctx, unconditioned_hidden);
    *conditioned   = voxcpm2_linear(ctx, weights.out_proj_weight, weights.out_proj_bias, cond_hidden_cont);
    *unconditioned = voxcpm2_linear(ctx, weights.out_proj_weight, weights.out_proj_bias, uncond_hidden_cont);
}

void LocDiTModel::forward_cfg_pair(ggml_context * ctx,
                                   ggml_tensor *  x,
                                   ggml_tensor *  mu,
                                   ggml_tensor *  t_scalar,
                                   ggml_tensor *  cond,
                                   ggml_tensor *  dt_scalar,
                                   ggml_tensor ** conditioned,
                                   ggml_tensor ** unconditioned) const {
    GGML_ASSERT(conditioned != nullptr);
    GGML_ASSERT(unconditioned != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(mu != nullptr);
    GGML_ASSERT(t_scalar != nullptr);
    GGML_ASSERT(dt_scalar != nullptr);
    GGML_ASSERT(x->ne[0] == config.feat_dim);
    GGML_ASSERT(mu->ne[0] % config.transformer.hidden_size == 0);
    GGML_ASSERT(!cond || cond->ne[0] == config.feat_dim);

    const int64_t prefix_len = cond ? cond->ne[1] : 0;
    ggml_tensor * x_proj     = project_input(ctx, x);
    ggml_tensor * cond_proj  = (cond && prefix_len > 0) ? project_condition(ctx, cond) : nullptr;
    ggml_tensor * time_token = build_time_token(ctx, t_scalar, dt_scalar);
    forward_cfg_pair_projected(ctx, x_proj, mu, time_token, cond_proj, static_cast<int>(prefix_len), conditioned,
                               unconditioned);
}

ggml_tensor * LocDiTModel::forward_cfg_pair(ggml_context * ctx,
                                            ggml_tensor *  x,
                                            ggml_tensor *  mu,
                                            float          t,
                                            ggml_tensor *  cond,
                                            float          dt,
                                            float          cfg_rate) const {
    ggml_tensor * t_scalar      = ggml_arange(ctx, t, t + 1.0f, 1.0f);
    ggml_tensor * dt_scalar     = ggml_arange(ctx, dt, dt + 1.0f, 1.0f);
    ggml_tensor * conditioned   = nullptr;
    ggml_tensor * unconditioned = nullptr;
    forward_cfg_pair(ctx, x, mu, t_scalar, cond, dt_scalar, &conditioned, &unconditioned);
    return ggml_add(ctx, unconditioned, ggml_scale(ctx, ggml_sub(ctx, conditioned, unconditioned), cfg_rate));
}

std::vector<float> LocDiTModel::precompute_cfg_time_table(const std::vector<float> & t_values,
                                                          ggml_backend_t             compute_backend) const {
    if (t_values.empty()) {
        return {};
    }

    ggml_backend_t be = compute_backend ? compute_backend : backend;
    if (!be) {
        LOG_ERR("LocDiTModel: cannot precompute time table without backend\n");
        return {};
    }

    ggml_init_params params{};
    params.mem_size   = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        LOG_ERR("LocDiTModel: failed to allocate time-table context\n");
        return {};
    }

    ggml_tensor * t_scalar = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_input(t_scalar);
    ggml_tensor * zero_scalar = ggml_arange(ctx.get(), 0.0f, 1.0f, 1.0f);
    ggml_tensor * combined    = build_time_embedding(ctx.get(), t_scalar);
    combined                  = ggml_add(ctx.get(), combined, build_delta_time_embedding(ctx.get(), zero_scalar));
    ggml_set_output(combined);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 4096, false);
    ggml_build_forward_expand(graph, combined);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), be));
    if (!buf) {
        LOG_ERR("LocDiTModel: failed to allocate time-table graph buffer\n");
        return {};
    }

    const int          hidden_size = config.transformer.hidden_size;
    std::vector<float> table(static_cast<size_t>(hidden_size) * t_values.size(), 0.0f);
    std::vector<float> scratch(static_cast<size_t>(hidden_size), 0.0f);

    for (size_t i = 0; i < t_values.size(); ++i) {
        const float t_value = t_values[i];
        ggml_backend_tensor_set(t_scalar, &t_value, 0, sizeof(t_value));
        if (ggml_backend_graph_compute(be, graph) != GGML_STATUS_SUCCESS) {
            LOG_ERR("LocDiTModel: time-table graph compute failed\n");
            return {};
        }
        ggml_backend_tensor_get(combined, scratch.data(), 0, scratch.size() * sizeof(float));
        std::memcpy(table.data() + i * scratch.size(), scratch.data(), scratch.size() * sizeof(float));
    }

    return table;
}
