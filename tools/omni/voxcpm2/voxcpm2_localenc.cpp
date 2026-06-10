#include "voxcpm2_localenc.h"

#include "log.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

LocEncModel::~LocEncModel() {
    free();
}

void LocEncModel::free() {
    store.reset();
    weights = {};
    config  = {};
    backend = nullptr;
}

bool LocEncModel::validate_weights() const {
    const int hidden_size = config.transformer.hidden_size;
    const int feat_dim    = config.feat_dim;

    if (!weights.in_proj_weight || !weights.in_proj_bias || !weights.cls_token) {
        LOG_ERR("LocEncModel: missing in_proj or cls_token tensors\n");
        return false;
    }
    if (!weights.transformer.norm ||
        weights.transformer.layers.size() != static_cast<size_t>(config.transformer.n_layer)) {
        LOG_ERR("LocEncModel: missing transformer weights\n");
        return false;
    }
    if (weights.in_proj_weight->ne[0] != feat_dim || weights.in_proj_weight->ne[1] != hidden_size) {
        LOG_ERR("LocEncModel: in_proj.weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                weights.in_proj_weight->ne[0], weights.in_proj_weight->ne[1], feat_dim, hidden_size);
        return false;
    }
    if (weights.in_proj_bias->ne[0] != hidden_size) {
        LOG_ERR("LocEncModel: in_proj.bias shape mismatch: got %" PRId64 ", expected %d\n", weights.in_proj_bias->ne[0],
                hidden_size);
        return false;
    }
    if (weights.cls_token->ne[0] != hidden_size) {
        LOG_ERR("LocEncModel: cls_token shape mismatch: got %" PRId64 ", expected %d\n", weights.cls_token->ne[0],
                hidden_size);
        return false;
    }
    return true;
}

bool LocEncModel::bind_from_store() {
    if (!store) {
        return false;
    }

    store->get_u32("voxcpm.patch_size", config.patch_size);
    store->get_u32("voxcpm.feat_dim", config.feat_dim);
    store->get_u32("voxcpm.locenc.n_layer", config.transformer.n_layer);
    store->get_u32("voxcpm.locenc.n_embd", config.transformer.hidden_size);
    config.transformer.max_length = 4096;
    config.transformer.no_rope    = false;

    weights.in_proj_weight = store->get("locenc.in_proj.weight");
    weights.in_proj_bias   = store->get("locenc.in_proj.bias");
    weights.cls_token      = store->get("locenc.cls_token.weight");

    if (!voxcpm2_bind_transformer_weights(store->tensors, "locenc", config.transformer, weights.transformer)) {
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

bool LocEncModel::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("LocEncModel: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "locenc." })) {
        free();
        return false;
    }

    if (!bind_from_store()) {
        free();
        return false;
    }

    LOG_INF("LocEncModel: loaded layers=%d hidden=%d feat_dim=%d patch_size=%d\n", config.transformer.n_layer,
            config.transformer.hidden_size, config.feat_dim, config.patch_size);
    return true;
}

bool LocEncModel::init_manual(const VoxCPM2LocEncConfig &       cfg,
                              ggml_tensor *                     in_proj_weight,
                              ggml_tensor *                     in_proj_bias,
                              ggml_tensor *                     cls_token,
                              const VoxCPM2TransformerWeights & transformer_weights) {
    free();

    config                 = cfg;
    weights.in_proj_weight = in_proj_weight;
    weights.in_proj_bias   = in_proj_bias;
    weights.cls_token      = cls_token;
    weights.transformer    = transformer_weights;

    if (!validate_weights()) {
        weights = {};
        return false;
    }
    return true;
}

ggml_tensor * LocEncModel::forward_patch(ggml_context * ctx, ggml_tensor * input) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(input->ne[1] > 0);
    GGML_ASSERT(input->ne[0] == config.feat_dim || input->ne[0] == config.transformer.hidden_size);

    ggml_tensor * projected = input;
    if (input->ne[0] != config.transformer.hidden_size) {
        projected = voxcpm2_linear(ctx, weights.in_proj_weight, weights.in_proj_bias, input);
    }

    ggml_tensor * cls = ggml_reshape_2d(ctx, weights.cls_token, config.transformer.hidden_size, 1);
    if (cls->type != projected->type) {
        cls = ggml_cast(ctx, cls, projected->type);
    }
    ggml_tensor * full_input = ggml_concat(ctx, cls, projected, 1);
    ggml_tensor * hidden =
        voxcpm2_transformer_forward(ctx, config.transformer, weights.transformer, full_input, nullptr, nullptr);
    return ggml_view_1d(ctx, hidden, config.transformer.hidden_size, 0);
}

ggml_tensor * LocEncModel::forward_sequence(ggml_context * ctx, ggml_tensor * input) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(ggml_n_dims(input) == 3);
    GGML_ASSERT(input->ne[1] > 0);
    GGML_ASSERT(input->ne[2] > 0);
    GGML_ASSERT(input->ne[0] == config.feat_dim || input->ne[0] == config.transformer.hidden_size);

    const int     hidden_size = config.transformer.hidden_size;
    const int64_t patch_size  = input->ne[1];
    const int64_t seq_len     = input->ne[2];

    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor * sync   = nullptr;

    for (int64_t idx = 0; idx < seq_len; ++idx) {
        ggml_tensor * patch_view =
            ggml_view_2d(ctx, input, input->ne[0], patch_size, input->nb[1], static_cast<size_t>(idx) * input->nb[2]);
        ggml_tensor * hidden     = forward_patch(ctx, patch_view);
        ggml_tensor * out_view   = ggml_view_1d(ctx, output, hidden_size, static_cast<size_t>(idx) * output->nb[1]);
        ggml_tensor * copied     = ggml_cpy(ctx, hidden, out_view);
        ggml_tensor * copied_sum = ggml_sum(ctx, copied);
        sync                     = sync ? ggml_add(ctx, sync, copied_sum) : copied_sum;
    }

    if (!sync) {
        return output;
    }

    sync = ggml_scale(ctx, sync, 0.0f);
    return ggml_add(ctx, output, sync);
}
