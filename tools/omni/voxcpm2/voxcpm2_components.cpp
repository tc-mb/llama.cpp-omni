#include "voxcpm2_components.h"

#include "log.h"

#include <cinttypes>
#include <cstdio>

namespace {

static bool validate_linear(const char *                 label,
                            const VoxCPM2LinearWeights & w,
                            int                          expected_in,
                            int                          expected_out,
                            bool                         bias_required = true) {
    if (!w.weight || (bias_required && !w.bias)) {
        LOG_ERR("%s: missing weight or bias tensor\n", label);
        return false;
    }
    if (w.weight->ne[0] != expected_in || w.weight->ne[1] != expected_out) {
        LOG_ERR("%s: weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n", label,
                w.weight->ne[0], w.weight->ne[1], expected_in, expected_out);
        return false;
    }
    if (w.bias && w.bias->ne[0] != expected_out) {
        LOG_ERR("%s: bias shape mismatch: got %" PRId64 ", expected %d\n", label, w.bias->ne[0], expected_out);
        return false;
    }
    return true;
}

static VoxCPM2LinearWeights get_linear(VoxCPM2GGUFWeightStore & store, const char * prefix) {
    VoxCPM2LinearWeights w;
    w.weight = store.get(std::string(prefix) + ".weight");
    w.bias   = store.get(std::string(prefix) + ".bias");
    return w;
}

}  // namespace

VoxCPM2Projections::~VoxCPM2Projections() {
    free();
}

void VoxCPM2Projections::free() {
    store.reset();
    weights = {};
    config  = {};
    backend = nullptr;
}

bool VoxCPM2Projections::validate_weights() const {
    if (!validate_linear("VoxCPM2Projections.enc_to_lm_proj", weights.enc_to_lm_proj, config.dit_hidden_size,
                         config.lm_hidden_size)) {
        return false;
    }
    if (!validate_linear("VoxCPM2Projections.lm_to_dit_proj", weights.lm_to_dit_proj, config.lm_hidden_size,
                         config.dit_hidden_size)) {
        return false;
    }
    if (!validate_linear("VoxCPM2Projections.res_to_dit_proj", weights.res_to_dit_proj, config.lm_hidden_size,
                         config.dit_hidden_size)) {
        return false;
    }
    // res_fusion_proj is optional (absent in v0.5 / v1.5)
    if (weights.res_fusion_proj.weight) {
        if (!validate_linear("VoxCPM2Projections.res_fusion_proj", weights.res_fusion_proj, config.lm_hidden_size * 2,
                             config.lm_hidden_size)) {
            return false;
        }
    }
    return true;
}

bool VoxCPM2Projections::bind_from_store() {
    if (!store) {
        return false;
    }

    weights.enc_to_lm_proj  = get_linear(*store, "projections.enc_to_lm_proj");
    weights.lm_to_dit_proj  = get_linear(*store, "projections.lm_to_dit_proj");
    weights.res_to_dit_proj = get_linear(*store, "projections.res_to_dit_proj");
    weights.res_fusion_proj = get_linear(*store, "projections.res_fusion_proj");

    if (weights.enc_to_lm_proj.weight) {
        config.dit_hidden_size = static_cast<int>(weights.enc_to_lm_proj.weight->ne[0]);
        config.lm_hidden_size  = static_cast<int>(weights.enc_to_lm_proj.weight->ne[1]);
    }

    return validate_weights();
}

bool VoxCPM2Projections::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("VoxCPM2Projections: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "projections." })) {
        free();
        return false;
    }
    if (!bind_from_store()) {
        free();
        return false;
    }

    LOG_INF("VoxCPM2Projections: loaded lm_hidden=%d dit_hidden=%d\n", config.lm_hidden_size, config.dit_hidden_size);
    return true;
}

bool VoxCPM2Projections::init_manual(const VoxCPM2ProjectionsConfig &  cfg,
                                     const VoxCPM2ProjectionsWeights & manual_weights) {
    free();
    config  = cfg;
    weights = manual_weights;
    if (!validate_weights()) {
        weights = {};
        return false;
    }
    return true;
}

ggml_tensor * VoxCPM2Projections::enc_to_lm(ggml_context * ctx, ggml_tensor * locenc_hidden) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(locenc_hidden != nullptr);
    GGML_ASSERT(locenc_hidden->ne[0] == weights.enc_to_lm_proj.weight->ne[0]);
    return voxcpm2_linear(ctx, weights.enc_to_lm_proj.weight, weights.enc_to_lm_proj.bias, locenc_hidden);
}

ggml_tensor * VoxCPM2Projections::lm_to_dit(ggml_context * ctx, ggml_tensor * lm_hidden) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(lm_hidden != nullptr);
    GGML_ASSERT(lm_hidden->ne[0] == weights.lm_to_dit_proj.weight->ne[0]);
    return voxcpm2_linear(ctx, weights.lm_to_dit_proj.weight, weights.lm_to_dit_proj.bias, lm_hidden);
}

ggml_tensor * VoxCPM2Projections::res_to_dit(ggml_context * ctx, ggml_tensor * residual_hidden) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(residual_hidden != nullptr);
    GGML_ASSERT(residual_hidden->ne[0] == weights.res_to_dit_proj.weight->ne[0]);
    return voxcpm2_linear(ctx, weights.res_to_dit_proj.weight, weights.res_to_dit_proj.bias, residual_hidden);
}

ggml_tensor * VoxCPM2Projections::res_fusion(ggml_context * ctx,
                                             ggml_tensor *  blended,
                                             ggml_tensor *  feat_embed) const {
    return build_residual_fusion(ctx, blended, feat_embed);
}

ggml_tensor * VoxCPM2Projections::build_dit_condition(ggml_context * ctx,
                                                      ggml_tensor *  lm_hidden,
                                                      ggml_tensor *  residual_hidden) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(lm_hidden != nullptr);
    GGML_ASSERT(residual_hidden != nullptr);

    ggml_tensor * dit_hidden_1 = lm_to_dit(ctx, lm_hidden);
    ggml_tensor * dit_hidden_2 = res_to_dit(ctx, residual_hidden);
    if (has_fusion()) {
        return ggml_concat(ctx, dit_hidden_1, dit_hidden_2, 0);   // → 2*dit_hidden
    }
    return ggml_add(ctx, dit_hidden_1, dit_hidden_2);              // → dit_hidden
}

ggml_tensor * VoxCPM2Projections::build_residual_fusion(ggml_context * ctx,
                                                        ggml_tensor *  blended,
                                                        ggml_tensor *  feat_embed) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(blended != nullptr);
    GGML_ASSERT(feat_embed != nullptr);
    GGML_ASSERT(blended->ne[0] == config.lm_hidden_size);
    GGML_ASSERT(feat_embed->ne[0] == config.lm_hidden_size);

    if (has_fusion()) {
        ggml_tensor * fused_input = ggml_concat(ctx, blended, feat_embed, 0);
        return voxcpm2_linear(ctx, weights.res_fusion_proj.weight, weights.res_fusion_proj.bias, fused_input);
    }
    return ggml_add(ctx, blended, feat_embed);  // no projection — element-wise sum
}

StopTokenPredictor::~StopTokenPredictor() {
    free();
}

void StopTokenPredictor::free() {
    store.reset();
    weights = {};
    config  = {};
    backend = nullptr;
}

bool StopTokenPredictor::validate_weights() const {
    if (!weights.linear1_weight || !weights.linear1_bias || !weights.linear2_weight) {
        LOG_ERR("StopTokenPredictor: missing one or more tensors\n");
        return false;
    }
    if (weights.linear1_weight->ne[0] != config.hidden_size || weights.linear1_weight->ne[1] != config.hidden_size) {
        LOG_ERR("StopTokenPredictor: linear1.weight shape mismatch: got [%" PRId64 ", %" PRId64
                "], expected [%d, %d]\n",
                weights.linear1_weight->ne[0], weights.linear1_weight->ne[1], config.hidden_size, config.hidden_size);
        return false;
    }
    if (weights.linear1_bias->ne[0] != config.hidden_size) {
        LOG_ERR("StopTokenPredictor: linear1.bias shape mismatch: got %" PRId64 ", expected %d\n",
                weights.linear1_bias->ne[0], config.hidden_size);
        return false;
    }
    if (weights.linear2_weight->ne[0] != config.hidden_size || weights.linear2_weight->ne[1] != config.n_classes) {
        LOG_ERR("StopTokenPredictor: linear2.weight shape mismatch: got [%" PRId64 ", %" PRId64
                "], expected [%d, %d]\n",
                weights.linear2_weight->ne[0], weights.linear2_weight->ne[1], config.hidden_size, config.n_classes);
        return false;
    }
    return true;
}

bool StopTokenPredictor::bind_from_store() {
    if (!store) {
        return false;
    }

    weights.linear1_weight = store->get("stop_predictor.linear1.weight");
    weights.linear1_bias   = store->get("stop_predictor.linear1.bias");
    weights.linear2_weight = store->get("stop_predictor.linear2.weight");

    if (weights.linear1_weight) {
        config.hidden_size = static_cast<int>(weights.linear1_weight->ne[0]);
    }
    if (weights.linear2_weight) {
        config.n_classes = static_cast<int>(weights.linear2_weight->ne[1]);
    }
    return validate_weights();
}

bool StopTokenPredictor::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("StopTokenPredictor: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "stop_predictor." })) {
        free();
        return false;
    }
    if (!bind_from_store()) {
        free();
        return false;
    }

    LOG_INF("StopTokenPredictor: loaded hidden=%d classes=%d\n", config.hidden_size, config.n_classes);
    return true;
}

bool StopTokenPredictor::init_manual(const VoxCPM2StopTokenPredictorConfig &  cfg,
                                     const VoxCPM2StopTokenPredictorWeights & manual_weights) {
    free();
    config  = cfg;
    weights = manual_weights;
    if (!validate_weights()) {
        weights = {};
        return false;
    }
    return true;
}

ggml_tensor * StopTokenPredictor::forward(ggml_context * ctx, ggml_tensor * input) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(input->ne[0] == config.hidden_size);

    ggml_tensor * hidden = voxcpm2_linear(ctx, weights.linear1_weight, weights.linear1_bias, input);
    hidden               = ggml_silu(ctx, hidden);
    return ggml_mul_mat(ctx, weights.linear2_weight, hidden);
}
