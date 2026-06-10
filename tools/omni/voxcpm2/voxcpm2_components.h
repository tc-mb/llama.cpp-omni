#pragma once

#include "voxcpm2_transformer.h"

#include <memory>
#include <string>

struct VoxCPM2LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct VoxCPM2ProjectionsConfig {
    int lm_hidden_size  = 2048;
    int dit_hidden_size = 1024;
};

struct VoxCPM2ProjectionsWeights {
    VoxCPM2LinearWeights enc_to_lm_proj;
    VoxCPM2LinearWeights lm_to_dit_proj;
    VoxCPM2LinearWeights res_to_dit_proj;
    VoxCPM2LinearWeights res_fusion_proj;
};

struct VoxCPM2Projections {
    VoxCPM2ProjectionsConfig  config;
    VoxCPM2ProjectionsWeights weights;

    ggml_backend_t                          backend = nullptr;  // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;

    VoxCPM2Projections() = default;
    ~VoxCPM2Projections();

    VoxCPM2Projections(const VoxCPM2Projections &)             = delete;
    VoxCPM2Projections & operator=(const VoxCPM2Projections &) = delete;
    VoxCPM2Projections(VoxCPM2Projections &&)                  = delete;
    VoxCPM2Projections & operator=(VoxCPM2Projections &&)      = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2ProjectionsConfig & cfg, const VoxCPM2ProjectionsWeights & manual_weights);

    ggml_tensor * enc_to_lm(ggml_context * ctx, ggml_tensor * locenc_hidden) const;
    ggml_tensor * lm_to_dit(ggml_context * ctx, ggml_tensor * lm_hidden) const;
    ggml_tensor * res_to_dit(ggml_context * ctx, ggml_tensor * residual_hidden) const;
    ggml_tensor * res_fusion(ggml_context * ctx, ggml_tensor * blended, ggml_tensor * feat_embed) const;

    // Whether the fusion_concat_proj linear layer is present.
    // Present in VoxCPM2 (v2.0); absent in VoxCPM-0.5B / VoxCPM-1.5.
    bool has_fusion() const { return weights.res_fusion_proj.weight != nullptr; }

    // Build LocDiT condition vector.
    //   v2.0: concat([lm_to_dit(lm_hidden), res_to_dit(residual_hidden)])  → 2*dit_hidden
    //   v0.5/1.5: add(lm_to_dit(lm_hidden), res_to_dit(residual_hidden))   → dit_hidden
    ggml_tensor * build_dit_condition(ggml_context * ctx, ggml_tensor * lm_hidden, ggml_tensor * residual_hidden) const;

    // Build ResidualLM input.
    //   v2.0: fusion_concat_proj(concat([blended, feat_embed]))  → lm_hidden
    //   v0.5/1.5: add(blended, feat_embed)                       → lm_hidden
    ggml_tensor * build_residual_fusion(ggml_context * ctx, ggml_tensor * blended, ggml_tensor * feat_embed) const;

    void free();

  private:
    bool bind_from_store();
    bool validate_weights() const;
};

struct VoxCPM2StopTokenPredictorConfig {
    int hidden_size = 2048;
    int n_classes   = 2;
};

struct VoxCPM2StopTokenPredictorWeights {
    ggml_tensor * linear1_weight = nullptr;
    ggml_tensor * linear1_bias   = nullptr;
    ggml_tensor * linear2_weight = nullptr;  // no bias
};

struct StopTokenPredictor {
    VoxCPM2StopTokenPredictorConfig  config;
    VoxCPM2StopTokenPredictorWeights weights;

    ggml_backend_t                          backend = nullptr;  // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;

    StopTokenPredictor() = default;
    ~StopTokenPredictor();

    StopTokenPredictor(const StopTokenPredictor &)             = delete;
    StopTokenPredictor & operator=(const StopTokenPredictor &) = delete;
    StopTokenPredictor(StopTokenPredictor &&)                  = delete;
    StopTokenPredictor & operator=(StopTokenPredictor &&)      = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2StopTokenPredictorConfig &  cfg,
                     const VoxCPM2StopTokenPredictorWeights & manual_weights);

    // input: [hidden_size] or [hidden_size, n_tokens]; output: [2] or [2, n_tokens]
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * input) const;

    ggml_tensor * predict(ggml_context * ctx, ggml_tensor * input) const { return forward(ctx, input); }

    void free();

  private:
    bool bind_from_store();
    bool validate_weights() const;
};
