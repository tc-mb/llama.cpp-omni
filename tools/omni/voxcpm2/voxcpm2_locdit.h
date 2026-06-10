#pragma once

#include "voxcpm2_transformer.h"

#include <memory>
#include <string>
#include <vector>

struct UnifiedCFMSolver;

struct VoxCPM2LocDiTConfig {
    int                      feat_dim   = 64;
    int                      patch_size = 4;
    float                    cfg_rate   = 2.0f;
    // VoxCPM v1/v1.5 LocDiT uses one generated prefix token:
    //   (mu + time)
    // VoxCPM2 uses separate prefix token(s):
    //   mu_token(s), time
    bool                     combine_mu_time = false;
    VoxCPM2TransformerConfig transformer;
};

struct VoxCPM2LocDiTWeights {
    ggml_tensor * in_proj_weight   = nullptr;  // [feat_dim, hidden_size]
    ggml_tensor * in_proj_bias     = nullptr;  // [hidden_size]
    ggml_tensor * cond_proj_weight = nullptr;  // [feat_dim, hidden_size]
    ggml_tensor * cond_proj_bias   = nullptr;  // [hidden_size]
    ggml_tensor * out_proj_weight  = nullptr;  // [hidden_size, feat_dim]
    ggml_tensor * out_proj_bias    = nullptr;  // [feat_dim]

    ggml_tensor * time_mlp_linear1_weight = nullptr;
    ggml_tensor * time_mlp_linear1_bias   = nullptr;
    ggml_tensor * time_mlp_linear2_weight = nullptr;
    ggml_tensor * time_mlp_linear2_bias   = nullptr;

    ggml_tensor * delta_time_mlp_linear1_weight = nullptr;
    ggml_tensor * delta_time_mlp_linear1_bias   = nullptr;
    ggml_tensor * delta_time_mlp_linear2_weight = nullptr;
    ggml_tensor * delta_time_mlp_linear2_bias   = nullptr;

    VoxCPM2TransformerWeights transformer;
};

struct LocDiTModel {
    VoxCPM2LocDiTConfig  config;
    VoxCPM2LocDiTWeights weights;

    ggml_backend_t                          backend = nullptr;  // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;

    LocDiTModel() = default;
    ~LocDiTModel();

    LocDiTModel(const LocDiTModel &)             = delete;
    LocDiTModel & operator=(const LocDiTModel &) = delete;
    LocDiTModel(LocDiTModel &&)                  = delete;
    LocDiTModel & operator=(LocDiTModel &&)      = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2LocDiTConfig & cfg, const VoxCPM2LocDiTWeights & manual_weights);

    ggml_tensor * build_time_embedding(ggml_context * ctx, ggml_tensor * t_scalar) const;
    ggml_tensor * build_delta_time_embedding(ggml_context * ctx, ggml_tensor * dt_scalar) const;
    ggml_tensor * build_time_embedding(ggml_context * ctx, float t) const;
    ggml_tensor * build_cfg_mask(ggml_context * ctx, int branch_len) const;

    // x / cond: [feat_dim, seq_len], mu: [hidden_size * n_mu_tokens]
    // output: [feat_dim, seq_len]
    ggml_tensor * forward_single(ggml_context * ctx,
                                 ggml_tensor *  x,
                                 ggml_tensor *  mu,
                                 ggml_tensor *  t_scalar,
                                 ggml_tensor *  cond,
                                 ggml_tensor *  dt_scalar) const;

    void forward_cfg_pair(ggml_context * ctx,
                          ggml_tensor *  x,
                          ggml_tensor *  mu,
                          ggml_tensor *  t_scalar,
                          ggml_tensor *  cond,
                          ggml_tensor *  dt_scalar,
                          ggml_tensor ** conditioned,
                          ggml_tensor ** unconditioned) const;

    // Convenience CFG blend: uncond + cfg_rate * (cond - uncond)
    ggml_tensor * forward_cfg_pair(ggml_context * ctx,
                                   ggml_tensor *  x,
                                   ggml_tensor *  mu,
                                   float          t,
                                   ggml_tensor *  cond,
                                   float          dt,
                                   float          cfg_rate) const;

    std::vector<float> precompute_cfg_time_table(const std::vector<float> & t_values,
                                                 ggml_backend_t             compute_backend) const;

    void free();

  private:
    friend struct UnifiedCFMSolver;

    bool bind_from_store();
    bool validate_weights() const;

    ggml_tensor * sinusoidal_embedding(ggml_context * ctx, ggml_tensor * scalar, int dim, float scale) const;
    ggml_tensor * timestep_mlp(ggml_context * ctx,
                               ggml_tensor *  input,
                               ggml_tensor *  linear1_w,
                               ggml_tensor *  linear1_b,
                               ggml_tensor *  linear2_w,
                               ggml_tensor *  linear2_b) const;
    ggml_tensor * project_input(ggml_context * ctx, ggml_tensor * x) const;
    ggml_tensor * project_condition(ggml_context * ctx, ggml_tensor * cond) const;
    ggml_tensor * build_time_token(ggml_context * ctx, ggml_tensor * t_scalar, ggml_tensor * dt_scalar) const;
    ggml_tensor * reshape_mu_tokens(ggml_context * ctx, ggml_tensor * mu) const;
    ggml_tensor * forward_projected(ggml_context * ctx,
                                    ggml_tensor *  x_proj,
                                    ggml_tensor *  prefix_tokens,
                                    ggml_tensor *  cond_proj,
                                    int            generated_prefix_tokens,
                                    int            prefix_len,
                                    int            seq_len) const;
    void          forward_cfg_pair_projected(ggml_context * ctx,
                                             ggml_tensor *  x_proj,
                                             ggml_tensor *  mu,
                                             ggml_tensor *  time_token,
                                             ggml_tensor *  cond_proj,
                                             int            prefix_len,
                                             ggml_tensor ** conditioned,
                                             ggml_tensor ** unconditioned) const;
};
