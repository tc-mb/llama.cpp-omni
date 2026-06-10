#pragma once

#include "ggml-alloc.h"
#include "ggml.h"
#include "voxcpm2_components.h"
#include "voxcpm2_locdit.h"

#include <vector>

struct VoxCPM2CFMConfig {
    int   n_steps            = 10;
    float cfg_rate           = 2.0f;
    float sigma_min          = 1.0e-6f;
    float temperature        = 1.0f;
    float sway_sampling_coef = 1.0f;
    bool  use_cfg_zero_star  = true;
};

struct UnifiedCFMSolver {
    VoxCPM2CFMConfig config;

    UnifiedCFMSolver() = default;

    explicit UnifiedCFMSolver(const VoxCPM2CFMConfig & cfg) : config(cfg) {}

    static std::vector<float> build_timesteps(int n_steps, float sway_sampling_coef = 1.0f);
    std::vector<float>        build_timesteps(int n_steps) const;

    // noise / cond: [feat_dim, patch_size], mu: [hidden_size] or [hidden_size * n_mu_tokens]
    // output: [feat_dim, patch_size]
    ggml_tensor * solve(ggml_context *      ctx,
                        ggml_tensor *       noise,
                        ggml_tensor *       mu,
                        ggml_tensor *       cond,
                        const LocDiTModel & dit_model) const;

    // Debug solver: one fresh ggml graph per Euler step (mu recomputed each step).
    // Used for precision isolation — verifies that single-graph solve() produces
    // the same result as per-step computation. Not used in production inference.
    std::vector<float> solve_per_step(ggml_backend_t             backend,
                                      const std::vector<float> & noise,
                                      const std::vector<float> & lm_hidden,
                                      const std::vector<float> & residual_hidden,
                                      const std::vector<float> & prefix_feat_cond,
                                      const LocDiTModel &        dit_model,
                                      const VoxCPM2Projections & projections,
                                      int                        n_steps,
                                      float                      cfg_rate,
                                      float                      temperature,
                                      float                      sway_sampling_coef,
                                      bool                       use_cfg_zero_star) const;

    ggml_tensor * solve(ggml_context *      ctx,
                        ggml_tensor *       noise,
                        ggml_tensor *       mu,
                        ggml_tensor *       cond,
                        const LocDiTModel & dit_model,
                        int                 n_steps,
                        float               cfg_rate,
                        float               temperature            = 1.0f,
                        float               sway_sampling_coef     = 1.0f,
                        bool                use_cfg_zero_star      = true,
                        ggml_tensor *       precomputed_time_table = nullptr) const;

  private:
    ggml_tensor * optimized_scale(ggml_context * ctx,
                                  ggml_tensor *  positive,
                                  ggml_tensor *  negative,
                                  float          eps = 1.0e-8f) const;

    ggml_tensor * time_embedding_from_table(ggml_context * ctx,
                                            ggml_tensor *  precomputed_time_table,
                                            int            hidden_size,
                                            int            step_index) const;

    ggml_tensor * compute_velocity_with_cfg(ggml_context *      ctx,
                                            ggml_tensor *       x,
                                            ggml_tensor *       mu,
                                            ggml_tensor *       cond_proj,
                                            int                 prefix_len,
                                            ggml_tensor *       delta_time_zero,
                                            ggml_tensor *       combined_time_embedding,
                                            float               t,
                                            float               cfg_rate,
                                            bool                use_cfg_zero_star,
                                            const LocDiTModel & dit_model) const;
};
