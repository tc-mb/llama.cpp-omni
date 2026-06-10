#include "voxcpm2_cfm.h"

#include <algorithm>
#include <cmath>
#include <cstring>

std::vector<float> UnifiedCFMSolver::build_timesteps(int n_steps, float sway_sampling_coef) {
    GGML_ASSERT(n_steps > 0);

    constexpr float    pi_half = 1.57079632679489661923f;
    std::vector<float> t_span(static_cast<size_t>(n_steps) + 1);
    for (int i = 0; i <= n_steps; ++i) {
        const float base               = 1.0f - static_cast<float>(i) / static_cast<float>(n_steps);
        t_span[static_cast<size_t>(i)] = base + sway_sampling_coef * (std::cos(pi_half * base) - 1.0f + base);
    }
    return t_span;
}

std::vector<float> UnifiedCFMSolver::build_timesteps(int n_steps) const {
    return build_timesteps(n_steps, config.sway_sampling_coef);
}

ggml_tensor * UnifiedCFMSolver::optimized_scale(ggml_context * ctx,
                                                ggml_tensor *  positive,
                                                ggml_tensor *  negative,
                                                float          eps) const {
    ggml_tensor * dot_product  = ggml_sum(ctx, ggml_mul(ctx, positive, negative));
    ggml_tensor * squared_norm = ggml_sum(ctx, ggml_mul(ctx, negative, negative));
    ggml_tensor * eps_tensor   = ggml_arange(ctx, eps, eps + 1.0f, 1.0f);
    return ggml_div(ctx, dot_product, ggml_add(ctx, squared_norm, eps_tensor));
}

ggml_tensor * UnifiedCFMSolver::time_embedding_from_table(ggml_context * ctx,
                                                          ggml_tensor *  precomputed_time_table,
                                                          int            hidden_size,
                                                          int            step_index) const {
    GGML_ASSERT(precomputed_time_table != nullptr);
    GGML_ASSERT(precomputed_time_table->ne[0] == hidden_size);
    return ggml_view_1d(
        ctx, precomputed_time_table, hidden_size,
        static_cast<size_t>(step_index) * static_cast<size_t>(hidden_size) * precomputed_time_table->nb[0]);
}

ggml_tensor * UnifiedCFMSolver::compute_velocity_with_cfg(ggml_context *      ctx,
                                                          ggml_tensor *       x,
                                                          ggml_tensor *       mu,
                                                          ggml_tensor *       cond_proj,
                                                          int                 prefix_len,
                                                          ggml_tensor *       delta_time_zero,
                                                          ggml_tensor *       combined_time_embedding,
                                                          float               t,
                                                          float               cfg_rate,
                                                          bool                use_cfg_zero_star,
                                                          const LocDiTModel & dit_model) const {
    ggml_tensor * x_proj = dit_model.project_input(ctx, x);
    if (!combined_time_embedding) {
        ggml_tensor * t_scalar  = ggml_arange(ctx, t, t + 1.0f, 1.0f);
        combined_time_embedding = dit_model.build_time_embedding(ctx, t_scalar);
        if (delta_time_zero) {
            combined_time_embedding = ggml_add(ctx, combined_time_embedding, delta_time_zero);
        }
    }

    ggml_tensor * time_token =
        ggml_reshape_2d(ctx, combined_time_embedding, dit_model.config.transformer.hidden_size, 1);

    ggml_tensor * conditioned   = nullptr;
    ggml_tensor * unconditioned = nullptr;
    dit_model.forward_cfg_pair_projected(ctx, x_proj, mu, time_token, cond_proj, prefix_len, &conditioned,
                                         &unconditioned);

    if (use_cfg_zero_star) {
        ggml_tensor * st_star       = optimized_scale(ctx, conditioned, unconditioned);
        ggml_tensor * scaled_uncond = ggml_mul(ctx, unconditioned, ggml_repeat(ctx, st_star, unconditioned));
        return ggml_add(ctx, scaled_uncond, ggml_scale(ctx, ggml_sub(ctx, conditioned, scaled_uncond), cfg_rate));
    }

    return ggml_add(ctx, unconditioned, ggml_scale(ctx, ggml_sub(ctx, conditioned, unconditioned), cfg_rate));
}

ggml_tensor * UnifiedCFMSolver::solve(ggml_context *      ctx,
                                      ggml_tensor *       noise,
                                      ggml_tensor *       mu,
                                      ggml_tensor *       cond,
                                      const LocDiTModel & dit_model) const {
    return solve(ctx, noise, mu, cond, dit_model, config.n_steps, config.cfg_rate, config.temperature,
                 config.sway_sampling_coef, config.use_cfg_zero_star, nullptr);
}

ggml_tensor * UnifiedCFMSolver::solve(ggml_context *      ctx,
                                      ggml_tensor *       noise,
                                      ggml_tensor *       mu,
                                      ggml_tensor *       cond,
                                      const LocDiTModel & dit_model,
                                      int                 n_steps,
                                      float               cfg_rate,
                                      float               temperature,
                                      float               sway_sampling_coef,
                                      bool                use_cfg_zero_star,
                                      ggml_tensor *       precomputed_time_table) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(noise != nullptr);
    GGML_ASSERT(mu != nullptr);
    GGML_ASSERT(cond != nullptr);
    GGML_ASSERT(noise->ne[0] == dit_model.config.feat_dim);
    GGML_ASSERT(cond->ne[0] == dit_model.config.feat_dim);
    GGML_ASSERT(mu->ne[0] % dit_model.config.transformer.hidden_size == 0);
    GGML_ASSERT(n_steps > 0);

    ggml_tensor *            x               = (temperature == 1.0f) ? noise : ggml_scale(ctx, noise, temperature);
    const std::vector<float> t_span          = build_timesteps(n_steps, sway_sampling_coef);
    float                    t               = t_span[0];
    float                    dt              = t_span[0] - t_span[1];
    const int                prefix_len      = static_cast<int>(cond->ne[1]);
    const int                zero_init_steps = n_steps > 1 ? std::max(1, static_cast<int>(t_span.size() * 0.04f)) : 0;

    ggml_tensor * cond_proj       = prefix_len > 0 ? dit_model.project_condition(ctx, cond) : nullptr;
    ggml_tensor * delta_time_zero = nullptr;
    if (!precomputed_time_table) {
        ggml_tensor * zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);
        delta_time_zero           = dit_model.build_delta_time_embedding(ctx, zero_scalar);
    }

    for (int step = 1; step <= n_steps; ++step) {
        ggml_tensor * velocity = nullptr;
        if (use_cfg_zero_star && step <= zero_init_steps) {
            velocity = ggml_scale(ctx, x, 0.0f);
        } else {
            ggml_tensor * combined_time_embedding = nullptr;
            if (precomputed_time_table) {
                combined_time_embedding = time_embedding_from_table(ctx, precomputed_time_table,
                                                                    dit_model.config.transformer.hidden_size, step - 1);
            }
            velocity = compute_velocity_with_cfg(ctx, x, mu, cond_proj, prefix_len, delta_time_zero,
                                                 combined_time_embedding, t, cfg_rate, use_cfg_zero_star, dit_model);
        }

        x = ggml_sub(ctx, x, ggml_scale(ctx, velocity, dt));
        t -= dt;
        if (step < n_steps) {
            dt = t - t_span[static_cast<size_t>(step + 1)];
        }
    }

    return x;
}

// ---------------------------------------------------------------------------
// Staged solver helpers
// ---------------------------------------------------------------------------

namespace {

struct CfmGgmlContextGuard {
    ggml_context * ctx = nullptr;

    explicit CfmGgmlContextGuard(size_t mem_size, bool no_alloc = true) {
        ggml_init_params params{};
        params.mem_size   = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc   = no_alloc;
        ctx               = ggml_init(params);
    }

    ~CfmGgmlContextGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    ggml_context * get() const { return ctx; }
};

struct CfmBackendBufferGuard {
    ggml_backend_buffer_t buffer = nullptr;

    explicit CfmBackendBufferGuard(ggml_backend_buffer_t buf) : buffer(buf) {}

    ~CfmBackendBufferGuard() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

static std::vector<float> cfm_tensor_to_vec(ggml_tensor * t) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

static bool cfm_finite_vec(const std::vector<float> & v) {
    return std::all_of(v.begin(), v.end(), [](float f) { return std::isfinite(f); });
}

constexpr size_t kCfmGraphMem   = 256ull * 1024 * 1024;
constexpr size_t kCfmGraphNodes = 65536;

}  // namespace

std::vector<float> UnifiedCFMSolver::solve_per_step(ggml_backend_t             backend,
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
                                                    bool                       use_cfg_zero_star) const {
    const int feat_dim    = dit_model.config.feat_dim;
    const int patch_size  = dit_model.config.patch_size;
    const int patch_elems = feat_dim * patch_size;

    if (!backend) {
        return {};
    }
    if (static_cast<int>(noise.size()) != patch_elems) {
        return {};
    }
    if (n_steps <= 0) {
        return noise;
    }

    const auto t_span          = build_timesteps(n_steps, sway_sampling_coef);
    const int  zero_init_steps = n_steps > 1 ? std::max(1, static_cast<int>(t_span.size() * 0.04f)) : 0;

    // Euler state (CPU-side)
    std::vector<float> x_data = noise;
    if (temperature != 1.0f) {
        for (float & v : x_data) {
            v *= temperature;
        }
    }
    float t  = t_span[0];
    float dt = t_span[0] - t_span[1];

    for (int step = 1; step <= n_steps; ++step) {
        std::vector<float> velocity(patch_elems, 0.0f);

        if (!(use_cfg_zero_star && step <= zero_init_steps)) {
            // Build a fresh ggml context + graph for this Euler step.
            // Everything (mu, cond_proj, time_token) computed in-graph —
            // no pre-computed tensor passing across graph boundaries.
            CfmGgmlContextGuard ctx_guard(kCfmGraphMem, true);
            ggml_context *      ctx = ctx_guard.get();
            if (!ctx) {
                return {};
            }

            // Input tensors
            ggml_tensor * x_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, patch_size);
            ggml_tensor * lm_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(lm_hidden.size()));
            ggml_tensor * res_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(residual_hidden.size()));
            ggml_tensor * cond_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, patch_size);
            // NOTE: deliberately NOT using ggml_set_input — matching the standalone
            // test_cfm_euler_step.cpp which achieves COS=0.999999 with Python.

            // Compute mu = build_dit_condition(lm, res) — FRESH in this graph
            ggml_tensor * mu = projections.build_dit_condition(ctx, lm_t, res_t);

            // Build time scalars
            ggml_tensor * t_scalar  = ggml_arange(ctx, t, t + 1.0f, 1.0f);
            ggml_tensor * dt_scalar = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);

            // forward_cfg_pair handles project_input, project_condition,
            // build_time_token, and forward_cfg_pair_projected internally
            ggml_tensor * conditioned   = nullptr;
            ggml_tensor * unconditioned = nullptr;
            dit_model.forward_cfg_pair(ctx, x_t, mu, t_scalar, cond_t, dt_scalar, &conditioned, &unconditioned);

            // CFG blending — also in-graph for bit-exactness
            ggml_tensor * vel;
            if (use_cfg_zero_star) {
                ggml_tensor * dot_p     = ggml_sum(ctx, ggml_mul(ctx, conditioned, unconditioned));
                ggml_tensor * sq_n      = ggml_sum(ctx, ggml_mul(ctx, unconditioned, unconditioned));
                ggml_tensor * eps_t     = ggml_arange(ctx, 1.0e-8f, 1.0e-8f + 1.0f, 1.0f);
                ggml_tensor * st_star   = ggml_div(ctx, dot_p, ggml_add(ctx, sq_n, eps_t));
                ggml_tensor * sc_uncond = ggml_mul(ctx, unconditioned, ggml_repeat(ctx, st_star, unconditioned));
                vel = ggml_add(ctx, sc_uncond, ggml_scale(ctx, ggml_sub(ctx, conditioned, sc_uncond), cfg_rate));
            } else {
                vel =
                    ggml_add(ctx, unconditioned, ggml_scale(ctx, ggml_sub(ctx, conditioned, unconditioned), cfg_rate));
            }

            ggml_cgraph * graph = ggml_new_graph_custom(ctx, kCfmGraphNodes, false);
            ggml_set_output(vel);
            ggml_build_forward_expand(graph, vel);

            CfmBackendBufferGuard buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
            if (!buf.buffer) {
                return {};
            }

            ggml_backend_tensor_set(x_t, x_data.data(), 0, x_data.size() * sizeof(float));
            ggml_backend_tensor_set(lm_t, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
            ggml_backend_tensor_set(res_t, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));
            ggml_backend_tensor_set(cond_t, prefix_feat_cond.data(), 0, prefix_feat_cond.size() * sizeof(float));

            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                return {};
            }

            velocity = cfm_tensor_to_vec(vel);
        }

        // CPU-side Euler update
        for (size_t i = 0; i < x_data.size(); ++i) {
            x_data[i] -= dt * velocity[i];
        }
        t -= dt;
        if (step < n_steps) {
            dt = t - t_span[static_cast<size_t>(step + 1)];
        }
    }

    return cfm_finite_vec(x_data) ? x_data : std::vector<float>{};
}
