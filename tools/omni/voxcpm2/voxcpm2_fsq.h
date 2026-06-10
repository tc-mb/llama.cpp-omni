#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>

/**
 * VoxCPM2 FSQ (Finite Scalar Quantization) module.
 *
 * HuggingFace reference:
 *   Linear(hidden_size -> latent_dim)
 *   tanh
 *   round(x * scale) / scale
 *   Linear(latent_dim -> hidden_size)
 *
 * GGML tensor layout follows the converter:
 *   fsq.in_proj.weight   [hidden_size, latent_dim]
 *   fsq.in_proj.bias     [latent_dim]
 *   fsq.out_proj.weight  [latent_dim, hidden_size]
 *   fsq.out_proj.bias    [hidden_size]
 */
struct VoxCPM2FSQConfig {
    int   hidden_size = 2048;
    int   latent_dim  = 512;
    float scale       = 9.0f;
};

struct VoxCPM2FSQWeights {
    ggml_tensor * in_proj_weight  = nullptr;
    ggml_tensor * in_proj_bias    = nullptr;
    ggml_tensor * out_proj_weight = nullptr;
    ggml_tensor * out_proj_bias   = nullptr;
};

struct FSQModule {
    VoxCPM2FSQConfig config;
    VoxCPM2FSQWeights weights;

    ggml_backend_t        backend       = nullptr; // not owned
    ggml_context *        ctx_meta      = nullptr;
    ggml_context *        ctx_weights   = nullptr;
    gguf_context *        ctx_gguf      = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    FSQModule() = default;
    ~FSQModule();

    FSQModule(const FSQModule &) = delete;
    FSQModule & operator=(const FSQModule &) = delete;
    FSQModule(FSQModule &&) = delete;
    FSQModule & operator=(FSQModule &&) = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2FSQConfig & cfg,
                     ggml_tensor * in_proj_weight,
                     ggml_tensor * in_proj_bias,
                     ggml_tensor * out_proj_weight,
                     ggml_tensor * out_proj_bias);

    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * x) const;
    void free();

private:
    bool load_metadata();
    bool load_weights(const std::string & path);
    bool validate_weights() const;
};

ggml_tensor * voxcpm2_fsq_quantize(ggml_context * ctx, ggml_tensor * x, float scale);
