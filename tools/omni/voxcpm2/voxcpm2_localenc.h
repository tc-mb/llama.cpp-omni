#pragma once

#include "voxcpm2_transformer.h"

#include <memory>
#include <string>

struct VoxCPM2LocEncConfig {
    int feat_dim   = 64;
    int patch_size = 4;
    VoxCPM2TransformerConfig transformer;
};

struct VoxCPM2LocEncWeights {
    ggml_tensor * in_proj_weight = nullptr; // [feat_dim, hidden_size]
    ggml_tensor * in_proj_bias   = nullptr; // [hidden_size]
    ggml_tensor * cls_token      = nullptr; // [hidden_size]
    VoxCPM2TransformerWeights transformer;
};

struct LocEncModel {
    VoxCPM2LocEncConfig config;
    VoxCPM2LocEncWeights weights;

    ggml_backend_t backend = nullptr; // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;

    LocEncModel() = default;
    ~LocEncModel();

    LocEncModel(const LocEncModel &) = delete;
    LocEncModel & operator=(const LocEncModel &) = delete;
    LocEncModel(LocEncModel &&) = delete;
    LocEncModel & operator=(LocEncModel &&) = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2LocEncConfig & cfg,
                     ggml_tensor * in_proj_weight,
                     ggml_tensor * in_proj_bias,
                     ggml_tensor * cls_token,
                     const VoxCPM2TransformerWeights & transformer_weights);

    // input: [feat_dim, patch_size] or [hidden_size, patch_size]
    // output: [hidden_size] CLS token
    ggml_tensor * forward_patch(ggml_context * ctx, ggml_tensor * input) const;

    // input: [feat_dim, patch_size, seq_len] or [hidden_size, patch_size, seq_len]
    // output: [hidden_size, seq_len]
    ggml_tensor * forward_sequence(ggml_context * ctx, ggml_tensor * input) const;

    void free();

private:
    bool bind_from_store();
    bool validate_weights() const;
};
