#pragma once

#include <cstdint>
#include <unordered_map>

struct ggml_backend_cann_context;
struct ggml_cgraph;
struct ggml_tensor;

struct ggml_cann_affine_layer_norm_stats_snapshot {
    uint64_t attempts = 0;
    uint64_t hits = 0;
};

bool ggml_cann_affine_layer_norm_enabled();
bool ggml_cann_affine_layer_norm_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * mul_index,
        int * add_index);
void ggml_cann_affine_layer_norm_stats_reset();
ggml_cann_affine_layer_norm_stats_snapshot
ggml_cann_affine_layer_norm_stats_get();
