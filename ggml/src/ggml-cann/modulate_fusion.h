#pragma once

#include "modulate-fusion-policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct ggml_backend_cann_context;
struct ggml_cgraph;
struct ggml_tensor;

using ggml_cann_unique_consumer_map =
    std::unordered_map<const ggml_tensor *, int>;

struct ggml_cann_modulate_fusion_stats_snapshot {
    uint64_t attempts = 0;
    uint64_t hits = 0;
    uint64_t modulate_hits = 0;
    uint64_t gate_residual_hits = 0;
    uint64_t launches_removed = 0;
    std::array<uint64_t,
               static_cast<size_t>(ggml_cann_modulate_fusion_fallback::COUNT)>
        fallback{};
};

bool ggml_cann_modulate_fusion_enabled();
bool ggml_cann_modulate_fusion_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const ggml_cann_unique_consumer_map & unique_consumers,
        int * add1_index,
        int * add2_index);
const char * ggml_cann_modulate_fusion_fallback_name(
        ggml_cann_modulate_fusion_fallback reason);
void ggml_cann_modulate_fusion_stats_reset();
ggml_cann_modulate_fusion_stats_snapshot
ggml_cann_modulate_fusion_stats_get();
