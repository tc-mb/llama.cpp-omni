#pragma once

#include "attn-time-pack-policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct ggml_backend_cann_context;
struct ggml_cgraph;
struct ggml_tensor;

struct ggml_cann_attn_time_pack_stats_snapshot {
    uint64_t attempts = 0;
    uint64_t hits = 0;
    std::array<uint64_t,
               static_cast<size_t>(ggml_cann_attn_time_pack_fallback::COUNT)>
        fallback{};
};

bool ggml_cann_attn_time_pack_enabled();
bool ggml_cann_attn_time_pack_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * permute_index,
        int * cont_index);
const char * ggml_cann_attn_time_pack_fallback_name(
        ggml_cann_attn_time_pack_fallback reason);
void ggml_cann_attn_time_pack_stats_reset();
ggml_cann_attn_time_pack_stats_snapshot
ggml_cann_attn_time_pack_stats_get();
