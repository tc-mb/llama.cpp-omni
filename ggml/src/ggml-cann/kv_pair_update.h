#pragma once

#include "kv-pair-update-policy.h"

#include <array>
#include <cstddef>
#include <cstdint>

struct ggml_backend_cann_context;
struct ggml_tensor;

struct ggml_cann_kv_pair_update_stats_snapshot {
    uint64_t attempts = 0;
    uint64_t hits = 0;
    std::array<uint64_t,
               static_cast<size_t>(ggml_cann_kv_pair_update_fallback::COUNT)>
        fallback{};
};

ggml_cann_kv_pair_update_fallback ggml_cann_kv_pair_update_validate(
        const ggml_tensor * first,
        const ggml_tensor * second);
bool ggml_cann_kv_pair_update_try(
        ggml_backend_cann_context & ctx,
        ggml_tensor * first,
        ggml_tensor * second);
const char * ggml_cann_kv_pair_update_fallback_name(
        ggml_cann_kv_pair_update_fallback reason);
void ggml_cann_kv_pair_update_stats_reset();
ggml_cann_kv_pair_update_stats_snapshot
ggml_cann_kv_pair_update_stats_get();
