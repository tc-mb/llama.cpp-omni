#pragma once

#include "set-rows-f32-f16-policy.h"

#include <cstdint>

enum class ggml_cann_kv_pair_update_fallback : uint32_t {
    NONE = 0,
    DISABLED,
    NOT_ADJACENT_SET_ROWS,
    FIRST_POLICY,
    SECOND_POLICY,
    ROW_WIDTH,
    CACHE_ROWS,
    PAIR_ALIAS,
    COUNT,
};

struct ggml_cann_kv_pair_update_policy_input {
    bool adjacent_set_rows = false;
    ggml_cann_set_rows_f32_f16_fallback first_reason =
        ggml_cann_set_rows_f32_f16_fallback::NONE;
    ggml_cann_set_rows_f32_f16_fallback second_reason =
        ggml_cann_set_rows_f32_f16_fallback::NONE;
    int64_t first_row_width = 0;
    int64_t second_row_width = 0;
    int64_t first_cache_rows = 0;
    int64_t second_cache_rows = 0;
    bool pair_buffers_disjoint = false;
};

inline ggml_cann_kv_pair_update_fallback
ggml_cann_kv_pair_update_validate_policy(
        const ggml_cann_kv_pair_update_policy_input & input) {
    if (!input.adjacent_set_rows) {
        return ggml_cann_kv_pair_update_fallback::NOT_ADJACENT_SET_ROWS;
    }
    if (input.first_reason != ggml_cann_set_rows_f32_f16_fallback::NONE) {
        return ggml_cann_kv_pair_update_fallback::FIRST_POLICY;
    }
    if (input.second_reason != ggml_cann_set_rows_f32_f16_fallback::NONE) {
        return ggml_cann_kv_pair_update_fallback::SECOND_POLICY;
    }
    if (input.first_row_width != input.second_row_width) {
        return ggml_cann_kv_pair_update_fallback::ROW_WIDTH;
    }
    if (input.first_cache_rows != input.second_cache_rows) {
        return ggml_cann_kv_pair_update_fallback::CACHE_ROWS;
    }
    if (!input.pair_buffers_disjoint) {
        return ggml_cann_kv_pair_update_fallback::PAIR_ALIAS;
    }
    return ggml_cann_kv_pair_update_fallback::NONE;
}

constexpr uint32_t ggml_cann_kv_pair_update_legacy_launches_per_pair = 4;
constexpr uint32_t ggml_cann_kv_pair_update_candidate_launches_per_pair = 1;
constexpr uint32_t ggml_cann_kv_pair_update_launches_removed_per_hit =
    ggml_cann_kv_pair_update_legacy_launches_per_pair -
    ggml_cann_kv_pair_update_candidate_launches_per_pair;
