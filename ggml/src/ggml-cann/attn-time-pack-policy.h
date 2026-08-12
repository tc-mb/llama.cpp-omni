#pragma once

#include <cstdint>

enum class ggml_cann_attn_time_pack_fallback : uint32_t {
    NONE = 0,
    DISABLED,
    NOT_EXACT_CHAIN,
    DTYPE,
    SHAPE,
    LAYOUT,
    ALIAS,
    ALIGNMENT,
    COUNT,
};

struct ggml_cann_attn_time_pack_policy_input {
    bool exact_chain = false;
    bool all_f32 = false;
    bool canonical_shape = false;
    bool supported_layout = false;
    bool buffers_disjoint = false;
    bool buffers_aligned = false;
};

inline ggml_cann_attn_time_pack_fallback
ggml_cann_attn_time_pack_validate_policy(
        const ggml_cann_attn_time_pack_policy_input & input) {
    if (!input.exact_chain) {
        return ggml_cann_attn_time_pack_fallback::NOT_EXACT_CHAIN;
    }
    if (!input.all_f32) {
        return ggml_cann_attn_time_pack_fallback::DTYPE;
    }
    if (!input.canonical_shape) {
        return ggml_cann_attn_time_pack_fallback::SHAPE;
    }
    if (!input.supported_layout) {
        return ggml_cann_attn_time_pack_fallback::LAYOUT;
    }
    if (!input.buffers_disjoint) {
        return ggml_cann_attn_time_pack_fallback::ALIAS;
    }
    if (!input.buffers_aligned) {
        return ggml_cann_attn_time_pack_fallback::ALIGNMENT;
    }
    return ggml_cann_attn_time_pack_fallback::NONE;
}

constexpr uint32_t ggml_cann_attn_time_pack_launches_removed_per_hit = 1;
