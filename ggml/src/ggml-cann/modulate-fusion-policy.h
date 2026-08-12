#pragma once

#include <cstdint>

enum class ggml_cann_modulate_fusion_fallback : uint32_t {
    NONE = 0,
    DISABLED,
    NOT_FUSIBLE_CHAIN,
    EDGE_IDENTITY,
    DTYPE,
    X_SHAPE,
    PARAM_SHAPE,
    LAYOUT,
    ALIAS,
    ALIGNMENT,
    COUNT,
};

struct ggml_cann_modulate_fusion_policy_input {
    bool fusible_chain = false;
    bool exact_edges = false;
    bool all_f32 = false;
    bool x_shape = false;
    bool param_shape = false;
    bool supported_layout = false;
    bool buffers_disjoint = false;
    bool buffers_aligned = false;
};

inline ggml_cann_modulate_fusion_fallback
ggml_cann_modulate_fusion_validate_policy(
        const ggml_cann_modulate_fusion_policy_input & input) {
    if (!input.fusible_chain) {
        return ggml_cann_modulate_fusion_fallback::NOT_FUSIBLE_CHAIN;
    }
    if (!input.exact_edges) {
        return ggml_cann_modulate_fusion_fallback::EDGE_IDENTITY;
    }
    if (!input.all_f32) {
        return ggml_cann_modulate_fusion_fallback::DTYPE;
    }
    if (!input.x_shape) {
        return ggml_cann_modulate_fusion_fallback::X_SHAPE;
    }
    if (!input.param_shape) {
        return ggml_cann_modulate_fusion_fallback::PARAM_SHAPE;
    }
    if (!input.supported_layout) {
        return ggml_cann_modulate_fusion_fallback::LAYOUT;
    }
    if (!input.buffers_disjoint) {
        return ggml_cann_modulate_fusion_fallback::ALIAS;
    }
    if (!input.buffers_aligned) {
        return ggml_cann_modulate_fusion_fallback::ALIGNMENT;
    }
    return ggml_cann_modulate_fusion_fallback::NONE;
}

constexpr uint32_t ggml_cann_modulate_fusion_legacy_launches_per_hit = 3;
constexpr uint32_t ggml_cann_modulate_fusion_candidate_launches_per_hit = 1;
constexpr uint32_t ggml_cann_modulate_fusion_launches_removed_per_hit =
    ggml_cann_modulate_fusion_legacy_launches_per_hit -
    ggml_cann_modulate_fusion_candidate_launches_per_hit;
