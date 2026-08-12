#pragma once

#include <cstddef>
#include <cstdint>

enum class ggml_cann_set_rows_f32_f16_fallback : uint32_t {
    NONE = 0,
    DISABLED,
    SRC_DTYPE,
    DST_DTYPE,
    INDEX_DTYPE,
    SRC_LAYOUT,
    DST_LAYOUT,
    INDEX_LAYOUT,
    BATCH,
    INDEX_COUNT,
    ROW_WIDTH,
    CACHE_ROWS,
    ALIAS,
    ALIGNMENT,
    COUNT,
};

struct ggml_cann_set_rows_f32_f16_policy_input {
    bool src_is_f32 = false;
    bool dst_is_f16 = false;
    bool index_is_i64 = false;

    bool src_row_dense = false;
    bool dst_rows_dense = false;
    bool index_dense = false;
    bool buffers_disjoint = false;

    int64_t src_ne[4] = {};
    int64_t dst_ne[4] = {};
    int64_t index_ne[4] = {};

    uintptr_t src_address = 0;
    uintptr_t dst_address = 0;
    uintptr_t index_address = 0;
};

inline ggml_cann_set_rows_f32_f16_fallback
ggml_cann_set_rows_f32_f16_validate_policy(
        const ggml_cann_set_rows_f32_f16_policy_input & input) {
    if (!input.src_is_f32) {
        return ggml_cann_set_rows_f32_f16_fallback::SRC_DTYPE;
    }
    if (!input.dst_is_f16) {
        return ggml_cann_set_rows_f32_f16_fallback::DST_DTYPE;
    }
    if (!input.index_is_i64) {
        return ggml_cann_set_rows_f32_f16_fallback::INDEX_DTYPE;
    }
    if (!input.src_row_dense) {
        return ggml_cann_set_rows_f32_f16_fallback::SRC_LAYOUT;
    }
    if (!input.dst_rows_dense) {
        return ggml_cann_set_rows_f32_f16_fallback::DST_LAYOUT;
    }
    if (!input.index_dense) {
        return ggml_cann_set_rows_f32_f16_fallback::INDEX_LAYOUT;
    }

    if (input.src_ne[2] != 1 || input.src_ne[3] != 1 ||
        input.dst_ne[2] != 1 || input.dst_ne[3] != 1 ||
        input.index_ne[1] != 1 || input.index_ne[2] != 1 ||
        input.index_ne[3] != 1) {
        return ggml_cann_set_rows_f32_f16_fallback::BATCH;
    }
    if (input.src_ne[1] != 1 || input.index_ne[0] != 1) {
        return ggml_cann_set_rows_f32_f16_fallback::INDEX_COUNT;
    }
    if (input.src_ne[0] != input.dst_ne[0] ||
        (input.dst_ne[0] != 768 && input.dst_ne[0] != 1024)) {
        return ggml_cann_set_rows_f32_f16_fallback::ROW_WIDTH;
    }
    if (input.dst_ne[1] != 4096) {
        return ggml_cann_set_rows_f32_f16_fallback::CACHE_ROWS;
    }
    if (!input.buffers_disjoint) {
        return ggml_cann_set_rows_f32_f16_fallback::ALIAS;
    }

    constexpr uintptr_t dma_alignment = 32;
    if (input.src_address == 0 || input.dst_address == 0 ||
        input.index_address == 0 ||
        (input.src_address % dma_alignment) != 0 ||
        (input.dst_address % dma_alignment) != 0 ||
        (input.index_address % alignof(int64_t)) != 0) {
        return ggml_cann_set_rows_f32_f16_fallback::ALIGNMENT;
    }

    return ggml_cann_set_rows_f32_f16_fallback::NONE;
}
