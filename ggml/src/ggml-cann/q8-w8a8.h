#pragma once

#include "../ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class ggml_cann_q8_w8a8_status {
    ok,
    invalid_shape,
    insufficient_capacity,
    non_finite_scale,
};

struct ggml_cann_q8_w8a8_result {
    ggml_cann_q8_w8a8_status status;
    size_t weight_bytes;
    size_t scale_offset;
    size_t scale_bytes;
};

struct ggml_cann_q8_w8a8_layout {
    size_t weight_bytes;
    size_t scale_offset;
    size_t scale_bytes;
    int64_t k;
    int64_t n;
};

struct ggml_cann_q8_w8a8_stats {
    uint64_t matmul_hits;
    uint64_t graph_workspace_allocations;
    uint64_t graph_workspace_frees;
};

enum class ggml_cann_q8_w8a8_workspace_status {
    ok,
    invalid_shape,
    overflow,
};

struct ggml_cann_q8_w8a8_workspace_plan {
    ggml_cann_q8_w8a8_workspace_status status;
    ggml_type input_type;
    ggml_type output_type;
    int64_t k;
    int64_t n;
    int64_t m;
    int64_t ne2;
    int64_t ne3;
    size_t input_f16_offset;
    size_t input_f16_bytes;
    size_t quant_offset;
    size_t quant_bytes;
    size_t token_scale_offset;
    size_t token_scale_bytes;
    size_t output_f16_offset;
    size_t output_f16_bytes;
    size_t total_bytes;
};

struct ggml_cann_q8_w8a8_graph_node_snapshot {
    bool registered = false;
    ggml_cann_q8_w8a8_layout layout = {};
    ggml_type input_type = GGML_TYPE_COUNT;
    ggml_type output_type = GGML_TYPE_COUNT;
    int64_t m = 0;
    int64_t ne2 = 0;
    int64_t ne3 = 0;
};

struct ggml_cann_q8_w8a8_graph_snapshot {
    std::vector<ggml_cann_q8_w8a8_graph_node_snapshot> nodes;

    void capture_from_cgraph(const ggml_cgraph * cgraph);
};

bool ggml_cann_q8_w8a8_graph_node_snapshot_matches(
    const ggml_cann_q8_w8a8_graph_node_snapshot & lhs,
    const ggml_cann_q8_w8a8_graph_node_snapshot & rhs);

enum class ggml_cann_q8_w8a8_reject {
    none,
    disabled,
    wrong_type,
    not_matmul_weight,
    batched,
    invalid_shape,
    insufficient_capacity,
};

GGML_API size_t ggml_cann_q8_w8a8_required_size(int64_t k, int64_t n);

GGML_API ggml_cann_q8_w8a8_workspace_plan ggml_cann_q8_w8a8_plan_workspace(
    ggml_type input_type,
    ggml_type output_type,
    int64_t k,
    int64_t n,
    int64_t m,
    int64_t ne2,
    int64_t ne3);

GGML_API bool ggml_cann_q8_w8a8_workspace_matches(
    const ggml_cann_q8_w8a8_workspace_plan & plan,
    ggml_type input_type,
    ggml_type output_type,
    int64_t k,
    int64_t n,
    int64_t m,
    int64_t ne2,
    int64_t ne3);

GGML_API ggml_cann_q8_w8a8_result ggml_cann_requantize_q8_0_per_channel(
    const void * src,
    int64_t k,
    int64_t n,
    void * dst,
    size_t dst_size);

GGML_API ggml_cann_q8_w8a8_reject ggml_cann_q8_w8a8_validate(
    bool enabled,
    ggml_type type,
    bool matmul_weight,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t allocation_size,
    ggml_cann_q8_w8a8_layout * layout);

GGML_API ggml_cann_q8_w8a8_result ggml_cann_restore_q8_0_from_per_channel(
    const void * src,
    int64_t k,
    int64_t n,
    void * dst,
    size_t dst_size);

GGML_API bool ggml_cann_get_q8_w8a8_layout(
    const ggml_tensor * tensor,
    ggml_cann_q8_w8a8_layout * layout);

GGML_API void ggml_cann_q8_w8a8_stats_reset();
GGML_API ggml_cann_q8_w8a8_stats ggml_cann_q8_w8a8_stats_get();
void ggml_cann_q8_w8a8_stats_note_matmul();
void ggml_cann_q8_w8a8_stats_note_graph_workspace_allocation();
void ggml_cann_q8_w8a8_stats_note_graph_workspace_free();

GGML_API bool ggml_cann_q8_w8a8_graph_compiled();
