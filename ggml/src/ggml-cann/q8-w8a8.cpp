#include "q8-w8a8.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

static std::atomic<uint64_t> g_q8_w8a8_matmul_hits{0};
static std::atomic<uint64_t> g_q8_w8a8_graph_workspace_allocations{0};
static std::atomic<uint64_t> g_q8_w8a8_graph_workspace_frees{0};

void ggml_cann_q8_w8a8_stats_reset() {
    g_q8_w8a8_matmul_hits.store(0, std::memory_order_relaxed);
    g_q8_w8a8_graph_workspace_allocations.store(0, std::memory_order_relaxed);
    g_q8_w8a8_graph_workspace_frees.store(0, std::memory_order_relaxed);
}

ggml_cann_q8_w8a8_stats ggml_cann_q8_w8a8_stats_get() {
    return {
        g_q8_w8a8_matmul_hits.load(std::memory_order_relaxed),
        g_q8_w8a8_graph_workspace_allocations.load(std::memory_order_relaxed),
        g_q8_w8a8_graph_workspace_frees.load(std::memory_order_relaxed),
    };
}

void ggml_cann_q8_w8a8_stats_note_matmul() {
    g_q8_w8a8_matmul_hits.fetch_add(1, std::memory_order_relaxed);
}

void ggml_cann_q8_w8a8_stats_note_graph_workspace_allocation() {
    g_q8_w8a8_graph_workspace_allocations.fetch_add(1, std::memory_order_relaxed);
}

void ggml_cann_q8_w8a8_stats_note_graph_workspace_free() {
    g_q8_w8a8_graph_workspace_frees.fetch_add(1, std::memory_order_relaxed);
}

bool ggml_cann_q8_w8a8_graph_compiled() {
#ifdef USE_ACL_GRAPH
    return true;
#else
    return false;
#endif
}

static ggml_cann_q8_w8a8_result make_result(
        ggml_cann_q8_w8a8_status status,
        size_t weight_bytes = 0,
        size_t scale_bytes = 0) {
    return { status, weight_bytes, weight_bytes, scale_bytes };
}

static bool checked_mul(size_t lhs, size_t rhs, size_t * result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static bool checked_add(size_t lhs, size_t rhs, size_t * result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

static bool checked_positive_to_size_t(int64_t value, size_t * result) {
    if (value <= 0 ||
        static_cast<uintmax_t>(value) > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *result = static_cast<size_t>(value);
    return true;
}

static bool align_128(size_t value, size_t * aligned) {
    const size_t remainder = value % 128;
    if (remainder == 0) {
        *aligned = value;
        return true;
    }
    return checked_add(value, 128 - remainder, aligned);
}

static bool append_workspace_region(size_t bytes, size_t * cursor, size_t * offset) {
    if (bytes == 0) {
        *offset = 0;
        return true;
    }
    size_t aligned = 0;
    if (!align_128(*cursor, &aligned) || !checked_add(aligned, bytes, cursor)) {
        return false;
    }
    *offset = aligned;
    return true;
}

ggml_cann_q8_w8a8_workspace_plan ggml_cann_q8_w8a8_plan_workspace(
        ggml_type input_type,
        ggml_type output_type,
        int64_t k,
        int64_t n,
        int64_t m,
        int64_t ne2,
        int64_t ne3) {
    ggml_cann_q8_w8a8_workspace_plan plan = {
        ggml_cann_q8_w8a8_workspace_status::overflow,
        input_type, output_type, k, n, m, ne2, ne3,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    if (k <= 0 || n <= 0 || m <= 0 || ne2 <= 0 || ne3 <= 0) {
        plan.status = ggml_cann_q8_w8a8_workspace_status::invalid_shape;
        return plan;
    }

    size_t sk = 0;
    size_t sn = 0;
    size_t sm = 0;
    size_t sne2 = 0;
    size_t sne3 = 0;
    size_t batches = 0;
    size_t input_elements = 0;
    size_t output_elements = 0;
    size_t token_scale_elements = 0;
    if (!checked_positive_to_size_t(k, &sk) ||
        !checked_positive_to_size_t(n, &sn) ||
        !checked_positive_to_size_t(m, &sm) ||
        !checked_positive_to_size_t(ne2, &sne2) ||
        !checked_positive_to_size_t(ne3, &sne3) ||
        !checked_mul(sne2, sne3, &batches) ||
        !checked_mul(sk, sm, &input_elements) ||
        !checked_mul(input_elements, batches, &input_elements) ||
        !checked_mul(sn, sm, &output_elements) ||
        !checked_mul(output_elements, batches, &output_elements) ||
        !checked_mul(sm, batches, &token_scale_elements)) {
        return plan;
    }

    size_t input_f16_bytes = 0;
    const size_t quant_bytes = input_elements;
    size_t token_scale_bytes = 0;
    size_t output_f16_bytes = 0;
    if (!checked_mul(token_scale_elements, sizeof(float), &token_scale_bytes) ||
        (input_type != GGML_TYPE_F16 &&
         !checked_mul(input_elements, sizeof(ggml_fp16_t), &input_f16_bytes)) ||
        (output_type != GGML_TYPE_F16 &&
         !checked_mul(output_elements, sizeof(ggml_fp16_t), &output_f16_bytes))) {
        return plan;
    }

    size_t cursor = 0;
    size_t input_f16_offset = 0;
    size_t quant_offset = 0;
    size_t token_scale_offset = 0;
    size_t output_f16_offset = 0;
    if (!append_workspace_region(input_f16_bytes, &cursor, &input_f16_offset) ||
        !append_workspace_region(quant_bytes, &cursor, &quant_offset) ||
        !append_workspace_region(token_scale_bytes, &cursor, &token_scale_offset) ||
        !append_workspace_region(output_f16_bytes, &cursor, &output_f16_offset)) {
        return plan;
    }
    plan.input_f16_offset = input_f16_offset;
    plan.input_f16_bytes = input_f16_bytes;
    plan.quant_offset = quant_offset;
    plan.quant_bytes = quant_bytes;
    plan.token_scale_offset = token_scale_offset;
    plan.token_scale_bytes = token_scale_bytes;
    plan.output_f16_offset = output_f16_offset;
    plan.output_f16_bytes = output_f16_bytes;
    plan.total_bytes = cursor;
    plan.status = ggml_cann_q8_w8a8_workspace_status::ok;
    return plan;
}

bool ggml_cann_q8_w8a8_workspace_matches(
        const ggml_cann_q8_w8a8_workspace_plan & plan,
        ggml_type input_type,
        ggml_type output_type,
        int64_t k,
        int64_t n,
        int64_t m,
        int64_t ne2,
        int64_t ne3) {
    return plan.status == ggml_cann_q8_w8a8_workspace_status::ok &&
           plan.input_type == input_type &&
           plan.output_type == output_type &&
           plan.k == k && plan.n == n && plan.m == m &&
           plan.ne2 == ne2 && plan.ne3 == ne3;
}

bool ggml_cann_q8_w8a8_graph_node_snapshot_matches(
        const ggml_cann_q8_w8a8_graph_node_snapshot & lhs,
        const ggml_cann_q8_w8a8_graph_node_snapshot & rhs) {
    if (lhs.registered != rhs.registered) {
        return false;
    }
    if (!lhs.registered) {
        return true;
    }
    return lhs.layout.weight_bytes == rhs.layout.weight_bytes &&
           lhs.layout.scale_offset == rhs.layout.scale_offset &&
           lhs.layout.scale_bytes == rhs.layout.scale_bytes &&
           lhs.layout.k == rhs.layout.k && lhs.layout.n == rhs.layout.n &&
           lhs.input_type == rhs.input_type && lhs.output_type == rhs.output_type &&
           lhs.m == rhs.m && lhs.ne2 == rhs.ne2 && lhs.ne3 == rhs.ne3;
}

size_t ggml_cann_q8_w8a8_required_size(int64_t k, int64_t n) {
    if (k <= 0 || n <= 0 || k % QK8_0 != 0) {
        return 0;
    }

    const size_t sk = static_cast<size_t>(k);
    const size_t sn = static_cast<size_t>(n);
    if (sk > std::numeric_limits<size_t>::max() / sn) {
        return 0;
    }
    const size_t weight_bytes = sk * sn;
    if (sn > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return 0;
    }
    const size_t scale_bytes = sn * sizeof(float);
    if (weight_bytes > std::numeric_limits<size_t>::max() - scale_bytes) {
        return 0;
    }
    return weight_bytes + scale_bytes;
}

ggml_cann_q8_w8a8_result ggml_cann_requantize_q8_0_per_channel(
        const void * src_data,
        int64_t k,
        int64_t n,
        void * dst,
        size_t dst_size) {
    const size_t required_size = ggml_cann_q8_w8a8_required_size(k, n);
    if (required_size == 0 || src_data == nullptr || dst == nullptr) {
        return make_result(ggml_cann_q8_w8a8_status::invalid_shape);
    }

    const auto * src = static_cast<const block_q8_0 *>(src_data);

    const size_t weight_bytes = static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t scale_bytes = static_cast<size_t>(n) * sizeof(float);
    if (dst_size < required_size) {
        return make_result(ggml_cann_q8_w8a8_status::insufficient_capacity, weight_bytes, scale_bytes);
    }

    auto * quant = static_cast<int8_t *>(dst);
    auto * scales = reinterpret_cast<float *>(static_cast<uint8_t *>(dst) + weight_bytes);
    const int64_t blocks_per_row = k / QK8_0;

    for (int64_t row = 0; row < n; ++row) {
        float max_abs = 0.0f;
        for (int64_t col = 0; col < k; ++col) {
            const block_q8_0 & block = src[row * blocks_per_row + col / QK8_0];
            const float value = ggml_fp16_to_fp32(block.d) * block.qs[col % QK8_0];
            if (!std::isfinite(value)) {
                return make_result(ggml_cann_q8_w8a8_status::non_finite_scale, weight_bytes, scale_bytes);
            }
            max_abs = std::max(max_abs, std::fabs(value));
        }

        const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
        if (!std::isfinite(scale) || scale <= 0.0f) {
            return make_result(ggml_cann_q8_w8a8_status::non_finite_scale, weight_bytes, scale_bytes);
        }
        scales[row] = scale;

        for (int64_t col = 0; col < k; ++col) {
            const block_q8_0 & block = src[row * blocks_per_row + col / QK8_0];
            const float value = ggml_fp16_to_fp32(block.d) * block.qs[col % QK8_0];
            const long rounded = std::lrint(value / scale);
            quant[row * k + col] = static_cast<int8_t>(std::max(-127L, std::min(127L, rounded)));
        }
    }

    return make_result(ggml_cann_q8_w8a8_status::ok, weight_bytes, scale_bytes);
}

ggml_cann_q8_w8a8_reject ggml_cann_q8_w8a8_validate(
        bool enabled,
        ggml_type type,
        bool matmul_weight,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t allocation_size,
        ggml_cann_q8_w8a8_layout * layout) {
    if (!enabled) {
        return ggml_cann_q8_w8a8_reject::disabled;
    }
    if (type != GGML_TYPE_Q8_0) {
        return ggml_cann_q8_w8a8_reject::wrong_type;
    }
    if (!matmul_weight) {
        return ggml_cann_q8_w8a8_reject::not_matmul_weight;
    }
    if (ne2 != 1 || ne3 != 1) {
        return ggml_cann_q8_w8a8_reject::batched;
    }

    const size_t required_size = ggml_cann_q8_w8a8_required_size(ne0, ne1);
    if (required_size == 0) {
        return ggml_cann_q8_w8a8_reject::invalid_shape;
    }
    if (allocation_size < required_size) {
        return ggml_cann_q8_w8a8_reject::insufficient_capacity;
    }

    if (layout != nullptr) {
        layout->weight_bytes = static_cast<size_t>(ne0) * static_cast<size_t>(ne1);
        layout->scale_offset = layout->weight_bytes;
        layout->scale_bytes = static_cast<size_t>(ne1) * sizeof(float);
        layout->k = ne0;
        layout->n = ne1;
    }
    return ggml_cann_q8_w8a8_reject::none;
}

ggml_cann_q8_w8a8_result ggml_cann_restore_q8_0_from_per_channel(
        const void * src_data,
        int64_t k,
        int64_t n,
        void * dst_data,
        size_t dst_size) {
    const size_t required_size = ggml_cann_q8_w8a8_required_size(k, n);
    if (required_size == 0 || src_data == nullptr || dst_data == nullptr) {
        return make_result(ggml_cann_q8_w8a8_status::invalid_shape);
    }

    const size_t block_count = static_cast<size_t>(k / QK8_0) * static_cast<size_t>(n);
    if (block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_0) ||
        dst_size < block_count * sizeof(block_q8_0)) {
        return make_result(ggml_cann_q8_w8a8_status::insufficient_capacity);
    }

    const size_t weight_bytes = static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t scale_bytes = static_cast<size_t>(n) * sizeof(float);
    const auto * quant = static_cast<const int8_t *>(src_data);
    const auto * scales = reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(src_data) + weight_bytes);
    auto * dst = static_cast<block_q8_0 *>(dst_data);
    const int64_t blocks_per_row = k / QK8_0;

    for (int64_t row = 0; row < n; ++row) {
        if (!std::isfinite(scales[row]) || scales[row] <= 0.0f) {
            return make_result(ggml_cann_q8_w8a8_status::non_finite_scale, weight_bytes, scale_bytes);
        }
        for (int64_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            block_q8_0 & block = dst[row * blocks_per_row + block_index];
            float max_abs = 0.0f;
            for (int i = 0; i < QK8_0; ++i) {
                const int64_t col = block_index * QK8_0 + i;
                max_abs = std::max(max_abs, std::fabs(quant[row * k + col] * scales[row]));
            }

            const float requested_scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
            block.d = ggml_fp32_to_fp16(requested_scale);
            const float stored_scale = ggml_fp16_to_fp32(block.d);
            if (!std::isfinite(stored_scale) || stored_scale <= 0.0f) {
                return make_result(ggml_cann_q8_w8a8_status::non_finite_scale, weight_bytes, scale_bytes);
            }
            for (int i = 0; i < QK8_0; ++i) {
                const int64_t col = block_index * QK8_0 + i;
                const float value = quant[row * k + col] * scales[row];
                const long rounded = std::lrint(value / stored_scale);
                block.qs[i] = static_cast<int8_t>(std::max(-127L, std::min(127L, rounded)));
            }
        }
    }

    return make_result(ggml_cann_q8_w8a8_status::ok, weight_bytes, scale_bytes);
}
