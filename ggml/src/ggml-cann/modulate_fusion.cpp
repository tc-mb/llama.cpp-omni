#include "modulate_fusion.h"

#include "aclrtlaunch_modulate_fusion.h"
#include "common.h"
#include "../ggml-impl.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {

constexpr size_t fallback_count =
    static_cast<size_t>(ggml_cann_modulate_fusion_fallback::COUNT);

const char * const fallback_names[] = {
    "none",
    "disabled",
    "not_fusible_chain",
    "edge_identity",
    "dtype",
    "x_shape",
    "param_shape",
    "layout",
    "alias",
    "alignment",
};

static_assert(sizeof(fallback_names) / sizeof(fallback_names[0]) == fallback_count,
              "fallback names must match ggml_cann_modulate_fusion_fallback");

struct atomic_stats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> modulate_hits{0};
    std::atomic<uint64_t> gate_residual_hits{0};
    std::atomic<uint64_t> launches_removed{0};
    std::array<std::atomic<uint64_t>, fallback_count> fallback;

    atomic_stats() {
        for (auto & value : fallback) {
            value.store(0, std::memory_order_relaxed);
        }
    }
};

struct buffer_range {
    uintptr_t address = 0;
    size_t size = 0;
};

atomic_stats g_stats;
std::once_flag g_stats_atexit_once;

bool range_disjoint(const buffer_range & a, const buffer_range & b) {
    if (a.address == 0 || b.address == 0 || a.size == 0 || b.size == 0 ||
        a.address > std::numeric_limits<uintptr_t>::max() - a.size ||
        b.address > std::numeric_limits<uintptr_t>::max() - b.size) {
        return false;
    }
    return a.address + a.size <= b.address ||
           b.address + b.size <= a.address;
}

template <size_t N>
bool all_ranges_disjoint(const std::array<buffer_range, N> & ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            if (!range_disjoint(ranges[i], ranges[j])) {
                return false;
            }
        }
    }
    return true;
}

bool aligned_32(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->data != nullptr &&
        reinterpret_cast<uintptr_t>(tensor->data) % 32 == 0;
}

bool same_shape(const ggml_tensor * tensor,
                int64_t n0,
                int64_t n1,
                int64_t n2,
                int64_t n3) {
    return tensor != nullptr && tensor->ne[0] == n0 &&
        tensor->ne[1] == n1 && tensor->ne[2] == n2 && tensor->ne[3] == n3;
}

void stats_fallback(ggml_cann_modulate_fusion_fallback reason) {
    g_stats.fallback[static_cast<size_t>(reason)].fetch_add(
        1, std::memory_order_relaxed);
}

void stats_print() {
    const ggml_cann_modulate_fusion_stats_snapshot stats =
        ggml_cann_modulate_fusion_stats_get();
    const double hit_rate = stats.attempts == 0
        ? 0.0
        : static_cast<double>(stats.hits) /
              static_cast<double>(stats.attempts);
    std::fprintf(stderr,
                 "[cann-modulate-fusion] attempts=%llu hits=%llu "
                 "modulate_hits=%llu gate_residual_hits=%llu "
                 "hit_rate=%.6f launches_removed=%llu",
                 static_cast<unsigned long long>(stats.attempts),
                 static_cast<unsigned long long>(stats.hits),
                 static_cast<unsigned long long>(stats.modulate_hits),
                 static_cast<unsigned long long>(stats.gate_residual_hits),
                 hit_rate,
                 static_cast<unsigned long long>(stats.launches_removed));
    for (size_t i = static_cast<size_t>(
             ggml_cann_modulate_fusion_fallback::DISABLED);
         i < fallback_count;
         ++i) {
        std::fprintf(stderr,
                     " fallback_%s=%llu",
                     fallback_names[i],
                     static_cast<unsigned long long>(stats.fallback[i]));
    }
    std::fputc('\n', stderr);
}

void register_stats_printer() {
    std::call_once(g_stats_atexit_once, [] {
        if (parse_bool(get_env_as_lowercase(
                "GGML_CANN_MODULATE_FUSION_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

enum class fusion_mode : uint32_t {
    MODULATE = 0,
    GATE_RESIDUAL = 1,
};

bool exact_alias_or_disjoint(const buffer_range & a, const buffer_range & b) {
    return (a.address == b.address && a.size == b.size) || range_disjoint(a, b);
}

int unique_consumer_index(
        const ggml_cann_unique_consumer_map & unique_consumers,
        const ggml_tensor * tensor) {
    const auto it = unique_consumers.find(tensor);
    return it == unique_consumers.end() ? -1 : it->second;
}

ggml_cann_modulate_fusion_fallback validate_modulate(
        ggml_cgraph * cgraph,
        int node_index,
        const ggml_cann_unique_consumer_map & unique_consumers,
        int * add1_index_out,
        int * add2_index_out,
        ggml_tensor ** x_out,
        ggml_tensor ** scale_out,
        ggml_tensor ** shift_out,
        ggml_tensor ** dst_out) {
    ggml_cann_modulate_fusion_policy_input input;
    int add1_index = -1;
    int add2_index = -1;
    if (cgraph != nullptr && node_index >= 0 && node_index < cgraph->n_nodes) {
        ggml_tensor * mul = cgraph->nodes[node_index];
        add1_index = unique_consumer_index(unique_consumers, mul);
        if (add1_index != -1) {
            ggml_tensor * add1 = cgraph->nodes[add1_index];
            add2_index = unique_consumer_index(unique_consumers, add1);
        }
    }
    const int node_indices[] = { node_index, add1_index, add2_index };
    const ggml_op ops[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD };
    input.fusible_chain = add1_index >= 0 && add2_index >= 0 &&
        ggml_can_fuse_ext(cgraph, node_indices, ops, 3);
    if (!input.fusible_chain) {
        return ggml_cann_modulate_fusion_validate_policy(input);
    }

    ggml_tensor * mul = cgraph->nodes[node_index];
    ggml_tensor * add1 = cgraph->nodes[add1_index];
    ggml_tensor * add2 = cgraph->nodes[add2_index];
    ggml_tensor * x = mul->src[0];
    ggml_tensor * scale = mul->src[1];
    ggml_tensor * shift = add2->src[1];

    input.exact_edges = x != nullptr && scale != nullptr && shift != nullptr &&
        add1->src[0] == x && add1->src[1] == mul &&
        add2->src[0] == add1;
    input.all_f32 = input.exact_edges &&
        x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 &&
        shift->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        add1->type == GGML_TYPE_F32 && add2->type == GGML_TYPE_F32;
    input.x_shape = input.all_f32 &&
        same_shape(x, 512, x->ne[1], 2, 1) &&
        (x->ne[1] == 50 || x->ne[1] == 56) &&
        ggml_are_same_shape(x, mul) && ggml_are_same_shape(x, add1) &&
        ggml_are_same_shape(x, add2);
    input.param_shape = input.x_shape &&
        same_shape(scale, 512, 1, 2, 1) &&
        same_shape(shift, 512, 1, 2, 1);
    const bool scale_rows_dense = input.param_shape &&
        scale->nb[0] == sizeof(float) &&
        scale->nb[2] >= 512 * sizeof(float) &&
        scale->nb[2] % 32 == 0;
    const bool shift_rows_dense = input.param_shape &&
        shift->nb[0] == sizeof(float) &&
        shift->nb[2] >= 512 * sizeof(float) &&
        shift->nb[2] % 32 == 0;
    input.supported_layout = input.param_shape &&
        ggml_is_contiguous(x) && ggml_is_contiguous(add2) &&
        scale_rows_dense && shift_rows_dense;
    if (input.supported_layout) {
        const uintptr_t scale_address =
            reinterpret_cast<uintptr_t>(scale->data);
        const uintptr_t shift_address =
            reinterpret_cast<uintptr_t>(shift->data);
        constexpr size_t param_row_bytes = 512 * sizeof(float);
        const buffer_range x_range = {
            reinterpret_cast<uintptr_t>(x->data), ggml_nbytes(x)
        };
        const buffer_range dst_range = {
            reinterpret_cast<uintptr_t>(add2->data), ggml_nbytes(add2)
        };
        const std::array<buffer_range, 5> input_ranges = {{
            x_range,
            { scale_address, param_row_bytes },
            { scale_address + scale->nb[2], param_row_bytes },
            { shift_address, param_row_bytes },
            { shift_address + shift->nb[2], param_row_bytes },
        }};
        const bool dst_is_exact_x_alias =
            dst_range.address == x_range.address && dst_range.size == x_range.size;
        bool dst_disjoint_from_params = true;
        for (size_t i = 1; i < input_ranges.size(); ++i) {
            dst_disjoint_from_params = dst_disjoint_from_params &&
                range_disjoint(dst_range, input_ranges[i]);
        }
        input.buffers_disjoint = all_ranges_disjoint(input_ranges) &&
            (dst_is_exact_x_alias || range_disjoint(x_range, dst_range)) &&
            dst_disjoint_from_params;
        input.buffers_aligned = aligned_32(x) && aligned_32(scale) &&
            aligned_32(shift) && aligned_32(add2);
    }

    const ggml_cann_modulate_fusion_fallback reason =
        ggml_cann_modulate_fusion_validate_policy(input);
    if (reason == ggml_cann_modulate_fusion_fallback::NONE) {
        *add1_index_out = add1_index;
        *add2_index_out = add2_index;
        *x_out = x;
        *scale_out = scale;
        *shift_out = shift;
        *dst_out = add2;
    }
    return reason;
}

ggml_cann_modulate_fusion_fallback validate_gate_residual(
        ggml_cgraph * cgraph,
        int node_index,
        const ggml_cann_unique_consumer_map & unique_consumers,
        int * add_index_out,
        ggml_tensor ** branch_out,
        ggml_tensor ** gate_out,
        ggml_tensor ** residual_out,
        ggml_tensor ** dst_out) {
    ggml_cann_modulate_fusion_policy_input input;
    int add_index = -1;
    if (cgraph != nullptr && node_index >= 0 && node_index < cgraph->n_nodes) {
        ggml_tensor * mul = cgraph->nodes[node_index];
        add_index = unique_consumer_index(unique_consumers, mul);
    }
    const int node_indices[] = { node_index, add_index };
    const ggml_op ops[] = { GGML_OP_MUL, GGML_OP_ADD };
    input.fusible_chain = add_index >= 0 &&
        ggml_can_fuse_ext(cgraph, node_indices, ops, 2);
    if (!input.fusible_chain) {
        return ggml_cann_modulate_fusion_validate_policy(input);
    }

    ggml_tensor * mul = cgraph->nodes[node_index];
    ggml_tensor * add = cgraph->nodes[add_index];
    ggml_tensor * branch = mul->src[0];
    ggml_tensor * gate = mul->src[1];
    ggml_tensor * residual = add->src[0];
    input.exact_edges = branch != nullptr && gate != nullptr &&
        residual != nullptr && add->src[1] == mul && branch != residual;
    input.all_f32 = input.exact_edges &&
        branch->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        residual->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32;
    input.x_shape = input.all_f32 &&
        same_shape(branch, 512, branch->ne[1], 2, 1) &&
        (branch->ne[1] == 50 || branch->ne[1] == 56) &&
        ggml_are_same_shape(branch, residual) &&
        ggml_are_same_shape(branch, mul) &&
        ggml_are_same_shape(branch, add);
    input.param_shape = input.x_shape &&
        same_shape(gate, 512, 1, 2, 1);
    const bool gate_rows_dense = input.param_shape &&
        gate->nb[0] == sizeof(float) &&
        gate->nb[2] >= 512 * sizeof(float) &&
        gate->nb[2] % 32 == 0;
    input.supported_layout = input.param_shape &&
        ggml_is_contiguous(branch) && ggml_is_contiguous(residual) &&
        ggml_is_contiguous(add) && gate_rows_dense;
    if (input.supported_layout) {
        const buffer_range branch_range = {
            reinterpret_cast<uintptr_t>(branch->data), ggml_nbytes(branch)
        };
        const buffer_range residual_range = {
            reinterpret_cast<uintptr_t>(residual->data), ggml_nbytes(residual)
        };
        const buffer_range dst_range = {
            reinterpret_cast<uintptr_t>(add->data), ggml_nbytes(add)
        };
        const uintptr_t gate_address =
            reinterpret_cast<uintptr_t>(gate->data);
        constexpr size_t param_row_bytes = 512 * sizeof(float);
        const std::array<buffer_range, 2> gate_ranges = {{
            { gate_address, param_row_bytes },
            { gate_address + gate->nb[2], param_row_bytes },
        }};
        bool gate_disjoint = all_ranges_disjoint(gate_ranges);
        for (const buffer_range & gate_range : gate_ranges) {
            gate_disjoint = gate_disjoint &&
                range_disjoint(gate_range, branch_range) &&
                range_disjoint(gate_range, residual_range) &&
                range_disjoint(gate_range, dst_range);
        }
        input.buffers_disjoint =
            exact_alias_or_disjoint(branch_range, residual_range) &&
            exact_alias_or_disjoint(branch_range, dst_range) &&
            exact_alias_or_disjoint(residual_range, dst_range) &&
            gate_disjoint;
        input.buffers_aligned = aligned_32(branch) && aligned_32(gate) &&
            aligned_32(residual) && aligned_32(add);
    }

    const ggml_cann_modulate_fusion_fallback reason =
        ggml_cann_modulate_fusion_validate_policy(input);
    if (reason == ggml_cann_modulate_fusion_fallback::NONE) {
        *add_index_out = add_index;
        *branch_out = branch;
        *gate_out = gate;
        *residual_out = residual;
        *dst_out = add;
    }
    return reason;
}

}  // namespace

const char * ggml_cann_modulate_fusion_fallback_name(
        ggml_cann_modulate_fusion_fallback reason) {
    const size_t index = static_cast<size_t>(reason);
    return index < fallback_count ? fallback_names[index] : "unknown";
}

bool ggml_cann_modulate_fusion_enabled() {
    static const bool enabled =
        get_env_as_lowercase("GGML_CANN_MODULATE_FUSION")
            .value_or("auto") == "auto";
    return enabled;
}

bool ggml_cann_modulate_fusion_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const ggml_cann_unique_consumer_map & unique_consumers,
        int * add1_index,
        int * add2_index) {
    register_stats_printer();
    g_stats.attempts.fetch_add(1, std::memory_order_relaxed);

    if (!ggml_cann_modulate_fusion_enabled()) {
        stats_fallback(ggml_cann_modulate_fusion_fallback::DISABLED);
        return false;
    }

    ggml_tensor * x = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * shift = nullptr;
    ggml_tensor * dst = nullptr;
    fusion_mode mode = fusion_mode::MODULATE;
    ggml_cann_modulate_fusion_fallback reason =
        validate_modulate(cgraph, node_index, unique_consumers,
                          add1_index, add2_index,
                          &x, &scale, &shift, &dst);
    if (reason != ggml_cann_modulate_fusion_fallback::NONE) {
        int gate_add_index = -1;
        reason = validate_gate_residual(
            cgraph, node_index, unique_consumers, &gate_add_index,
            &x, &scale, &shift, &dst);
        if (reason == ggml_cann_modulate_fusion_fallback::NONE) {
            mode = fusion_mode::GATE_RESIDUAL;
            *add1_index = gate_add_index;
            *add2_index = -1;
        }
    }
    if (reason != ggml_cann_modulate_fusion_fallback::NONE) {
        stats_fallback(reason);
        return false;
    }

    const uint64_t time = static_cast<uint64_t>(x->ne[1]);
    const uint64_t batch = static_cast<uint64_t>(x->ne[2]);
    const uint64_t rows = time * batch;
    const uint32_t block_count = static_cast<uint32_t>(rows < 48 ? rows : 48);
    ACL_CHECK(ACLRT_LAUNCH_KERNEL(modulate_fusion)(
        block_count,
        ctx.stream(),
        x->data,
        scale->data,
        shift->data,
        dst->data,
        time,
        batch,
        static_cast<uint64_t>(scale->nb[2] / sizeof(float)),
        mode == fusion_mode::MODULATE
            ? static_cast<uint64_t>(shift->nb[2] / sizeof(float))
            : 0,
        static_cast<uint32_t>(mode)));

    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    if (mode == fusion_mode::MODULATE) {
        g_stats.modulate_hits.fetch_add(1, std::memory_order_relaxed);
        g_stats.launches_removed.fetch_add(2, std::memory_order_relaxed);
    } else {
        g_stats.gate_residual_hits.fetch_add(1, std::memory_order_relaxed);
        g_stats.launches_removed.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void ggml_cann_modulate_fusion_stats_reset() {
    g_stats.attempts.store(0, std::memory_order_relaxed);
    g_stats.hits.store(0, std::memory_order_relaxed);
    g_stats.modulate_hits.store(0, std::memory_order_relaxed);
    g_stats.gate_residual_hits.store(0, std::memory_order_relaxed);
    g_stats.launches_removed.store(0, std::memory_order_relaxed);
    for (auto & value : g_stats.fallback) {
        value.store(0, std::memory_order_relaxed);
    }
}

ggml_cann_modulate_fusion_stats_snapshot
ggml_cann_modulate_fusion_stats_get() {
    ggml_cann_modulate_fusion_stats_snapshot snapshot;
    snapshot.attempts = g_stats.attempts.load(std::memory_order_relaxed);
    snapshot.hits = g_stats.hits.load(std::memory_order_relaxed);
    snapshot.modulate_hits =
        g_stats.modulate_hits.load(std::memory_order_relaxed);
    snapshot.gate_residual_hits =
        g_stats.gate_residual_hits.load(std::memory_order_relaxed);
    snapshot.launches_removed =
        g_stats.launches_removed.load(std::memory_order_relaxed);
    for (size_t i = 0; i < fallback_count; ++i) {
        snapshot.fallback[i] =
            g_stats.fallback[i].load(std::memory_order_relaxed);
    }
    return snapshot;
}
