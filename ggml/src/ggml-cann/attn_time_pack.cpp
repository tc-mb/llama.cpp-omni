#include "attn_time_pack.h"

#include "aclrtlaunch_attn_time_pack.h"
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
    static_cast<size_t>(ggml_cann_attn_time_pack_fallback::COUNT);

const char * const fallback_names[] = {
    "none",
    "disabled",
    "not_exact_chain",
    "dtype",
    "shape",
    "layout",
    "alias",
    "alignment",
};

static_assert(sizeof(fallback_names) / sizeof(fallback_names[0]) == fallback_count,
              "fallback names must match attn time pack policy");

struct atomic_stats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
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

bool checked_add(size_t a, size_t b, size_t * result) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    *result = a + b;
    return true;
}

bool tensor_span(const ggml_tensor * tensor, size_t * span) {
    if (tensor == nullptr || tensor->data == nullptr || span == nullptr) {
        return false;
    }
    size_t value = ggml_type_size(tensor->type);
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] <= 0) {
            return false;
        }
        const uint64_t count = static_cast<uint64_t>(tensor->ne[dim] - 1);
        if (count != 0 && tensor->nb[dim] >
                              std::numeric_limits<size_t>::max() / count) {
            return false;
        }
        if (!checked_add(value, static_cast<size_t>(count) * tensor->nb[dim],
                         &value)) {
            return false;
        }
    }
    *span = value;
    return true;
}

bool ranges_disjoint(const buffer_range & a, const buffer_range & b) {
    if (a.address == 0 || b.address == 0 || a.size == 0 || b.size == 0 ||
        a.address > std::numeric_limits<uintptr_t>::max() - a.size ||
        b.address > std::numeric_limits<uintptr_t>::max() - b.size) {
        return false;
    }
    return a.address + a.size <= b.address ||
           b.address + b.size <= a.address;
}

bool aligned_32(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->data != nullptr &&
           reinterpret_cast<uintptr_t>(tensor->data) % 32 == 0;
}

int unique_consumer_index(
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        const ggml_tensor * tensor) {
    const auto it = unique_consumers.find(tensor);
    return it == unique_consumers.end() ? -1 : it->second;
}

bool same_shape(const ggml_tensor * tensor,
                int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return tensor != nullptr && tensor->ne[0] == ne0 &&
           tensor->ne[1] == ne1 && tensor->ne[2] == ne2 &&
           tensor->ne[3] == ne3;
}

void stats_fallback(ggml_cann_attn_time_pack_fallback reason) {
    g_stats.fallback[static_cast<size_t>(reason)].fetch_add(
        1, std::memory_order_relaxed);
}

void stats_print() {
    const auto stats = ggml_cann_attn_time_pack_stats_get();
    const double hit_rate = stats.attempts == 0
        ? 0.0
        : static_cast<double>(stats.hits) /
              static_cast<double>(stats.attempts);
    std::fprintf(stderr,
                 "[cann-attn-time-pack] attempts=%llu hits=%llu "
                 "hit_rate=%.6f launches_removed=%llu",
                 static_cast<unsigned long long>(stats.attempts),
                 static_cast<unsigned long long>(stats.hits),
                 hit_rate,
                 static_cast<unsigned long long>(
                     stats.hits *
                     ggml_cann_attn_time_pack_launches_removed_per_hit));
    for (size_t i = static_cast<size_t>(
             ggml_cann_attn_time_pack_fallback::DISABLED);
         i < fallback_count; ++i) {
        std::fprintf(stderr, " fallback_%s=%llu", fallback_names[i],
                     static_cast<unsigned long long>(stats.fallback[i]));
    }
    std::fputc('\n', stderr);
}

void register_stats_printer() {
    std::call_once(g_stats_atexit_once, [] {
        if (parse_bool(get_env_as_lowercase(
                "GGML_CANN_ATTN_TIME_PACK_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

ggml_cann_attn_time_pack_fallback validate(
        ggml_cgraph * cgraph,
        int node_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * permute_index_out,
        int * cont_index_out,
        ggml_tensor ** src0_out,
        ggml_tensor ** src1_out,
        ggml_tensor ** dst_out) {
    ggml_cann_attn_time_pack_policy_input input;
    if (cgraph == nullptr || node_index < 0 || node_index >= cgraph->n_nodes) {
        return ggml_cann_attn_time_pack_validate_policy(input);
    }

    ggml_tensor * concat = cgraph->nodes[node_index];
    const int permute_index = unique_consumer_index(unique_consumers, concat);
    ggml_tensor * permute = permute_index >= 0
        ? cgraph->nodes[permute_index]
        : nullptr;
    const int cont_index = permute != nullptr
        ? unique_consumer_index(unique_consumers, permute)
        : -1;
    ggml_tensor * cont = cont_index >= 0 ? cgraph->nodes[cont_index] : nullptr;

    input.exact_chain = concat->op == GGML_OP_CONCAT &&
        ggml_get_op_params_i32(concat, 0) == 2 &&
        permute != nullptr && permute->op == GGML_OP_PERMUTE &&
        permute->src[0] == concat &&
        cont != nullptr && cont->op == GGML_OP_CONT &&
        cont->src[0] == permute &&
        (concat->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
        (permute->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    if (!input.exact_chain) {
        return ggml_cann_attn_time_pack_validate_policy(input);
    }

    ggml_tensor * src0 = concat->src[0];
    ggml_tensor * src1 = concat->src[1];
    input.all_f32 = src0 != nullptr && src1 != nullptr &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
        concat->type == GGML_TYPE_F32 && permute->type == GGML_TYPE_F32 &&
        cont->type == GGML_TYPE_F32;
    if (!input.all_f32) {
        return ggml_cann_attn_time_pack_validate_policy(input);
    }

    const int64_t dt = src0->ne[2];
    const int64_t cache = src1->ne[2];
    const int64_t total = dt + cache;
    input.canonical_shape =
        (dt == 50 || dt == 56) && cache == 302 &&
        same_shape(src0, 64, 8, dt, 2) &&
        same_shape(src1, 64, 8, cache, 2) &&
        same_shape(concat, 64, 8, total, 2) &&
        same_shape(permute, 64, total, 8, 2) &&
        same_shape(cont, 64, total, 8, 2);
    if (!input.canonical_shape) {
        return ggml_cann_attn_time_pack_validate_policy(input);
    }

    const auto row_aligned = [](const ggml_tensor * tensor) {
        return tensor->nb[0] == sizeof(float) &&
               tensor->nb[1] >= 64 * sizeof(float) &&
               tensor->nb[1] % 32 == 0 &&
               tensor->nb[2] % 32 == 0 && tensor->nb[3] % 32 == 0;
    };
    input.supported_layout = row_aligned(src0) && row_aligned(src1) &&
        permute->nb[0] == concat->nb[0] &&
        permute->nb[1] == concat->nb[2] &&
        permute->nb[2] == concat->nb[1] &&
        permute->nb[3] == concat->nb[3] &&
        ggml_is_contiguous(cont);
    if (!input.supported_layout) {
        return ggml_cann_attn_time_pack_validate_policy(input);
    }

    size_t src0_span = 0;
    size_t src1_span = 0;
    size_t dst_span = 0;
    const bool spans_ok = tensor_span(src0, &src0_span) &&
        tensor_span(src1, &src1_span) && tensor_span(cont, &dst_span);
    if (spans_ok) {
        const buffer_range ranges[] = {
            { reinterpret_cast<uintptr_t>(src0->data), src0_span },
            { reinterpret_cast<uintptr_t>(src1->data), src1_span },
            { reinterpret_cast<uintptr_t>(cont->data), dst_span },
        };
        input.buffers_disjoint = ranges_disjoint(ranges[0], ranges[1]) &&
            ranges_disjoint(ranges[0], ranges[2]) &&
            ranges_disjoint(ranges[1], ranges[2]);
    }
    input.buffers_aligned = aligned_32(src0) && aligned_32(src1) &&
        aligned_32(cont);

    const auto reason = ggml_cann_attn_time_pack_validate_policy(input);
    if (reason == ggml_cann_attn_time_pack_fallback::NONE) {
        *permute_index_out = permute_index;
        *cont_index_out = cont_index;
        *src0_out = src0;
        *src1_out = src1;
        *dst_out = cont;
    }
    return reason;
}

}  // namespace

const char * ggml_cann_attn_time_pack_fallback_name(
        ggml_cann_attn_time_pack_fallback reason) {
    const size_t index = static_cast<size_t>(reason);
    return index < fallback_count ? fallback_names[index] : "unknown";
}

bool ggml_cann_attn_time_pack_enabled() {
    static const bool enabled =
        get_env_as_lowercase("GGML_CANN_ATTN_TIME_PACK")
            .value_or("auto") == "auto";
    return enabled;
}

bool ggml_cann_attn_time_pack_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * permute_index,
        int * cont_index) {
    register_stats_printer();
    g_stats.attempts.fetch_add(1, std::memory_order_relaxed);
    if (!ggml_cann_attn_time_pack_enabled()) {
        stats_fallback(ggml_cann_attn_time_pack_fallback::DISABLED);
        return false;
    }

    ggml_tensor * src0 = nullptr;
    ggml_tensor * src1 = nullptr;
    ggml_tensor * dst = nullptr;
    const auto reason = validate(
        cgraph, node_index, unique_consumers, permute_index, cont_index,
        &src0, &src1, &dst);
    if (reason != ggml_cann_attn_time_pack_fallback::NONE) {
        stats_fallback(reason);
        return false;
    }

    const uint64_t time0 = static_cast<uint64_t>(src0->ne[2]);
    const uint64_t time1 = static_cast<uint64_t>(src1->ne[2]);
    const uint32_t block_count = static_cast<uint32_t>(
        static_cast<uint64_t>(src0->ne[3]) *
        static_cast<uint64_t>(src0->ne[1]));
    ACL_CHECK(ACLRT_LAUNCH_KERNEL(attn_time_pack)(
        block_count,
        ctx.stream(),
        src0->data,
        src1->data,
        dst->data,
        time0,
        time1,
        static_cast<uint64_t>(src0->ne[1]),
        static_cast<uint64_t>(src0->ne[3]),
        static_cast<uint64_t>(src0->nb[1] / sizeof(float)),
        static_cast<uint64_t>(src0->nb[2] / sizeof(float)),
        static_cast<uint64_t>(src0->nb[3] / sizeof(float)),
        static_cast<uint64_t>(src1->nb[1] / sizeof(float)),
        static_cast<uint64_t>(src1->nb[2] / sizeof(float)),
        static_cast<uint64_t>(src1->nb[3] / sizeof(float))));

    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_cann_attn_time_pack_stats_reset() {
    g_stats.attempts.store(0, std::memory_order_relaxed);
    g_stats.hits.store(0, std::memory_order_relaxed);
    for (auto & value : g_stats.fallback) {
        value.store(0, std::memory_order_relaxed);
    }
}

ggml_cann_attn_time_pack_stats_snapshot
ggml_cann_attn_time_pack_stats_get() {
    ggml_cann_attn_time_pack_stats_snapshot snapshot;
    snapshot.attempts = g_stats.attempts.load(std::memory_order_relaxed);
    snapshot.hits = g_stats.hits.load(std::memory_order_relaxed);
    for (size_t i = 0; i < fallback_count; ++i) {
        snapshot.fallback[i] =
            g_stats.fallback[i].load(std::memory_order_relaxed);
    }
    return snapshot;
}
