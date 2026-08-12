#include "kv_pair_update.h"

#include "aclrtlaunch_kv_pair_update.h"
#include "common.h"
#include "set_rows_f32_f16.h"
#include "../ggml-impl.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {

constexpr size_t fallback_count =
    static_cast<size_t>(ggml_cann_kv_pair_update_fallback::COUNT);

const char * const fallback_names[] = {
    "none",
    "disabled",
    "not_adjacent_set_rows",
    "first_policy",
    "second_policy",
    "row_width",
    "cache_rows",
    "pair_alias",
};

static_assert(sizeof(fallback_names) / sizeof(fallback_names[0]) == fallback_count,
              "fallback names must match ggml_cann_kv_pair_update_fallback");

struct ggml_cann_kv_pair_update_atomic_stats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
    std::array<std::atomic<uint64_t>, fallback_count> fallback;

    ggml_cann_kv_pair_update_atomic_stats() {
        for (auto & value : fallback) {
            value.store(0, std::memory_order_relaxed);
        }
    }
};

struct buffer_range {
    uintptr_t address = 0;
    size_t size = 0;
};

ggml_cann_kv_pair_update_atomic_stats g_stats;
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

bool all_ranges_disjoint(const std::array<buffer_range, 6> & ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            if (!range_disjoint(ranges[i], ranges[j])) {
                return false;
            }
        }
    }
    return true;
}

buffer_range source_range(const ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    return {
        reinterpret_cast<uintptr_t>(src->data),
        static_cast<size_t>(src->ne[0]) * sizeof(float),
    };
}

buffer_range index_range(const ggml_tensor * dst) {
    return {
        reinterpret_cast<uintptr_t>(dst->src[1]->data),
        sizeof(int64_t),
    };
}

buffer_range destination_range(const ggml_tensor * dst) {
    return {
        reinterpret_cast<uintptr_t>(dst->data),
        static_cast<size_t>(dst->ne[0]) *
            static_cast<size_t>(dst->ne[1]) * sizeof(ggml_fp16_t),
    };
}

void stats_fallback(ggml_cann_kv_pair_update_fallback reason) {
    g_stats.fallback[static_cast<size_t>(reason)].fetch_add(
        1, std::memory_order_relaxed);
}

void stats_print() {
    const ggml_cann_kv_pair_update_stats_snapshot stats =
        ggml_cann_kv_pair_update_stats_get();
    const double hit_rate = stats.attempts == 0
        ? 0.0
        : static_cast<double>(stats.hits) /
              static_cast<double>(stats.attempts);
    std::fprintf(stderr,
                 "[cann-kv-pair-update] attempts=%llu hits=%llu "
                 "hit_rate=%.6f launches_removed=%llu",
                 static_cast<unsigned long long>(stats.attempts),
                 static_cast<unsigned long long>(stats.hits),
                 hit_rate,
                 static_cast<unsigned long long>(
                     stats.hits *
                     ggml_cann_kv_pair_update_launches_removed_per_hit));
    for (size_t i = static_cast<size_t>(
             ggml_cann_kv_pair_update_fallback::DISABLED);
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
                "GGML_CANN_KV_PAIR_UPDATE_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

}  // namespace

const char * ggml_cann_kv_pair_update_fallback_name(
        ggml_cann_kv_pair_update_fallback reason) {
    const size_t index = static_cast<size_t>(reason);
    return index < fallback_count ? fallback_names[index] : "unknown";
}

ggml_cann_kv_pair_update_fallback ggml_cann_kv_pair_update_validate(
        const ggml_tensor * first,
        const ggml_tensor * second) {
    ggml_cann_kv_pair_update_policy_input input;
    input.adjacent_set_rows =
        first != nullptr && second != nullptr &&
        first->op == GGML_OP_SET_ROWS && second->op == GGML_OP_SET_ROWS &&
        (first->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
        (second->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    if (!input.adjacent_set_rows) {
        return ggml_cann_kv_pair_update_validate_policy(input);
    }

    input.first_reason = ggml_cann_set_rows_f32_f16_validate(first);
    input.second_reason = ggml_cann_set_rows_f32_f16_validate(second);
    input.first_row_width = first->ne[0];
    input.second_row_width = second->ne[0];
    input.first_cache_rows = first->ne[1];
    input.second_cache_rows = second->ne[1];

    if (input.first_reason == ggml_cann_set_rows_f32_f16_fallback::NONE &&
        input.second_reason == ggml_cann_set_rows_f32_f16_fallback::NONE) {
        const std::array<buffer_range, 6> ranges = {
            source_range(first),
            index_range(first),
            destination_range(first),
            source_range(second),
            index_range(second),
            destination_range(second),
        };
        input.pair_buffers_disjoint = all_ranges_disjoint(ranges);
    }

    return ggml_cann_kv_pair_update_validate_policy(input);
}

bool ggml_cann_kv_pair_update_try(
        ggml_backend_cann_context & ctx,
        ggml_tensor * first,
        ggml_tensor * second) {
    register_stats_printer();
    g_stats.attempts.fetch_add(1, std::memory_order_relaxed);

    static const bool disabled =
        get_env_as_lowercase("GGML_CANN_KV_PAIR_UPDATE")
            .value_or("off") != "auto";
    if (disabled) {
        stats_fallback(ggml_cann_kv_pair_update_fallback::DISABLED);
        return false;
    }

    const ggml_cann_kv_pair_update_fallback reason =
        ggml_cann_kv_pair_update_validate(first, second);
    if (reason != ggml_cann_kv_pair_update_fallback::NONE) {
        stats_fallback(reason);
        return false;
    }

    const ggml_tensor * first_src = first->src[0];
    const ggml_tensor * first_index = first->src[1];
    const ggml_tensor * second_src = second->src[0];
    const ggml_tensor * second_index = second->src[1];
    ACL_CHECK(ACLRT_LAUNCH_KERNEL(kv_pair_update)(
        1,
        ctx.stream(),
        first_src->data,
        first_index->data,
        first->data,
        second_src->data,
        second_index->data,
        second->data,
        static_cast<uint64_t>(first->ne[0]),
        static_cast<uint64_t>(first->ne[1])));

    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_cann_kv_pair_update_stats_reset() {
    g_stats.attempts.store(0, std::memory_order_relaxed);
    g_stats.hits.store(0, std::memory_order_relaxed);
    for (auto & value : g_stats.fallback) {
        value.store(0, std::memory_order_relaxed);
    }
}

ggml_cann_kv_pair_update_stats_snapshot
ggml_cann_kv_pair_update_stats_get() {
    ggml_cann_kv_pair_update_stats_snapshot snapshot;
    snapshot.attempts = g_stats.attempts.load(std::memory_order_relaxed);
    snapshot.hits = g_stats.hits.load(std::memory_order_relaxed);
    for (size_t i = 0; i < fallback_count; ++i) {
        snapshot.fallback[i] =
            g_stats.fallback[i].load(std::memory_order_relaxed);
    }
    return snapshot;
}
