#include "set_rows_f32_f16.h"

#include "aclrtlaunch_set_rows_f32_f16.h"
#include "common.h"
#include "../ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {

constexpr size_t fallback_count =
    static_cast<size_t>(ggml_cann_set_rows_f32_f16_fallback::COUNT);

const char * const fallback_names[] = {
    "none",
    "disabled",
    "src_dtype",
    "dst_dtype",
    "index_dtype",
    "src_layout",
    "dst_layout",
    "index_layout",
    "batch",
    "index_count",
    "row_width",
    "cache_rows",
    "alias",
    "alignment",
};

static_assert(sizeof(fallback_names) / sizeof(fallback_names[0]) == fallback_count,
              "fallback names must match ggml_cann_set_rows_f32_f16_fallback");

struct ggml_cann_set_rows_f32_f16_atomic_stats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
    std::array<std::atomic<uint64_t>, fallback_count> fallback;

    ggml_cann_set_rows_f32_f16_atomic_stats() {
        for (auto & value : fallback) {
            value.store(0, std::memory_order_relaxed);
        }
    }
};

ggml_cann_set_rows_f32_f16_atomic_stats g_stats;
std::once_flag g_stats_atexit_once;

bool range_disjoint(uintptr_t a, size_t a_size, uintptr_t b, size_t b_size) {
    if (a_size == 0 || b_size == 0 ||
        a > std::numeric_limits<uintptr_t>::max() - a_size ||
        b > std::numeric_limits<uintptr_t>::max() - b_size) {
        return false;
    }
    return a + a_size <= b || b + b_size <= a;
}

void stats_fallback(ggml_cann_set_rows_f32_f16_fallback reason) {
    g_stats.fallback[static_cast<size_t>(reason)].fetch_add(
        1, std::memory_order_relaxed);
}

void stats_print() {
    const ggml_cann_set_rows_f32_f16_stats_snapshot stats =
        ggml_cann_set_rows_f32_f16_stats_get();
    const double hit_rate = stats.attempts == 0
        ? 0.0
        : static_cast<double>(stats.hits) /
              static_cast<double>(stats.attempts);
    std::fprintf(stderr,
                 "[cann-set-rows-f32-f16] attempts=%llu hits=%llu "
                 "hit_rate=%.6f cast_launches_removed=%llu "
                 "scatter_launches_removed=%llu",
                 static_cast<unsigned long long>(stats.attempts),
                 static_cast<unsigned long long>(stats.hits),
                 hit_rate,
                 static_cast<unsigned long long>(stats.hits),
                 static_cast<unsigned long long>(stats.hits));
    for (size_t i = static_cast<size_t>(
             ggml_cann_set_rows_f32_f16_fallback::DISABLED);
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
                "GGML_CANN_SET_ROWS_F32_F16_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

}  // namespace

const char * ggml_cann_set_rows_f32_f16_fallback_name(
        ggml_cann_set_rows_f32_f16_fallback reason) {
    const size_t index = static_cast<size_t>(reason);
    return index < fallback_count ? fallback_names[index] : "unknown";
}

ggml_cann_set_rows_f32_f16_fallback ggml_cann_set_rows_f32_f16_validate(
        const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return ggml_cann_set_rows_f32_f16_fallback::BATCH;
    }

    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * index = dst->src[1];
    ggml_cann_set_rows_f32_f16_policy_input input;
    input.src_is_f32 = src->type == GGML_TYPE_F32;
    input.dst_is_f16 = dst->type == GGML_TYPE_F16;
    input.index_is_i64 = index->type == GGML_TYPE_I64;
    input.src_row_dense = src->nb[0] == sizeof(float);
    input.dst_rows_dense =
        dst->nb[0] == sizeof(ggml_fp16_t) &&
        dst->ne[0] > 0 &&
        static_cast<uint64_t>(dst->ne[0]) <=
            std::numeric_limits<size_t>::max() / sizeof(ggml_fp16_t) &&
        dst->nb[1] == static_cast<size_t>(dst->ne[0]) * sizeof(ggml_fp16_t);
    input.index_dense = index->nb[0] == sizeof(int64_t);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        input.src_ne[d] = src->ne[d];
        input.dst_ne[d] = dst->ne[d];
        input.index_ne[d] = index->ne[d];
    }
    input.src_address = reinterpret_cast<uintptr_t>(src->data);
    input.dst_address = reinterpret_cast<uintptr_t>(dst->data);
    input.index_address = reinterpret_cast<uintptr_t>(index->data);
    const size_t src_bytes = src->ne[0] > 0
        ? static_cast<size_t>(src->ne[0]) * sizeof(float)
        : 0;
    const size_t dst_bytes = dst->ne[0] > 0 && dst->ne[1] > 0
        ? static_cast<size_t>(dst->ne[0]) *
              static_cast<size_t>(dst->ne[1]) * sizeof(ggml_fp16_t)
        : 0;
    const size_t index_bytes = sizeof(int64_t);
    input.buffers_disjoint =
        range_disjoint(input.src_address, src_bytes,
                       input.dst_address, dst_bytes) &&
        range_disjoint(input.src_address, src_bytes,
                       input.index_address, index_bytes) &&
        range_disjoint(input.dst_address, dst_bytes,
                       input.index_address, index_bytes);
    return ggml_cann_set_rows_f32_f16_validate_policy(input);
}

bool ggml_cann_set_rows_f32_f16_try(
        ggml_backend_cann_context & ctx,
        ggml_tensor * dst) {
    register_stats_printer();
    g_stats.attempts.fetch_add(1, std::memory_order_relaxed);

    static const bool disabled =
        get_env_as_lowercase("GGML_CANN_SET_ROWS_F32_F16")
            .value_or("off") == "off";
    if (disabled) {
        stats_fallback(ggml_cann_set_rows_f32_f16_fallback::DISABLED);
        return false;
    }

    const ggml_cann_set_rows_f32_f16_fallback reason =
        ggml_cann_set_rows_f32_f16_validate(dst);
    if (reason != ggml_cann_set_rows_f32_f16_fallback::NONE) {
        stats_fallback(reason);
        return false;
    }

    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * index = dst->src[1];
    ACL_CHECK(ACLRT_LAUNCH_KERNEL(set_rows_f32_f16)(
        1,
        ctx.stream(),
        src->data,
        index->data,
        dst->data,
        static_cast<uint64_t>(dst->ne[0]),
        static_cast<uint64_t>(dst->ne[1])));

    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_cann_set_rows_f32_f16_stats_reset() {
    g_stats.attempts.store(0, std::memory_order_relaxed);
    g_stats.hits.store(0, std::memory_order_relaxed);
    for (auto & value : g_stats.fallback) {
        value.store(0, std::memory_order_relaxed);
    }
}

ggml_cann_set_rows_f32_f16_stats_snapshot
ggml_cann_set_rows_f32_f16_stats_get() {
    ggml_cann_set_rows_f32_f16_stats_snapshot snapshot;
    snapshot.attempts = g_stats.attempts.load(std::memory_order_relaxed);
    snapshot.hits = g_stats.hits.load(std::memory_order_relaxed);
    for (size_t i = 0; i < fallback_count; ++i) {
        snapshot.fallback[i] =
            g_stats.fallback[i].load(std::memory_order_relaxed);
    }
    return snapshot;
}
