#include "im2col1d.h"

#include "common.h"
#include "aclrtlaunch_im2col1d_f32.h"
#include "../ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <vector>

namespace {

constexpr size_t fallback_count = static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT);

constexpr uint64_t gm_cache_line_words = 64 / sizeof(uint32_t);

const char * const fallback_names[] = {
    "none",
    "unmarked",
    "disabled",
    "is_2d",
    "secondary_params",
    "src_dtype",
    "dst_dtype",
    "src_layout",
    "dst_layout",
    "shape",
    "conv_params",
    "output_shape",
    "index_overflow",
};

static_assert(sizeof(fallback_names) / sizeof(fallback_names[0]) == fallback_count,
              "fallback names must match ggml_cann_im2col1d_fallback");

struct ggml_cann_im2col1d_atomic_kind_stats {
    std::atomic<uint64_t> marked{0};
    std::atomic<uint64_t> hits{0};
    std::array<std::atomic<uint64_t>, fallback_count> fallback;

    ggml_cann_im2col1d_atomic_kind_stats() {
        for (auto & value : fallback) {
            value.store(0, std::memory_order_relaxed);
        }
    }
};

struct ggml_cann_im2col1d_atomic_stats : ggml_cann_im2col1d_atomic_kind_stats {
    std::atomic<uint64_t> temp_bytes_removed{0};
    std::atomic<uint64_t> permutes_removed{0};
    std::atomic<uint64_t> d2d_copies_removed{0};
    std::atomic<uint64_t> d2d_bytes_removed{0};
    std::atomic<uint64_t> ctb_split_hits{0};
    std::atomic<uint64_t> ctb_static_offset_hits{0};
};

ggml_cann_im2col1d_atomic_stats g_stats;
ggml_cann_im2col1d_atomic_kind_stats g_causal_stats;
ggml_cann_im2col1d_atomic_kind_stats g_vocoder_stats;
std::once_flag g_stats_atexit_once;
std::mutex g_bench_mutex;
std::vector<float> g_kernel_ms;

void record_kernel_ms(float value) {
    std::lock_guard<std::mutex> lock(g_bench_mutex);
    g_kernel_ms.push_back(value);
}

bool exact_contiguous(const ggml_tensor * tensor) {
    const int64_t block_size = ggml_blck_size(tensor->type);
    if (block_size <= 0 || tensor->ne[0] <= 0 || tensor->ne[0] % block_size != 0) {
        return false;
    }

    size_t expected = ggml_type_size(tensor->type);
    if (tensor->nb[0] != expected) {
        return false;
    }

    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        const int64_t extent = d == 1 ? tensor->ne[0] / block_size : tensor->ne[d - 1];
        if (extent <= 0 ||
            static_cast<uint64_t>(extent) > std::numeric_limits<size_t>::max() / expected) {
            return false;
        }
        expected *= static_cast<size_t>(extent);
        if (tensor->nb[d] != expected) {
            return false;
        }
    }

    return true;
}

bool tensor_ranges_overlap(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr || lhs->data == nullptr || rhs->data == nullptr) {
        return false;
    }
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs->data);
    const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs->data);
    const size_t lhs_size = ggml_nbytes(lhs);
    const size_t rhs_size = ggml_nbytes(rhs);
    const uintptr_t lhs_end = lhs_size > UINTPTR_MAX - lhs_begin ? UINTPTR_MAX : lhs_begin + lhs_size;
    const uintptr_t rhs_end = rhs_size > UINTPTR_MAX - rhs_begin ? UINTPTR_MAX : rhs_begin + rhs_size;
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

ggml_cann_im2col1d_atomic_kind_stats & stats_for_kind(ggml_cann_im2col1d_kind kind) {
    return kind == ggml_cann_im2col1d_kind::CAUSAL ? g_causal_stats : g_vocoder_stats;
}

void stats_marked(ggml_cann_im2col1d_kind kind) {
    g_stats.marked.fetch_add(1, std::memory_order_relaxed);
    stats_for_kind(kind).marked.fetch_add(1, std::memory_order_relaxed);
}

void stats_fallback(ggml_cann_im2col1d_kind kind, ggml_cann_im2col1d_fallback reason) {
    g_stats.fallback[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    stats_for_kind(kind).fallback[static_cast<size_t>(reason)].fetch_add(
        1, std::memory_order_relaxed);
}

void stats_hit(ggml_cann_im2col1d_kind kind,
               uint64_t temp_bytes, uint64_t permutes,
               uint64_t d2d_copies, uint64_t d2d_bytes) {
    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    stats_for_kind(kind).hits.fetch_add(1, std::memory_order_relaxed);
    g_stats.temp_bytes_removed.fetch_add(temp_bytes, std::memory_order_relaxed);
    g_stats.permutes_removed.fetch_add(permutes, std::memory_order_relaxed);
    g_stats.d2d_copies_removed.fetch_add(d2d_copies, std::memory_order_relaxed);
    g_stats.d2d_bytes_removed.fetch_add(d2d_bytes, std::memory_order_relaxed);
}

void stats_print() {
    const ggml_cann_im2col1d_stats_snapshot stats = ggml_cann_im2col1d_stats_get();
    const double hit_rate = stats.marked == 0 ? 0.0 :
        static_cast<double>(stats.hits) / static_cast<double>(stats.marked);

    std::fprintf(stderr,
                 "[cann-im2col1d] marked=%llu hits=%llu hit_rate=%.6f "
                 "temp_bytes_removed=%llu permutes_removed=%llu "
                 "d2d_copies_removed=%llu d2d_bytes_removed=%llu "
                 "ctb_split_hits=%llu ctb_static_offset_hits=%llu "
                 "causal_marked=%llu causal_hits=%llu "
                 "vocoder_marked=%llu vocoder_hits=%llu",
                 static_cast<unsigned long long>(stats.marked),
                 static_cast<unsigned long long>(stats.hits),
                 hit_rate,
                 static_cast<unsigned long long>(stats.temp_bytes_removed),
                 static_cast<unsigned long long>(stats.permutes_removed),
                 static_cast<unsigned long long>(stats.d2d_copies_removed),
                 static_cast<unsigned long long>(stats.d2d_bytes_removed),
                 static_cast<unsigned long long>(stats.ctb_split_hits),
                 static_cast<unsigned long long>(stats.ctb_static_offset_hits),
                 static_cast<unsigned long long>(stats.causal.marked),
                 static_cast<unsigned long long>(stats.causal.hits),
                 static_cast<unsigned long long>(stats.vocoder.marked),
                 static_cast<unsigned long long>(stats.vocoder.hits));
    for (size_t i = static_cast<size_t>(ggml_cann_im2col1d_fallback::DISABLED);
         i < fallback_count;
         ++i) {
        std::fprintf(stderr,
                     " fallback_%s=%llu causal_fallback_%s=%llu vocoder_fallback_%s=%llu",
                     fallback_names[i],
                     static_cast<unsigned long long>(stats.fallback[i]),
                     fallback_names[i],
                     static_cast<unsigned long long>(stats.causal.fallback[i]),
                     fallback_names[i],
                     static_cast<unsigned long long>(stats.vocoder.fallback[i]));
    }
    std::fputc('\n', stderr);

}

void register_stats_printer() {
    std::call_once(g_stats_atexit_once, [] {
        if (parse_bool(get_env_as_lowercase("GGML_CANN_IM2COL1D_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

uint32_t block_dim_for_kind(
        ggml_cann_im2col1d_kind kind,
        uint64_t line_count,
        uint64_t output_rows) {
    const uint64_t work_items =
        kind == ggml_cann_im2col1d_kind::VOCODER
            ? output_rows : line_count;
    const uint64_t cap = kind == ggml_cann_im2col1d_kind::VOCODER ||
            get_env_as_lowercase("GGML_CANN_IM2COL1D_CAUSAL_BLOCKS")
                    .value_or("40") != "20"
        ? 40
        : 20;
    return static_cast<uint32_t>(std::min(work_items, cap));
}

}  // namespace

const char * ggml_cann_im2col1d_fallback_name(ggml_cann_im2col1d_fallback reason) {
    const size_t index = static_cast<size_t>(reason);
    return index < fallback_count ? fallback_names[index] : "unknown";
}

ggml_cann_im2col1d_fallback ggml_cann_im2col1d_validate(
        const ggml_tensor * dst,
        ggml_cann_im2col1d_params * params) {
    if (dst == nullptr) {
        return ggml_cann_im2col1d_fallback::UNMARKED;
    }
    const int32_t marker = reinterpret_cast<const int32_t *>(dst->op_params)[7];
    ggml_cann_im2col1d_kind kind;
    bool causal_ctb = false;
    if (marker == GGML_IM2COL_CAUSAL_1D_MARKER_V1) {
        kind = ggml_cann_im2col1d_kind::CAUSAL;
    } else if (marker == GGML_IM2COL_CAUSAL_CTB_1D_MARKER_V1) {
        kind = ggml_cann_im2col1d_kind::CAUSAL;
        causal_ctb = true;
    } else if (marker == GGML_IM2COL_VOCODER_1D_MARKER_V1) {
        kind = ggml_cann_im2col1d_kind::VOCODER;
    } else {
        return ggml_cann_im2col1d_fallback::UNMARKED;
    }
    if (params != nullptr) {
        params->kind = kind;
        params->causal_ctb = causal_ctb;
    }

    const int32_t * op = reinterpret_cast<const int32_t *>(dst->op_params);
    if (op[6] != 0) {
        return ggml_cann_im2col1d_fallback::IS_2D;
    }
    if (op[1] != 0 || op[3] != 0 || op[5] != 0) {
        return ggml_cann_im2col1d_fallback::SECONDARY_PARAMS;
    }

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src = dst->src[1];
    if (src == nullptr || kernel == nullptr) {
        return ggml_cann_im2col1d_fallback::SHAPE;
    }
    if (src->type != GGML_TYPE_F32) {
        return ggml_cann_im2col1d_fallback::SRC_DTYPE;
    }
    if (dst->type != GGML_TYPE_F32) {
        return ggml_cann_im2col1d_fallback::DST_DTYPE;
    }
    if (causal_ctb) {
        const bool exact_ctb_view = src->nb[1] == sizeof(float) &&
            src->ne[1] > 0 &&
            static_cast<uint64_t>(src->ne[1]) <=
                std::numeric_limits<size_t>::max() / sizeof(float) &&
            src->nb[0] == static_cast<size_t>(src->ne[1]) * sizeof(float) &&
            src->ne[0] > 0 &&
            static_cast<uint64_t>(src->ne[0]) <=
                std::numeric_limits<size_t>::max() / src->nb[0] &&
            src->nb[2] == static_cast<size_t>(src->ne[0]) * src->nb[0] &&
            src->ne[2] > 0 &&
            static_cast<uint64_t>(src->ne[2]) <=
                std::numeric_limits<size_t>::max() / src->nb[2] &&
            src->nb[3] == static_cast<size_t>(src->ne[2]) * src->nb[2];
        if (!exact_ctb_view) {
            return ggml_cann_im2col1d_fallback::SRC_LAYOUT;
        }
    } else if (!exact_contiguous(src)) {
        return ggml_cann_im2col1d_fallback::SRC_LAYOUT;
    }
    if (!exact_contiguous(dst)) {
        return ggml_cann_im2col1d_fallback::DST_LAYOUT;
    }

    if (src->ne[3] != 1 || dst->ne[3] != 1 ||
        kernel->ne[0] <= 0 || kernel->ne[1] <= 0 || kernel->ne[2] <= 0 || kernel->ne[3] != 1 ||
        src->ne[0] <= 0 || src->ne[1] != kernel->ne[1] || src->ne[2] <= 0 ||
        dst->ne[2] != src->ne[2] ||
        (kind == ggml_cann_im2col1d_kind::VOCODER && src->ne[2] != 1)) {
        return ggml_cann_im2col1d_fallback::SHAPE;
    }

    const __int128 dst_ne0 = static_cast<__int128>(kernel->ne[0]) * kernel->ne[1];
    if (dst_ne0 != dst->ne[0]) {
        return ggml_cann_im2col1d_fallback::SHAPE;
    }

    const int32_t s0 = op[0];
    const int32_t p0 = op[2];
    const int32_t d0 = op[4];
    const bool valid_conv = kind == ggml_cann_im2col1d_kind::CAUSAL
        ? kernel->ne[0] == 3 && src->ne[0] >= 3 && s0 == 1 && p0 == 0 && d0 == 1
        : (((kernel->ne[0] == 3 || kernel->ne[0] == 7 || kernel->ne[0] == 11) &&
            s0 == 1 &&
            (d0 == 1 || d0 == 3 || d0 == 5) &&
            p0 == d0 * (kernel->ne[0] - 1) / 2) ||
           (kernel->ne[0] == 30 && s0 == 15 && p0 == 7 && d0 == 1));
    if (!valid_conv) {
        return ggml_cann_im2col1d_fallback::CONV_PARAMS;
    }

    const __int128 numerator =
        static_cast<__int128>(src->ne[0]) +
        static_cast<__int128>(2) * p0 -
        static_cast<__int128>(d0) * (kernel->ne[0] - 1) - 1;
    if (numerator < 0) {
        return ggml_cann_im2col1d_fallback::OUTPUT_SHAPE;
    }
    const __int128 calculated_ow = numerator / s0 + 1;
    if (calculated_ow != dst->ne[1]) {
        return ggml_cann_im2col1d_fallback::OUTPUT_SHAPE;
    }

    const uint64_t t = static_cast<uint64_t>(src->ne[0]);
    const uint64_t c = static_cast<uint64_t>(src->ne[1]);
    const uint64_t b = static_cast<uint64_t>(src->ne[2]);
    const uint64_t k = static_cast<uint64_t>(kernel->ne[0]);
    const uint64_t ow = static_cast<uint64_t>(calculated_ow);
    const uint64_t stride = static_cast<uint64_t>(s0);
    const uint64_t dilation = static_cast<uint64_t>(d0);
    using uint128_t = unsigned __int128;
    const uint128_t max_u64 = std::numeric_limits<uint64_t>::max();
    const uint128_t max_size_words =
        std::numeric_limits<size_t>::max() / sizeof(float);
    const uint128_t max_words = std::min(max_u64, max_size_words);
    const auto checked_multiply = [](
            uint128_t lhs, uint128_t rhs, uint128_t limit, uint128_t & result) {
        if (lhs == 0 || rhs == 0) {
            result = 0;
            return true;
        }
        if (lhs > limit / rhs) {
            return false;
        }
        result = lhs * rhs;
        return true;
    };

    uint128_t units_wide;
    if (!checked_multiply(b, ow, max_u64, units_wide) ||
        !checked_multiply(units_wide, c, max_u64, units_wide) ||
        units_wide == 0) {
        return ggml_cann_im2col1d_fallback::INDEX_OVERFLOW;
    }

    uint128_t dst_words_wide;
    uint128_t src_words_wide;
    if (!checked_multiply(units_wide, k, max_words, dst_words_wide) ||
        !checked_multiply(b, c, max_words, src_words_wide) ||
        !checked_multiply(src_words_wide, t, max_words, src_words_wide)) {
        return ggml_cann_im2col1d_fallback::INDEX_OVERFLOW;
    }

    uint128_t dst_bytes_wide;
    if (!checked_multiply(dst_words_wide, sizeof(float), max_u64, dst_bytes_wide) ||
        dst_bytes_wide > max_u64 / 6) {
        return ggml_cann_im2col1d_fallback::INDEX_OVERFLOW;
    }

    const uint64_t dst_words = static_cast<uint64_t>(dst_words_wide);
    const uint64_t line_count = (dst_words - 1) / gm_cache_line_words + 1;

    const __int128 max_time =
        static_cast<__int128>(ow - 1) * stride +
        static_cast<__int128>(k - 1) * dilation;
    if (max_time > std::numeric_limits<int64_t>::max()) {
        return ggml_cann_im2col1d_fallback::INDEX_OVERFLOW;
    }

    if (params != nullptr) {
        params->t = t;
        params->c = c;
        params->b = b;
        params->k = k;
        params->ow = ow;
        params->stride = stride;
        params->padding = p0;
        params->dilation = dilation;
        params->block_dim = block_dim_for_kind(kind, line_count, b * ow);
    }
    return ggml_cann_im2col1d_fallback::NONE;
}

void ggml_cann_im2col1d_bench_reset() {
    std::lock_guard<std::mutex> lock(g_bench_mutex);
    g_kernel_ms.clear();
}

double ggml_cann_im2col1d_bench_p50(size_t warmup) {
    std::vector<float> values;
    {
        std::lock_guard<std::mutex> lock(g_bench_mutex);
        values = g_kernel_ms;
    }
    if (values.size() <= warmup) {
        return 0.0;
    }
    values.erase(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(warmup));
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool ggml_cann_im2col1d_ctb_split_enabled() {
    return get_env_as_lowercase("GGML_CANN_IM2COL1D_CTB_SPLIT")
               .value_or("auto") == "auto";
}

static bool ggml_cann_im2col1d_ctb_static_offsets_enabled() {
    return get_env_as_lowercase("GGML_CANN_IM2COL1D_CTB_STATIC_OFFSETS")
               .value_or("auto") == "auto";
}

static bool ggml_cann_im2col1d_vocoder_tile_enabled() {
    return get_env_as_lowercase("GGML_CANN_IM2COL1D_VOCODER_TILE")
               .value_or("auto") == "auto";
}

static uint64_t ggml_cann_im2col1d_ctb_mode(
        const ggml_cann_im2col1d_params & params,
        bool split) {
    if (!params.causal_ctb) {
        if (params.kind == ggml_cann_im2col1d_kind::VOCODER &&
            params.k == 30 && params.stride == 15 &&
            params.padding == 7 && params.dilation == 1) {
            return UINT64_C(5);
        }
        if (params.kind == ggml_cann_im2col1d_kind::VOCODER &&
            ggml_cann_im2col1d_vocoder_tile_enabled() &&
            params.stride == 1 && params.ow >= 320 &&
            (params.k == 3 || params.k == 7 || params.k == 11) &&
            (params.dilation == 1 || params.dilation == 3 ||
             params.dilation == 5)) {
            return UINT64_C(6);
        }
        return 0;
    }
    const bool use_static_offsets =
        ggml_cann_im2col1d_ctb_static_offsets_enabled() &&
        params.c == 512 && params.b == 2 &&
        (params.ow == 50 || params.ow == 56) &&
        params.t == params.ow + 2;
    if (split) {
        return use_static_offsets ? UINT64_C(4) : UINT64_C(2);
    }
    return use_static_offsets ? UINT64_C(3) : UINT64_C(1);
}

bool ggml_cann_im2col1d_try_ctb_split(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int concat_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * permute_index,
        int * im2col_index) {
    if (!ggml_cann_im2col1d_ctb_split_enabled() || cgraph == nullptr ||
        concat_index < 0 || concat_index >= cgraph->n_nodes ||
        permute_index == nullptr || im2col_index == nullptr) {
        return false;
    }

    ggml_tensor * concat = cgraph->nodes[concat_index];
    if (concat == nullptr || concat->op != GGML_OP_CONCAT ||
        ggml_get_op_params_i32(concat, 0) != 1 ||
        concat->src[0] == nullptr || concat->src[1] == nullptr) {
        return false;
    }

    const auto concat_consumer = unique_consumers.find(concat);
    if (concat_consumer == unique_consumers.end() ||
        concat_consumer->second < 0 ||
        concat_consumer->second >= cgraph->n_nodes) {
        return false;
    }
    const int permute_node_index = concat_consumer->second;
    ggml_tensor * permute = cgraph->nodes[permute_node_index];
    if (permute == nullptr || permute->op != GGML_OP_PERMUTE ||
        permute->src[0] != concat ||
        ggml_get_op_params_i32(permute, 0) != 1 ||
        ggml_get_op_params_i32(permute, 1) != 0 ||
        ggml_get_op_params_i32(permute, 2) != 2 ||
        ggml_get_op_params_i32(permute, 3) != 3) {
        return false;
    }

    const auto permute_consumer = unique_consumers.find(permute);
    if (permute_consumer == unique_consumers.end() ||
        permute_consumer->second < 0 ||
        permute_consumer->second >= cgraph->n_nodes) {
        return false;
    }
    const int im2col_node_index = permute_consumer->second;
    ggml_tensor * im2col = cgraph->nodes[im2col_node_index];
    if (im2col == nullptr || im2col->op != GGML_OP_IM2COL ||
        im2col->src[1] != permute) {
        return false;
    }

    ggml_cann_im2col1d_params params;
    if (ggml_cann_im2col1d_validate(im2col, &params) !=
            ggml_cann_im2col1d_fallback::NONE ||
        params.kind != ggml_cann_im2col1d_kind::CAUSAL ||
        !params.causal_ctb || params.k != 3 || params.stride != 1 ||
        params.padding != 0 || params.dilation != 1 ||
        params.c != 512 || params.b != 2 ||
        (params.ow != 50 && params.ow != 56)) {
        return false;
    }
    if (get_env_as_lowercase("GGML_CANN_IM2COL1D").value_or("auto") ==
        "off") {
        return false;
    }

    const ggml_tensor * cache = concat->src[0];
    const ggml_tensor * x = concat->src[1];
    if (cache->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        !exact_contiguous(cache) || !exact_contiguous(x) ||
        cache->data == nullptr || x->data == nullptr ||
        cache->data == x->data || im2col->data == nullptr ||
        tensor_ranges_overlap(im2col, cache) ||
        tensor_ranges_overlap(im2col, x) ||
        cache->ne[0] != static_cast<int64_t>(params.c) ||
        x->ne[0] != static_cast<int64_t>(params.c) ||
        cache->ne[1] != 2 ||
        x->ne[1] != static_cast<int64_t>(params.ow) ||
        cache->ne[2] != static_cast<int64_t>(params.b) ||
        x->ne[2] != static_cast<int64_t>(params.b) ||
        cache->ne[3] != 1 || x->ne[3] != 1 ||
        concat->ne[0] != x->ne[0] ||
        concat->ne[1] != cache->ne[1] + x->ne[1] ||
        concat->ne[2] != x->ne[2] || concat->ne[3] != 1) {
        return false;
    }

    register_stats_printer();
    stats_marked(params.kind);
    const uint64_t ctb_mode = ggml_cann_im2col1d_ctb_mode(params, true);
    ACL_CHECK(ACLRT_LAUNCH_KERNEL(im2col1d_f32)(
        params.block_dim,
        ctx.stream(),
        x->data,
        cache->data,
        im2col->data,
        params.t,
        params.c,
        params.b,
        params.k,
        params.ow,
        params.stride,
        params.padding,
        params.dilation,
        ctb_mode));

    const uint64_t dst_bytes =
        params.b * params.ow * params.c * params.k * sizeof(float);
    const uint64_t concat_bytes = ggml_nbytes(concat);
    stats_hit(
        params.kind,
        6 * dst_bytes + concat_bytes,
        1,
        params.b * params.ow + 1,
        dst_bytes + concat_bytes);
    g_stats.ctb_split_hits.fetch_add(1, std::memory_order_relaxed);
    if (ctb_mode == 4) {
        g_stats.ctb_static_offset_hits.fetch_add(1, std::memory_order_relaxed);
    }
    *permute_index = permute_node_index;
    *im2col_index = im2col_node_index;
    return true;
}

bool ggml_cann_im2col1d_try(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_cann_im2col1d_params params;
    const ggml_cann_im2col1d_fallback reason = ggml_cann_im2col1d_validate(dst, &params);
    if (reason == ggml_cann_im2col1d_fallback::UNMARKED) {
        return false;
    }

    stats_marked(params.kind);
    register_stats_printer();

    if (get_env_as_lowercase("GGML_CANN_IM2COL1D").value_or("auto") == "off") {
        stats_fallback(params.kind, ggml_cann_im2col1d_fallback::DISABLED);
        return false;
    }
    if (params.kind == ggml_cann_im2col1d_kind::VOCODER &&
        get_env_as_lowercase("GGML_CANN_IM2COL1D_VOCODER").value_or("auto") == "off") {
        stats_fallback(params.kind, ggml_cann_im2col1d_fallback::DISABLED);
        return false;
    }
    if (params.kind == ggml_cann_im2col1d_kind::VOCODER &&
        params.k == 30 && params.stride == 15 &&
        params.padding == 7 && params.dilation == 1 &&
        get_env_as_lowercase("GGML_CANN_IM2COL1D_HIFT_STRIDED")
                .value_or("auto") == "off") {
        stats_fallback(params.kind, ggml_cann_im2col1d_fallback::DISABLED);
        return false;
    }
    if (reason != ggml_cann_im2col1d_fallback::NONE) {
        stats_fallback(params.kind, reason);
        return false;
    }

    const bool bench_enabled =
#ifdef USE_ACL_GRAPH
        !ctx.acl_graph_mode &&
#endif
        parse_bool(get_env_as_lowercase("GGML_CANN_IM2COL1D_BENCH").value_or(""));
    const uint64_t ctb_mode = ggml_cann_im2col1d_ctb_mode(params, false);
    if (!bench_enabled) {
        ACL_CHECK(ACLRT_LAUNCH_KERNEL(im2col1d_f32)(
            params.block_dim,
            ctx.stream(),
            dst->src[1]->data,
            nullptr,
            dst->data,
            params.t,
            params.c,
            params.b,
            params.k,
            params.ow,
            params.stride,
            params.padding,
            params.dilation,
            ctb_mode));
    } else {
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
        ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
        ACL_CHECK(aclrtRecordEvent(start, ctx.stream()));
        ACL_CHECK(ACLRT_LAUNCH_KERNEL(im2col1d_f32)(
            params.block_dim,
            ctx.stream(),
            dst->src[1]->data,
            nullptr,
            dst->data,
            params.t,
            params.c,
            params.b,
            params.k,
            params.ow,
            params.stride,
            params.padding,
            params.dilation,
            ctb_mode));
        ACL_CHECK(aclrtRecordEvent(end, ctx.stream()));
        ACL_CHECK(aclrtSynchronizeEvent(end));
        float elapsed_ms = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
        record_kernel_ms(elapsed_ms);
        ACL_CHECK(aclrtDestroyEvent(start));
        ACL_CHECK(aclrtDestroyEvent(end));
    }

    const uint64_t dst_bytes = params.b * params.ow * params.c * params.k * sizeof(float);
    stats_hit(params.kind, 6 * dst_bytes, 1, params.b * params.ow, dst_bytes);
    if (ctb_mode == 3) {
        g_stats.ctb_static_offset_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

static void reset_kind_stats(ggml_cann_im2col1d_atomic_kind_stats & stats) {
    stats.marked.store(0, std::memory_order_relaxed);
    stats.hits.store(0, std::memory_order_relaxed);
    for (auto & value : stats.fallback) {
        value.store(0, std::memory_order_relaxed);
    }
}

void ggml_cann_im2col1d_stats_reset() {
    reset_kind_stats(g_stats);
    reset_kind_stats(g_causal_stats);
    reset_kind_stats(g_vocoder_stats);
    g_stats.temp_bytes_removed.store(0, std::memory_order_relaxed);
    g_stats.permutes_removed.store(0, std::memory_order_relaxed);
    g_stats.d2d_copies_removed.store(0, std::memory_order_relaxed);
    g_stats.d2d_bytes_removed.store(0, std::memory_order_relaxed);
    g_stats.ctb_split_hits.store(0, std::memory_order_relaxed);
    g_stats.ctb_static_offset_hits.store(0, std::memory_order_relaxed);
}

static void load_kind_stats(ggml_cann_im2col1d_kind_stats_snapshot & snapshot,
                     const ggml_cann_im2col1d_atomic_kind_stats & stats) {
    snapshot.marked = stats.marked.load(std::memory_order_relaxed);
    snapshot.hits = stats.hits.load(std::memory_order_relaxed);
    for (size_t i = 0; i < fallback_count; ++i) {
        snapshot.fallback[i] = stats.fallback[i].load(std::memory_order_relaxed);
    }
}

ggml_cann_im2col1d_stats_snapshot ggml_cann_im2col1d_stats_get() {
    ggml_cann_im2col1d_stats_snapshot snapshot;
    load_kind_stats(snapshot, g_stats);
    load_kind_stats(snapshot.causal, g_causal_stats);
    load_kind_stats(snapshot.vocoder, g_vocoder_stats);
    snapshot.temp_bytes_removed = g_stats.temp_bytes_removed.load(std::memory_order_relaxed);
    snapshot.permutes_removed = g_stats.permutes_removed.load(std::memory_order_relaxed);
    snapshot.d2d_copies_removed = g_stats.d2d_copies_removed.load(std::memory_order_relaxed);
    snapshot.d2d_bytes_removed = g_stats.d2d_bytes_removed.load(std::memory_order_relaxed);
    snapshot.ctb_split_hits = g_stats.ctb_split_hits.load(std::memory_order_relaxed);
    snapshot.ctb_static_offset_hits =
        g_stats.ctb_static_offset_hits.load(std::memory_order_relaxed);
    return snapshot;
}
