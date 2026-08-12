#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct ggml_backend_cann_context;

enum class ggml_cann_im2col1d_fallback : uint32_t {
    NONE = 0,
    UNMARKED,
    DISABLED,
    IS_2D,
    SECONDARY_PARAMS,
    SRC_DTYPE,
    DST_DTYPE,
    SRC_LAYOUT,
    DST_LAYOUT,
    SHAPE,
    CONV_PARAMS,
    OUTPUT_SHAPE,
    INDEX_OVERFLOW,
    COUNT,
};

enum class ggml_cann_im2col1d_kind : uint32_t {
    CAUSAL = 0,
    VOCODER,
};

struct ggml_cann_im2col1d_params {
    uint64_t t = 0;
    ggml_cann_im2col1d_kind kind = ggml_cann_im2col1d_kind::CAUSAL;
    uint64_t c = 0;
    uint64_t b = 0;
    uint64_t k = 0;
    uint64_t ow = 0;
    uint64_t stride = 0;
    int64_t padding = 0;
    uint64_t dilation = 0;
    uint32_t block_dim = 0;
    bool causal_ctb = false;
};

struct ggml_cann_im2col1d_kind_stats_snapshot {
    uint64_t marked = 0;
    uint64_t hits = 0;
    std::array<uint64_t, static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT)> fallback{};
};

struct ggml_cann_im2col1d_stats_snapshot : ggml_cann_im2col1d_kind_stats_snapshot {
    ggml_cann_im2col1d_kind_stats_snapshot causal;
    ggml_cann_im2col1d_kind_stats_snapshot vocoder;
    uint64_t temp_bytes_removed = 0;
    uint64_t permutes_removed = 0;
    uint64_t d2d_copies_removed = 0;
    uint64_t d2d_bytes_removed = 0;
    uint64_t ctb_split_hits = 0;
    uint64_t ctb_static_offset_hits = 0;
};

ggml_cann_im2col1d_fallback ggml_cann_im2col1d_validate(
        const ggml_tensor * dst,
        ggml_cann_im2col1d_params * params);
bool ggml_cann_im2col1d_try(ggml_backend_cann_context & ctx, ggml_tensor * dst);
bool ggml_cann_im2col1d_ctb_split_enabled();
bool ggml_cann_im2col1d_try_ctb_split(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int concat_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * permute_index,
        int * im2col_index);
const char * ggml_cann_im2col1d_fallback_name(ggml_cann_im2col1d_fallback reason);
void ggml_cann_im2col1d_stats_reset();
ggml_cann_im2col1d_stats_snapshot ggml_cann_im2col1d_stats_get();
void ggml_cann_im2col1d_bench_reset();
double ggml_cann_im2col1d_bench_p50(size_t warmup);
