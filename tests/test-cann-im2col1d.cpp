#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "im2col1d.h"
#include "im2col1d_vocoder_gather_mask.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

static void test_vocoder_gather_mask_patterns() {
    struct mask_case {
        uint32_t kernel;
        uint32_t dilation;
        uint32_t word0;
        uint32_t word1;
    };
    const mask_case cases[] = {
        { 3, 1, UINT32_C(0x00000007), UINT32_C(0x00000000) },
        { 3, 3, UINT32_C(0x00000049), UINT32_C(0x00000000) },
        { 3, 5, UINT32_C(0x00000421), UINT32_C(0x00000000) },
        { 7, 1, UINT32_C(0x0000007f), UINT32_C(0x00000000) },
        { 7, 3, UINT32_C(0x00049249), UINT32_C(0x00000000) },
        { 7, 5, UINT32_C(0x42108421), UINT32_C(0x00000000) },
        { 11, 1, UINT32_C(0x000007ff), UINT32_C(0x00000000) },
        { 11, 3, UINT32_C(0x49249249), UINT32_C(0x00000000) },
        { 11, 5, UINT32_C(0x42108421), UINT32_C(0x00042108) },
    };
    for (const mask_case & tc : cases) {
        const vocoder_gather_mask_pattern actual =
            make_vocoder_gather_mask_pattern(tc.kernel, tc.dilation);
        assert(actual.word0 == tc.word0);
        assert(actual.word1 == tc.word1);
        assert(actual.selected == tc.kernel);
    }
}

static ggml_backend_t init_cann() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(dev)).find("CANN") != std::string::npos) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static void assert_words_equal(const ggml_tensor * old_out, const ggml_tensor * fast_out) {
    assert(ggml_nbytes(old_out) == ggml_nbytes(fast_out));
    std::vector<uint32_t> old_words(ggml_nbytes(old_out) / sizeof(uint32_t));
    std::vector<uint32_t> fast_words(old_words.size());
    ggml_backend_tensor_get(old_out, old_words.data(), 0, ggml_nbytes(old_out));
    ggml_backend_tensor_get(fast_out, fast_words.data(), 0, ggml_nbytes(fast_out));
    for (size_t i = 0; i < old_words.size(); ++i) {
        if (old_words[i] != fast_words[i]) {
            std::fprintf(stderr, "word mismatch i=%zu old=%08x fast=%08x\n",
                i, old_words[i], fast_words[i]);
            std::abort();
        }
    }
}

static void poison_output_pair(ggml_tensor * old_out, ggml_tensor * fast_out) {
    assert(ggml_nbytes(old_out) == ggml_nbytes(fast_out));
    assert(ggml_nbytes(old_out) % sizeof(uint32_t) == 0);
    std::vector<uint32_t> old_canary(
        ggml_nbytes(old_out) / sizeof(uint32_t), UINT32_C(0xdeadbeef));
    std::vector<uint32_t> fast_canary(
        ggml_nbytes(fast_out) / sizeof(uint32_t), UINT32_C(0xa5a5a5a5));
    ggml_backend_tensor_set(old_out, old_canary.data(), 0, ggml_nbytes(old_out));
    ggml_backend_tensor_set(fast_out, fast_canary.data(), 0, ggml_nbytes(fast_out));
}

static void test_marker_helper() {
    ggml_init_params params = { 4u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 8);
    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 18, 4, 2);
    ggml_tensor * plain  = ggml_im2col(ctx, weight, input, 2, 0, 1, 0, 3, 0, false, GGML_TYPE_F32);
    ggml_tensor * marked = ggml_im2col_causal_1d(ctx, weight, input, 2, 1, 3, GGML_TYPE_F32);
    ggml_tensor * vocoder = ggml_im2col_vocoder_1d(
        ctx, weight, input, 2, 1, 3, GGML_TYPE_F32);

    const int32_t * p0 = reinterpret_cast<const int32_t *>(plain->op_params);
    const int32_t * p1 = reinterpret_cast<const int32_t *>(marked->op_params);
    const int32_t * p2 = reinterpret_cast<const int32_t *>(vocoder->op_params);
    for (int i = 0; i < 7; ++i) {
        assert(p0[i] == p1[i]);
        assert(p0[i] == p2[i]);
    }
    assert(p0[7] == 0);
    assert(p1[7] == GGML_IM2COL_CAUSAL_1D_MARKER_V1);
    assert(p2[7] == GGML_IM2COL_VOCODER_1D_MARKER_V1);
    assert(p2[7] != GGML_IM2COL_CAUSAL_1D_MARKER_V1);
    assert(p1[0] == 2 && p1[1] == 0);
    assert(p1[2] == 1 && p1[3] == 0);
    assert(p1[4] == 3 && p1[5] == 0);
    assert(p1[6] == 0);
    ggml_free(ctx);
}

static void expect_reason(
        const ggml_tensor * node,
        ggml_cann_im2col1d_fallback expected) {
    ggml_cann_im2col1d_params params;
    const ggml_cann_im2col1d_fallback actual = ggml_cann_im2col1d_validate(node, &params);
    if (actual != expected) {
        std::fprintf(stderr, "expected %s, got %s\n",
                     ggml_cann_im2col1d_fallback_name(expected),
                     ggml_cann_im2col1d_fallback_name(actual));
    }
    assert(actual == expected);
}

static void test_split_stats_api() {
    ggml_cann_im2col1d_stats_reset();
    const auto stats = ggml_cann_im2col1d_stats_get();
    assert(stats.marked == 0);
    assert(stats.hits == 0);
    assert(stats.causal.marked == 0);
    assert(stats.causal.hits == 0);
    assert(stats.vocoder.marked == 0);
    assert(stats.vocoder.hits == 0);
    for (size_t i = 0; i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        assert(stats.causal.fallback[i] == 0);
        assert(stats.vocoder.fallback[i] == 0);
    }
}

static void make_exact_contiguous(ggml_tensor * tensor) {
    tensor->nb[0] = ggml_type_size(tensor->type);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        tensor->nb[d] = tensor->nb[d - 1] * static_cast<size_t>(tensor->ne[d - 1]);
    }
}

struct scoped_env_restore {
    std::string name;
    std::string previous;
    bool had_previous;

    explicit scoped_env_restore(const char * env_name)
        : name(env_name),
          previous(std::getenv(env_name) != nullptr ? std::getenv(env_name) : ""),
          had_previous(std::getenv(env_name) != nullptr) {
    }

    ~scoped_env_restore() {
        if (had_previous) {
            assert(setenv(name.c_str(), previous.c_str(), 1) == 0);
        } else {
            assert(unsetenv(name.c_str()) == 0);
        }
    }
};

static void test_validation_matrix() {
    ggml_init_params init_params = { 4u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 8);
    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 18, 4, 2);
    ggml_tensor * plain  = ggml_im2col(ctx, kernel, input, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * marked = ggml_im2col_causal_1d(ctx, kernel, input, 1, 0, 1, GGML_TYPE_F32);

    expect_reason(marked, ggml_cann_im2col1d_fallback::NONE);
    expect_reason(plain, ggml_cann_im2col1d_fallback::UNMARKED);

    ggml_tensor * kernel_k5 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 4, 8);
    ggml_tensor * marked_k5 = ggml_im2col_causal_1d(
        ctx, kernel_k5, input, 1, 0, 1, GGML_TYPE_F32);
    expect_reason(marked_k5, ggml_cann_im2col1d_fallback::CONV_PARAMS);

    int32_t * op = reinterpret_cast<int32_t *>(marked->op_params);

    op[6] = 1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::IS_2D);
    op[6] = 0;

    op[1] = 1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SECONDARY_PARAMS);
    op[1] = 0;
    op[3] = 1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SECONDARY_PARAMS);
    op[3] = 0;
    op[5] = 1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SECONDARY_PARAMS);
    op[5] = 0;

    input->type = GGML_TYPE_F16;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SRC_DTYPE);
    input->type = GGML_TYPE_BF16;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SRC_DTYPE);
    input->type = GGML_TYPE_F32;

    marked->type = GGML_TYPE_F16;
    expect_reason(marked, ggml_cann_im2col1d_fallback::DST_DTYPE);
    marked->type = GGML_TYPE_F32;

    input->nb[1] += sizeof(float);
    expect_reason(marked, ggml_cann_im2col1d_fallback::SRC_LAYOUT);
    make_exact_contiguous(input);

    marked->nb[1] += sizeof(float);
    expect_reason(marked, ggml_cann_im2col1d_fallback::DST_LAYOUT);
    make_exact_contiguous(marked);

    input->ne[3] = 2;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SHAPE);
    input->ne[3] = 1;

    kernel->ne[1] = 5;
    expect_reason(marked, ggml_cann_im2col1d_fallback::SHAPE);
    kernel->ne[1] = 4;

    const int64_t dst_ne0 = marked->ne[0];
    marked->ne[0] = dst_ne0 + 1;
    make_exact_contiguous(marked);
    expect_reason(marked, ggml_cann_im2col1d_fallback::SHAPE);
    marked->ne[0] = dst_ne0;
    make_exact_contiguous(marked);

    op[0] = 0;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[0] = 2;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[0] = 1;
    op[2] = -1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[2] = 1;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[2] = 0;
    op[4] = 0;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[4] = 2;
    expect_reason(marked, ggml_cann_im2col1d_fallback::CONV_PARAMS);
    op[4] = 1;

    const int64_t dst_ow = marked->ne[1];
    marked->ne[1] = dst_ow + 1;
    make_exact_contiguous(marked);
    expect_reason(marked, ggml_cann_im2col1d_fallback::OUTPUT_SHAPE);
    marked->ne[1] = dst_ow;
    make_exact_contiguous(marked);

    expect_reason(marked, ggml_cann_im2col1d_fallback::NONE);
    ggml_free(ctx);
}

static void test_vocoder_validation_matrix() {
    ggml_init_params init_params = { 32u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    const auto make_node = [&](int64_t k, int64_t t, int64_t c, int64_t b,
                               int stride, int padding, int dilation) {
        ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, c, c);
        ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t, c, b);
        return ggml_im2col_vocoder_1d(
            ctx, kernel, input, stride, padding, dilation, GGML_TYPE_F32);
    };

    for (int k : { 3, 7, 11 }) {
        for (int d : { 1, 3, 5 }) {
            const int p = d * (k - 1) / 2;
            ggml_tensor * node = make_node(k, 37, 64, 1, 1, p, d);
            expect_reason(node, ggml_cann_im2col1d_fallback::NONE);
            assert(node->ne[0] == k * 64);
            assert(node->ne[1] == 37);
            assert(node->ne[2] == 1);
            assert(node->ne[3] == 1);
        }
    }

    expect_reason(make_node(5, 37, 64, 1, 1, 2, 1),
                  ggml_cann_im2col1d_fallback::CONV_PARAMS);
    expect_reason(make_node(3, 37, 64, 1, 1, 1, 2),
                  ggml_cann_im2col1d_fallback::CONV_PARAMS);
    expect_reason(make_node(3, 37, 64, 1, 2, 1, 1),
                  ggml_cann_im2col1d_fallback::CONV_PARAMS);
    expect_reason(make_node(3, 37, 64, 1, 1, 2, 1),
                  ggml_cann_im2col1d_fallback::CONV_PARAMS);
    expect_reason(make_node(3, 37, 64, 1, 1, -1, 1),
                  ggml_cann_im2col1d_fallback::CONV_PARAMS);
    expect_reason(make_node(3, 37, 64, 2, 1, 1, 1),
                  ggml_cann_im2col1d_fallback::SHAPE);

    ggml_tensor * output_shape = make_node(7, 37, 64, 1, 1, 3, 1);
    output_shape->ne[1] += 1;
    make_exact_contiguous(output_shape);
    expect_reason(output_shape, ggml_cann_im2col1d_fallback::OUTPUT_SHAPE);

    ggml_free(ctx);
}

static void test_blockdim_schedule() {
    const scoped_env_restore enable_restore(
        "GGML_CANN_IM2COL1D_BLOCK_DIM_DIAG_ENABLE");
    const scoped_env_restore diag_restore(
        "GGML_CANN_IM2COL1D_BLOCK_DIM_DIAG");
    assert(setenv("GGML_CANN_IM2COL1D_BLOCK_DIM_DIAG_ENABLE", "1", 1) == 0);
    assert(setenv("GGML_CANN_IM2COL1D_BLOCK_DIM_DIAG", "80", 1) == 0);

    ggml_init_params init_params = { 32u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    const auto make_vocoder_node = [&](int64_t t, int64_t c) {
        ggml_tensor * kernel =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, c, c);
        ggml_tensor * input =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t, c, 1);
        return ggml_im2col_vocoder_1d(
            ctx, kernel, input, 1, 1, 1, GGML_TYPE_F32);
    };
    const auto make_causal_node = [&](int64_t t, int64_t c) {
        ggml_tensor * kernel =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, c, c);
        ggml_tensor * input =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t, c, 1);
        return ggml_im2col_causal_1d(
            ctx, kernel, input, 1, 0, 1, GGML_TYPE_F32);
    };
    const auto expect_block_dim = [&](
            const ggml_tensor * node,
            ggml_cann_im2col1d_kind expected_kind,
            uint32_t expected) {
        ggml_cann_im2col1d_params params;
        assert(ggml_cann_im2col1d_validate(node, &params) ==
               ggml_cann_im2col1d_fallback::NONE);
        assert(params.kind == expected_kind);
        if (params.block_dim != expected) {
            std::fprintf(stderr, "block_dim expected=%u actual=%u\n",
                         expected, params.block_dim);
        }
        assert(params.block_dim == expected);
    };

    expect_block_dim(make_vocoder_node(128, 64),
                     ggml_cann_im2col1d_kind::VOCODER, 40);
    expect_block_dim(make_vocoder_node(30, 64),
                     ggml_cann_im2col1d_kind::VOCODER, 30);
    expect_block_dim(make_vocoder_node(2, 1),
                     ggml_cann_im2col1d_kind::VOCODER, 2);
    expect_block_dim(make_causal_node(128, 64),
                     ggml_cann_im2col1d_kind::CAUSAL, 40);
    expect_block_dim(make_causal_node(3, 1),
                     ggml_cann_im2col1d_kind::CAUSAL, 1);

    {
        const scoped_env_restore legacy_blocks(
            "GGML_CANN_IM2COL1D_CAUSAL_BLOCKS");
        assert(setenv("GGML_CANN_IM2COL1D_CAUSAL_BLOCKS", "20", 1) == 0);
        expect_block_dim(make_causal_node(128, 64),
                         ggml_cann_im2col1d_kind::CAUSAL, 20);
    }

    ggml_free(ctx);
}

struct direct_case {
    int64_t t;
    int64_t c;
    int64_t b;
    int64_t k;
    int s;
    int p;
    int d;
};

enum class direct_marker_kind {
    CAUSAL,
    VOCODER,
};

enum class cann_execution_profile {
    EAGER,
    ACL_GRAPH_CAPTURE,
};

struct direct_graph_pair {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * legacy_out = nullptr;
    ggml_tensor * native_out = nullptr;
};

static direct_graph_pair make_direct_graph_pair(
        ggml_backend_t backend,
        const direct_case & tc,
        direct_marker_kind marker_kind) {
    direct_graph_pair pair;
    ggml_init_params params = { 64u * 1024u * 1024u, nullptr, true };
    pair.ctx = ggml_init(params);
    assert(pair.ctx != nullptr);

    ggml_tensor * kernel =
        ggml_new_tensor_3d(pair.ctx, GGML_TYPE_F32, tc.k, tc.c, 1);
    pair.input =
        ggml_new_tensor_3d(pair.ctx, GGML_TYPE_F32, tc.t, tc.c, tc.b);
    pair.legacy_out = ggml_im2col(pair.ctx, kernel, pair.input,
        tc.s, 0, tc.p, 0, tc.d, 0, false, GGML_TYPE_F32);
    pair.native_out = marker_kind == direct_marker_kind::CAUSAL
        ? ggml_im2col_causal_1d(
            pair.ctx, kernel, pair.input, tc.s, tc.p, tc.d, GGML_TYPE_F32)
        : ggml_im2col_vocoder_1d(
            pair.ctx, kernel, pair.input, tc.s, tc.p, tc.d, GGML_TYPE_F32);

    pair.graph = ggml_new_graph(pair.ctx);
    ggml_build_forward_expand(pair.graph, pair.legacy_out);
    ggml_build_forward_expand(pair.graph, pair.native_out);
    pair.buffer = ggml_backend_alloc_ctx_tensors(pair.ctx, backend);
    assert(pair.buffer != nullptr);

    std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
    return pair;
}

static void run_direct_graph_pair(
        ggml_backend_t backend,
        const direct_graph_pair & pair,
        const std::vector<float> & input_values,
        int repeats = 1) {
    assert(input_values.size() ==
           static_cast<size_t>(ggml_nelements(pair.input)));
    ggml_backend_tensor_set(
        pair.input, input_values.data(), 0, ggml_nbytes(pair.input));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        poison_output_pair(pair.legacy_out, pair.native_out);
        assert(ggml_backend_graph_compute(backend, pair.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        assert_words_equal(pair.legacy_out, pair.native_out);
    }
}

static void free_direct_graph_pair(direct_graph_pair & pair) {
    ggml_backend_buffer_free(pair.buffer);
    ggml_free(pair.ctx);
    pair = {};
}

static void run_direct_case(
        ggml_backend_t backend,
        const direct_case & tc,
        const std::vector<float> & input_values,
        int repeats = 1,
        direct_marker_kind marker_kind = direct_marker_kind::CAUSAL) {
    direct_graph_pair pair = make_direct_graph_pair(backend, tc, marker_kind);
    run_direct_graph_pair(backend, pair, input_values, repeats);
    free_direct_graph_pair(pair);
}

static float float_from_word(uint32_t word) {
    float value;
    static_assert(sizeof(value) == sizeof(word), "F32 word size");
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

static size_t tcb_index(const direct_case & tc, int64_t t, int64_t c, int64_t b) {
    return static_cast<size_t>(t + tc.t * c + tc.t * tc.c * b);
}

static size_t ctb_offset(int64_t c_size, int64_t t_size,
                         int64_t c, int64_t t, int64_t b) {
    return static_cast<size_t>(c + c_size * t + c_size * t_size * b);
}

static uint32_t word_from_float(float value) {
    uint32_t word;
    static_assert(sizeof(value) == sizeof(word), "F32 word size");
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

static uint64_t run_case_payloads(
        ggml_backend_t backend,
        const direct_case & tc,
        std::mt19937 & rng,
        int random_repeats = 1,
        direct_marker_kind marker_kind = direct_marker_kind::CAUSAL,
        bool reuse_graph = false,
        bool exhaustive_special_words = false) {
    const size_t n = static_cast<size_t>(tc.t * tc.c * tc.b);
    uint64_t compute_count = 0;
    direct_graph_pair pair;
    if (reuse_graph) {
        pair = make_direct_graph_pair(backend, tc, marker_kind);
    }
    const auto execute = [&](const std::vector<float> & values, int repeats = 1) {
        compute_count += static_cast<uint64_t>(repeats);
        if (reuse_graph) {
            run_direct_graph_pair(backend, pair, values, repeats);
        } else {
            run_direct_case(backend, tc, values, repeats, marker_kind);
        }
    };

    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> values(n);
    for (float & value : values) {
        value = dist(rng);
    }
    execute(values, random_repeats);

    std::fill(values.begin(), values.end(), 0.0f);
    execute(values);

    const int64_t t_edges[] = { 0, tc.t - 1 };
    const int64_t c_edges[] = { 0, tc.c - 1 };
    const int64_t b_edges[] = { 0, tc.b - 1 };
    for (int64_t t : t_edges) {
        for (int64_t c : c_edges) {
            for (int64_t b : b_edges) {
                std::fill(values.begin(), values.end(), 0.0f);
                values[tcb_index(tc, t, c, b)] = 1.0f;
                execute(values);
            }
        }
    }

    for (int64_t b = 0; b < tc.b; ++b) {
        for (int64_t c = 0; c < tc.c; ++c) {
            for (int64_t t = 0; t < tc.t; ++t) {
                values[tcb_index(tc, t, c, b)] =
                    static_cast<float>(100000 * b + 1000 * c + t);
            }
        }
    }
    execute(values);

    const uint32_t payloads[] = {
        UINT32_C(0x00000000), UINT32_C(0x80000000),
        UINT32_C(0x7f800000), UINT32_C(0xff800000),
        UINT32_C(0x7fc12345), UINT32_C(0x7fa54321),
        UINT32_C(0xffc12345), UINT32_C(0xffa54321),
        UINT32_C(0x00000001), UINT32_C(0x80000001),
        UINT32_C(0x007fffff), UINT32_C(0x807fffff),
        UINT32_C(0x00800000), UINT32_C(0x80800000),
        UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff),
    };
    const size_t payload_count = sizeof(payloads) / sizeof(payloads[0]);
    const size_t words_to_cover =
        exhaustive_special_words ? payload_count : std::min(n, payload_count);
    for (size_t start = 0; start < words_to_cover; start += n) {
        std::fill(values.begin(), values.end(), 1.0f);
        const size_t batch_size = std::min(n, words_to_cover - start);
        for (size_t i = 0; i < batch_size; ++i) {
            values[i] = float_from_word(payloads[start + i]);
        }
        execute(values);
    }

    if (reuse_graph) {
        free_direct_graph_pair(pair);
    }
    return compute_count;
}

static cann_execution_profile run_direct_cases(ggml_backend_t backend) {
    ggml_cann_im2col1d_stats_reset();

    const direct_case cases[] = {
        { 5, 1, 1, 1, 1, 0, 1 },
        { 8, 2, 1, 3, 1, 0, 1 },
        { 9, 3, 2, 3, 2, 1, 1 },
        { 12, 4, 3, 5, 1, 2, 2 },
        { 7, 5, 2, 3, 2, 3, 2 },
        { 2, 2, 1, 5, 1, 2, 1 },
        { 8, 511, 2, 3, 1, 0, 1 },
        { 8, 513, 2, 3, 1, 0, 1 },
        { 52, 512, 2, 3, 1, 0, 1 },
    };

    std::mt19937 rng(0x1d);
    bool first_graph = true;

    for (const direct_case & tc : cases) {
        run_case_payloads(backend, tc, rng, first_graph ? 3 : 1);
        first_graph = false;
    }

    const auto stats = ggml_cann_im2col1d_stats_get();
    assert(stats.fallback[static_cast<size_t>(
        ggml_cann_im2col1d_fallback::SRC_DTYPE)] == 0);

    if (stats.marked == 110 && stats.hits == 48 &&
        stats.fallback[static_cast<size_t>(
            ggml_cann_im2col1d_fallback::CONV_PARAMS)] == 62) {
        return cann_execution_profile::EAGER;
    }
    if (stats.marked == 9 && stats.hits == 4 &&
        stats.fallback[static_cast<size_t>(
            ggml_cann_im2col1d_fallback::CONV_PARAMS)] == 5) {
        return cann_execution_profile::ACL_GRAPH_CAPTURE;
    }

    std::fprintf(stderr,
        "unexpected direct execution profile marked=%llu hits=%llu\n",
        static_cast<unsigned long long>(stats.marked),
        static_cast<unsigned long long>(stats.hits));
    std::abort();
}

static void run_vocoder_direct_cases(ggml_backend_t backend) {
    const direct_case cases[] = {
        { 2, 1, 1, 3, 1, 1, 1 },
        { 7, 64, 1, 3, 1, 3, 3 },
        { 17, 64, 1, 3, 1, 5, 5 },
        { 31, 128, 1, 7, 1, 3, 1 },
        { 37, 128, 1, 7, 1, 9, 3 },
        { 41, 256, 1, 7, 1, 15, 5 },
        { 53, 64, 1, 11, 1, 5, 1 },
        { 59, 128, 1, 11, 1, 15, 3 },
        { 67, 256, 1, 11, 1, 25, 5 },
        { 73, 511, 1, 11, 1, 25, 5 },
        { 73, 513, 1, 11, 1, 25, 5 },
    };
    ggml_cann_im2col1d_stats_reset();
    bool execution_profile_inferred = false;
    bool acl_graph_capture = false;
    std::mt19937 rng(0x51d0c0de);
    for (const direct_case & tc : cases) {
        const auto before = ggml_cann_im2col1d_stats_get();
        const uint64_t graph_compute_count = run_case_payloads(
            backend, tc, rng, 1, direct_marker_kind::VOCODER, true, true);
        const auto after = ggml_cann_im2col1d_stats_get();
        const uint64_t expected_graph_computes =
            11 + (16 + static_cast<uint64_t>(tc.t * tc.c * tc.b) - 1) /
                static_cast<uint64_t>(tc.t * tc.c * tc.b);
        assert(graph_compute_count == expected_graph_computes);
        assert(after.vocoder.marked >= before.vocoder.marked);
        assert(after.vocoder.hits >= before.vocoder.hits);
        const uint64_t marked_delta =
            after.vocoder.marked - before.vocoder.marked;
        const uint64_t hits_delta =
            after.vocoder.hits - before.vocoder.hits;
        assert(marked_delta == hits_delta);
        if (!execution_profile_inferred) {
            assert(marked_delta == 1 || marked_delta == graph_compute_count);
            acl_graph_capture = marked_delta == 1;
            execution_profile_inferred = true;
        }
        const uint64_t expected_hits =
            acl_graph_capture ? 1 : graph_compute_count;
        assert(marked_delta == expected_hits);

        assert(after.marked == before.marked + expected_hits);
        assert(after.hits == before.hits + expected_hits);
        assert(after.vocoder.marked == before.vocoder.marked + expected_hits);
        assert(after.vocoder.hits == before.vocoder.hits + expected_hits);
        assert(after.causal.marked == before.causal.marked);
        assert(after.causal.hits == before.causal.hits);
        for (size_t i = 0;
             i < static_cast<size_t>(
                 ggml_cann_im2col1d_fallback::COUNT); ++i) {
            assert(after.fallback[i] == before.fallback[i]);
            assert(after.vocoder.fallback[i] == before.vocoder.fallback[i]);
            assert(after.causal.fallback[i] == before.causal.fallback[i]);
        }
    }
}

static void run_padding_case(ggml_backend_t backend) {
    constexpr int64_t T = 7;
    constexpr int64_t C = 3;
    constexpr int64_t B = 2;
    constexpr int64_t K = 3;
    ggml_init_params params = { 32u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C, 1);
    ggml_tensor * x_ctb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, B);
    ggml_tensor * x_tcb = ggml_cont(ctx, ggml_permute(ctx, x_ctb, 1, 0, 2, 3));
    ggml_tensor * x_pad = ggml_pad_ext(ctx, x_tcb, K - 1, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * old_out = ggml_im2col(ctx, kernel, x_pad,
        1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * fast_out = ggml_im2col_causal_1d(ctx, kernel, x_pad,
        1, 0, 1, GGML_TYPE_F32);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, old_out);
    ggml_build_forward_expand(graph, fast_out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
    std::vector<float> values(ggml_nelements(x_ctb));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                values[ctb_offset(C, T, c, t, b)] =
                    static_cast<float>(10000 * b + 100 * c + t + 1);
            }
        }
    }
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
    ggml_backend_tensor_set(x_ctb, values.data(), 0, ggml_nbytes(x_ctb));
    poison_output_pair(old_out, fast_out);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    assert_words_equal(old_out, fast_out);

    std::vector<uint32_t> words(ggml_nbytes(fast_out) / sizeof(uint32_t));
    ggml_backend_tensor_get(fast_out, words.data(), 0, ggml_nbytes(fast_out));
    assert(words[0] == UINT32_C(0x00000000));
    assert(words[1] == UINT32_C(0x00000000));
    assert(words[2] == word_from_float(values[ctb_offset(C, T, 0, 0, 0)]));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_streaming_concat_cases(ggml_backend_t backend) {
    constexpr int64_t K = 3;
    constexpr int64_t C = 4;
    constexpr int64_t B = 2;
    constexpr int64_t CACHE_T = K - 1;
    const int64_t chunk_lengths[] = { 1, 4, 7 };
    std::vector<float> cache(static_cast<size_t>(C * CACHE_T * B), 0.0f);

    for (size_t chunk_id = 0; chunk_id < sizeof(chunk_lengths) / sizeof(chunk_lengths[0]); ++chunk_id) {
        const int64_t dt = chunk_lengths[chunk_id];
        std::vector<float> current(static_cast<size_t>(C * dt * B));
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < dt; ++t) {
                for (int64_t c = 0; c < C; ++c) {
                    current[ctb_offset(C, dt, c, t, b)] = static_cast<float>(
                        100000 * static_cast<int64_t>(chunk_id + 1) + 10000 * b + 100 * c + t + 1);
                }
            }
        }

        ggml_init_params params = { 32u * 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        assert(ctx != nullptr);
        ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C, 1);
        ggml_tensor * cache_ctb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, CACHE_T, B);
        ggml_tensor * current_ctb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, dt, B);
        ggml_tensor * cache_tcb = ggml_cont(ctx, ggml_permute(ctx, cache_ctb, 1, 0, 2, 3));
        ggml_tensor * current_tcb = ggml_cont(ctx, ggml_permute(ctx, current_ctb, 1, 0, 2, 3));
        ggml_tensor * x_cat_tcb = ggml_concat(ctx, cache_tcb, current_tcb, 0);
        ggml_tensor * old_out = ggml_im2col(ctx, kernel, x_cat_tcb,
            1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor * x_cat_ctb = ggml_concat(ctx, cache_ctb, current_ctb, 1);
        ggml_tensor * ctb_out = ggml_im2col_causal_ctb_1d(
            ctx, kernel, x_cat_ctb, 1, 0, 1, GGML_TYPE_F32);
        ggml_set_output(old_out);
        ggml_set_output(x_cat_ctb);
        ggml_set_output(ctb_out);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, old_out);
        ggml_build_forward_expand(graph, ctb_out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer != nullptr);
        std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
        ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
        ggml_backend_tensor_set(cache_ctb, cache.data(), 0, ggml_nbytes(cache_ctb));
        ggml_backend_tensor_set(current_ctb, current.data(), 0, ggml_nbytes(current_ctb));
        poison_output_pair(old_out, ctb_out);
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        assert_words_equal(old_out, ctb_out);

        std::vector<uint32_t> words(ggml_nbytes(ctb_out) / sizeof(uint32_t));
        ggml_backend_tensor_get(ctb_out, words.data(), 0, ggml_nbytes(ctb_out));
        assert(words[0] == word_from_float(cache[ctb_offset(C, CACHE_T, 0, 0, 0)]));
        assert(words[1] == word_from_float(cache[ctb_offset(C, CACHE_T, 0, 1, 0)]));
        assert(words[2] == word_from_float(current[ctb_offset(C, dt, 0, 0, 0)]));

        std::vector<float> next_cache(cache.size());
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t out_t = 0; out_t < CACHE_T; ++out_t) {
                    const int64_t concat_t = dt + out_t;
                    const float value = concat_t < CACHE_T
                        ? cache[ctb_offset(C, CACHE_T, c, concat_t, b)]
                        : current[ctb_offset(C, dt, c, concat_t - CACHE_T, b)];
                    next_cache[ctb_offset(C, CACHE_T, c, out_t, b)] = value;
                }
            }
        }
        cache.swap(next_cache);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void run_dtype_fallback_case(ggml_backend_t backend, ggml_type src_type) {
    const direct_case tc{ 8, 2, 2, 3, 1, 0, 1 };
    ggml_init_params params = { 16u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.k, tc.c, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, src_type, tc.t, tc.c, tc.b);
    ggml_tensor * old_out = ggml_im2col(ctx, kernel, input,
        tc.s, 0, tc.p, 0, tc.d, 0, false, GGML_TYPE_F32);
    ggml_tensor * marked_out = ggml_im2col_causal_1d(ctx, kernel, input,
        tc.s, tc.p, tc.d, GGML_TYPE_F32);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, old_out);
    ggml_build_forward_expand(graph, marked_out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
    std::vector<float> source_values(ggml_nelements(input));
    for (size_t i = 0; i < source_values.size(); ++i) {
        source_values[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 8.0f;
    }
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
    if (src_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(source_values.size());
        ggml_fp32_to_fp16_row(source_values.data(), converted.data(),
                              static_cast<int64_t>(converted.size()));
        ggml_backend_tensor_set(input, converted.data(), 0, ggml_nbytes(input));
    } else {
        assert(src_type == GGML_TYPE_BF16);
        std::vector<ggml_bf16_t> converted(source_values.size());
        ggml_fp32_to_bf16_row(source_values.data(), converted.data(),
                              static_cast<int64_t>(converted.size()));
        ggml_backend_tensor_set(input, converted.data(), 0, ggml_nbytes(input));
    }

    poison_output_pair(old_out, marked_out);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    assert_words_equal(old_out, marked_out);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_dtype_fallback_cases(ggml_backend_t backend) {
    const auto before = ggml_cann_im2col1d_stats_get();
    run_dtype_fallback_case(backend, GGML_TYPE_F16);
    run_dtype_fallback_case(backend, GGML_TYPE_BF16);
    const auto after = ggml_cann_im2col1d_stats_get();
    const size_t i = static_cast<size_t>(ggml_cann_im2col1d_fallback::SRC_DTYPE);
    assert(after.marked == before.marked + 2);
    assert(after.hits == before.hits);
    assert(after.fallback[i] == before.fallback[i] + 2);
}

static void assert_normal_stats(cann_execution_profile profile) {
    const auto stats = ggml_cann_im2col1d_stats_get();
    const bool acl_graph = profile == cann_execution_profile::ACL_GRAPH_CAPTURE;
    assert(stats.marked == (acl_graph ? 15 : 116));
    assert(stats.hits == (acl_graph ? 8 : 52));

    const size_t src_dtype = static_cast<size_t>(
        ggml_cann_im2col1d_fallback::SRC_DTYPE);
    const size_t conv_params = static_cast<size_t>(
        ggml_cann_im2col1d_fallback::CONV_PARAMS);
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        const uint64_t expected = i == src_dtype ? 2 :
            (i == conv_params ? (acl_graph ? 5 : 62) : 0);
        assert(stats.fallback[i] == expected);
    }
}

static void run_fallback_only_cases(ggml_backend_t backend) {
    ggml_cann_im2col1d_stats_reset();
    const direct_case tc{ 8, 2, 2, 3, 1, 0, 1 };
    std::vector<float> values(static_cast<size_t>(tc.t * tc.c * tc.b));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    run_direct_case(backend, tc, values);
    const auto stats = ggml_cann_im2col1d_stats_get();
    assert(stats.marked == 1);
    assert(stats.hits == 0);
    assert(stats.fallback[static_cast<size_t>(
        ggml_cann_im2col1d_fallback::DISABLED)] == 1);
}

struct scoped_env_override {
    std::string name;
    std::string previous;
    bool had_previous;

    scoped_env_override(const char * env_name, const char * value)
        : name(env_name),
          previous(std::getenv(env_name) != nullptr ? std::getenv(env_name) : ""),
          had_previous(std::getenv(env_name) != nullptr) {
        assert(setenv(name.c_str(), value, 1) == 0);
    }

    ~scoped_env_override() {
        if (had_previous) {
            assert(setenv(name.c_str(), previous.c_str(), 1) == 0);
        } else {
            assert(unsetenv(name.c_str()) == 0);
        }
    }
};

static void assert_kind_stats_equal(
        const ggml_cann_im2col1d_kind_stats_snapshot & expected,
        const ggml_cann_im2col1d_kind_stats_snapshot & actual) {
    assert(actual.marked == expected.marked);
    assert(actual.hits == expected.hits);
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        assert(actual.fallback[i] == expected.fallback[i]);
    }
}

static void assert_vocoder_fallback_delta(
        const ggml_cann_im2col1d_stats_snapshot & before,
        const ggml_cann_im2col1d_stats_snapshot & after,
        ggml_cann_im2col1d_fallback reason) {
    assert(after.marked == before.marked + 1);
    assert(after.hits == before.hits);
    assert(after.vocoder.marked == before.vocoder.marked + 1);
    assert(after.vocoder.hits == before.vocoder.hits);
    assert_kind_stats_equal(before.causal, after.causal);

    assert(after.temp_bytes_removed == before.temp_bytes_removed);
    assert(after.permutes_removed == before.permutes_removed);
    assert(after.d2d_copies_removed == before.d2d_copies_removed);
    assert(after.d2d_bytes_removed == before.d2d_bytes_removed);
    const size_t reason_index = static_cast<size_t>(reason);
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        const uint64_t delta = i == reason_index ? 1 : 0;
        assert(after.fallback[i] == before.fallback[i] + delta);
        assert(after.vocoder.fallback[i] == before.vocoder.fallback[i] + delta);
    }
}

static std::vector<float> make_special_values(const direct_case & tc) {
    std::vector<float> values(static_cast<size_t>(tc.t * tc.c * tc.b));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] =
            static_cast<float>(static_cast<int>(i % 251) - 125) / 32.0f;
    }
    const uint32_t payloads[] = {
        UINT32_C(0x00000000), UINT32_C(0x80000000),
        UINT32_C(0x7f800000), UINT32_C(0xff800000),
        UINT32_C(0x7fc12345), UINT32_C(0x7fa54321),
        UINT32_C(0xffc12345), UINT32_C(0xffa54321),
        UINT32_C(0x00000001), UINT32_C(0x80000001),
        UINT32_C(0x007fffff), UINT32_C(0x807fffff),
        UINT32_C(0x00800000), UINT32_C(0x80800000),
        UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff),
    };
    const size_t payload_count = sizeof(payloads) / sizeof(payloads[0]);
    for (size_t i = 0; i < std::min(values.size(), payload_count); ++i) {
        values[i] = float_from_word(payloads[i]);
    }
    return values;
}

static void run_vocoder_unsupported_fallback_cases(ggml_backend_t backend) {
    const struct {
        direct_case tc;
        ggml_cann_im2col1d_fallback reason;
    } cases[] = {
        { { 17, 64, 2, 3, 1, 1, 1 },
          ggml_cann_im2col1d_fallback::SHAPE },
        { { 17, 64, 1, 5, 1, 2, 1 },
          ggml_cann_im2col1d_fallback::CONV_PARAMS },
        { { 17, 64, 1, 3, 1, 2, 2 },
          ggml_cann_im2col1d_fallback::CONV_PARAMS },
    };

    ggml_cann_im2col1d_stats_reset();
    for (const auto & fallback_case : cases) {
        const auto before = ggml_cann_im2col1d_stats_get();
        run_direct_case(
            backend,
            fallback_case.tc,
            make_special_values(fallback_case.tc),
            1,
            direct_marker_kind::VOCODER);
        const auto after = ggml_cann_im2col1d_stats_get();
        assert_vocoder_fallback_delta(
            before, after, fallback_case.reason);
    }
}

static void run_vocoder_switch_case(ggml_backend_t backend) {
    const scoped_env_override general_switch("GGML_CANN_IM2COL1D", "auto");
    const scoped_env_override vocoder_switch(
        "GGML_CANN_IM2COL1D_VOCODER", "off");
    ggml_cann_im2col1d_stats_reset();

    const direct_case causal_case{ 12, 4, 1, 3, 1, 0, 1 };
    run_direct_case(
        backend, causal_case, make_special_values(causal_case));
    const auto before_vocoder = ggml_cann_im2col1d_stats_get();
    assert(before_vocoder.marked == 1);
    assert(before_vocoder.hits == 1);
    assert(before_vocoder.causal.marked == 1);
    assert(before_vocoder.causal.hits == 1);
    assert(before_vocoder.vocoder.marked == 0);
    assert(before_vocoder.vocoder.hits == 0);

    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        assert(before_vocoder.fallback[i] == 0);
        assert(before_vocoder.causal.fallback[i] == 0);
        assert(before_vocoder.vocoder.fallback[i] == 0);
    }

    const direct_case vocoder_case{ 12, 4, 1, 7, 1, 3, 1 };
    run_direct_case(
        backend,
        vocoder_case,
        make_special_values(vocoder_case),
        1,
        direct_marker_kind::VOCODER);
    const auto after_vocoder = ggml_cann_im2col1d_stats_get();
    assert_vocoder_fallback_delta(
        before_vocoder,
        after_vocoder,
        ggml_cann_im2col1d_fallback::DISABLED);
}

static void run_vocoder_gather_mask_contract_cases(ggml_backend_t backend) {
    const int64_t channels[] = { 1, 63, 64, 255, 256, 257, 511, 513 };
    std::mt19937 rng(UINT32_C(0x67415448));
    ggml_cann_im2col1d_stats_reset();
    for (int64_t c : channels) {
        for (int64_t k : { INT64_C(3), INT64_C(7), INT64_C(11) }) {
            for (int d : { 1, 3, 5 }) {
                const int p = d * static_cast<int>(k - 1) / 2;
                const direct_case tc{ 67, c, 1, k, 1, p, d };
                const auto before = ggml_cann_im2col1d_stats_get();
                const uint64_t compute_count = run_case_payloads(
                    backend,
                    tc,
                    rng,
                    1,
                    direct_marker_kind::VOCODER,
                    false,
                    true);
                const auto after = ggml_cann_im2col1d_stats_get();
                const uint64_t marked_delta =
                    after.vocoder.marked - before.vocoder.marked;
                const uint64_t hits_delta =
                    after.vocoder.hits - before.vocoder.hits;
                assert(marked_delta == hits_delta);
                assert(marked_delta > 0);
                assert(marked_delta <= compute_count);
                assert(after.marked == before.marked + marked_delta);
                assert(after.hits == before.hits + marked_delta);
                assert(after.causal.marked == before.causal.marked);
                assert(after.causal.hits == before.causal.hits);
                for (size_t i = 0;
                     i < static_cast<size_t>(
                         ggml_cann_im2col1d_fallback::COUNT); ++i) {
                    assert(after.fallback[i] == before.fallback[i]);
                    assert(after.vocoder.fallback[i] ==
                           before.vocoder.fallback[i]);
                    assert(after.causal.fallback[i] ==
                           before.causal.fallback[i]);
                }
            }
        }
    }
}

static void run_vocoder_only_cases(ggml_backend_t backend) {
    const scoped_env_override general_switch("GGML_CANN_IM2COL1D", "auto");
    const scoped_env_override vocoder_switch(
        "GGML_CANN_IM2COL1D_VOCODER", "auto");
    run_vocoder_direct_cases(backend);
    run_vocoder_gather_mask_contract_cases(backend);
    run_vocoder_unsupported_fallback_cases(backend);
    run_vocoder_switch_case(backend);
}

struct benchmark_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
};

struct benchmark_result {
    double legacy_node_p50_ms = 0.0;
    double native_node_p50_ms = 0.0;
    double native_kernel_p50_ms = 0.0;
    double speedup = 0.0;
};

static benchmark_graph make_benchmark_graph(ggml_backend_t backend, bool marked) {
    benchmark_graph result;
    ggml_init_params params = { 64u * 1024u * 1024u, nullptr, true };
    result.ctx = ggml_init(params);
    assert(result.ctx != nullptr);

    ggml_tensor * kernel = ggml_new_tensor_3d(result.ctx, GGML_TYPE_F32, 3, 512, 1);
    ggml_tensor * input = ggml_new_tensor_3d(result.ctx, GGML_TYPE_F32, 52, 512, 2);
    result.output = marked
        ? ggml_im2col_causal_1d(result.ctx, kernel, input, 1, 0, 1, GGML_TYPE_F32)
        : ggml_im2col(result.ctx, kernel, input,
            1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    assert(result.output->ne[0] == 1536);
    assert(result.output->ne[1] == 50);
    assert(result.output->ne[2] == 2);
    assert(result.output->ne[3] == 1);

    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.output);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, backend);
    assert(result.buffer != nullptr);

    std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
    std::vector<float> input_values(ggml_nelements(input));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] =
            static_cast<float>(static_cast<int>(i % 251) - 125) / 64.0f;
    }
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
    ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));
    return result;
}

static void free_benchmark_graph(benchmark_graph & graph) {
    ggml_backend_buffer_free(graph.buffer);
    ggml_free(graph.ctx);
    graph = {};
}

static void verify_benchmark_outputs(
        ggml_backend_t backend,
        const benchmark_graph & legacy,
        const benchmark_graph & native) {
    poison_output_pair(legacy.output, native.output);
    assert(ggml_backend_graph_compute(backend, legacy.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    assert(ggml_backend_graph_compute(backend, native.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    assert_words_equal(legacy.output, native.output);
}

static double sample_p50(std::vector<double> samples) {
    assert(!samples.empty());
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static double run_graph_p50(
        ggml_backend_t backend,
        const benchmark_graph & graph,
        size_t warmup,
        size_t runs) {
    for (size_t i = 0; i < warmup; ++i) {
        assert(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
    }

    std::vector<double> samples;
    samples.reserve(runs);
    for (size_t i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        assert(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    return sample_p50(samples);
}

struct benchmark_sample_summary {
    size_t samples = 0;
    double p50_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct benchmark_paired_samples {
    std::vector<double> legacy;
    std::vector<double> native;
};

enum class vocoder_benchmark_execution_profile {
    UNKNOWN = 0,
    EAGER,
    ACL_GRAPH_CAPTURE,
};

static const char * vocoder_benchmark_execution_profile_name(
        vocoder_benchmark_execution_profile profile) {
    switch (profile) {
        case vocoder_benchmark_execution_profile::EAGER:
            return "eager";
        case vocoder_benchmark_execution_profile::ACL_GRAPH_CAPTURE:
            return "acl_graph_capture";
        case vocoder_benchmark_execution_profile::UNKNOWN:
            break;
    }
    assert(false);
    return "unknown";
}

struct vocoder_production_correctness_result {
    uint64_t compute_count = 0;
    uint64_t host_dispatches = 0;
};

struct vocoder_benchmark_shape_result {
    direct_case tc;
    benchmark_sample_summary legacy;
    benchmark_sample_summary native;
    double speedup = 0.0;
    uint64_t native_marked = 0;
    uint64_t native_hits = 0;
};

struct vocoder_benchmark_result {
    std::vector<vocoder_benchmark_shape_result> shapes;
    benchmark_sample_summary legacy;
    benchmark_sample_summary native;
    double speedup = 0.0;
};

struct vocoder_production_shape_result {
    std::string set;
    int stage = 0;
    direct_case tc = {};
    int call_weight = 0;
    uint32_t block_dim = 0;
    benchmark_sample_summary legacy;
    benchmark_sample_summary native;
    double speedup = 0.0;
    uint64_t correctness_computes = 0;
    uint64_t correctness_host_dispatches = 0;
    uint64_t native_marked = 0;
    uint64_t native_hits = 0;
    uint64_t native_fallbacks = 0;
};

struct vocoder_production_weighted_summary {
    size_t weighted_call_count = 0;
    double legacy_total_mean_ms = 0.0;
    double native_total_mean_ms = 0.0;
    double legacy_mean_per_call_ms = 0.0;
    double native_mean_per_call_ms = 0.0;
    double speedup = 0.0;
};

struct vocoder_production_stage_result {
    int stage = 0;
    int64_t t = 0;
    int64_t c = 0;
    int k_weights[3] = {};
    vocoder_production_weighted_summary weighted;
};

struct vocoder_production_set_result {
    std::string set;
    std::vector<vocoder_production_stage_result> stages;
    vocoder_production_weighted_summary weighted;
};

struct vocoder_production_benchmark_result {
    std::vector<vocoder_production_shape_result> shapes;
    std::vector<vocoder_production_set_result> sets;
    vocoder_benchmark_execution_profile execution_profile =
        vocoder_benchmark_execution_profile::UNKNOWN;
};

struct vocoder_production_stage_spec {
    const char * set;
    int stage;
    int64_t t;
    int64_t c;
    int k_weights[3];
};

static benchmark_sample_summary summarize_samples(
        const std::vector<double> & samples) {
    assert(!samples.empty());
    benchmark_sample_summary result;
    result.samples = samples.size();
    result.p50_ms = sample_p50(samples);
    const auto minmax = std::minmax_element(samples.begin(), samples.end());
    result.min_ms = *minmax.first;
    result.max_ms = *minmax.second;
    double sum = 0.0;
    for (double sample : samples) {
        sum += sample;
    }
    result.mean_ms = sum / static_cast<double>(samples.size());
    return result;
}

static void assert_finite_positive(double value) {
    assert(std::isfinite(value));
    assert(value > 0.0);
}

static void assert_sample_summary_finite_positive(
        const benchmark_sample_summary & summary) {
    assert(summary.samples > 0);
    assert_finite_positive(summary.p50_ms);
    assert_finite_positive(summary.mean_ms);
}

static std::vector<double> run_graph_samples(
        ggml_backend_t backend,
        const benchmark_graph & graph,
        size_t warmup,
        size_t runs) {
    for (size_t i = 0; i < warmup; ++i) {
        assert(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
    }

    std::vector<double> samples;
    samples.reserve(runs);
    for (size_t i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        assert(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    return samples;
}

static void run_graph_once(
        ggml_backend_t backend,
        const benchmark_graph & graph) {
    assert(ggml_backend_graph_compute(backend, graph.graph) ==
           GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
}

static double run_graph_sample(
        ggml_backend_t backend,
        const benchmark_graph & graph) {
    const auto start = std::chrono::steady_clock::now();
    run_graph_once(backend, graph);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static benchmark_paired_samples run_graph_abba_samples(
        ggml_backend_t backend,
        const benchmark_graph & legacy,
        const benchmark_graph & native,
        size_t warmup,
        size_t runs) {
    assert(warmup > 0 && warmup % 2 == 0);
    assert(runs > 0 && runs % 2 == 0);

    const size_t warmup_cycles = warmup / 2;
    for (size_t cycle = 0; cycle < warmup_cycles; ++cycle) {
        run_graph_once(backend, legacy);
        run_graph_once(backend, native);
        run_graph_once(backend, native);
        run_graph_once(backend, legacy);
    }

    benchmark_paired_samples result;
    result.legacy.reserve(runs);
    result.native.reserve(runs);
    const size_t timing_cycles = runs / 2;
    for (size_t cycle = 0; cycle < timing_cycles; ++cycle) {
        result.legacy.push_back(run_graph_sample(backend, legacy));
        result.native.push_back(run_graph_sample(backend, native));
        result.native.push_back(run_graph_sample(backend, native));
        result.legacy.push_back(run_graph_sample(backend, legacy));
    }
    assert(result.legacy.size() == runs);
    assert(result.native.size() == runs);
    return result;
}

static benchmark_graph make_vocoder_benchmark_graph(
        ggml_backend_t backend,
        const direct_case & tc,
        bool marked) {
    assert(tc.b == 1);
    assert(tc.s == 1);
    benchmark_graph result;
    ggml_init_params params = { 64u * 1024u * 1024u, nullptr, true };
    result.ctx = ggml_init(params);
    assert(result.ctx != nullptr);

    ggml_tensor * kernel = ggml_new_tensor_3d(
        result.ctx, GGML_TYPE_F32, tc.k, tc.c, 1);
    ggml_tensor * input = ggml_new_tensor_3d(
        result.ctx, GGML_TYPE_F32, tc.t, tc.c, tc.b);
    result.output = marked
        ? ggml_im2col_vocoder_1d(
            result.ctx, kernel, input, tc.s, tc.p, tc.d, GGML_TYPE_F32)
        : ggml_im2col(result.ctx, kernel, input,
            tc.s, 0, tc.p, 0, tc.d, 0, false, GGML_TYPE_F32);
    assert(result.output->ne[0] == tc.k * tc.c);
    assert(result.output->ne[1] == tc.t);
    assert(result.output->ne[2] == tc.b);
    assert(result.output->ne[3] == 1);

    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.output);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, backend);
    assert(result.buffer != nullptr);

    std::vector<float> kernel_values(ggml_nelements(kernel), 0.0f);
    const std::vector<float> input_values = make_special_values(tc);
    ggml_backend_tensor_set(kernel, kernel_values.data(), 0, ggml_nbytes(kernel));
    ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));
    return result;
}

static void assert_vocoder_benchmark_stats(
        const ggml_cann_im2col1d_stats_snapshot & stats,
        uint64_t expected_hits) {
    assert(stats.marked == expected_hits);
    assert(stats.hits == expected_hits);
    assert(stats.vocoder.marked == expected_hits);
    assert(stats.vocoder.hits == expected_hits);
    assert(stats.causal.marked == 0);
    assert(stats.causal.hits == 0);
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        assert(stats.fallback[i] == 0);
        assert(stats.vocoder.fallback[i] == 0);
        assert(stats.causal.fallback[i] == 0);
    }
}

static uint64_t assert_vocoder_production_stats(
        const ggml_cann_im2col1d_stats_snapshot & stats,
        uint64_t compute_count,
        vocoder_benchmark_execution_profile & profile) {
    assert(stats.marked == stats.hits);
    assert(stats.vocoder.marked == stats.vocoder.hits);
    assert(stats.marked == stats.vocoder.marked);
    assert(stats.causal.marked == 0);
    assert(stats.causal.hits == 0);
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        assert(stats.fallback[i] == 0);
        assert(stats.vocoder.fallback[i] == 0);
        assert(stats.causal.fallback[i] == 0);
    }

    const uint64_t host_dispatches = stats.vocoder.marked;
    if (profile == vocoder_benchmark_execution_profile::UNKNOWN) {
        assert(host_dispatches == 1 || host_dispatches == compute_count);
        profile = host_dispatches == 1
            ? vocoder_benchmark_execution_profile::ACL_GRAPH_CAPTURE
            : vocoder_benchmark_execution_profile::EAGER;
    }
    const uint64_t expected =
        profile == vocoder_benchmark_execution_profile::ACL_GRAPH_CAPTURE
            ? 1
            : compute_count;
    assert(host_dispatches == expected);
    return host_dispatches;
}

static vocoder_benchmark_result run_vocoder_benchmark(ggml_backend_t backend) {
    constexpr size_t warmup = 10;
    constexpr size_t runs = 100;
    const direct_case cases[] = {
        { 53,  64, 1,  3, 1,  1, 1 },
        { 53,  64, 1,  3, 1,  3, 3 },
        { 53,  64, 1,  3, 1,  5, 5 },
        { 97, 128, 1,  7, 1,  3, 1 },
        { 97, 128, 1,  7, 1,  9, 3 },
        { 97, 128, 1,  7, 1, 15, 5 },
        { 193, 256, 1, 11, 1,  5, 1 },
        { 193, 256, 1, 11, 1, 15, 3 },
        { 193, 256, 1, 11, 1, 25, 5 },
    };

    const scoped_env_override general_switch("GGML_CANN_IM2COL1D", "auto");
    const scoped_env_override vocoder_switch(
        "GGML_CANN_IM2COL1D_VOCODER", "auto");
    unsetenv("GGML_CANN_IM2COL1D_BENCH");

    vocoder_benchmark_result result;
    std::vector<double> all_legacy_samples;
    std::vector<double> all_native_samples;
    all_legacy_samples.reserve(runs * 9);
    all_native_samples.reserve(runs * 9);

    for (const direct_case & tc : cases) {
        assert(tc.p == tc.d * (tc.k - 1) / 2);
        benchmark_graph legacy =
            make_vocoder_benchmark_graph(backend, tc, false);
        benchmark_graph native =
            make_vocoder_benchmark_graph(backend, tc, true);

        verify_benchmark_outputs(backend, legacy, native);
        ggml_cann_im2col1d_stats_reset();

        const std::vector<double> legacy_samples =
            run_graph_samples(backend, legacy, warmup, runs);
        assert_vocoder_benchmark_stats(
            ggml_cann_im2col1d_stats_get(), 0);
        const std::vector<double> native_samples =
            run_graph_samples(backend, native, warmup, runs);
        const auto stats = ggml_cann_im2col1d_stats_get();
        assert_vocoder_benchmark_stats(stats, warmup + runs);

        vocoder_benchmark_shape_result shape_result;
        shape_result.tc = tc;
        shape_result.legacy = summarize_samples(legacy_samples);
        shape_result.native = summarize_samples(native_samples);
        shape_result.speedup =
            shape_result.legacy.p50_ms / shape_result.native.p50_ms;
        shape_result.native_marked = stats.vocoder.marked;
        shape_result.native_hits = stats.vocoder.hits;
        result.shapes.push_back(shape_result);
        all_legacy_samples.insert(
            all_legacy_samples.end(), legacy_samples.begin(), legacy_samples.end());
        all_native_samples.insert(
            all_native_samples.end(), native_samples.begin(), native_samples.end());

        free_benchmark_graph(native);
        free_benchmark_graph(legacy);
    }

    assert(result.shapes.size() == 9);
    result.legacy = summarize_samples(all_legacy_samples);
    result.native = summarize_samples(all_native_samples);
    result.speedup = result.legacy.p50_ms / result.native.p50_ms;
    return result;
}

static uint64_t count_vocoder_fallbacks(
        const ggml_cann_im2col1d_stats_snapshot & stats) {
    uint64_t count = 0;
    for (size_t i = 0;
         i < static_cast<size_t>(ggml_cann_im2col1d_fallback::COUNT); ++i) {
        count += stats.vocoder.fallback[i];
    }
    return count;
}

static void add_weighted_shape(
        vocoder_production_weighted_summary & summary,
        const vocoder_production_shape_result & shape) {
    assert(shape.call_weight > 0);
    summary.weighted_call_count += static_cast<size_t>(shape.call_weight);
    summary.legacy_total_mean_ms +=
        shape.legacy.mean_ms * static_cast<double>(shape.call_weight);
    summary.native_total_mean_ms +=
        shape.native.mean_ms * static_cast<double>(shape.call_weight);
}

static void finalize_weighted_summary(
        vocoder_production_weighted_summary & summary) {
    assert(summary.weighted_call_count > 0);
    assert_finite_positive(summary.legacy_total_mean_ms);
    assert_finite_positive(summary.native_total_mean_ms);
    const double count = static_cast<double>(summary.weighted_call_count);
    summary.legacy_mean_per_call_ms =
        summary.legacy_total_mean_ms / count;
    summary.native_mean_per_call_ms =
        summary.native_total_mean_ms / count;
    summary.speedup =
        summary.legacy_total_mean_ms / summary.native_total_mean_ms;
    assert_finite_positive(summary.legacy_mean_per_call_ms);
    assert_finite_positive(summary.native_mean_per_call_ms);
    assert_finite_positive(summary.speedup);
}

static vocoder_production_correctness_result
run_vocoder_production_correctness(
        ggml_backend_t backend,
        const direct_case & tc,
        uint32_t seed,
        vocoder_benchmark_execution_profile & profile) {
    ggml_cann_im2col1d_stats_reset();
    std::mt19937 rng(seed);
    const uint64_t compute_count = run_case_payloads(
        backend,
        tc,
        rng,
        1,
        direct_marker_kind::VOCODER,
        true,
        true);
    const auto stats = ggml_cann_im2col1d_stats_get();
    const uint64_t host_dispatches = assert_vocoder_production_stats(
        stats, compute_count, profile);
    vocoder_production_correctness_result result;
    result.compute_count = compute_count;
    result.host_dispatches = host_dispatches;
    return result;
}

static vocoder_production_benchmark_result
run_vocoder_production_benchmark(ggml_backend_t backend) {
    constexpr size_t warmup = 10;
    constexpr size_t runs = 100;
    const vocoder_production_stage_spec stages[] = {
        { "nominal", 1,  400, 256, { 6, 12,  6 } },
        { "nominal", 2, 2000, 128, { 6, 12,  6 } },
        { "nominal", 3, 6001,  64, { 6,  6, 12 } },
        { "upper",   1,  464, 256, { 6, 12,  6 } },
        { "upper",   2, 2320, 128, { 6, 12,  6 } },
        { "upper",   3, 6961,  64, { 6,  6, 12 } },
    };
    const int64_t kernel_sizes[] = { 3, 7, 11 };
    const int dilations[] = { 1, 3, 5 };
    const int dilation_weights[] = { 4, 1, 1 };

    const scoped_env_override general_switch("GGML_CANN_IM2COL1D", "auto");
    const scoped_env_override vocoder_switch(
        "GGML_CANN_IM2COL1D_VOCODER", "auto");
    const char * bench_value = std::getenv("GGML_CANN_IM2COL1D_BENCH");
    const bool had_bench_value = bench_value != nullptr;
    const std::string previous_bench_value =
        had_bench_value ? bench_value : "";
    assert(unsetenv("GGML_CANN_IM2COL1D_BENCH") == 0);

    vocoder_production_benchmark_result result;
    result.shapes.reserve(54);
    result.sets.reserve(2);

    for (const vocoder_production_stage_spec & stage : stages) {
        if (result.sets.empty() || result.sets.back().set != stage.set) {
            vocoder_production_set_result set_result;
            set_result.set = stage.set;
            result.sets.push_back(set_result);
        }
        vocoder_production_set_result & set_result = result.sets.back();

        vocoder_production_stage_result stage_result;
        stage_result.stage = stage.stage;
        stage_result.t = stage.t;
        stage_result.c = stage.c;
        for (size_t k_index = 0; k_index < 3; ++k_index) {
            stage_result.k_weights[k_index] = stage.k_weights[k_index];
            assert(stage.k_weights[k_index] % 6 == 0);
            for (size_t d_index = 0; d_index < 3; ++d_index) {
                const int64_t k = kernel_sizes[k_index];
                const int d = dilations[d_index];
                const int p = d * static_cast<int>(k - 1) / 2;
                const direct_case tc{
                    stage.t, stage.c, 1, k, 1, p, d
                };

                vocoder_production_shape_result shape;
                shape.set = stage.set;
                shape.stage = stage.stage;
                shape.tc = tc;
                shape.call_weight =
                    (stage.k_weights[k_index] / 6) *
                    dilation_weights[d_index];
                const vocoder_production_correctness_result correctness =
                    run_vocoder_production_correctness(
                        backend,
                        tc,
                        UINT32_C(0x51d00000) ^
                            static_cast<uint32_t>(
                                stage.t + 31 * stage.c +
                                131 * k + 17 * d),
                        result.execution_profile);
                shape.correctness_computes = correctness.compute_count;
                shape.correctness_host_dispatches =
                    correctness.host_dispatches;

                benchmark_graph legacy =
                    make_vocoder_benchmark_graph(backend, tc, false);
                benchmark_graph native =
                    make_vocoder_benchmark_graph(backend, tc, true);
                ggml_cann_im2col1d_params native_params;
                assert(ggml_cann_im2col1d_validate(
                           native.output, &native_params) ==
                       ggml_cann_im2col1d_fallback::NONE);
                shape.block_dim = native_params.block_dim;

                ggml_cann_im2col1d_stats_reset();
                const benchmark_paired_samples samples =
                    run_graph_abba_samples(
                        backend, legacy, native, warmup, runs);
                const auto stats = ggml_cann_im2col1d_stats_get();
                const uint64_t timing_host_dispatches =
                    assert_vocoder_production_stats(
                        stats, warmup + runs, result.execution_profile);

                shape.legacy = summarize_samples(samples.legacy);
                shape.native = summarize_samples(samples.native);
                assert(shape.legacy.samples == runs);
                assert(shape.native.samples == runs);
                assert_sample_summary_finite_positive(shape.legacy);
                assert_sample_summary_finite_positive(shape.native);
                shape.speedup =
                    shape.legacy.mean_ms / shape.native.mean_ms;
                assert_finite_positive(shape.speedup);
                shape.native_marked = timing_host_dispatches;
                shape.native_hits = stats.vocoder.hits;
                shape.native_fallbacks =
                    count_vocoder_fallbacks(stats);
                assert(shape.native_fallbacks == 0);

                add_weighted_shape(stage_result.weighted, shape);
                add_weighted_shape(set_result.weighted, shape);
                result.shapes.push_back(shape);

                free_benchmark_graph(native);
                free_benchmark_graph(legacy);
            }
        }
        finalize_weighted_summary(stage_result.weighted);
        assert(stage_result.weighted.weighted_call_count == 24);
        set_result.stages.push_back(stage_result);
    }

    assert(result.shapes.size() == 54);
    assert(result.sets.size() == 2);
    for (vocoder_production_set_result & set_result : result.sets) {
        assert(set_result.stages.size() == 3);
        finalize_weighted_summary(set_result.weighted);
        assert(set_result.weighted.weighted_call_count == 72);
    }
    assert(result.execution_profile !=
           vocoder_benchmark_execution_profile::UNKNOWN);
    if (had_bench_value) {
        assert(setenv("GGML_CANN_IM2COL1D_BENCH",
                      previous_bench_value.c_str(), 1) == 0);
    } else {
        assert(unsetenv("GGML_CANN_IM2COL1D_BENCH") == 0);
    }
    return result;
}

static benchmark_result run_benchmark(ggml_backend_t backend) {
    constexpr size_t warmup = 10;
    constexpr size_t runs = 100;
    benchmark_result result;
    unsetenv("GGML_CANN_IM2COL1D_BENCH");
    ggml_cann_im2col1d_bench_reset();
    assert(ggml_cann_im2col1d_bench_p50(0) == 0.0);

    benchmark_graph legacy = make_benchmark_graph(backend, false);
    benchmark_graph native = make_benchmark_graph(backend, true);
    verify_benchmark_outputs(backend, legacy, native);

    result.legacy_node_p50_ms =
        run_graph_p50(backend, legacy, warmup, runs);
    result.native_node_p50_ms =
        run_graph_p50(backend, native, warmup, runs);

    setenv("GGML_CANN_IM2COL1D_BENCH", "1", 1);
    ggml_cann_im2col1d_bench_reset();
    for (size_t i = 0; i < warmup + runs; ++i) {
        assert(ggml_backend_graph_compute(backend, native.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
    }
    result.native_kernel_p50_ms =
        ggml_cann_im2col1d_bench_p50(warmup);
    assert(result.native_kernel_p50_ms > 0.0);
    unsetenv("GGML_CANN_IM2COL1D_BENCH");
    result.speedup = result.legacy_node_p50_ms / result.native_node_p50_ms;

    free_benchmark_graph(native);
    free_benchmark_graph(legacy);
    return result;
}

static void print_benchmark_result(const benchmark_result & result) {
    std::printf(
        "{\"shape\":{\"T\":52,\"C\":512,\"B\":2,\"K\":3,\"OW\":50},"
        "\"warmup\":10,\"runs\":100,"
        "\"legacy_node_p50_ms\":%.9f,\"native_node_p50_ms\":%.9f,"
        "\"native_kernel_p50_ms\":%.9f,\"speedup\":%.9f,"
        "\"temp_bytes_removed_per_node\":3686400,"
        "\"permutes_removed_per_node\":1,"
        "\"d2d_copies_removed_per_node\":100,"
        "\"d2d_bytes_removed_per_node\":614400}\n",
        result.legacy_node_p50_ms,
        result.native_node_p50_ms,
        result.native_kernel_p50_ms,
        result.speedup);
}

static void print_sample_summary(const benchmark_sample_summary & summary) {
    std::printf(
        "{\"samples\":%zu,\"p50_ms\":%.9f,\"mean_ms\":%.9f,"
        "\"min_ms\":%.9f,\"max_ms\":%.9f}",
        summary.samples,
        summary.p50_ms,
        summary.mean_ms,
        summary.min_ms,
        summary.max_ms);
}

static void print_vocoder_benchmark_result(
        const vocoder_benchmark_result & result) {
    std::printf(
        "{\"mode\":\"vocoder\",\"warmup\":10,\"runs\":100,"
        "\"correctness\":\"bitwise\",\"shapes\":[");
    for (size_t i = 0; i < result.shapes.size(); ++i) {
        const vocoder_benchmark_shape_result & shape = result.shapes[i];
        std::printf(
            "%s{\"t\":%lld,\"c\":%lld,\"k\":%lld,"
            "\"dilation\":%d,\"padding\":%d,\"legacy\":",
            i == 0 ? "" : ",",
            static_cast<long long>(shape.tc.t),
            static_cast<long long>(shape.tc.c),
            static_cast<long long>(shape.tc.k),
            shape.tc.d,
            shape.tc.p);
        print_sample_summary(shape.legacy);
        std::printf(",\"native\":");
        print_sample_summary(shape.native);
        std::printf(
            ",\"speedup\":%.9f,\"bitwise_equal\":true,"
            "\"native_marked\":%llu,\"native_hits\":%llu,"
            "\"native_fallbacks\":0}",
            shape.speedup,
            static_cast<unsigned long long>(shape.native_marked),
            static_cast<unsigned long long>(shape.native_hits));
    }
    std::printf("],\"aggregate\":{\"legacy\":");
    print_sample_summary(result.legacy);
    std::printf(",\"native\":");
    print_sample_summary(result.native);
    std::printf(
        ",\"speedup\":%.9f},\"native_fallbacks\":0}\n",
        result.speedup);
}

static void print_vocoder_production_weighted_summary(
        const vocoder_production_weighted_summary & summary) {
    std::printf(
        "{\"weighted_call_count\":%zu,"
        "\"weighted_legacy_total_mean_ms\":%.9f,"
        "\"weighted_native_total_mean_ms\":%.9f,"
        "\"weighted_legacy_mean_per_call_ms\":%.9f,"
        "\"weighted_native_mean_per_call_ms\":%.9f,"
        "\"weighted_speedup\":%.9f}",
        summary.weighted_call_count,
        summary.legacy_total_mean_ms,
        summary.native_total_mean_ms,
        summary.legacy_mean_per_call_ms,
        summary.native_mean_per_call_ms,
        summary.speedup);
}

static void print_vocoder_production_benchmark_result(
        const vocoder_production_benchmark_result & result) {
    std::printf(
        "{\"mode\":\"vocoder-production\","
        "\"execution_profile\":\"%s\","
        "\"warmup\":10,\"runs\":100,"
        "\"timing_order\":\"ABBA\",\"timing_cycles\":50,"
        "\"correctness\":\"bitwise_full_output\","
        "\"speedup_basis\":\"legacy_mean_ms/native_mean_ms\","
        "\"weighted_latency_basis\":"
        "\"sum(shape_mean_ms*call_weight)\","
        "\"shape_count\":%zu,\"shapes\":[",
        vocoder_benchmark_execution_profile_name(result.execution_profile),
        result.shapes.size());
    for (size_t i = 0; i < result.shapes.size(); ++i) {
        const vocoder_production_shape_result & shape = result.shapes[i];
        std::printf(
            "%s{\"mode\":\"vocoder-production\","
            "\"set\":\"%s\",\"stage\":%d,"
            "\"T\":%lld,\"C\":%lld,\"K\":%lld,"
            "\"d\":%d,\"padding\":%d,\"B\":1,\"stride\":1,"
            "\"call_weight\":%d,\"block_dim\":%u,"
            "\"block_dim_params\":%u,\"block_dim_effective\":%u,"
            "\"legacy\":",
            i == 0 ? "" : ",",
            shape.set.c_str(),
            shape.stage,
            static_cast<long long>(shape.tc.t),
            static_cast<long long>(shape.tc.c),
            static_cast<long long>(shape.tc.k),
            shape.tc.d,
            shape.tc.p,
            shape.call_weight,
            shape.block_dim,
            shape.block_dim,
            shape.block_dim);
        print_sample_summary(shape.legacy);
        std::printf(",\"native\":");
        print_sample_summary(shape.native);
        std::printf(
            ",\"speedup\":%.9f,\"bitwise\":true,"
            "\"bitwise_equal\":true,\"correctness_computes\":%llu,"
            "\"correctness_host_dispatches\":%llu,"
            "\"native_marked\":%llu,\"hits\":%llu,"
            "\"fallback\":%llu}",
            shape.speedup,
            static_cast<unsigned long long>(shape.correctness_computes),
            static_cast<unsigned long long>(
                shape.correctness_host_dispatches),
            static_cast<unsigned long long>(shape.native_marked),
            static_cast<unsigned long long>(shape.native_hits),
            static_cast<unsigned long long>(shape.native_fallbacks));
    }
    std::printf("],\"summaries\":{");
    for (size_t set_index = 0; set_index < result.sets.size(); ++set_index) {
        const vocoder_production_set_result & set_result =
            result.sets[set_index];
        std::printf(
            "%s\"%s\":{\"weighted\":",
            set_index == 0 ? "" : ",",
            set_result.set.c_str());
        print_vocoder_production_weighted_summary(set_result.weighted);
        std::printf(",\"stages\":[");
        for (size_t stage_index = 0;
             stage_index < set_result.stages.size(); ++stage_index) {
            const vocoder_production_stage_result & stage =
                set_result.stages[stage_index];
            std::printf(
                "%s{\"stage\":%d,\"T\":%lld,\"C\":%lld,"
                "\"K_weights\":{\"3\":%d,\"7\":%d,\"11\":%d},"
                "\"weighted\":",
                stage_index == 0 ? "" : ",",
                stage.stage,
                static_cast<long long>(stage.t),
                static_cast<long long>(stage.c),
                stage.k_weights[0],
                stage.k_weights[1],
                stage.k_weights[2]);
            print_vocoder_production_weighted_summary(stage.weighted);
            std::printf("}");
        }
        std::printf("]}");
    }
    std::printf("}}\n");
}

static int redirect_stdout_to_stderr() {
    assert(std::fflush(stdout) == 0);
    const int saved_stdout = dup(STDOUT_FILENO);
    assert(saved_stdout >= 0);
    assert(dup2(STDERR_FILENO, STDOUT_FILENO) >= 0);
    return saved_stdout;
}

static void restore_stdout(int saved_stdout) {
    assert(std::fflush(stdout) == 0);
    assert(dup2(saved_stdout, STDOUT_FILENO) >= 0);
    assert(close(saved_stdout) == 0);
}

int main(int argc, char ** argv) {
    test_vocoder_gather_mask_patterns();
    test_marker_helper();
    test_split_stats_api();
    if (argc == 2 && std::string(argv[1]) == "--marker-only") {
        return 0;
    }
    test_validation_matrix();
    test_vocoder_validation_matrix();
    test_blockdim_schedule();
    if (argc == 2 && std::string(argv[1]) == "--validation-only") {
        return 0;
    }
    const bool benchmark_requested =
        argc == 2 && std::string(argv[1]) == "--benchmark";
    const bool benchmark_vocoder_requested =
        argc == 2 && std::string(argv[1]) == "--benchmark-vocoder";
    const bool benchmark_vocoder_production_requested =
        argc == 2 &&
        std::string(argv[1]) == "--benchmark-vocoder-production";
    const int saved_stdout =
        (benchmark_requested || benchmark_vocoder_requested ||
         benchmark_vocoder_production_requested)
            ? redirect_stdout_to_stderr()
            : -1;
    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    if (argc == 2 && std::string(argv[1]) == "--ctb-only") {
        run_streaming_concat_cases(backend);
        ggml_backend_free(backend);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--vocoder-only") {
        run_vocoder_only_cases(backend);
        ggml_backend_free(backend);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--vocoder-switch-only") {
        run_vocoder_switch_case(backend);
        ggml_backend_free(backend);
        return 0;
    }
    if (benchmark_requested) {
        const benchmark_result result = run_benchmark(backend);
        ggml_backend_free(backend);
        std::cout.flush();
        restore_stdout(saved_stdout);
        print_benchmark_result(result);
        const bool performance_pass =
            result.native_node_p50_ms < result.legacy_node_p50_ms;
        if (!performance_pass) {
            std::fprintf(stderr,
                "CANN im2col1d performance gate failed: native %.9f ms, legacy %.9f ms\n",
                result.native_node_p50_ms,
                result.legacy_node_p50_ms);
        }
        return performance_pass ? 0 : 2;
    }
    if (benchmark_vocoder_requested) {
        const vocoder_benchmark_result result =
            run_vocoder_benchmark(backend);
        ggml_backend_free(backend);
        std::cout.flush();
        restore_stdout(saved_stdout);
        print_vocoder_benchmark_result(result);
        return 0;
    }
    if (benchmark_vocoder_production_requested) {
        const vocoder_production_benchmark_result result =
            run_vocoder_production_benchmark(backend);
        ggml_backend_free(backend);
        std::cout.flush();
        restore_stdout(saved_stdout);
        print_vocoder_production_benchmark_result(result);
        return 0;
    }

    if (argc == 2 && std::string(argv[1]) == "--fallback-only") {
        run_fallback_only_cases(backend);
    } else {
        const cann_execution_profile profile = run_direct_cases(backend);
        run_padding_case(backend);
        run_streaming_concat_cases(backend);
        run_dtype_fallback_cases(backend);
        assert_normal_stats(profile);
    }
    ggml_backend_free(backend);
    return 0;
}
