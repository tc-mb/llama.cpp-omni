#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "graph-transaction.h"
#include "q8-w8a8.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static void test_graph_device_gate_allows_concurrent_replay() {
    ggml_cann_graph_device_gate device_gate;
    ggml_cann_graph_context_gate first_context;
    ggml_cann_graph_context_gate second_context;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    int entered = 0;
    bool release = false;

    auto replay = [&](ggml_cann_graph_context_gate & context_gate) {
        ggml_cann_graph_transaction transaction(context_gate, device_gate);
        std::unique_lock<std::mutex> state_lock(state_mutex);
        ++entered;
        state_cv.notify_all();
        state_cv.wait(state_lock, [&]() { return release; });
    };

    std::thread first(replay, std::ref(first_context));
    std::thread second(replay, std::ref(second_context));
    bool overlapped = false;
    {
        std::unique_lock<std::mutex> state_lock(state_mutex);
        overlapped = state_cv.wait_for(
            state_lock, std::chrono::seconds(2), [&]() { return entered == 2; });
        release = true;
        state_cv.notify_all();
    }
    first.join();
    second.join();
    assert(overlapped);
}

static void test_graph_context_gate_serializes_transactions() {
    ggml_cann_graph_device_gate device_gate;
    ggml_cann_graph_context_gate context_gate;
    auto first_transaction = std::make_unique<ggml_cann_graph_transaction>(context_gate, device_gate);
    std::promise<void> started;
    std::future<void> started_future = started.get_future();
    std::promise<void> acquired;
    std::future<void> acquired_future = acquired.get_future();
    std::thread second([&]() {
        started.set_value();
        ggml_cann_graph_transaction second_transaction(context_gate, device_gate);
        acquired.set_value();
    });

    assert(started_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(acquired_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
    first_transaction.reset();
    assert(acquired_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    second.join();
}

static void test_graph_device_gate_excludes_capture_and_replay() {
    ggml_cann_graph_device_gate device_gate;
    ggml_cann_graph_context_gate context_gate;
    auto replay_transaction = std::make_unique<ggml_cann_graph_transaction>(context_gate, device_gate);
    std::promise<void> exclusive_started;
    std::future<void> exclusive_started_future = exclusive_started.get_future();
    std::promise<void> exclusive_acquired;
    std::future<void> exclusive_future = exclusive_acquired.get_future();
    std::thread capture([&]() {
        exclusive_started.set_value();
        auto exclusive_guard = device_gate.lock_exclusive();
        exclusive_acquired.set_value();
    });

    assert(exclusive_started_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(exclusive_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
    replay_transaction.reset();
    assert(exclusive_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    capture.join();

    auto exclusive_guard = device_gate.lock_exclusive();
    std::promise<void> replay_started;
    std::future<void> replay_started_future = replay_started.get_future();
    std::promise<void> replay_acquired;
    std::future<void> replay_future = replay_acquired.get_future();
    std::thread replay([&]() {
        replay_started.set_value();
        ggml_cann_graph_transaction next_replay_transaction(context_gate, device_gate);
        replay_acquired.set_value();
    });
    assert(replay_started_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(replay_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
    exclusive_guard.unlock();
    assert(replay_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    replay.join();
}

static std::vector<block_q8_0> make_source(int64_t k, int64_t n) {
    std::vector<block_q8_0> blocks(static_cast<size_t>(k / QK8_0 * n));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t block = 0; block < k / QK8_0; ++block) {
            block_q8_0 & b = blocks[static_cast<size_t>(row * (k / QK8_0) + block)];
            b.d = GGML_FP32_TO_FP16(row == 0 ? 0.25f : 0.5f);
            for (int i = 0; i < QK8_0; ++i) {
                b.qs[i] = static_cast<int8_t>(((i + block * 7 + row * 3) % 31) - 15);
            }
        }
    }
    return blocks;
}

static std::vector<ggml_fp16_t> make_deterministic_input(int64_t k, int64_t m) {
    std::vector<ggml_fp16_t> input_data(static_cast<size_t>(k * m));
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            const float value = static_cast<float>(((row * 11 + col * 7) % 23) - 11) / 128.0f;
            input_data[static_cast<size_t>(row * k + col)] = GGML_FP32_TO_FP16(value);
        }
    }
    return input_data;
}

static void test_invalid_arguments() {
    uint8_t dst[256] = {};
    auto src = make_source(64, 2);
    assert(ggml_cann_requantize_q8_0_per_channel(src.data(), 33, 2, dst, sizeof(dst)).status ==
           ggml_cann_q8_w8a8_status::invalid_shape);
    assert(ggml_cann_requantize_q8_0_per_channel(src.data(), 64, 0, dst, sizeof(dst)).status ==
           ggml_cann_q8_w8a8_status::invalid_shape);
    assert(ggml_cann_requantize_q8_0_per_channel(src.data(), 64, 2, dst, 8).status ==
           ggml_cann_q8_w8a8_status::insufficient_capacity);
}

static void assert_workspace_layout_is_empty(const ggml_cann_q8_w8a8_workspace_plan & plan) {
    assert(plan.input_f16_offset == 0);
    assert(plan.input_f16_bytes == 0);
    assert(plan.quant_offset == 0);
    assert(plan.quant_bytes == 0);
    assert(plan.token_scale_offset == 0);
    assert(plan.token_scale_bytes == 0);
    assert(plan.output_f16_offset == 0);
    assert(plan.output_f16_bytes == 0);
    assert(plan.total_bytes == 0);
}

static void test_workspace_plan_f16() {
    const auto plan = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F16, 128, 96, 17, 2, 3);
    assert(plan.status == ggml_cann_q8_w8a8_workspace_status::ok);
    assert(plan.input_f16_bytes == 0);
    assert(plan.output_f16_bytes == 0);
    assert(plan.quant_offset == 0);
    assert(plan.quant_bytes == 13056);
    assert(plan.token_scale_offset == 13056);
    assert(plan.token_scale_bytes == 408);
    assert(plan.total_bytes == 13464);
}

static void test_workspace_plan_casts_and_match() {
    const auto plan = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 17, 1, 1);
    assert(plan.status == ggml_cann_q8_w8a8_workspace_status::ok);
    assert(plan.input_f16_offset == 0 && plan.input_f16_bytes == 4352);
    assert(plan.quant_offset == 4352 && plan.quant_bytes == 2176);
    assert(plan.token_scale_offset == 6528 && plan.token_scale_bytes == 68);
    assert(plan.output_f16_offset == 6656 && plan.output_f16_bytes == 3264);
    assert(plan.total_bytes == 9920);
    assert(ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 17, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F16, GGML_TYPE_F32, 128, 96, 17, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F16, 128, 96, 17, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 127, 96, 17, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 95, 17, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 18, 1, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 17, 2, 1));
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 17, 1, 2));
    auto failed_plan = plan;
    failed_plan.status = ggml_cann_q8_w8a8_workspace_status::overflow;
    assert(!ggml_cann_q8_w8a8_workspace_matches(
        failed_plan, GGML_TYPE_F32, GGML_TYPE_F32, 128, 96, 17, 1, 1));
}

static void test_workspace_plan_rejects_invalid_and_overflow() {
    const auto invalid = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F16, 0, 96, 1, 1, 1);
    assert(invalid.status == ggml_cann_q8_w8a8_workspace_status::invalid_shape);
    assert_workspace_layout_is_empty(invalid);

    const auto batch_overflow = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F16,
        1, 1, 1, std::numeric_limits<int64_t>::max(), 2);
    assert(batch_overflow.status == ggml_cann_q8_w8a8_workspace_status::overflow);
    assert_workspace_layout_is_empty(batch_overflow);

    const auto scale_overflow = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F16,
        1, 1, std::numeric_limits<int64_t>::max(), 1, 1);
    assert(scale_overflow.status == ggml_cann_q8_w8a8_workspace_status::overflow);
    assert_workspace_layout_is_empty(scale_overflow);

    const auto input_cast_overflow = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F32, GGML_TYPE_F16,
        std::numeric_limits<int64_t>::max(), 1, 2, 1, 1);
    assert(input_cast_overflow.status == ggml_cann_q8_w8a8_workspace_status::overflow);
    assert_workspace_layout_is_empty(input_cast_overflow);

    const auto output_cast_overflow = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F32,
        1, std::numeric_limits<int64_t>::max(), 2, 1, 1);
    assert(output_cast_overflow.status == ggml_cann_q8_w8a8_workspace_status::overflow);
    assert_workspace_layout_is_empty(output_cast_overflow);

    const auto alignment_overflow = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F16,
        INT64_C(4294967295), 1, INT64_C(4294967297), 1, 1);
    assert(alignment_overflow.status == ggml_cann_q8_w8a8_workspace_status::overflow);
    assert_workspace_layout_is_empty(alignment_overflow);
}

static void test_workspace_plan_mixed_types() {
    const auto f32_to_f16 = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F32, GGML_TYPE_F16, 128, 96, 17, 1, 1);
    assert(f32_to_f16.status == ggml_cann_q8_w8a8_workspace_status::ok);
    assert(f32_to_f16.input_f16_bytes == 4352);
    assert(f32_to_f16.output_f16_offset == 0);
    assert(f32_to_f16.output_f16_bytes == 0);

    const auto f16_to_f32 = ggml_cann_q8_w8a8_plan_workspace(
        GGML_TYPE_F16, GGML_TYPE_F32, 128, 96, 17, 1, 1);
    assert(f16_to_f32.status == ggml_cann_q8_w8a8_workspace_status::ok);
    assert(f16_to_f32.input_f16_offset == 0);
    assert(f16_to_f32.input_f16_bytes == 0);
    assert(f16_to_f32.output_f16_bytes == 3264);
}

static void test_known_rows_and_reconstruction() {
    constexpr int64_t k = 64;
    constexpr int64_t n = 2;
    auto src = make_source(k, n);
    std::vector<uint8_t> dst(ggml_cann_q8_w8a8_required_size(k, n));
    const auto result = ggml_cann_requantize_q8_0_per_channel(src.data(), k, n, dst.data(), dst.size());
    assert(result.status == ggml_cann_q8_w8a8_status::ok);
    assert(result.weight_bytes == static_cast<size_t>(k * n));
    assert(result.scale_offset == result.weight_bytes);
    const int8_t * q = reinterpret_cast<const int8_t *>(dst.data());
    const float * scales = reinterpret_cast<const float *>(dst.data() + result.scale_offset);
    for (int64_t row = 0; row < n; ++row) {
        assert(std::isfinite(scales[row]));
        assert(scales[row] > 0.0f);
        for (int64_t col = 0; col < k; ++col) {
            const block_q8_0 & b = src[static_cast<size_t>(row * (k / QK8_0) + col / QK8_0)];
            const float original = GGML_FP16_TO_FP32(b.d) * b.qs[col % QK8_0];
            const float rebuilt = scales[row] * q[row * k + col];
            assert(std::fabs(original - rebuilt) <= scales[row] * 0.51f + 1e-6f);
            assert(q[row * k + col] >= -127);
        }
    }
}

static void test_zero_row_uses_unit_scale() {
    constexpr int64_t k = 64;
    block_q8_0 src[k / QK8_0] = {};
    for (auto & b : src) {
        b.d = GGML_FP32_TO_FP16(1.0f);
        std::memset(b.qs, 0, sizeof(b.qs));
    }
    std::vector<uint8_t> dst(ggml_cann_q8_w8a8_required_size(k, 1));
    const auto result = ggml_cann_requantize_q8_0_per_channel(src, k, 1, dst.data(), dst.size());
    assert(result.status == ggml_cann_q8_w8a8_status::ok);
    const float scale = *reinterpret_cast<const float *>(dst.data() + result.scale_offset);
    assert(scale == 1.0f);
    for (int64_t i = 0; i < k; ++i) {
        assert(reinterpret_cast<const int8_t *>(dst.data())[i] == 0);
    }
}

static void test_non_finite_scale_is_rejected() {
    constexpr int64_t k = 64;
    block_q8_0 src[k / QK8_0] = {};
    for (auto & b : src) {
        b.d = UINT16_C(0x7c00);
        b.qs[0] = 1;
    }
    assert(!std::isfinite(GGML_FP16_TO_FP32(src[0].d)));
    std::vector<uint8_t> dst(ggml_cann_q8_w8a8_required_size(k, 1));
    assert(ggml_cann_requantize_q8_0_per_channel(src, k, 1, dst.data(), dst.size()).status ==
           ggml_cann_q8_w8a8_status::non_finite_scale);
}

static void test_layout_validation() {
    ggml_cann_q8_w8a8_layout layout = {};
    constexpr size_t capacity = 64 * 2 + 2 * sizeof(float);
    assert(ggml_cann_q8_w8a8_validate(
               false, GGML_TYPE_Q8_0, true, 64, 2, 1, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::disabled);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q4_0, true, 64, 2, 1, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::wrong_type);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q8_0, false, 64, 2, 1, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::not_matmul_weight);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q8_0, true, 64, 2, 2, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::batched);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q8_0, true, 33, 2, 1, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::invalid_shape);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q8_0, true, 64, 2, 1, 1, capacity - 1, &layout) ==
           ggml_cann_q8_w8a8_reject::insufficient_capacity);
    assert(ggml_cann_q8_w8a8_validate(
               true, GGML_TYPE_Q8_0, true, 64, 2, 1, 1, capacity, &layout) ==
           ggml_cann_q8_w8a8_reject::none);
    assert(layout.k == 64 && layout.n == 2);
    assert(layout.weight_bytes == 128);
    assert(layout.scale_offset == 128);
    assert(layout.scale_bytes == 2 * sizeof(float));
}

static void test_restore_to_standard_q8_0() {
    constexpr int64_t k = 64;
    constexpr int64_t n = 2;
    const auto original = make_source(k, n);
    std::vector<uint8_t> w8(ggml_cann_q8_w8a8_required_size(k, n));
    const auto converted = ggml_cann_requantize_q8_0_per_channel(
        original.data(), k, n, w8.data(), w8.size());
    assert(converted.status == ggml_cann_q8_w8a8_status::ok);

    std::vector<block_q8_0> restored(static_cast<size_t>(k / QK8_0 * n));
    const auto result = ggml_cann_restore_q8_0_from_per_channel(
        w8.data(), k, n, restored.data(), restored.size() * sizeof(block_q8_0));
    assert(result.status == ggml_cann_q8_w8a8_status::ok);

    const int8_t * q = reinterpret_cast<const int8_t *>(w8.data());
    const float * scales = reinterpret_cast<const float *>(w8.data() + converted.scale_offset);
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            const block_q8_0 & block = restored[static_cast<size_t>(row * (k / QK8_0) + col / QK8_0)];
            const float expected = q[row * k + col] * scales[row];
            const float actual = GGML_FP16_TO_FP32(block.d) * block.qs[col % QK8_0];
            const float block_scale = GGML_FP16_TO_FP32(block.d);
            assert(std::fabs(expected - actual) <= block_scale * 0.55f + 1e-5f);
        }
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

struct graph_run_result {
    std::vector<std::vector<uint8_t>> outputs;
    ggml_cann_q8_w8a8_stats stats;
};

struct graph_fixture {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_cgraph * graph;
    ggml_tensor * weight;
    ggml_tensor * input;
    ggml_tensor * output;
};

static graph_fixture make_deterministic_w8a8_graph(ggml_backend_t backend, int64_t m) {
    constexpr int64_t k = 128;
    constexpr int64_t n = 96;

    ggml_init_params params = { 8u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(weight, "blk.7.attn_q.weight");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    const auto source = make_source(k, n);
    const auto input_data = make_deterministic_input(k, m);
    ggml_backend_tensor_set(weight, source.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    assert(ggml_cann_get_q8_w8a8_layout(weight, nullptr));
    return { ctx, buffer, graph, weight, input, output };
}

static std::vector<uint8_t> compute_graph_output(ggml_backend_t backend, const graph_fixture & fixture) {
    assert(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<uint8_t> output(ggml_nbytes(fixture.output));
    ggml_backend_tensor_get(fixture.output, output.data(), 0, output.size());
    return output;
}

static void free_graph_fixture(graph_fixture fixture) {
    ggml_backend_buffer_free(fixture.buffer);
    ggml_free(fixture.ctx);
}

static graph_run_result run_w8a8_graph_case(
        bool acl_graph_enabled, size_t graph_cache_capacity, int64_t m, int repeats) {
    const std::string capacity = std::to_string(graph_cache_capacity);
    assert(setenv("GGML_CANN_ACL_GRAPH", acl_graph_enabled ? "on" : "off", 1) == 0);
    assert(setenv("GGML_CANN_GRAPH_CACHE_CAPACITY", capacity.c_str(), 1) == 0);
    ggml_cann_q8_w8a8_stats_reset();

    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    const graph_fixture fixture = make_deterministic_w8a8_graph(backend, m);
    std::vector<std::vector<uint8_t>> outputs;
    outputs.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        outputs.push_back(compute_graph_output(backend, fixture));
    }
    free_graph_fixture(fixture);
    ggml_backend_free(backend);
    return { std::move(outputs), ggml_cann_q8_w8a8_stats_get() };
}

struct graph_cache_run_result {
    std::vector<uint8_t> first_a;
    std::vector<uint8_t> second_a;
    ggml_cann_q8_w8a8_stats before_backend_free;
    ggml_cann_q8_w8a8_stats after_backend_free;
};

static graph_cache_run_result run_w8a8_graph_cache_case(const char * graph_cache_capacity);

static graph_cache_run_result run_w8a8_graph_cache_case(size_t graph_cache_capacity) {
    const std::string capacity = std::to_string(graph_cache_capacity);
    return run_w8a8_graph_cache_case(capacity.c_str());
}

static graph_cache_run_result run_w8a8_graph_cache_case(const char * graph_cache_capacity) {
    assert(setenv("GGML_CANN_ACL_GRAPH", "on", 1) == 0);
    assert(setenv("GGML_CANN_GRAPH_CACHE_CAPACITY", graph_cache_capacity, 1) == 0);
    ggml_cann_q8_w8a8_stats_reset();

    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    const graph_fixture a = make_deterministic_w8a8_graph(backend, 1);
    const graph_fixture b = make_deterministic_w8a8_graph(backend, 17);
    std::vector<uint8_t> first_a = compute_graph_output(backend, a);
    (void) compute_graph_output(backend, b);
    std::vector<uint8_t> second_a = compute_graph_output(backend, a);
    const ggml_cann_q8_w8a8_stats before_backend_free = ggml_cann_q8_w8a8_stats_get();
    free_graph_fixture(b);
    free_graph_fixture(a);
    ggml_backend_free(backend);
    return { std::move(first_a), std::move(second_a), before_backend_free,
             ggml_cann_q8_w8a8_stats_get() };
}

static void test_graph_cache_invalid_capacities_clamp_to_one() {
    const char * invalid_capacities[] = { "0", "-1", "invalid" };
    for (const char * capacity : invalid_capacities) {
        const graph_cache_run_result result = run_w8a8_graph_cache_case(capacity);
        assert(result.first_a == result.second_a);
        assert(result.before_backend_free.matmul_hits == 3);
        assert(result.before_backend_free.graph_workspace_allocations == 3);
        assert(result.before_backend_free.graph_workspace_frees == 2);
        assert(result.after_backend_free.graph_workspace_frees == 3);
    }
}

static void test_w8a8_graph_replay_is_bitwise_equal_to_eager() {
    const graph_run_result eager = run_w8a8_graph_case(false, 2, 17, 4);
    const graph_run_result graph = run_w8a8_graph_case(true, 2, 17, 4);

    assert(eager.outputs.size() == 4);
    assert(graph.outputs.size() == 4);
    const std::vector<uint8_t> & eager_reference = eager.outputs.front();
    for (const auto & output : eager.outputs) {
        assert(output == eager_reference);
    }
    for (const auto & output : graph.outputs) {
        assert(output == eager_reference);
    }
    assert(eager.stats.matmul_hits == 4);
    assert(eager.stats.graph_workspace_allocations == 0);
    assert(eager.stats.graph_workspace_frees == 0);
    assert(graph.stats.matmul_hits == 1);
    assert(graph.stats.graph_workspace_allocations == 1);
    assert(graph.stats.graph_workspace_frees == 1);
}

static void test_w8a8_graph_cache_capacity_two_reuses_a() {
    const graph_cache_run_result result = run_w8a8_graph_cache_case(2);

    assert(result.first_a == result.second_a);
    assert(result.before_backend_free.matmul_hits == 2);
    assert(result.before_backend_free.graph_workspace_allocations == 2);
    assert(result.before_backend_free.graph_workspace_frees == 0);
    assert(result.after_backend_free.graph_workspace_frees == 2);
}

static void test_w8a8_graph_cache_capacity_one_evicts_a() {
    const graph_cache_run_result result = run_w8a8_graph_cache_case(1);

    assert(result.first_a == result.second_a);
    assert(result.before_backend_free.matmul_hits == 3);
    assert(result.before_backend_free.graph_workspace_allocations == 3);
    assert(result.before_backend_free.graph_workspace_frees == 2);
    assert(result.after_backend_free.graph_workspace_frees == 3);
}

static void test_w8a8_graph_cache_tracks_registry_transitions() {
    assert(setenv("GGML_CANN_ACL_GRAPH", "on", 1) == 0);
    assert(setenv("GGML_CANN_GRAPH_CACHE_CAPACITY", "1", 1) == 0);
    ggml_cann_q8_w8a8_stats_reset();

    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    const graph_fixture fixture = make_deterministic_w8a8_graph(backend, 17);

    const std::vector<uint8_t> first_output = compute_graph_output(backend, fixture);
    ggml_cann_q8_w8a8_stats stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 1);
    assert(stats.graph_workspace_allocations == 1);
    assert(stats.graph_workspace_frees == 0);

    ggml_backend_buffer_clear(fixture.buffer, 0);
    assert(!ggml_cann_get_q8_w8a8_layout(fixture.weight, nullptr));
    const std::vector<uint8_t> cleared_output = compute_graph_output(backend, fixture);
    assert(std::all_of(cleared_output.begin(), cleared_output.end(), [](uint8_t value) { return value == 0; }));
    stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 1);
    assert(stats.graph_workspace_allocations == 1);
    assert(stats.graph_workspace_frees == 1);

    const auto source = make_source(fixture.weight->ne[0], fixture.weight->ne[1]);
    const auto input_data = make_deterministic_input(fixture.input->ne[0], fixture.input->ne[1]);
    ggml_backend_tensor_set(fixture.weight, source.data(), 0, ggml_nbytes(fixture.weight));
    ggml_backend_tensor_set(fixture.input, input_data.data(), 0, ggml_nbytes(fixture.input));
    assert(ggml_cann_get_q8_w8a8_layout(fixture.weight, nullptr));
    const std::vector<uint8_t> reuploaded_output = compute_graph_output(backend, fixture);
    assert(reuploaded_output == first_output);
    stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 2);
    assert(stats.graph_workspace_allocations == 2);
    assert(stats.graph_workspace_frees == 1);

    free_graph_fixture(fixture);
    ggml_backend_free(backend);
    stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 2);
    assert(stats.graph_workspace_allocations == 2);
    assert(stats.graph_workspace_frees == 2);
}

static void test_w8a8_graph_snapshot_is_immutable() {
    assert(setenv("GGML_CANN_ACL_GRAPH", "on", 1) == 0);
    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    const graph_fixture fixture = make_deterministic_w8a8_graph(backend, 17);

    ggml_cann_q8_w8a8_graph_snapshot snapshot_a;
    snapshot_a.capture_from_cgraph(fixture.graph);
    assert(std::count_if(
        snapshot_a.nodes.begin(), snapshot_a.nodes.end(),
        [](const ggml_cann_q8_w8a8_graph_node_snapshot & node) { return node.registered; }) == 1);
    assert(!snapshot_a.nodes.empty() && snapshot_a.nodes[0].registered);
    const ggml_cann_q8_w8a8_graph_node_snapshot captured_node = snapshot_a.nodes[0];

    ggml_cann_q8_w8a8_graph_snapshot reusable_snapshot;
    reusable_snapshot.capture_from_cgraph(fixture.graph);
    const ggml_cann_q8_w8a8_graph_node_snapshot * snapshot_storage = reusable_snapshot.nodes.data();

    ggml_backend_buffer_clear(fixture.buffer, 0);
    ggml_cann_q8_w8a8_graph_snapshot snapshot_b;
    snapshot_b.capture_from_cgraph(fixture.graph);
    assert(!ggml_cann_q8_w8a8_graph_node_snapshot_matches(snapshot_a.nodes[0], snapshot_b.nodes[0]));
    reusable_snapshot.capture_from_cgraph(fixture.graph);
    assert(reusable_snapshot.nodes.data() == snapshot_storage);

    const auto source = make_source(fixture.weight->ne[0], fixture.weight->ne[1]);
    ggml_backend_tensor_set(fixture.weight, source.data(), 0, ggml_nbytes(fixture.weight));
    ggml_cann_q8_w8a8_graph_snapshot snapshot_c;
    snapshot_c.capture_from_cgraph(fixture.graph);
    assert(ggml_cann_q8_w8a8_graph_node_snapshot_matches(snapshot_a.nodes[0], snapshot_c.nodes[0]));

    ggml_backend_buffer_clear(fixture.buffer, 0);
    const ggml_cann_q8_w8a8_workspace_plan plan = ggml_cann_q8_w8a8_plan_workspace(
        captured_node.input_type, captured_node.output_type,
        captured_node.layout.k, captured_node.layout.n,
        captured_node.m, captured_node.ne2, captured_node.ne3);
    assert(plan.status == ggml_cann_q8_w8a8_workspace_status::ok);
    assert(snapshot_a.nodes[0].registered);
    assert(snapshot_a.nodes[0].layout.k == captured_node.layout.k);
    assert(snapshot_a.nodes[0].layout.n == captured_node.layout.n);

    free_graph_fixture(fixture);
    ggml_backend_free(backend);
}

static void test_same_device_backend_free_preserves_peer_graph() {
    assert(setenv("GGML_CANN_GRAPH_CACHE_CAPACITY", "2", 1) == 0);
    ggml_cann_q8_w8a8_stats_reset();

    assert(setenv("GGML_CANN_ACL_GRAPH", "off", 1) == 0);
    ggml_backend_t backend_a = init_cann();
    assert(backend_a != nullptr);
    const graph_fixture fixture_a = make_deterministic_w8a8_graph(backend_a, 17);

    const std::vector<uint8_t> first_a = compute_graph_output(backend_a, fixture_a);

    assert(setenv("GGML_CANN_ACL_GRAPH", "on", 1) == 0);
    ggml_backend_t backend_b = init_cann();
    assert(backend_b != nullptr);
    const graph_fixture fixture_b = make_deterministic_w8a8_graph(backend_b, 17);

    const std::vector<uint8_t> first_b = compute_graph_output(backend_b, fixture_b);
    assert(first_a == first_b);
    ggml_cann_q8_w8a8_stats stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 2);
    assert(stats.graph_workspace_allocations == 1);
    assert(stats.graph_workspace_frees == 0);

    free_graph_fixture(fixture_a);
    ggml_backend_free(backend_a);
    stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.graph_workspace_frees == 0);

    const std::vector<uint8_t> second_b = compute_graph_output(backend_b, fixture_b);
    assert(second_b == first_b);
    stats = ggml_cann_q8_w8a8_stats_get();
    assert(stats.matmul_hits == 2);
    assert(stats.graph_workspace_allocations == 1);
    assert(stats.graph_workspace_frees == 0);

    free_graph_fixture(fixture_b);
    ggml_backend_free(backend_b);
    assert(ggml_cann_q8_w8a8_stats_get().graph_workspace_frees == 1);
}

static void test_cann_buffer_full_upload_round_trip(ggml_backend_t backend, bool expect_w8a8) {
    ggml_init_params params = { 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 2);
    ggml_tensor * chunked = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 2);
    ggml_tensor * copied = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 2);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    ggml_set_name(chunked, "blk.1.attn_q.weight");
    ggml_set_name(copied, "blk.2.attn_q.weight");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    const auto source = make_source(64, 2);
    ggml_backend_tensor_set(weight, source.data(), 0, ggml_nbytes(weight));
    ggml_cann_q8_w8a8_layout layout = {};
    assert(ggml_cann_get_q8_w8a8_layout(weight, &layout) == expect_w8a8);
    if (expect_w8a8) {
        assert(layout.k == 64 && layout.n == 2);
    }

    std::vector<block_q8_0> restored(source.size());
    ggml_backend_tensor_get(weight, restored.data(), 0, ggml_nbytes(weight));
    for (size_t block = 0; block < source.size(); ++block) {
        for (int i = 0; i < QK8_0; ++i) {
            const float expected = GGML_FP16_TO_FP32(source[block].d) * source[block].qs[i];
            const float actual = GGML_FP16_TO_FP32(restored[block].d) * restored[block].qs[i];
            assert(std::fabs(expected - actual) <= 0.1f);
        }
    }

    const size_t half = ggml_nbytes(chunked) / 2;
    ggml_backend_tensor_set(chunked, source.data(), 0, half);
    assert(!ggml_cann_get_q8_w8a8_layout(chunked, nullptr));
    ggml_backend_tensor_set(
        chunked, reinterpret_cast<const uint8_t *>(source.data()) + half,
        half, ggml_nbytes(chunked) - half);
    assert(ggml_cann_get_q8_w8a8_layout(chunked, &layout) == expect_w8a8);

    std::fill(restored.begin(), restored.end(), block_q8_0{});
    ggml_backend_tensor_get(chunked, restored.data(), 0, ggml_nbytes(chunked));
    for (size_t block = 0; block < source.size(); ++block) {
        for (int i = 0; i < QK8_0; ++i) {
            const float expected = GGML_FP16_TO_FP32(source[block].d) * source[block].qs[i];
            const float actual = GGML_FP16_TO_FP32(restored[block].d) * restored[block].qs[i];
            assert(std::fabs(expected - actual) <= 0.1f);
        }
    }

    ggml_backend_tensor_copy(chunked, copied);
    assert(ggml_cann_get_q8_w8a8_layout(copied, &layout) == expect_w8a8);
    std::fill(restored.begin(), restored.end(), block_q8_0{});
    ggml_backend_tensor_get(copied, restored.data(), 0, ggml_nbytes(copied));
    for (size_t block = 0; block < source.size(); ++block) {
        for (int i = 0; i < QK8_0; ++i) {
            const float expected = GGML_FP16_TO_FP32(source[block].d) * source[block].qs[i];
            const float actual = GGML_FP16_TO_FP32(restored[block].d) * restored[block].qs[i];
            assert(std::fabs(expected - actual) <= 0.1f);
        }
    }

    ggml_backend_tensor_set(copied, source.data(), 0, half);
    assert(!ggml_cann_get_q8_w8a8_layout(copied, nullptr));
    ggml_backend_tensor_set(
        copied, reinterpret_cast<const uint8_t *>(source.data()) + half,
        half, ggml_nbytes(copied) - half);
    assert(ggml_cann_get_q8_w8a8_layout(copied, nullptr) == expect_w8a8);

    ggml_backend_tensor_memset(copied, 0, 0, half);
    assert(!ggml_cann_get_q8_w8a8_layout(copied, nullptr));
    std::vector<uint8_t> partially_zeroed(ggml_nbytes(copied));
    ggml_backend_tensor_get(copied, partially_zeroed.data(), 0, partially_zeroed.size());
    for (size_t i = 0; i < half; ++i) {
        assert(partially_zeroed[i] == 0);
    }
    assert(std::memcmp(
        partially_zeroed.data() + half,
        reinterpret_cast<const uint8_t *>(restored.data()) + half,
        partially_zeroed.size() - half) == 0);

    ggml_backend_tensor_set(copied, source.data(), 0, ggml_nbytes(copied));
    assert(ggml_cann_get_q8_w8a8_layout(copied, nullptr) == expect_w8a8);
    ggml_backend_tensor_memset(copied, 0, 0, ggml_nbytes(copied));
    assert(!ggml_cann_get_q8_w8a8_layout(copied, nullptr));

    ggml_backend_tensor_set(copied, source.data(), 0, ggml_nbytes(copied));
    assert(ggml_cann_get_q8_w8a8_layout(copied, nullptr) == expect_w8a8);
    ggml_backend_buffer_clear(buffer, 0);
    assert(!ggml_cann_get_q8_w8a8_layout(copied, nullptr));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_cann_w8a8_matmul(ggml_backend_t backend, bool expect_w8a8, bool eligible_name) {
    struct matmul_case {
        int64_t k;
        int64_t n;
        int64_t m;
    };
    const matmul_case cases[] = {
        { 128, 96, 1 },
        { 128, 96, 17 },
        { 128, 65568, 1 },
    };

    ggml_cann_q8_w8a8_stats_reset();

    for (const matmul_case & tc : cases) {
        ggml_init_params params = { 8u * 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        assert(ctx != nullptr);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, tc.k, tc.n);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tc.k, tc.m);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        ggml_set_name(weight, eligible_name ? "blk.3.attn_q.weight" : "token_embd.weight");
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer != nullptr);

        auto source = make_source(tc.k, tc.n);
        std::vector<ggml_fp16_t> input_data(static_cast<size_t>(tc.k * tc.m));
        for (int64_t row = 0; row < tc.m; ++row) {
            for (int64_t col = 0; col < tc.k; ++col) {
                const float value = static_cast<float>(((row * 11 + col * 7) % 23) - 11) / 128.0f;
                input_data[static_cast<size_t>(row * tc.k + col)] = GGML_FP32_TO_FP16(value);
            }
        }
        ggml_backend_tensor_set(weight, source.data(), 0, ggml_nbytes(weight));
        ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
        assert(ggml_cann_get_q8_w8a8_layout(weight, nullptr) == expect_w8a8);
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

        std::vector<float> actual(static_cast<size_t>(tc.n * tc.m));
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        double sum_abs_error = 0.0;
        float max_abs_error = 0.0f;
        for (int64_t row = 0; row < tc.m; ++row) {
            for (int64_t out = 0; out < tc.n; ++out) {
                float expected = 0.0f;
                for (int64_t col = 0; col < tc.k; ++col) {
                    const block_q8_0 & block = source[static_cast<size_t>(out * (tc.k / QK8_0) + col / QK8_0)];
                    const float w = GGML_FP16_TO_FP32(block.d) * block.qs[col % QK8_0];
                    const float x = GGML_FP16_TO_FP32(input_data[static_cast<size_t>(row * tc.k + col)]);
                    expected += x * w;
                }
                const float got = actual[static_cast<size_t>(row * tc.n + out)];
                const float abs_error = std::fabs(got - expected);
                assert(std::isfinite(got));
                assert(abs_error <= 0.08f + 0.03f * std::fabs(expected));
                sum_abs_error += abs_error;
                max_abs_error = std::max(max_abs_error, abs_error);
            }
        }
        std::printf(
            "Q8 matmul k=%lld n=%lld m=%lld path=%s mean_abs_error=%.8f max_abs_error=%.8f\n",
            static_cast<long long>(tc.k), static_cast<long long>(tc.n), static_cast<long long>(tc.m),
            expect_w8a8 ? "w8a8" : "w8a16",
            sum_abs_error / static_cast<double>(tc.n * tc.m), max_abs_error);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    const uint64_t expected_hits = expect_w8a8 ? sizeof(cases) / sizeof(cases[0]) : 0;
    assert(ggml_cann_q8_w8a8_stats_get().matmul_hits == expected_hits);
}

static void test_cann_batched_q8_falls_back(ggml_backend_t backend) {
    constexpr int64_t k = 128;
    constexpr int64_t n = 96;
    constexpr int64_t m = 1;
    constexpr int64_t batches = 2;

    ggml_cann_q8_w8a8_stats_reset();
    ggml_init_params params = { 8u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, k, n, batches);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, batches);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(weight, "blk.4.attn_q.weight");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);

    const auto source = make_source(k, n * batches);
    std::vector<ggml_fp16_t> input_data(static_cast<size_t>(k * m * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t col = 0; col < k; ++col) {
            const float value = static_cast<float>(((batch * 13 + col * 7) % 23) - 11) / 128.0f;
            input_data[static_cast<size_t>(batch * k + col)] = GGML_FP32_TO_FP16(value);
        }
    }
    ggml_backend_tensor_set(weight, source.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    assert(!ggml_cann_get_q8_w8a8_layout(weight, nullptr));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(static_cast<size_t>(n * m * batches));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t out = 0; out < n; ++out) {
            float expected = 0.0f;
            for (int64_t col = 0; col < k; ++col) {
                const size_t block_index = static_cast<size_t>(
                    (batch * n + out) * (k / QK8_0) + col / QK8_0);
                const block_q8_0 & block = source[block_index];
                const float w = GGML_FP16_TO_FP32(block.d) * block.qs[col % QK8_0];
                const float x = GGML_FP16_TO_FP32(input_data[static_cast<size_t>(batch * k + col)]);
                expected += x * w;
            }
            const float got = actual[static_cast<size_t>(batch * n + out)];
            assert(std::isfinite(got));
            assert(std::fabs(got - expected) <= 0.08f + 0.03f * std::fabs(expected));
        }
    }
    assert(ggml_cann_q8_w8a8_stats_get().matmul_hits == 0);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    test_graph_device_gate_allows_concurrent_replay();
    test_graph_context_gate_serializes_transactions();
    test_graph_device_gate_excludes_capture_and_replay();
    test_invalid_arguments();
    test_workspace_plan_f16();
    test_workspace_plan_casts_and_match();
    test_workspace_plan_rejects_invalid_and_overflow();
    test_workspace_plan_mixed_types();
    test_known_rows_and_reconstruction();
    test_zero_row_uses_unit_scale();
    test_non_finite_scale_is_rejected();
    test_layout_validation();
    test_restore_to_standard_q8_0();
    if (argc > 1 && std::strcmp(argv[1], "--host-only") == 0) {
        return 0;
    }
    const bool enabled = std::getenv("GGML_CANN_Q8_W8A8") != nullptr &&
                         std::strcmp(std::getenv("GGML_CANN_Q8_W8A8"), "on") == 0;
    const bool eligible_name = argc < 2 || std::strcmp(argv[1], "--ineligible-name") != 0;
    const bool expect_w8a8 = enabled && eligible_name;
    if (argc > 1 && std::strcmp(argv[1], "--invalid-capacity") == 0) {
        assert(enabled && ggml_cann_q8_w8a8_graph_compiled());
        test_graph_cache_invalid_capacities_clamp_to_one();
        return 0;
    }
    ggml_backend_t backend = init_cann();
    assert(backend != nullptr);
    test_cann_buffer_full_upload_round_trip(backend, enabled);
    test_cann_w8a8_matmul(backend, expect_w8a8, eligible_name);
    test_cann_batched_q8_falls_back(backend);
    ggml_backend_free(backend);
    if (expect_w8a8 && ggml_cann_q8_w8a8_graph_compiled()) {
        test_w8a8_graph_replay_is_bitwise_equal_to_eager();
        test_w8a8_graph_cache_capacity_two_reuses_a();
        test_w8a8_graph_cache_capacity_one_evicts_a();
        test_w8a8_graph_cache_tracks_registry_transitions();
        test_w8a8_graph_snapshot_is_immutable();
        test_same_device_backend_free_preserves_peer_graph();
    }
    return 0;
}
