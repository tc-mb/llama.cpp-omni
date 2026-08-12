#include "affine_layer_norm.h"

#include "aclnn_ops.h"
#include "common.h"
#include "../ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {

struct atomic_stats {
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> hits{0};
};

atomic_stats g_stats;
std::once_flag g_stats_atexit_once;

int unique_consumer_index(
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        const ggml_tensor * tensor) {
    const auto it = unique_consumers.find(tensor);
    return it == unique_consumers.end() ? -1 : it->second;
}

bool exact_contiguous_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
        tensor->nb[0] != sizeof(float)) {
        return false;
    }
    size_t stride = sizeof(float);
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim - 1] <= 0 ||
            static_cast<uint64_t>(tensor->ne[dim - 1]) >
                std::numeric_limits<size_t>::max() / stride) {
            return false;
        }
        stride *= static_cast<size_t>(tensor->ne[dim - 1]);
        if (tensor->nb[dim] != stride) {
            return false;
        }
    }
    return true;
}

bool canonical_shape_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    const bool flow = tensor->ne[0] == 512 &&
        (tensor->ne[1] == 50 || tensor->ne[1] == 56) &&
        tensor->ne[2] == 2 && tensor->ne[3] == 1;
    const bool attention = tensor->ne[0] == 64 && tensor->ne[1] == 8 &&
        (tensor->ne[2] == 50 || tensor->ne[2] == 56) && tensor->ne[3] == 2;
    return flow || attention;
}

bool canonical_tensor(const ggml_tensor * tensor) {
    return canonical_shape_f32(tensor) && exact_contiguous_f32(tensor);
}

bool canonical_affine(const ggml_tensor * tensor, int64_t width) {
    return exact_contiguous_f32(tensor) && tensor->ne[0] == width &&
        tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool disjoint_tensor(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
        return false;
    }
    const auto tensor_span = [](const ggml_tensor * tensor, size_t * span) {
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
            const size_t offset = static_cast<size_t>(count) * tensor->nb[dim];
            if (value > std::numeric_limits<size_t>::max() - offset) {
                return false;
            }
            value += offset;
        }
        *span = value;
        return true;
    };
    size_t a_size = 0;
    size_t b_size = 0;
    if (!tensor_span(a, &a_size) || !tensor_span(b, &b_size)) {
        return false;
    }
    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b->data);
    if (a_size == 0 || b_size == 0 ||
        a_begin > std::numeric_limits<uintptr_t>::max() - a_size ||
        b_begin > std::numeric_limits<uintptr_t>::max() - b_size) {
        return false;
    }
    return a_begin + a_size <= b_begin || b_begin + b_size <= a_begin;
}

void stats_print() {
    const auto stats = ggml_cann_affine_layer_norm_stats_get();
    std::fprintf(stderr,
                 "[cann-affine-layer-norm] attempts=%llu hits=%llu "
                 "launches_removed=%llu\n",
                 static_cast<unsigned long long>(stats.attempts),
                 static_cast<unsigned long long>(stats.hits),
                 static_cast<unsigned long long>(stats.hits * 2));
}

void register_stats_printer() {
    std::call_once(g_stats_atexit_once, [] {
        if (parse_bool(get_env_as_lowercase(
                "GGML_CANN_AFFINE_LAYERNORM_STATS").value_or("off"))) {
            std::atexit(stats_print);
        }
    });
}

}  // namespace

bool ggml_cann_affine_layer_norm_enabled() {
    return get_env_as_lowercase("GGML_CANN_AFFINE_LAYERNORM")
        .value_or("auto") == "auto";
}

bool ggml_cann_affine_layer_norm_try(
        ggml_backend_cann_context & ctx,
        ggml_cgraph * cgraph,
        int node_index,
        const std::unordered_map<const ggml_tensor *, int> & unique_consumers,
        int * mul_index_out,
        int * add_index_out) {
    register_stats_printer();
    g_stats.attempts.fetch_add(1, std::memory_order_relaxed);
    if (!ggml_cann_affine_layer_norm_enabled() || cgraph == nullptr ||
        node_index < 0 || node_index >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * norm = cgraph->nodes[node_index];
    const int mul_index = unique_consumer_index(unique_consumers, norm);
    ggml_tensor * mul = mul_index >= 0 ? cgraph->nodes[mul_index] : nullptr;
    const int add_index = mul != nullptr
        ? unique_consumer_index(unique_consumers, mul)
        : -1;
    ggml_tensor * add = add_index >= 0 ? cgraph->nodes[add_index] : nullptr;
    if (norm->op != GGML_OP_NORM || mul == nullptr || mul->op != GGML_OP_MUL ||
        mul->src[0] != norm || add == nullptr || add->op != GGML_OP_ADD ||
        add->src[0] != mul || (norm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        (mul->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    ggml_tensor * input = norm->src[0];
    ggml_tensor * weight = mul->src[1];
    ggml_tensor * bias = add->src[1];
    if (!canonical_shape_f32(input) || !canonical_tensor(norm) ||
        !canonical_tensor(mul) || !canonical_tensor(add) ||
        !canonical_affine(weight, input->ne[0]) ||
        !canonical_affine(bias, input->ne[0]) ||
        !disjoint_tensor(input, add) || !disjoint_tensor(weight, add) ||
        !disjoint_tensor(bias, add)) {
        return false;
    }

    ggml_cann_norm_affine(ctx, norm, weight, bias, add);
    *mul_index_out = mul_index;
    *add_index_out = add_index;
    g_stats.hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_cann_affine_layer_norm_stats_reset() {
    g_stats.attempts.store(0, std::memory_order_relaxed);
    g_stats.hits.store(0, std::memory_order_relaxed);
}

ggml_cann_affine_layer_norm_stats_snapshot
ggml_cann_affine_layer_norm_stats_get() {
    ggml_cann_affine_layer_norm_stats_snapshot snapshot;
    snapshot.attempts = g_stats.attempts.load(std::memory_order_relaxed);
    snapshot.hits = g_stats.hits.load(std::memory_order_relaxed);
    return snapshot;
}
