#include "token2wav-frontend-gguf.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omni {
namespace flow {
namespace {

class GGUFWeightStore {
  public:
    bool load(const std::string & path, ggml_backend_t backend, std::string * error) {
        reset();
        if (!backend) {
            return fail(error, "GGUF frontend: backend is null");
        }

        ggml_context * meta = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &meta,
        };
        ctx_gguf_.reset(gguf_init_from_file(path.c_str(), params));
        if (!ctx_gguf_ || !meta) {
            if (meta) {
                ggml_free(meta);
            }
            return fail(error, "GGUF frontend: failed to load " + path);
        }
        ctx_meta_.reset(meta);

        if (gguf_get_version(ctx_gguf_.get()) != GGUF_VERSION) {
            return fail(error, "GGUF frontend: unsupported GGUF version in " + path);
        }
        const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf_.get());
        if (n_tensors <= 0) {
            return fail(error, "GGUF frontend: no tensors in " + path);
        }

        ggml_init_params data_params = {
            /*.mem_size   =*/ static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_data_.reset(ggml_init(data_params));
        if (!ctx_data_) {
            return fail(error, "GGUF frontend: failed to allocate tensor metadata for " + path);
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            return fail(error, "GGUF frontend: failed to open " + path);
        }

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf_.get(), i);
            if (!name || !*name) {
                return fail(error, "GGUF frontend: tensor has no name in " + path);
            }
            ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta_.get(), name);
            if (!meta_tensor) {
                return fail(error, "GGUF frontend: missing tensor metadata for " + std::string(name));
            }
            if (meta_tensor->type != GGML_TYPE_F32) {
                return fail(error, "GGUF frontend: expected F32 tensor " + std::string(name));
            }
            ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data_.get(), meta_tensor);
            if (!data_tensor) {
                return fail(error, "GGUF frontend: failed to allocate tensor " + std::string(name));
            }
            ggml_set_name(data_tensor, name);
            tensors_.emplace(name, data_tensor);
            offsets_.emplace(name, gguf_get_data_offset(ctx_gguf_.get()) +
                                      gguf_get_tensor_offset(ctx_gguf_.get(), i));
        }

        const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffer_.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_data_.get(), buft));
        if (!buffer_) {
            return fail(error, "GGUF frontend: failed to allocate tensor buffer for " + path);
        }
        ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<uint8_t> read_buffer;
        for (const auto & entry : tensors_) {
            ggml_tensor * tensor = entry.second;
            const size_t nbytes = ggml_nbytes(tensor);
            const auto offset_it = offsets_.find(entry.first);
            if (offset_it == offsets_.end()) {
                return fail(error, "GGUF frontend: missing offset for tensor " + entry.first);
            }
            file.seekg(static_cast<std::streamoff>(offset_it->second), std::ios::beg);
            if (!file.good()) {
                return fail(error, "GGUF frontend: failed to seek tensor " + entry.first);
            }
            read_buffer.resize(nbytes);
            if (nbytes > 0 &&
                (!file.read(reinterpret_cast<char *>(read_buffer.data()),
                            static_cast<std::streamsize>(nbytes)) ||
                 file.gcount() != static_cast<std::streamsize>(nbytes))) {
                return fail(error, "GGUF frontend: failed to read tensor " + entry.first);
            }
            ggml_backend_tensor_set(tensor, read_buffer.data(), 0, nbytes);
        }
        return true;
    }

    void reset() {
        buffer_.reset();
        ctx_data_.reset();
        ctx_meta_.reset();
        ctx_gguf_.reset();
        tensors_.clear();
        offsets_.clear();
    }

    ggml_tensor * get(const std::string & name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            std::fprintf(stderr, "GGUF frontend: missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return it->second;
    }

  private:
    static bool fail(std::string * error, const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    }

    gguf_context_ptr ctx_gguf_;
    ggml_context_ptr ctx_meta_;
    ggml_context_ptr ctx_data_;
    ggml_backend_buffer_ptr buffer_;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    std::unordered_map<std::string, size_t> offsets_;
};

static ggml_tensor * scalar_tensor(ggml_context * ctx, float value) {
    return ggml_arange(ctx, value, value + 1.0f, 1.0f);
}

static ggml_tensor * linear(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * weight_onnx,
                            ggml_tensor * bias) {
    if (!ctx || !x || !weight_onnx) {
        return nullptr;
    }
    // ONNX MatMul stores [in, out], while ggml_mul_mat consumes [in, out]
    // as [K, M]. GGUF exposes the reversed logical dimensions, so transpose
    // the 2-D view before multiplication.
    ggml_tensor * weight = ggml_cont(ctx, ggml_transpose(ctx, weight_onnx));
    ggml_tensor * result = ggml_mul_mat(ctx, weight, x);
    if (bias) {
        result = ggml_add(ctx, result, bias);
    }
    return result;
}

static ggml_tensor * conv1d(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * weight,
                            ggml_tensor * bias,
                            int stride,
                            int padding,
                            int dilation) {
    if (!ctx || !x || !weight) {
        return nullptr;
    }
    // ggml_conv_1d consumes [time, channels, batch] and returns
    // [time, output_channels, batch]. The frontend graph uses
    // [channels, time, batch] everywhere else.
    ggml_tensor * x_tcb = ggml_cast(ctx, ggml_permute(ctx, x, 1, 0, 2, 3), GGML_TYPE_F16);
    ggml_tensor * conv_weight = weight->type == GGML_TYPE_F16 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F16);
    ggml_tensor * result = ggml_conv_1d(ctx, conv_weight, x_tcb, stride, padding, dilation);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
    if (bias) {
        result = ggml_add(ctx, result, bias);
    }
    return result;
}

static ggml_tensor * conv1d_depthwise(ggml_context * ctx,
                                      ggml_tensor * x,
                                      ggml_tensor * weight,
                                      int stride,
                                      int padding,
                                      int dilation) {
    if (!ctx || !x || !weight) {
        return nullptr;
    }
    ggml_tensor * x_tcb = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * conv_weight = weight->type == GGML_TYPE_F16 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F16);
    ggml_tensor * result = ggml_conv_1d_dw(ctx, conv_weight, x_tcb, stride, padding, dilation);
    return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

static ggml_tensor * conv2d(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * weight,
                            ggml_tensor * bias,
                            int stride_w,
                            int stride_h,
                            int padding_w,
                            int padding_h) {
    if (!ctx || !x || !weight) {
        return nullptr;
    }
    ggml_tensor * conv_weight = weight->type == GGML_TYPE_F16 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F16);
    ggml_tensor * raw = ggml_conv_2d(ctx, conv_weight, x, stride_w, stride_h,
                                     padding_w, padding_h, 1, 1);
    if (!bias) {
        return raw;
    }
    std::fprintf(stderr, "conv2d before bias reshape\n");
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    std::fprintf(stderr, "conv2d before bias add\n");
    return ggml_add(ctx, raw, bias_4d);
}

static ggml_tensor * layer_norm(ggml_context * ctx,
                                ggml_tensor * x,
                                ggml_tensor * weight,
                                ggml_tensor * bias,
                                float epsilon = 1e-5f) {
    ggml_tensor * result = ggml_norm(ctx, x, epsilon);
    result = ggml_mul(ctx, result, weight);
    return bias ? ggml_add(ctx, result, bias) : result;
}

static ggml_tensor * relu(ggml_context * ctx, ggml_tensor * x) {
    return x ? ggml_relu(ctx, x) : nullptr;
}

static ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x) {
    return x ? ggml_gelu_erf(ctx, x) : nullptr;
}

static ggml_tensor * make_rotary_factor(ggml_context * ctx, int64_t dimension, int64_t time) {
    ggml_tensor * factor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dimension, time);
    ggml_set_input(factor);
    return factor;
}

static ggml_tensor * rotate_half(ggml_context * ctx, ggml_tensor * x) {
    const int64_t dimension = x->ne[0];
    const int64_t half = dimension / 2;
    ggml_tensor * left = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2],
                                      x->nb[1], x->nb[2], 0);
    ggml_tensor * right = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2],
                                       x->nb[1], x->nb[2], half * x->nb[0]);
    return ggml_concat(ctx, ggml_neg(ctx, right), left, 0);
}

static ggml_tensor * rotary_apply(ggml_context * ctx,
                                  ggml_tensor * x,
                                  ggml_tensor * cos_factor,
                                  ggml_tensor * sin_factor) {
    cos_factor = ggml_repeat(ctx, ggml_reshape_3d(ctx, cos_factor, cos_factor->ne[0], 1, cos_factor->ne[1]), x);
    sin_factor = ggml_repeat(ctx, ggml_reshape_3d(ctx, sin_factor, sin_factor->ne[0], 1, sin_factor->ne[1]), x);
    return ggml_add(ctx,
                    ggml_mul(ctx, x, cos_factor),
                    ggml_mul(ctx, rotate_half(ctx, x), sin_factor));
}

struct S3BlockNames {
    const char * fsmn;
    const char * attn_ln_w;
    const char * attn_ln_b;
    const char * q_w;
    const char * q_b;
    const char * k_w;
    const char * v_w;
    const char * v_b;
    const char * out_w;
    const char * out_b;
    const char * mlp_ln_w;
    const char * mlp_ln_b;
    const char * mlp1_w;
    const char * mlp1_b;
    const char * mlp2_w;
    const char * mlp2_b;
};

static const std::array<S3BlockNames, 6> & s3_block_names() {
    static const std::array<S3BlockNames, 6> names = {{
        {"blocks.0.attn.fsmn_block.weight", "onnx::LayerNormalization_2224", "onnx::LayerNormalization_2225",
         "onnx::MatMul_2228", "onnx::Add_2227", "onnx::MatMul_2230", "onnx::MatMul_2233", "onnx::Add_2232",
         "onnx::MatMul_2267", "onnx::Add_2266", "onnx::LayerNormalization_2268", "onnx::LayerNormalization_2269",
         "onnx::MatMul_2272", "onnx::Add_2271", "onnx::MatMul_2275", "onnx::Add_2274"},
        {"blocks.1.attn.fsmn_block.weight", "onnx::LayerNormalization_2276", "onnx::LayerNormalization_2277",
         "onnx::MatMul_2280", "onnx::Add_2279", "onnx::MatMul_2282", "onnx::MatMul_2285", "onnx::Add_2284",
         "onnx::MatMul_2319", "onnx::Add_2318", "onnx::LayerNormalization_2320", "onnx::LayerNormalization_2321",
         "onnx::MatMul_2324", "onnx::Add_2323", "onnx::MatMul_2327", "onnx::Add_2326"},
        {"blocks.2.attn.fsmn_block.weight", "onnx::LayerNormalization_2328", "onnx::LayerNormalization_2329",
         "onnx::MatMul_2332", "onnx::Add_2331", "onnx::MatMul_2334", "onnx::MatMul_2337", "onnx::Add_2336",
         "onnx::MatMul_2371", "onnx::Add_2370", "onnx::LayerNormalization_2372", "onnx::LayerNormalization_2373",
         "onnx::MatMul_2376", "onnx::Add_2375", "onnx::MatMul_2379", "onnx::Add_2378"},
        {"blocks.3.attn.fsmn_block.weight", "onnx::LayerNormalization_2380", "onnx::LayerNormalization_2381",
         "onnx::MatMul_2384", "onnx::Add_2383", "onnx::MatMul_2386", "onnx::MatMul_2389", "onnx::Add_2388",
         "onnx::MatMul_2423", "onnx::Add_2422", "onnx::LayerNormalization_2424", "onnx::LayerNormalization_2425",
         "onnx::MatMul_2428", "onnx::Add_2427", "onnx::MatMul_2431", "onnx::Add_2430"},
        {"blocks.4.attn.fsmn_block.weight", "onnx::LayerNormalization_2432", "onnx::LayerNormalization_2433",
         "onnx::MatMul_2436", "onnx::Add_2435", "onnx::MatMul_2438", "onnx::MatMul_2441", "onnx::Add_2440",
         "onnx::MatMul_2475", "onnx::Add_2474", "onnx::LayerNormalization_2476", "onnx::LayerNormalization_2477",
         "onnx::MatMul_2480", "onnx::Add_2479", "onnx::MatMul_2483", "onnx::Add_2482"},
        {"blocks.5.attn.fsmn_block.weight", "onnx::LayerNormalization_2484", "onnx::LayerNormalization_2485",
         "onnx::MatMul_2488", "onnx::Add_2487", "onnx::MatMul_2490", "onnx::MatMul_2493", "onnx::Add_2492",
         "onnx::MatMul_2527", "onnx::Add_2526", "onnx::LayerNormalization_2528", "onnx::LayerNormalization_2529",
         "onnx::MatMul_2532", "onnx::Add_2531", "onnx::MatMul_2535", "onnx::Add_2534"},
    }};
    return names;
}

static ggml_tensor * flatten_attention_qk(ggml_context * ctx,
                                          ggml_tensor *  heads,
                                          int            head_dim,
                                          int64_t        time,
                                          int64_t        batch,
                                          int            head_count) {
    ggml_tensor * permuted = ggml_permute(ctx, heads, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, head_dim, time, static_cast<int64_t>(head_count) * batch);
}

static ggml_tensor * flatten_attention_v(ggml_context * ctx,
                                         ggml_tensor *  heads,
                                         int            head_dim,
                                         int64_t        time,
                                         int64_t        batch,
                                         int            head_count) {
    ggml_tensor * flat_qk = flatten_attention_qk(ctx, heads, head_dim, time, batch, head_count);
    ggml_tensor * permuted = ggml_permute(ctx, flat_qk, 1, 0, 2, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, time, head_dim, static_cast<int64_t>(head_count) * batch);
}

static ggml_tensor * merge_attention_heads(ggml_context * ctx,
                                            ggml_tensor *  heads_flat,
                                            int            head_dim,
                                            int            head_count,
                                            int64_t        time,
                                            int64_t        batch) {
    ggml_tensor * view4d = ggml_reshape_4d(ctx, heads_flat, head_dim, time, head_count, batch);
    ggml_tensor * permuted = ggml_permute(ctx, view4d, 0, 2, 1, 3);
    ggml_tensor * contiguous = ggml_cont(ctx, permuted);
    return ggml_reshape_3d(ctx, contiguous, static_cast<int64_t>(head_dim) * head_count, time, batch);
}

static ggml_tensor * add_bias_time_channels(ggml_context * ctx,
                                            ggml_tensor * x,
                                            ggml_tensor * bias) {
    if (!bias) {
        return x;
    }
    ggml_tensor * bias_tc = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, x, ggml_repeat(ctx, bias_tc, x));
}

static ggml_tensor * conv1d_time_major(ggml_context * ctx,
                                       ggml_tensor * x,
                                       ggml_tensor * weight,
                                       ggml_tensor * bias,
                                       int stride,
                                       int padding,
                                       int dilation) {
    if (!ctx || !x || !weight) {
        return nullptr;
    }
    ggml_tensor * conv_weight = weight->type == GGML_TYPE_F16 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F16);
    ggml_tensor * result = ggml_conv_1d(ctx, conv_weight, x, stride, padding, dilation);
    if (bias) {
        result = ggml_add(ctx, result, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    }
    return result;
}

static ggml_tensor * batch_norm_time_major(ggml_context * ctx,
                                           ggml_tensor * x,
                                           ggml_tensor * weight,
                                           ggml_tensor * bias,
                                           ggml_tensor * running_mean,
                                           ggml_tensor * running_var,
                                           float epsilon = 1e-5f) {
    if (!ctx || !x || !weight || !bias || !running_mean || !running_var) {
        return nullptr;
    }
    const int64_t channels = weight->ne[0];
    auto broadcast = [&](ggml_tensor * value) {
        return ggml_reshape_2d(ctx, value, 1, channels);
    };
    ggml_tensor * normalized = ggml_div(
        ctx,
        ggml_sub(ctx, x, broadcast(running_mean)),
        ggml_sqrt(ctx, ggml_add(ctx, broadcast(running_var), scalar_tensor(ctx, epsilon))));
    return ggml_add(ctx, ggml_mul(ctx, normalized, broadcast(weight)), broadcast(bias));
}

static ggml_tensor * batch_norm_time_major_no_affine(ggml_context * ctx,
                                                      ggml_tensor * x,
                                                      ggml_tensor * running_mean,
                                                      ggml_tensor * running_var,
                                                      float epsilon = 1e-5f) {
    if (!ctx || !x || !running_mean || !running_var) {
        return nullptr;
    }
    const int64_t channels = running_mean->ne[0];
    auto broadcast = [&](ggml_tensor * value) {
        return ggml_reshape_2d(ctx, value, 1, channels);
    };
    return ggml_div(
        ctx,
        ggml_sub(ctx, x, broadcast(running_mean)),
        ggml_sqrt(ctx, ggml_add(ctx, broadcast(running_var), scalar_tensor(ctx, epsilon))));
}

static ggml_tensor * segment_average_time_major(ggml_context * ctx,
                                                ggml_tensor * x,
                                                int64_t segment_length) {
    std::vector<ggml_tensor *> segments;
    for (int64_t start = 0; start < x->ne[0]; start += segment_length) {
        const int64_t length = std::min(segment_length, x->ne[0] - start);
        ggml_tensor * segment = ggml_view_2d(ctx, x, length, x->ne[1], x->nb[1],
                                             static_cast<size_t>(start) * x->nb[0]);
        ggml_tensor * mean = ggml_scale(
            ctx, ggml_sum_rows(ctx, segment), 1.0f / static_cast<float>(segment_length));
        segments.push_back(ggml_repeat(ctx, mean, segment));
    }
    while (segments.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((segments.size() + 1) / 2);
        for (size_t i = 0; i < segments.size(); i += 2) {
            next.push_back(i + 1 < segments.size() ? ggml_concat(ctx, segments[i], segments[i + 1], 0)
                                                    : segments[i]);
        }
        segments.swap(next);
    }
    return segments.empty() ? nullptr : segments.front();
}

static ggml_tensor * campplus_dense_layer(ggml_context * ctx,
                                          ggml_tensor * x,
                                          GGUFWeightStore & weights,
                                          int block,
                                          int layer,
                                          const char * linear_weight_name,
                                          const char * linear_bias_name,
                                          int dilation) {
    const std::string prefix = "xvector.block" + std::to_string(block) + ".tdnnd" + std::to_string(layer);
    ggml_tensor * hidden = batch_norm_time_major(
        ctx, x,
        weights.get(prefix + ".nonlinear1.batchnorm.weight"),
        weights.get(prefix + ".nonlinear1.batchnorm.bias"),
        weights.get(prefix + ".nonlinear1.batchnorm.running_mean"),
        weights.get(prefix + ".nonlinear1.batchnorm.running_var"));
    hidden = relu(ctx, hidden);
    hidden = conv1d_time_major(ctx, hidden, weights.get(linear_weight_name),
                                  weights.get(linear_bias_name), 1, 0, 1);
    hidden = relu(ctx, hidden);

    const std::string local_name = prefix + ".cam_layer.linear_local.weight";
    ggml_tensor * local = conv1d_time_major(ctx, hidden, weights.get(local_name), nullptr, 1, dilation, dilation);

    // CAM combines the global mean with a 100-frame segment mean, then
    // predicts a channel gate for the local convolution output.
    ggml_tensor * segment_mean = segment_average_time_major(ctx, hidden, 100);
    ggml_tensor * global_mean = ggml_mean(ctx, hidden);
    ggml_tensor * context = ggml_add(ctx, segment_mean, global_mean);
    context = relu(ctx, conv1d_time_major(
        ctx, context,
        weights.get(prefix + ".cam_layer.linear1.weight"),
        weights.get(prefix + ".cam_layer.linear1.bias"), 1, 0, 1));
    context = ggml_sigmoid(ctx, conv1d_time_major(
        ctx, context,
        weights.get(prefix + ".cam_layer.linear2.weight"),
        weights.get(prefix + ".cam_layer.linear2.bias"), 1, 0, 1));
    return ggml_mul(ctx, local, context);
}

static ggml_tensor * campplus_residual_block(ggml_context * ctx,
                                             ggml_tensor * x,
                                             GGUFWeightStore & weights,
                                             const char * conv1_weight,
                                             const char * conv1_bias,
                                             const char * conv2_weight,
                                             const char * conv2_bias,
                                             const char * shortcut_weight,
                                             const char * shortcut_bias,
                                             int stride) {
    const int64_t input_channels = x->ne[2];
    const int64_t output_channels = weights.get(conv1_bias)->ne[0];
    ggml_tensor * y = conv2d(ctx, x, weights.get(conv1_weight), weights.get(conv1_bias),
                             1, stride, 1, 1);
    y = relu(ctx, y);
    y = conv2d(ctx, y, weights.get(conv2_weight), weights.get(conv2_bias),
               1, 1, 1, 1);
    ggml_tensor * shortcut = x;
    if (stride != 1 || input_channels != output_channels) {
        shortcut = conv2d(ctx, x, weights.get(shortcut_weight), weights.get(shortcut_bias),
                           1, stride, 0, 0);
    }
    return relu(ctx, ggml_add(ctx, y, shortcut));
}

static ggml_tensor * build_campplus_fcm(ggml_context * ctx,
                                        ggml_tensor * input,
                                        GGUFWeightStore & weights) {
    ggml_tensor * x = conv2d(ctx, input,
                                      weights.get("onnx::Conv_4423"),
                                      weights.get("onnx::Conv_4424"), 1, 1, 1, 1);
    x = relu(ctx, x);
    x = campplus_residual_block(ctx, x, weights,
                                "onnx::Conv_4426", "onnx::Conv_4427",
                                "onnx::Conv_4429", "onnx::Conv_4430",
                                "onnx::Conv_4432", "onnx::Conv_4433", 2);
    x = campplus_residual_block(ctx, x, weights,
                                "onnx::Conv_4435", "onnx::Conv_4436",
                                "onnx::Conv_4438", "onnx::Conv_4439",
                                nullptr, nullptr, 1);
    x = campplus_residual_block(ctx, x, weights,
                                "onnx::Conv_4441", "onnx::Conv_4442",
                                "onnx::Conv_4444", "onnx::Conv_4445",
                                "onnx::Conv_4447", "onnx::Conv_4448", 2);
    x = campplus_residual_block(ctx, x, weights,
                                "onnx::Conv_4450", "onnx::Conv_4451",
                                "onnx::Conv_4453", "onnx::Conv_4454",
                                nullptr, nullptr, 1);
    x = relu(ctx, conv2d(ctx, x,
                         weights.get("onnx::Conv_4456"),
                         weights.get("onnx::Conv_4457"), 1, 2, 1, 1));
    // Reinterpret the convolution buffer as time-major [time, channels].
    // Its physical order is already [channel, frequency, time], which is
    // exactly the contiguous layout required by a ggml [time, channels]
    // tensor.
    ggml_tensor * output = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    return output;
}

static ggml_tensor * build_campplus_graph(ggml_context * ctx,
                                          ggml_tensor * input,
                                          GGUFWeightStore & weights) {
    ggml_tensor * x = build_campplus_fcm(ctx, input, weights);
    x = relu(ctx, conv1d_time_major(ctx, x,
                                       weights.get("onnx::Conv_4459"),
                                       weights.get("onnx::Conv_4460"), 2, 2, 1));
    constexpr std::array<int, 3> layer_counts = { 12, 24, 16 };
    constexpr std::array<int, 3> linear_bases = { 4462, 4498, 4570 };
    constexpr std::array<int, 3> dilations = { 1, 2, 2 };
    constexpr std::array<int, 3> block_ids = { 1, 2, 3 };

    for (size_t block_index = 0; block_index < layer_counts.size(); ++block_index) {
        const int block = block_ids[block_index];
        for (int layer = 1; layer <= layer_counts[block_index]; ++layer) {
            const int weight_id = linear_bases[block_index] + 3 * (layer - 1);
            const std::string linear_weight = "onnx::Conv_" + std::to_string(weight_id);
            const std::string linear_bias = "onnx::Conv_" + std::to_string(weight_id + 1);
            ggml_tensor * dense = campplus_dense_layer(
                ctx, x, weights, block, layer, linear_weight.c_str(),
                linear_bias.c_str(), dilations[block_index]);
            x = ggml_concat(ctx, x, dense, 1);
        }

        const std::string transit = "xvector.transit" + std::to_string(block);
        x = batch_norm_time_major(
            ctx, x,
            weights.get(transit + ".nonlinear.batchnorm.weight"),
            weights.get(transit + ".nonlinear.batchnorm.bias"),
            weights.get(transit + ".nonlinear.batchnorm.running_mean"),
            weights.get(transit + ".nonlinear.batchnorm.running_var"));
        x = relu(ctx, x);
        const std::string transit_weight =
            block == 3 ? "onnx::Conv_4618"
                       : ("xvector.transit" + std::to_string(block) + ".linear.weight");
        const char * transit_bias_name = block == 3 ? "onnx::Conv_4619" : nullptr;
        x = conv1d_time_major(ctx, x, weights.get(transit_weight),
                                 transit_bias_name ? weights.get(transit_bias_name) : nullptr,
                                 1, 0, 1);
    }

    x = relu(ctx, x);

    const int64_t time = x->ne[0];
    ggml_tensor * mean = ggml_mean(ctx, x);
    ggml_tensor * centered = ggml_sub(ctx, x, mean);
    ggml_tensor * variance = ggml_mean(ctx, ggml_sqr(ctx, centered));
    variance = ggml_scale(ctx, variance, static_cast<float>(time) /
                                         static_cast<float>(std::max<int64_t>(1, time - 1)));
    ggml_tensor * stdev = ggml_sqrt(ctx, variance);
    ggml_tensor * statistics = ggml_concat(ctx, mean, stdev, 1);

    ggml_tensor * dense = conv1d_time_major(
        ctx, statistics, weights.get("xvector.dense.linear.weight"), nullptr, 1, 0, 1);
    return batch_norm_time_major_no_affine(
        ctx, dense,
        weights.get("xvector.dense.nonlinear.batchnorm.running_mean"),
        weights.get("xvector.dense.nonlinear.batchnorm.running_var"));
}

class FrontendGGUFModels {
  public:
    bool load(const std::string & speech_path, const std::string & campplus_path, std::string * error) {
        backend_.reset(ggml_backend_cpu_init());
        if (!backend_) {
            return fail(error, "GGUF frontend: failed to initialize CPU backend");
        }
        if (!speech_.load(speech_path, backend_.get(), error)) {
            return false;
        }
        if (!campplus_.load(campplus_path, backend_.get(), error)) {
            return false;
        }
        return true;
    }

    bool run_speech(const AudioFeatures & features,
                    int num_threads,
                    std::vector<int32_t> & tokens,
                    std::string * error);

    bool run_campplus(const AudioFeatures & features,
                      int num_threads,
                      std::vector<float> & embedding,
                      std::string * error);

  private:
    static bool fail(std::string * error, const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    }

    ggml_backend_ptr backend_;
    GGUFWeightStore speech_;
    GGUFWeightStore campplus_;
    std::mutex run_mutex_;
};

static std::shared_ptr<FrontendGGUFModels> get_frontend_models(
        const std::string & speech_path, const std::string & campplus_path, std::string * error) {
    static std::mutex cache_mutex;
    static std::shared_ptr<FrontendGGUFModels> cached;
    static std::string cached_speech_path;
    static std::string cached_campplus_path;

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cached && cached_speech_path == speech_path && cached_campplus_path == campplus_path) {
        return cached;
    }
    auto candidate = std::make_shared<FrontendGGUFModels>();
    if (!candidate->load(speech_path, campplus_path, error)) {
        return nullptr;
    }
    cached = std::move(candidate);
    cached_speech_path = speech_path;
    cached_campplus_path = campplus_path;
    return cached;
}

static ggml_tensor * s3_attention(ggml_context * ctx,
                                  ggml_tensor * x,
                                  const S3BlockNames & names,
                                  GGUFWeightStore & weights,
                                  ggml_tensor * cos_factor,
                                  ggml_tensor * sin_factor) {
    constexpr int64_t n_head = 20;
    constexpr int64_t head_dim = 64;
    const int64_t time = x->ne[1];
    ggml_tensor * normalized = layer_norm(ctx, x, weights.get(names.attn_ln_w), weights.get(names.attn_ln_b));
    ggml_tensor * q = linear(ctx, normalized, weights.get(names.q_w), weights.get(names.q_b));
    ggml_tensor * k = linear(ctx, normalized, weights.get(names.k_w), nullptr);
    ggml_tensor * v = linear(ctx, normalized, weights.get(names.v_w), weights.get(names.v_b));
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, time);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, time);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, time);
    q = rotary_apply(ctx, q, cos_factor, sin_factor);
    k = rotary_apply(ctx, k, cos_factor, sin_factor);

    ggml_tensor * q_flat = flatten_attention_qk(ctx, q, head_dim, time, 1, n_head);
    ggml_tensor * k_flat = flatten_attention_qk(ctx, k, head_dim, time, 1, n_head);
    ggml_tensor * scores = ggml_mul_mat(ctx, k_flat, q_flat);  // [T, T, H]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, std::pow(static_cast<float>(head_dim), -0.5f));
    ggml_tensor * probabilities = ggml_soft_max(ctx, scores);

    ggml_tensor * v_flat = flatten_attention_v(ctx, v, head_dim, time, 1, n_head);
    ggml_tensor * context = ggml_mul_mat(ctx, v_flat, probabilities); // [D, T, H]
    context = merge_attention_heads(ctx, context, head_dim, n_head, time, 1);
    ggml_tensor * projected = linear(ctx, context, weights.get(names.out_w), weights.get(names.out_b));

    ggml_tensor * v_for_fsmn = linear(ctx, normalized, weights.get(names.v_w), weights.get(names.v_b));
    ggml_tensor * fsmn_input = v_for_fsmn;
    v_for_fsmn = ggml_pad_ext(ctx, v_for_fsmn, 0, 0, 15, 15, 0, 0, 0, 0);
    ggml_tensor * fsmn = conv1d_depthwise(ctx, v_for_fsmn, weights.get(names.fsmn), 1, 0, 1);
    fsmn = ggml_add(ctx, fsmn, fsmn_input);
    return ggml_add(ctx, projected, fsmn);
}

static ggml_tensor * build_s3_graph(ggml_context * ctx,
                                     ggml_tensor * input,
                                     GGUFWeightStore & weights,
                                     ggml_tensor * cos_factor,
                                     ggml_tensor * sin_factor) {
    ggml_tensor * x = conv1d(ctx, input, weights.get("onnx::Conv_2216"),
                             weights.get("onnx::Conv_2217"), 2, 1, 1);
    x = gelu(ctx, x);
    x = conv1d(ctx, x, weights.get("onnx::Conv_2218"),
               weights.get("onnx::Conv_2219"), 2, 1, 1);
    x = gelu(ctx, x);
    x = ggml_cont_2d(ctx, x, x->ne[0], x->ne[1]);

    for (const auto & block : s3_block_names()) {
        ggml_tensor * residual = x;
        ggml_tensor * attention = s3_attention(ctx, x, block, weights, cos_factor, sin_factor);
        x = ggml_add(ctx, residual, attention);
        residual = x;
        ggml_tensor * mlp_input = layer_norm(ctx, x, weights.get(block.mlp_ln_w), weights.get(block.mlp_ln_b));
        mlp_input = linear(ctx, mlp_input, weights.get(block.mlp1_w), weights.get(block.mlp1_b));
        mlp_input = gelu(ctx, mlp_input);
        mlp_input = linear(ctx, mlp_input, weights.get(block.mlp2_w), weights.get(block.mlp2_b));
        x = ggml_add(ctx, residual, mlp_input);
    }
    ggml_tensor * output = linear(ctx, x, weights.get("onnx::MatMul_2536"),
                                  weights.get("quantizer.project_in.bias"));
    return output;
}

}  // namespace

bool FrontendGGUFModels::run_speech(const AudioFeatures & features,
                                    int num_threads,
                                    std::vector<int32_t> & tokens,
                                    std::string * error) {
    std::lock_guard<std::mutex> lock(run_mutex_);
    tokens.clear();
    if (features.channels != 128 || features.frames <= 0 ||
        features.values.size() != static_cast<size_t>(features.channels) * features.frames) {
        return fail(error, "GGUF speech tokenizer input must have shape [128, T]");
    }

    ggml_backend_cpu_set_n_threads(backend_.get(), std::max(1, num_threads));
    ggml_init_params params = {
        /*.mem_size   =*/ 1024ull * 1024ull * 1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail(error, "GGUF speech tokenizer: failed to create compute context");
    }
    const int64_t time = features.frames;
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, features.channels, time);
    ggml_set_input(input);
    const int64_t token_time = ((time + 1) / 2 + 1) / 2;
    ggml_tensor * cos_factor = make_rotary_factor(ctx.get(), 64, token_time);
    ggml_tensor * sin_factor = make_rotary_factor(ctx.get(), 64, token_time);
    ggml_tensor * output = build_s3_graph(ctx.get(), input, speech_, cos_factor, sin_factor);
    if (!output) {
        return fail(error, "GGUF speech tokenizer: failed to build graph");
    }
    const int64_t output_time = output->ne[1];
    // The graph output is [8, T]. Quantization uses tanh -> round + 1 and
    // base-3 positional encoding over the eight scalar code dimensions.
    ggml_tensor * quantized = ggml_tanh(ctx.get(), output);
    quantized = ggml_scale(ctx.get(), quantized, 0.9990000128746033f);
    quantized = ggml_add(ctx.get(), quantized, scalar_tensor(ctx.get(), 1.0f));
    quantized = ggml_round(ctx.get(), quantized);
    ggml_tensor * powers = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_set_input(powers);
    std::array<float, 8> power_values{};
    power_values[0] = 1.0f;
    for (size_t i = 1; i < power_values.size(); ++i) {
        power_values[i] = power_values[i - 1] * 3.0f;
    }
    ggml_tensor * code_values = ggml_sum_rows(ctx.get(), ggml_mul(ctx.get(), quantized, powers));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE * 16, false);
    ggml_build_forward_expand(graph, code_values);
    ggml_set_output(code_values);

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_get_default_buffer_type(backend_.get())));
    if (!buffer) {
        return fail(error, "GGUF speech tokenizer: failed to allocate compute buffer");
    }
    std::vector<float> input_values(features.values.size());
    for (int64_t frame = 0; frame < time; ++frame) {
        for (int64_t channel = 0; channel < features.channels; ++channel) {
            input_values[static_cast<size_t>(frame) * features.channels + channel] =
                features.values[static_cast<size_t>(channel) * time + frame];
        }
    }
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    std::array<float, 8> powers_host = power_values;
    ggml_backend_tensor_set(powers, powers_host.data(), 0, powers_host.size() * sizeof(float));
    // make_rotary_factor wrote host values before allocation; copy them again
    // through the backend API because no_alloc contexts do not retain data.
    std::vector<float> cos_values(static_cast<size_t>(64 * cos_factor->ne[1]));
    std::vector<float> sin_values(static_cast<size_t>(64 * sin_factor->ne[1]));
    for (int64_t t = 0; t < cos_factor->ne[1]; ++t) {
        for (int64_t d = 0; d < 64; ++d) {
            const int64_t half = d % 32;
            const float frequency = std::pow(10000.0f, -2.0f * static_cast<float>(half) / 64.0f);
            cos_values[static_cast<size_t>(d + 64 * t)] = std::cos(static_cast<float>(t) * frequency);
            sin_values[static_cast<size_t>(d + 64 * t)] = std::sin(static_cast<float>(t) * frequency);
        }
    }
    ggml_backend_tensor_set(cos_factor, cos_values.data(), 0, cos_values.size() * sizeof(float));
    ggml_backend_tensor_set(sin_factor, sin_values.data(), 0, sin_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend_.get(), graph) != GGML_STATUS_SUCCESS) {
        return fail(error, "GGUF speech tokenizer: graph compute failed");
    }

    std::vector<float> code_host(static_cast<size_t>(code_values->ne[0] * code_values->ne[1]));
    ggml_backend_tensor_get(code_values, code_host.data(), 0, code_host.size() * sizeof(float));
    tokens.resize(static_cast<size_t>(output_time));
    for (int64_t t = 0; t < output_time; ++t) {
        const float value = code_host[static_cast<size_t>(t)];
        if (!std::isfinite(value) || value < 0.0f || value > 6560.0f) {
            return fail(error, "GGUF speech tokenizer: invalid quantized token");
        }
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>(value);
    }
    return true;
}

bool FrontendGGUFModels::run_campplus(const AudioFeatures & features,
                                      int num_threads,
                                      std::vector<float> & embedding,
                                      std::string * error) {
    embedding.clear();
    if (features.channels != 80 || features.frames <= 0 ||
        features.values.size() != static_cast<size_t>(features.channels) * features.frames) {
        return fail(error, "GGUF CampPlus input must have shape [T, 80]");
    }

    ggml_backend_cpu_set_n_threads(backend_.get(), std::max(1, num_threads));
    ggml_init_params params = {
        /*.mem_size   =*/ 1024ull * 1024ull * 1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail(error, "GGUF CampPlus: failed to create compute context");
    }

    const int64_t time = features.frames;
    // GGML conv2d uses [time, frequency, channels, batch], while the public
    // AudioFeatures buffer is stored as [frame, frequency].
    ggml_tensor * input = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, time, 80, 1, 1);
    ggml_set_input(input);
    ggml_tensor * output = build_campplus_graph(ctx.get(), input, campplus_);
    if (!output || output->ne[0] != 1 || output->ne[1] != 192) {
        return fail(error, "GGUF CampPlus: graph produced an invalid output shape");
    }
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE * 512, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_get_default_buffer_type(backend_.get())));
    if (!buffer) {
        return fail(error, "GGUF CampPlus: failed to allocate compute buffer");
    }

    std::vector<float> input_values(static_cast<size_t>(time) * 80);
    for (int64_t feature = 0; feature < 80; ++feature) {
        for (int64_t frame = 0; frame < time; ++frame) {
            input_values[static_cast<size_t>(frame) + static_cast<size_t>(time) * feature] =
                features.values[static_cast<size_t>(frame) * 80 + static_cast<size_t>(feature)];
        }
    }
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend_.get(), graph) != GGML_STATUS_SUCCESS) {
        return fail(error, "GGUF CampPlus: graph compute failed");
    }

    embedding.resize(192);
    ggml_backend_tensor_get(output, embedding.data(), 0, embedding.size() * sizeof(float));
    for (const float value : embedding) {
        if (!std::isfinite(value)) {
            embedding.clear();
            return fail(error, "GGUF CampPlus: output contains a non-finite value");
        }
    }
    return true;
}

bool prepare_prompt_bundle_gguf(const AudioFeatures & speech_features,
                                const AudioFeatures & campplus_features,
                                const std::string &  speech_model_path,
                                const std::string &  campplus_model_path,
                                int                    num_threads,
                                std::vector<int32_t> & speech_tokens,
                                std::vector<float> &   speaker_embedding,
                                std::string *           error) {
    auto models = get_frontend_models(speech_model_path, campplus_model_path, error);
    if (!models) {
        return false;
    }
    if (!models->run_speech(speech_features, num_threads, speech_tokens, error)) {
        return false;
    }
    return models->run_campplus(campplus_features, num_threads, speaker_embedding, error);
}

}  // namespace flow
}  // namespace omni
