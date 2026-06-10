#include "voxcpm2_fsq.h"

#include "log.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace {

static bool gguf_get_u32_opt(gguf_context * ctx, const char * key, int & dst) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    dst = static_cast<int>(gguf_get_val_u32(ctx, id));
    return true;
}

static bool gguf_get_f32_opt(gguf_context * ctx, const char * key, float & dst) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    dst = gguf_get_val_f32(ctx, id);
    return true;
}

static ggml_tensor * reshape_bias_for(ggml_context * ctx, ggml_tensor * bias, ggml_tensor * output) {
    const int n_dims = ggml_n_dims(output);
    if (n_dims <= 1) {
        return bias;
    }
    if (n_dims == 2) {
        return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    }
    if (n_dims == 3) {
        return ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1);
    }
    return ggml_reshape_4d(ctx, bias, bias->ne[0], 1, 1, 1);
}

static ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * output, ggml_tensor * bias) {
    if (!bias) {
        return output;
    }
    ggml_tensor * bias_view = reshape_bias_for(ctx, bias, output);
    if (bias_view->type != output->type) {
        bias_view = ggml_cast(ctx, bias_view, output->type);
    }
    return ggml_add(ctx, output, bias_view);
}

static ggml_tensor * round_limited(ggml_context * ctx, ggml_tensor * x, float max_abs) {
    const int max_level = std::max(0, static_cast<int>(std::floor(max_abs + 0.5f)));
    if (max_level == 0) {
        return ggml_scale(ctx, x, 0.0f);
    }

    ggml_tensor * neg_x = ggml_scale(ctx, x, -1.0f);
    ggml_tensor * pos_count = nullptr;
    ggml_tensor * neg_count = nullptr;

    // Current llama.cpp-omni ggml does not expose ggml_round. Since FSQ rounds
    // tanh(x) * scale, its range is bounded, so a finite step-sum is exact for
    // roundf-style inference quantization over [-scale, scale].
    constexpr float eps = 1.0e-6f;
    for (int level = 1; level <= max_level; ++level) {
        const float threshold = static_cast<float>(level) - 0.5f - eps;
        ggml_tensor * pos = ggml_step(ctx, ggml_scale_bias(ctx, x, 1.0f, -threshold));
        ggml_tensor * neg = ggml_step(ctx, ggml_scale_bias(ctx, neg_x, 1.0f, -threshold));
        pos_count = pos_count ? ggml_add(ctx, pos_count, pos) : pos;
        neg_count = neg_count ? ggml_add(ctx, neg_count, neg) : neg;
    }

    return ggml_sub(ctx, pos_count, neg_count);
}

} // namespace

FSQModule::~FSQModule() {
    free();
}

void FSQModule::free() {
    if (weight_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        weight_buffer = nullptr;
    }
    if (ctx_weights) {
        ggml_free(ctx_weights);
        ctx_weights = nullptr;
    }
    if (ctx_meta) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (ctx_gguf) {
        gguf_free(ctx_gguf);
        ctx_gguf = nullptr;
    }

    weights = {};
    backend = nullptr;
}

bool FSQModule::load_metadata() {
    if (!ctx_gguf) {
        return false;
    }

    gguf_get_u32_opt(ctx_gguf, "voxcpm.fsq.latent_dim", config.latent_dim);
    gguf_get_f32_opt(ctx_gguf, "voxcpm.fsq.scale", config.scale);

    return true;
}

bool FSQModule::validate_weights() const {
    if (!weights.in_proj_weight || !weights.in_proj_bias ||
        !weights.out_proj_weight || !weights.out_proj_bias) {
        LOG_ERR("FSQModule: missing one or more FSQ tensors\n");
        return false;
    }

    if (weights.in_proj_weight->ne[0] <= 0 || weights.in_proj_weight->ne[1] <= 0) {
        LOG_ERR("FSQModule: invalid in_proj.weight shape\n");
        return false;
    }

    const int hidden_size = static_cast<int>(weights.in_proj_weight->ne[0]);
    const int latent_dim  = static_cast<int>(weights.in_proj_weight->ne[1]);

    if (weights.in_proj_bias->ne[0] != latent_dim) {
        LOG_ERR("FSQModule: in_proj.bias shape mismatch: got %" PRId64 ", expected %d\n",
                weights.in_proj_bias->ne[0], latent_dim);
        return false;
    }
    if (weights.out_proj_weight->ne[0] != latent_dim ||
        weights.out_proj_weight->ne[1] != hidden_size) {
        LOG_ERR("FSQModule: out_proj.weight shape mismatch: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                weights.out_proj_weight->ne[0], weights.out_proj_weight->ne[1], latent_dim, hidden_size);
        return false;
    }
    if (weights.out_proj_bias->ne[0] != hidden_size) {
        LOG_ERR("FSQModule: out_proj.bias shape mismatch: got %" PRId64 ", expected %d\n",
                weights.out_proj_bias->ne[0], hidden_size);
        return false;
    }
    if (config.scale <= 0.0f || !std::isfinite(config.scale)) {
        LOG_ERR("FSQModule: invalid quantization scale: %f\n", config.scale);
        return false;
    }

    return true;
}

bool FSQModule::load_weights(const std::string & path) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors <= 0) {
        LOG_ERR("FSQModule: no tensors in GGUF: %s\n", path.c_str());
        return false;
    }

    ggml_init_params data_params{};
    data_params.mem_size   = static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead();
    data_params.mem_buffer = nullptr;
    data_params.no_alloc   = true;
    ctx_weights = ggml_init(data_params);
    if (!ctx_weights) {
        LOG_ERR("FSQModule: failed to initialize weight context\n");
        return false;
    }

    std::unordered_map<std::string, size_t> offsets;
    offsets.reserve(static_cast<size_t>(n_tensors));

    std::vector<ggml_tensor *> tensors_to_load;
    tensors_to_load.reserve(4);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (!name || std::strncmp(name, "fsq.", 4) != 0) {
            continue;
        }

        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta, name);
        if (!meta_tensor) {
            LOG_ERR("FSQModule: missing tensor metadata for %s\n", name);
            return false;
        }

        ggml_tensor * tensor = ggml_dup_tensor(ctx_weights, meta_tensor);
        ggml_set_name(tensor, name);
        tensors_to_load.push_back(tensor);
        offsets.emplace(std::string(name), gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i));

        if (std::strcmp(name, "fsq.in_proj.weight") == 0) {
            weights.in_proj_weight = tensor;
        } else if (std::strcmp(name, "fsq.in_proj.bias") == 0) {
            weights.in_proj_bias = tensor;
        } else if (std::strcmp(name, "fsq.out_proj.weight") == 0) {
            weights.out_proj_weight = tensor;
        } else if (std::strcmp(name, "fsq.out_proj.bias") == 0) {
            weights.out_proj_bias = tensor;
        }
    }

    if (tensors_to_load.empty()) {
        LOG_ERR("FSQModule: no fsq.* tensors found in GGUF: %s\n", path.c_str());
        return false;
    }

    if (!validate_weights()) {
        return false;
    }

    config.hidden_size = static_cast<int>(weights.in_proj_weight->ne[0]);
    config.latent_dim  = static_cast<int>(weights.in_proj_weight->ne[1]);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, buft);
    if (!weight_buffer) {
        LOG_ERR("FSQModule: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        LOG_ERR("FSQModule: failed to open GGUF for reading: %s\n", path.c_str());
        return false;
    }

    std::vector<uint8_t> staging;
    for (ggml_tensor * tensor : tensors_to_load) {
        const auto it = offsets.find(std::string(tensor->name));
        if (it == offsets.end()) {
            LOG_ERR("FSQModule: missing file offset for %s\n", tensor->name);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        fin.seekg(static_cast<std::streamoff>(it->second), std::ios::beg);
        if (!fin) {
            LOG_ERR("FSQModule: seek failed for tensor %s\n", tensor->name);
            return false;
        }

        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(tensor->data), static_cast<std::streamsize>(nbytes));
        } else {
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(nbytes));
            if (!fin) {
                LOG_ERR("FSQModule: read failed for tensor %s\n", tensor->name);
                return false;
            }
            ggml_backend_tensor_set(tensor, staging.data(), 0, nbytes);
        }

        if (!fin) {
            LOG_ERR("FSQModule: read failed for tensor %s\n", tensor->name);
            return false;
        }
    }

    return true;
}

bool FSQModule::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("FSQModule: backend is null\n");
        return false;
    }
    backend = backend_in;

    ggml_context * meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta;

    ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("FSQModule: failed to open GGUF: %s\n", path.c_str());
        free();
        return false;
    }
    ctx_meta = meta;
    if (!ctx_meta) {
        LOG_ERR("FSQModule: GGUF metadata context is null: %s\n", path.c_str());
        free();
        return false;
    }

    if (!load_metadata() || !load_weights(path)) {
        free();
        return false;
    }

    LOG_INF("FSQModule: loaded hidden_size=%d latent_dim=%d scale=%.3f\n",
            config.hidden_size, config.latent_dim, config.scale);
    return true;
}

bool FSQModule::init_manual(const VoxCPM2FSQConfig & cfg,
                            ggml_tensor * in_proj_weight,
                            ggml_tensor * in_proj_bias,
                            ggml_tensor * out_proj_weight,
                            ggml_tensor * out_proj_bias) {
    free();

    config = cfg;
    weights.in_proj_weight  = in_proj_weight;
    weights.in_proj_bias    = in_proj_bias;
    weights.out_proj_weight = out_proj_weight;
    weights.out_proj_bias   = out_proj_bias;

    if (!validate_weights()) {
        weights = {};
        return false;
    }

    config.hidden_size = static_cast<int>(weights.in_proj_weight->ne[0]);
    config.latent_dim  = static_cast<int>(weights.in_proj_weight->ne[1]);
    return true;
}

ggml_tensor * FSQModule::forward(ggml_context * ctx, ggml_tensor * x) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(weights.in_proj_weight != nullptr);
    GGML_ASSERT(weights.out_proj_weight != nullptr);
    GGML_ASSERT(x->ne[0] == weights.in_proj_weight->ne[0]);

    ggml_tensor * latent = ggml_mul_mat(ctx, weights.in_proj_weight, x);
    latent = add_bias(ctx, latent, weights.in_proj_bias);
    latent = ggml_tanh(ctx, latent);
    latent = voxcpm2_fsq_quantize(ctx, latent, config.scale);

    ggml_tensor * output = ggml_mul_mat(ctx, weights.out_proj_weight, latent);
    output = add_bias(ctx, output, weights.out_proj_bias);
    return output;
}

ggml_tensor * voxcpm2_fsq_quantize(ggml_context * ctx, ggml_tensor * x, float scale) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(scale > 0.0f);

    ggml_tensor * scaled  = ggml_scale(ctx, x, scale);
    ggml_tensor * rounded = round_limited(ctx, scaled, scale);
    return ggml_scale(ctx, rounded, 1.0f / scale);
}
