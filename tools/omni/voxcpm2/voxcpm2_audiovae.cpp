#include "voxcpm2_audiovae.h"

#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace {

struct Conv1DSpec {
    int64_t kernel       = 0;
    int64_t in_channels  = 0;
    int64_t out_channels = 0;
};

static Conv1DSpec resolve_conv1d_spec(const ggml_tensor * weight, int expected_kernel) {
    GGML_ASSERT(weight != nullptr);

    Conv1DSpec spec;
    if (ggml_n_dims(weight) == 3) {
        spec.kernel       = weight->ne[0];
        spec.in_channels  = weight->ne[1];
        spec.out_channels = weight->ne[2];
        return spec;
    }

    GGML_ASSERT(ggml_n_dims(weight) == 2);
    GGML_ASSERT(expected_kernel > 0);
    if (weight->ne[0] == expected_kernel) {
        spec.kernel       = weight->ne[0];
        spec.in_channels  = weight->ne[1];
        spec.out_channels = 1;
        return spec;
    }

    GGML_ASSERT(weight->ne[0] % expected_kernel == 0);
    spec.kernel       = expected_kernel;
    spec.in_channels  = weight->ne[0] / expected_kernel;
    spec.out_channels = weight->ne[1];
    return spec;
}

static ggml_tensor * reshape_conv1d_weight_2d(ggml_context * ctx, ggml_tensor * weight, const Conv1DSpec & spec) {
    if (ggml_n_dims(weight) == 2 && !(weight->ne[0] == spec.kernel && spec.out_channels == 1)) {
        return weight;
    }
    ggml_tensor * reshaped = ggml_reshape_2d(ctx, weight, spec.kernel * spec.in_channels, spec.out_channels);
    ggml_set_name(reshaped, weight->name);
    return reshaped;
}

static ggml_tensor * conv1d_mul_mat_impl(ggml_context * ctx,
                                         ggml_tensor *  weight,
                                         ggml_tensor *  input,
                                         int            expected_kernel,
                                         int            stride,
                                         int            dilation) {
    const Conv1DSpec spec      = resolve_conv1d_spec(weight, expected_kernel);
    ggml_tensor *    weight_2d = reshape_conv1d_weight_2d(ctx, weight, spec);

    ggml_tensor * kernel_shape =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.kernel, spec.in_channels, spec.out_channels);
    ggml_tensor * im2col = ggml_im2col(ctx, kernel_shape, input, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * activations = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1] * im2col->ne[2]);
    ggml_tensor * result      = ggml_mul_mat(ctx, weight_2d, activations);
    result                    = ggml_reshape_3d(ctx, result, spec.out_channels, im2col->ne[1], im2col->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

static ggml_tensor * reshape_bias_3d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
}

static ggml_tensor * cast_like(ggml_context * ctx, ggml_tensor * src, ggml_tensor * like) {
    return src->type == like->type ? src : ggml_cast(ctx, src, like->type);
}

static ggml_tensor * add_bias_3d(ggml_context * ctx, ggml_tensor * output, ggml_tensor * bias) {
    if (!bias) {
        return output;
    }
    ggml_tensor * bias_view = reshape_bias_3d(ctx, bias);
    bias_view               = cast_like(ctx, bias_view, output);
    return ggml_add(ctx, output, bias_view);
}

static std::string encoder_res_prefix(int block_idx, int res_idx) {
    return "audio_vae.encoder.block." + std::to_string(block_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static std::string decoder_res_prefix(int model_idx, int res_idx) {
    return "audio_vae.decoder.model." + std::to_string(model_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static int decoder_final_snake_model_idx(const VoxCPM2AudioVAEConfig & cfg) {
    return cfg.num_decoder_blocks() + 2;
}

static int decoder_final_conv_model_idx(const VoxCPM2AudioVAEConfig & cfg) {
    return cfg.num_decoder_blocks() + 3;
}

static bool all_present(std::initializer_list<ggml_tensor *> tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](ggml_tensor * t) { return t != nullptr; });
}

static bool any_sr_conditioning(const std::vector<VoxCPM2AudioVAEDecoderBlockWeights> & blocks) {
    return std::any_of(blocks.begin(), blocks.end(),
                       [](const VoxCPM2AudioVAEDecoderBlockWeights & block) { return block.sr_cond.active(); });
}

}  // namespace

int VoxCPM2AudioVAEConfig::hop_length() const {
    return std::accumulate(encoder_rates.begin(), encoder_rates.end(), 1, std::multiplies<int>());
}

int VoxCPM2AudioVAEConfig::decode_hop_length() const {
    return std::accumulate(decoder_rates.begin(), decoder_rates.end(), 1, std::multiplies<int>());
}

int VoxCPM2AudioVAEConfig::sample_rate_bucket(int sample_rate_hz) const {
    int bucket = 0;
    while (bucket < static_cast<int>(sr_bin_boundaries.size()) &&
           sample_rate_hz > sr_bin_boundaries[static_cast<size_t>(bucket)]) {
        ++bucket;
    }
    return bucket;
}

ggml_tensor * voxcpm2_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(alpha != nullptr);

    const int64_t channels        = alpha->ne[1] > 1 ? alpha->ne[1] : alpha->ne[0];
    ggml_tensor * alpha_view      = ggml_reshape_3d(ctx, alpha, 1, channels, 1);
    alpha_view                    = cast_like(ctx, alpha_view, x);
    ggml_tensor * alpha_broadcast = ggml_repeat(ctx, alpha_view, x);
    ggml_tensor * ax              = ggml_mul(ctx, x, alpha_broadcast);
    ggml_tensor * sin_sq          = ggml_sqr(ctx, ggml_sin(ctx, ax));
    ggml_tensor * eps_tensor      = ggml_arange(ctx, eps, eps + 1.0f, 1.0f);
    ggml_tensor * alpha_eps       = ggml_add(ctx, alpha_broadcast, eps_tensor);
    return ggml_add(ctx, x, ggml_div(ctx, sin_sq, alpha_eps));
}

AudioVAEModel::~AudioVAEModel() {
    free();
}

void AudioVAEModel::free() {
    store.reset();
    weights                    = {};
    config                     = {};
    backend                    = nullptr;
    last_decode_sr_cond_tensor = nullptr;
    last_decode_sr_bucket      = 0;
}

ggml_tensor * AudioVAEModel::get_optional(const std::string & name) const {
    if (!store) {
        return nullptr;
    }
    ggml_tensor * t = store->get(name);
    if (!t && name.size() > 7 && name.rfind(".weight") == name.size() - 7) {
        t = store->get(name.substr(0, name.size() - 7));
    } else if (!t && name.find(".weight") == std::string::npos) {
        t = store->get(name + ".weight");
    }
    return t;
}

ggml_tensor * AudioVAEModel::get_required(const std::string & name) const {
    ggml_tensor * t = get_optional(name);
    if (!t) {
        LOG_ERR("AudioVAEModel: missing tensor %s\n", name.c_str());
    }
    return t;
}

bool AudioVAEModel::load_encoder_weights() {
    bool ok                        = true;
    weights.encoder_block_0_weight = get_required("audio_vae.encoder.block.0.weight");
    weights.encoder_block_0_bias   = get_required("audio_vae.encoder.block.0.bias");
    weights.encoder_fc_mu_weight   = get_required("audio_vae.encoder.fc_mu.weight");
    weights.encoder_fc_mu_bias     = get_required("audio_vae.encoder.fc_mu.bias");
    ok &= all_present({ weights.encoder_block_0_weight, weights.encoder_block_0_bias, weights.encoder_fc_mu_weight,
                        weights.encoder_fc_mu_bias });

    weights.encoder_blocks.resize(static_cast<size_t>(config.num_encoder_blocks()));
    for (int i = 0; i < config.num_encoder_blocks(); ++i) {
        VoxCPM2AudioVAEEncoderBlockWeights & block     = weights.encoder_blocks[static_cast<size_t>(i)];
        const int                            block_idx = i + 1;
        const std::string block_prefix = "audio_vae.encoder.block." + std::to_string(block_idx) + ".block.";

        auto load_res = [&](VoxCPM2AudioVAEResidualUnitWeights & res, int res_idx) {
            const std::string prefix = encoder_res_prefix(block_idx, res_idx);
            res.snake1_alpha         = get_required(prefix + "0.alpha");
            res.conv1_weight         = get_required(prefix + "1.weight");
            res.conv1_bias           = get_required(prefix + "1.bias");
            res.snake2_alpha         = get_required(prefix + "2.alpha");
            res.conv2_weight         = get_required(prefix + "3.weight");
            res.conv2_bias           = get_required(prefix + "3.bias");
            ok &= all_present({ res.snake1_alpha, res.conv1_weight, res.conv1_bias, res.snake2_alpha, res.conv2_weight,
                                res.conv2_bias });
        };

        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
        block.snake_alpha = get_required(block_prefix + "3.alpha");
        block.conv_weight = get_required(block_prefix + "4.weight");
        block.conv_bias   = get_required(block_prefix + "4.bias");
        ok &= all_present({ block.snake_alpha, block.conv_weight, block.conv_bias });
    }

    return ok;
}

bool AudioVAEModel::load_decoder_weights() {
    bool ok                        = true;
    weights.decoder_model_0_weight = get_required("audio_vae.decoder.model.0.weight");
    weights.decoder_model_0_bias   = get_required("audio_vae.decoder.model.0.bias");
    weights.decoder_model_1_weight = get_required("audio_vae.decoder.model.1.weight");
    weights.decoder_model_1_bias   = get_required("audio_vae.decoder.model.1.bias");
    ok &= all_present({ weights.decoder_model_0_weight, weights.decoder_model_0_bias, weights.decoder_model_1_weight,
                        weights.decoder_model_1_bias });

    const int final_snake_idx = decoder_final_snake_model_idx(config);
    const int final_conv_idx  = decoder_final_conv_model_idx(config);
    weights.decoder_final_snake_alpha =
        get_required("audio_vae.decoder.model." + std::to_string(final_snake_idx) + ".alpha");
    weights.decoder_final_conv_weight =
        get_required("audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".weight");
    weights.decoder_final_conv_bias =
        get_required("audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".bias");
    ok &= all_present(
        { weights.decoder_final_snake_alpha, weights.decoder_final_conv_weight, weights.decoder_final_conv_bias });

    weights.decoder_blocks.resize(static_cast<size_t>(config.num_decoder_blocks()));
    for (int i = 0; i < config.num_decoder_blocks(); ++i) {
        VoxCPM2AudioVAEDecoderBlockWeights & block     = weights.decoder_blocks[static_cast<size_t>(i)];
        const int                            model_idx = i + 2;
        const std::string block_prefix = "audio_vae.decoder.model." + std::to_string(model_idx) + ".block.";
        const std::string sr_prefix    = "audio_vae.decoder.sr_cond_model." + std::to_string(model_idx) + ".";

        block.sr_cond.scale_embed     = get_optional(sr_prefix + "scale_embed.weight");
        block.sr_cond.bias_embed      = get_optional(sr_prefix + "bias_embed.weight");
        block.sr_cond.cond_embed      = get_optional(sr_prefix + "cond_embed.weight");
        block.sr_cond.out_snake_alpha = get_optional(sr_prefix + "out_layer.0.alpha");
        block.sr_cond.out_weight      = get_optional(sr_prefix + "out_layer.1.weight");
        block.sr_cond.out_bias        = get_optional(sr_prefix + "out_layer.1.bias");

        block.snake_alpha = get_required(block_prefix + "0.alpha");
        block.conv_weight = get_required(block_prefix + "1.weight");
        block.conv_bias   = get_required(block_prefix + "1.bias");
        ok &= all_present({ block.snake_alpha, block.conv_weight, block.conv_bias });

        auto load_res = [&](VoxCPM2AudioVAEResidualUnitWeights & res, int res_idx) {
            const std::string prefix = decoder_res_prefix(model_idx, res_idx + 2);
            res.snake1_alpha         = get_required(prefix + "0.alpha");
            res.conv1_weight         = get_required(prefix + "1.weight");
            res.conv1_bias           = get_required(prefix + "1.bias");
            res.snake2_alpha         = get_required(prefix + "2.alpha");
            res.conv2_weight         = get_required(prefix + "3.weight");
            res.conv2_bias           = get_required(prefix + "3.bias");
            ok &= all_present({ res.snake1_alpha, res.conv1_weight, res.conv1_bias, res.snake2_alpha, res.conv2_weight,
                                res.conv2_bias });
        };
        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
    }

    return ok;
}

bool AudioVAEModel::validate_weights() const {
    if (config.encoder_rates.empty() || config.decoder_rates.empty()) {
        LOG_ERR("AudioVAEModel: empty encoder/decoder rates\n");
        return false;
    }
    if (!weights.encoder_block_0_weight || !weights.encoder_fc_mu_weight || !weights.decoder_model_0_weight ||
        !weights.decoder_model_1_weight || !weights.decoder_final_snake_alpha || !weights.decoder_final_conv_weight) {
        LOG_ERR("AudioVAEModel: missing top-level weights\n");
        return false;
    }
    if (static_cast<int>(weights.encoder_blocks.size()) != config.num_encoder_blocks() ||
        static_cast<int>(weights.decoder_blocks.size()) != config.num_decoder_blocks()) {
        LOG_ERR("AudioVAEModel: block count mismatch\n");
        return false;
    }
    return true;
}

bool AudioVAEModel::bind_from_store() {
    if (!store) {
        return false;
    }

    store->get_u32("voxcpm.audiovae.encoder_dim", config.encoder_dim);
    store->get_u32("voxcpm.audiovae.latent_dim", config.latent_dim);
    store->get_u32("voxcpm.audiovae.decoder_dim", config.decoder_dim);
    store->get_u32("voxcpm.audiovae.sample_rate", config.sample_rate);
    store->get_u32("voxcpm.audiovae.out_sample_rate", config.out_sample_rate);
    store->get_i32_array("voxcpm.audiovae.encoder_rates", config.encoder_rates);
    store->get_i32_array("voxcpm.audiovae.decoder_rates", config.decoder_rates);
    store->get_i32_array("voxcpm.audiovae.sr_bin_boundaries", config.sr_bin_boundaries);
    store->get_string("voxcpm.audiovae.cond_type", config.cond_type);

    if (!load_encoder_weights() || !load_decoder_weights()) {
        return false;
    }
    return validate_weights();
}

bool AudioVAEModel::init_from_gguf(const std::string & path, ggml_backend_t backend_in) {
    free();

    if (!backend_in) {
        LOG_ERR("AudioVAEModel: backend is null\n");
        return false;
    }
    backend = backend_in;

    store = std::make_unique<VoxCPM2GGUFWeightStore>();
    if (!store->load(path, backend, { "audio_vae." })) {
        free();
        return false;
    }
    if (!bind_from_store()) {
        free();
        return false;
    }

    LOG_INF("AudioVAEModel: loaded enc_dim=%d dec_dim=%d latent=%d sr=%d out_sr=%d enc_blocks=%d dec_blocks=%d\n",
            config.encoder_dim, config.decoder_dim, config.latent_dim, config.sample_rate, config.output_sample_rate(),
            config.num_encoder_blocks(), config.num_decoder_blocks());
    return true;
}

bool AudioVAEModel::init_manual(const VoxCPM2AudioVAEConfig & cfg, const VoxCPM2AudioVAEWeights & manual_weights) {
    free();
    config  = cfg;
    weights = manual_weights;
    if (!validate_weights()) {
        weights = {};
        return false;
    }
    return true;
}

ggml_tensor * AudioVAEModel::causal_conv1d(ggml_context * ctx,
                                           ggml_tensor *  x,
                                           ggml_tensor *  weight,
                                           ggml_tensor *  bias,
                                           int            kernel_size,
                                           int            stride,
                                           int            dilation,
                                           int            padding) const {
    ggml_tensor * padded = x;
    if (padding > 0) {
        padded = ggml_pad_ext(ctx, x, padding * 2, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor * result = conv1d_mul_mat_impl(ctx, weight, padded, kernel_size, stride, dilation);
    return add_bias_3d(ctx, result, bias);
}

ggml_tensor * AudioVAEModel::causal_conv1d_dw(ggml_context * ctx,
                                              ggml_tensor *  x,
                                              ggml_tensor *  weight,
                                              ggml_tensor *  bias,
                                              int            stride,
                                              int            dilation,
                                              int            padding) const {
    ggml_tensor * padded = x;
    if (padding > 0) {
        padded = ggml_pad_ext(ctx, x, padding * 2, 0, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * result = ggml_conv_1d_dw(ctx, weight, padded, stride, 0, dilation);
    result               = add_bias_3d(ctx, result, bias);
    return ggml_cont(ctx, result);
}

ggml_tensor * AudioVAEModel::causal_transpose_conv1d(ggml_context * ctx,
                                                     ggml_tensor *  x,
                                                     ggml_tensor *  weight,
                                                     ggml_tensor *  bias,
                                                     int            stride,
                                                     int            padding,
                                                     int            output_padding) const {
    ggml_tensor * x_matrix = x;
    if (ggml_n_dims(x_matrix) == 3 && x_matrix->ne[2] == 1) {
        x_matrix = ggml_reshape_2d(ctx, x_matrix, x_matrix->ne[0], x_matrix->ne[1]);
    }

    // CUDA conv-transpose-1d-gemm only supports F32 weight tensors. GGUF stores
    // AudioVAE weights as F16 for memory efficiency, so we cast here when needed.
    //
    // TODO: add a half-precision kernel path (conv_transpose_1d_load_weight<half> template) to avoid this cast.
    ggml_tensor * weight_for_op = weight->type == GGML_TYPE_F32 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor * result        = ggml_conv_transpose_1d(ctx, weight_for_op, x_matrix, stride, 0, 1);
    if (result->ne[3] == 1) {
        result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[1], result->ne[2]);
    }

    const int crop = padding * 2 - output_padding;
    if (crop > 0) {
        result = ggml_view_3d(ctx, result, result->ne[0] - crop, result->ne[1], result->ne[2], result->nb[1],
                              result->nb[2], 0);
    }
    return add_bias_3d(ctx, result, bias);
}

ggml_tensor * AudioVAEModel::residual_unit_forward(ggml_context *                             ctx,
                                                   ggml_tensor *                              x,
                                                   const VoxCPM2AudioVAEResidualUnitWeights & w,
                                                   int                                        dilation) const {
    ggml_tensor * h = voxcpm2_snake(ctx, x, w.snake1_alpha);
    h               = causal_conv1d_dw(ctx, h, w.conv1_weight, w.conv1_bias, 1, dilation, ((7 - 1) * dilation) / 2);
    h               = voxcpm2_snake(ctx, h, w.snake2_alpha);
    h               = causal_conv1d(ctx, h, w.conv2_weight, w.conv2_bias, 1, 1, 1, 0);

    if (x->ne[0] != h->ne[0]) {
        const int64_t target = std::min<int64_t>(x->ne[0], h->ne[0]);
        x                    = ggml_view_3d(ctx, x, target, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        h                    = ggml_view_3d(ctx, h, target, h->ne[1], h->ne[2], h->nb[1], h->nb[2], 0);
    }
    return ggml_add(ctx, x, h);
}

ggml_tensor * AudioVAEModel::encoder_block_forward(ggml_context *                             ctx,
                                                   ggml_tensor *                              x,
                                                   const VoxCPM2AudioVAEEncoderBlockWeights & w,
                                                   int                                        stride) const {
    x = residual_unit_forward(ctx, x, w.res0, 1);
    x = residual_unit_forward(ctx, x, w.res1, 3);
    x = residual_unit_forward(ctx, x, w.res2, 9);
    x = voxcpm2_snake(ctx, x, w.snake_alpha);
    return causal_conv1d(ctx, x, w.conv_weight, w.conv_bias, stride * 2, stride, 1,
                         static_cast<int>(std::ceil(stride / 2.0f)));
}

ggml_tensor * AudioVAEModel::sample_rate_condition_forward(ggml_context *                                    ctx,
                                                           ggml_tensor *                                     x,
                                                           const VoxCPM2AudioVAESampleRateConditionWeights & w,
                                                           ggml_tensor * sr_bucket) const {
    if (!w.active()) {
        return x;
    }

    GGML_ASSERT(sr_bucket != nullptr);
    GGML_ASSERT(sr_bucket->type == GGML_TYPE_I32);

    ggml_tensor * conditioned = x;
    if (config.cond_type == "scale_bias" || config.cond_type == "scale_bias_init") {
        GGML_ASSERT(w.scale_embed != nullptr);
        GGML_ASSERT(w.bias_embed != nullptr);
        ggml_tensor * scale    = ggml_get_rows(ctx, w.scale_embed, sr_bucket);
        ggml_tensor * bias     = ggml_get_rows(ctx, w.bias_embed, sr_bucket);
        scale                  = cast_like(ctx, scale, conditioned);
        bias                   = cast_like(ctx, bias, conditioned);
        ggml_tensor * scale_3d = ggml_reshape_3d(ctx, scale, 1, scale->ne[0], 1);
        ggml_tensor * bias_3d  = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        conditioned            = ggml_add(ctx, ggml_mul(ctx, conditioned, ggml_repeat(ctx, scale_3d, conditioned)),
                                          ggml_repeat(ctx, bias_3d, conditioned));
    } else if (config.cond_type == "add") {
        GGML_ASSERT(w.cond_embed != nullptr);
        ggml_tensor * cond    = ggml_get_rows(ctx, w.cond_embed, sr_bucket);
        cond                  = cast_like(ctx, cond, conditioned);
        ggml_tensor * cond_3d = ggml_reshape_3d(ctx, cond, 1, cond->ne[0], 1);
        conditioned           = ggml_add(ctx, conditioned, ggml_repeat(ctx, cond_3d, conditioned));
    } else {
        GGML_ABORT("unsupported VoxCPM2 AudioVAE sample-rate conditioning type");
    }

    if (w.out_weight) {
        GGML_ASSERT(w.out_snake_alpha != nullptr);
        conditioned = voxcpm2_snake(ctx, conditioned, w.out_snake_alpha);
        conditioned = causal_conv1d(ctx, conditioned, w.out_weight, w.out_bias, 1, 1, 1, 0);
    }
    return conditioned;
}

ggml_tensor * AudioVAEModel::decoder_block_forward(ggml_context *                             ctx,
                                                   ggml_tensor *                              x,
                                                   const VoxCPM2AudioVAEDecoderBlockWeights & w,
                                                   ggml_tensor *                              sr_bucket,
                                                   int                                        stride) const {
    x = sample_rate_condition_forward(ctx, x, w.sr_cond, sr_bucket);
    x = voxcpm2_snake(ctx, x, w.snake_alpha);
    x = causal_transpose_conv1d(ctx, x, w.conv_weight, w.conv_bias, stride, static_cast<int>(std::ceil(stride / 2.0f)),
                                stride % 2);
    x = residual_unit_forward(ctx, x, w.res0, 1);
    x = residual_unit_forward(ctx, x, w.res1, 3);
    x = residual_unit_forward(ctx, x, w.res2, 9);
    return x;
}

ggml_tensor * AudioVAEModel::encode(ggml_context * ctx, ggml_tensor * waveform) const {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(waveform != nullptr);

    ggml_tensor * x = waveform;
    if (ggml_n_dims(x) == 1) {
        x = ggml_reshape_3d(ctx, x, x->ne[0], 1, 1);
    } else if (ggml_n_dims(x) == 2) {
        x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);
    }
    GGML_ASSERT(x->ne[1] == 1);

    x = causal_conv1d(ctx, x, weights.encoder_block_0_weight, weights.encoder_block_0_bias, 7, 1, 1, 3);
    for (int i = 0; i < config.num_encoder_blocks(); ++i) {
        x = encoder_block_forward(ctx, x, weights.encoder_blocks[static_cast<size_t>(i)],
                                  config.encoder_rates[static_cast<size_t>(i)]);
    }
    return causal_conv1d(ctx, x, weights.encoder_fc_mu_weight, weights.encoder_fc_mu_bias, 3, 1, 1, 1);
}

ggml_tensor * AudioVAEModel::decode(ggml_context * ctx, ggml_tensor * latents, int target_sr) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(latents != nullptr);

    last_decode_sr_cond_tensor = nullptr;
    last_decode_sr_bucket      = 0;

    ggml_tensor * x = latents;
    if (ggml_n_dims(x) == 2) {
        x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);
    }
    GGML_ASSERT(x->ne[1] == config.latent_dim);
    GGML_ASSERT(x->ne[2] == 1);

    x = causal_conv1d_dw(ctx, x, weights.decoder_model_0_weight, weights.decoder_model_0_bias, 1, 1, 3);
    x = causal_conv1d(ctx, x, weights.decoder_model_1_weight, weights.decoder_model_1_bias, 1, 1, 1, 0);

    ggml_tensor * sr_bucket = nullptr;
    if (any_sr_conditioning(weights.decoder_blocks)) {
        const int sr               = target_sr > 0 ? target_sr : config.output_sample_rate();
        last_decode_sr_bucket      = static_cast<int32_t>(config.sample_rate_bucket(sr));
        last_decode_sr_cond_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(last_decode_sr_cond_tensor);
        sr_bucket = last_decode_sr_cond_tensor;
    }

    for (int i = 0; i < config.num_decoder_blocks(); ++i) {
        x = decoder_block_forward(ctx, x, weights.decoder_blocks[static_cast<size_t>(i)], sr_bucket,
                                  config.decoder_rates[static_cast<size_t>(i)]);
    }

    x = voxcpm2_snake(ctx, x, weights.decoder_final_snake_alpha);
    x = causal_conv1d(ctx, x, weights.decoder_final_conv_weight, weights.decoder_final_conv_bias, 7, 1, 1, 3);
    return ggml_tanh(ctx, x);
}

void AudioVAEModel::prepare_decode_inputs() const {
    if (last_decode_sr_cond_tensor) {
        ggml_backend_tensor_set(last_decode_sr_cond_tensor, &last_decode_sr_bucket, 0, sizeof(last_decode_sr_bucket));
    }
}
