#pragma once

#include "voxcpm2_transformer.h"

#include <memory>
#include <string>
#include <vector>

ggml_tensor * voxcpm2_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps = 1.0e-9f);

struct VoxCPM2AudioVAEConfig {
    int encoder_dim     = 128;
    int latent_dim      = 64;
    int decoder_dim     = 2048;
    int sample_rate     = 16000;
    int out_sample_rate = 48000;

    std::vector<int> encoder_rates     = { 2, 5, 8, 8 };
    std::vector<int> decoder_rates     = { 8, 6, 5, 2, 2, 2 };
    std::vector<int> sr_bin_boundaries = { 20000, 30000, 40000 };
    std::string      cond_type         = "scale_bias";

    int hop_length() const;
    int decode_hop_length() const;

    int num_encoder_blocks() const { return static_cast<int>(encoder_rates.size()); }

    int num_decoder_blocks() const { return static_cast<int>(decoder_rates.size()); }

    int output_sample_rate() const { return out_sample_rate > 0 ? out_sample_rate : sample_rate; }

    int sample_rate_bucket(int sample_rate_hz) const;
};

struct VoxCPM2AudioVAEResidualUnitWeights {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * conv1_weight = nullptr;
    ggml_tensor * conv1_bias   = nullptr;
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * conv2_weight = nullptr;
    ggml_tensor * conv2_bias   = nullptr;
};

struct VoxCPM2AudioVAEEncoderBlockWeights {
    VoxCPM2AudioVAEResidualUnitWeights res0;
    VoxCPM2AudioVAEResidualUnitWeights res1;
    VoxCPM2AudioVAEResidualUnitWeights res2;
    ggml_tensor *                      snake_alpha = nullptr;
    ggml_tensor *                      conv_weight = nullptr;
    ggml_tensor *                      conv_bias   = nullptr;
};

struct VoxCPM2AudioVAESampleRateConditionWeights {
    ggml_tensor * scale_embed     = nullptr;
    ggml_tensor * bias_embed      = nullptr;
    ggml_tensor * cond_embed      = nullptr;
    ggml_tensor * out_snake_alpha = nullptr;
    ggml_tensor * out_weight      = nullptr;
    ggml_tensor * out_bias        = nullptr;

    bool active() const { return scale_embed || bias_embed || cond_embed; }
};

struct VoxCPM2AudioVAEDecoderBlockWeights {
    VoxCPM2AudioVAESampleRateConditionWeights sr_cond;
    ggml_tensor *                             snake_alpha = nullptr;
    ggml_tensor *                             conv_weight = nullptr;
    ggml_tensor *                             conv_bias   = nullptr;
    VoxCPM2AudioVAEResidualUnitWeights        res0;
    VoxCPM2AudioVAEResidualUnitWeights        res1;
    VoxCPM2AudioVAEResidualUnitWeights        res2;
};

struct VoxCPM2AudioVAEWeights {
    ggml_tensor *                                   encoder_block_0_weight = nullptr;
    ggml_tensor *                                   encoder_block_0_bias   = nullptr;
    std::vector<VoxCPM2AudioVAEEncoderBlockWeights> encoder_blocks;
    ggml_tensor *                                   encoder_fc_mu_weight = nullptr;
    ggml_tensor *                                   encoder_fc_mu_bias   = nullptr;

    ggml_tensor *                                   decoder_model_0_weight = nullptr;
    ggml_tensor *                                   decoder_model_0_bias   = nullptr;
    ggml_tensor *                                   decoder_model_1_weight = nullptr;
    ggml_tensor *                                   decoder_model_1_bias   = nullptr;
    std::vector<VoxCPM2AudioVAEDecoderBlockWeights> decoder_blocks;
    ggml_tensor *                                   decoder_final_snake_alpha = nullptr;
    ggml_tensor *                                   decoder_final_conv_weight = nullptr;
    ggml_tensor *                                   decoder_final_conv_bias   = nullptr;
};

struct AudioVAEModel {
    VoxCPM2AudioVAEConfig  config;
    VoxCPM2AudioVAEWeights weights;

    ggml_backend_t                          backend = nullptr;  // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;
    ggml_tensor *                           last_decode_sr_cond_tensor = nullptr;
    int32_t                                 last_decode_sr_bucket      = 0;

    AudioVAEModel() = default;
    ~AudioVAEModel();

    AudioVAEModel(const AudioVAEModel &)             = delete;
    AudioVAEModel & operator=(const AudioVAEModel &) = delete;
    AudioVAEModel(AudioVAEModel &&)                  = delete;
    AudioVAEModel & operator=(AudioVAEModel &&)      = delete;

    bool init_from_gguf(const std::string & path, ggml_backend_t backend);
    bool init_manual(const VoxCPM2AudioVAEConfig & cfg, const VoxCPM2AudioVAEWeights & manual_weights);

    // waveform: [n_samples], [n_samples, 1], or [n_samples, 1, 1]
    // output: [frames, latent_dim, 1]
    ggml_tensor * encode(ggml_context * ctx, ggml_tensor * waveform) const;

    // latents: [frames, latent_dim] or [frames, latent_dim, 1]
    // output: [n_samples, 1, 1]
    ggml_tensor * decode(ggml_context * ctx, ggml_tensor * latents, int target_sr = -1);

    // Call after graph allocation and before compute if decode() created an SR bucket input.
    void prepare_decode_inputs() const;

    void free();

  private:
    bool bind_from_store();
    bool validate_weights() const;

    ggml_tensor * get_required(const std::string & name) const;
    ggml_tensor * get_optional(const std::string & name) const;
    bool          load_encoder_weights();
    bool          load_decoder_weights();

    ggml_tensor * causal_conv1d(ggml_context * ctx,
                                ggml_tensor *  x,
                                ggml_tensor *  weight,
                                ggml_tensor *  bias,
                                int            kernel_size,
                                int            stride,
                                int            dilation,
                                int            padding) const;

    ggml_tensor * causal_conv1d_dw(ggml_context * ctx,
                                   ggml_tensor *  x,
                                   ggml_tensor *  weight,
                                   ggml_tensor *  bias,
                                   int            stride,
                                   int            dilation,
                                   int            padding) const;

    ggml_tensor * causal_transpose_conv1d(ggml_context * ctx,
                                          ggml_tensor *  x,
                                          ggml_tensor *  weight,
                                          ggml_tensor *  bias,
                                          int            stride,
                                          int            padding,
                                          int            output_padding) const;

    ggml_tensor * residual_unit_forward(ggml_context *                             ctx,
                                        ggml_tensor *                              x,
                                        const VoxCPM2AudioVAEResidualUnitWeights & weights,
                                        int                                        dilation) const;

    ggml_tensor * encoder_block_forward(ggml_context *                             ctx,
                                        ggml_tensor *                              x,
                                        const VoxCPM2AudioVAEEncoderBlockWeights & weights,
                                        int                                        stride) const;

    ggml_tensor * decoder_block_forward(ggml_context *                             ctx,
                                        ggml_tensor *                              x,
                                        const VoxCPM2AudioVAEDecoderBlockWeights & weights,
                                        ggml_tensor *                              sr_bucket,
                                        int                                        stride) const;

    ggml_tensor * sample_rate_condition_forward(ggml_context *                                    ctx,
                                                ggml_tensor *                                     x,
                                                const VoxCPM2AudioVAESampleRateConditionWeights & weights,
                                                ggml_tensor *                                     sr_bucket) const;
};
