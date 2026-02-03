#pragma once


#include <cstdint>
#include "ggml.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <new>
#include "ggml-backend.h"
#include "ggml-alloc.h"

namespace omni {
namespace flow_matching {
class fmDiT;
class fmFlowMatchingModelLoaderGGUF;
}  // namespace flow_matching
namespace upsample_encoder_v2 {
class ueUpsampleConformerEncoderV2;
class ueUpsampleEncoderModelLoaderGGUF;
}  // namespace upsample_encoder_v2
namespace flow {
class flowCausalMaskedDiffWithXvec;
class flowExtraModelLoaderGGUF;
bool bind_flow_extra_weights(const flowExtraModelLoaderGGUF & loader, flowCausalMaskedDiffWithXvec & flow);
bool bind_flow_matching_weights(const flow_matching::fmFlowMatchingModelLoaderGGUF & loader,
                                flow_matching::fmDiT &                               dit);
bool bind_upsample_encoder_weights(const upsample_encoder_v2::ueUpsampleEncoderModelLoaderGGUF & loader,
                                   upsample_encoder_v2::ueUpsampleConformerEncoderV2 &           encoder);
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow_matching {
class fmCausalConditionalCFM;
struct fmCFMCache;
}  // namespace flow_matching
namespace upsample_encoder_v2 {
class ueUpsampleConformerEncoderV2;
}
namespace flow {
struct flowCache {
    ggml_tensor * conformer_cnn_cache = nullptr;
    ggml_tensor * conformer_att_cache = nullptr;
    flow_matching::fmCFMCache * estimator_cache = nullptr;
    flow_matching::fmCFMCache * estimator_cache_out = nullptr;
};
struct flowSetupCacheOut {
    ggml_tensor * feat_ctb = nullptr;
    ggml_tensor * mu_ctb   = nullptr;
    ggml_tensor * conformer_cnn_cache = nullptr;
    ggml_tensor * conformer_att_cache = nullptr;
    flow_matching::fmCFMCache * estimator_cache = nullptr;
};
struct flowInferenceChunkOut {
    ggml_tensor * feat_ctb = nullptr;
    ggml_tensor * conformer_cnn_cache = nullptr;
    ggml_tensor * conformer_att_cache = nullptr;
    flow_matching::fmCFMCache * estimator_cache = nullptr;
};
class flowCausalMaskedDiffWithXvec {
  public:
    flowCausalMaskedDiffWithXvec(int32_t                                                            input_size,
                                 int32_t                                                            output_size,
                                 int32_t                                                            spk_embed_dim,
                                 int32_t                                                            vocab_size,
                                 std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> encoder,
                                 std::shared_ptr<flow_matching::fmCausalConditionalCFM>             decoder);
    void set_parameters(ggml_tensor * token_embedding_weight,
                        ggml_tensor * spk_affine_weight,
                        ggml_tensor * spk_affine_bias,
                        ggml_tensor * encoder_proj_weight,
                        ggml_tensor * encoder_proj_bias);
    int32_t input_size() const { return input_size_; }
    int32_t output_size() const { return output_size_; }
    int32_t vocab_size() const { return vocab_size_; }
    int32_t spk_embed_dim() const { return spk_embed_dim_; }
    flowSetupCacheOut build_setup_cache_graph(ggml_context *              ctx,
                                              ggml_tensor *               token_ids_tb_i32,
                                              ggml_tensor *               mel_ctb_f32,
                                              ggml_tensor *               spk_cb_f32,
                                              int                         n_timesteps,
                                              float                       temperature,
                                              flow_matching::fmCFMCache * estimator_cache_out) const;
    flowInferenceChunkOut build_inference_chunk_graph(ggml_context *                    ctx,
                                                      ggml_tensor *                     token_ids_tb_i32,
                                                      ggml_tensor *                     spk_cb_f32,
                                                      bool                              last_chunk,
                                                      ggml_tensor *                     conformer_cnn_cache_in,
                                                      ggml_tensor *                     conformer_att_cache_in,
                                                      const flow_matching::fmCFMCache * estimator_cache_in,
                                                      int                               n_timesteps,
                                                      float                             temperature,
                                                      flow_matching::fmCFMCache *       estimator_cache_out) const;
  private:
    int32_t input_size_    = 0;
    int32_t output_size_   = 0;
    int32_t spk_embed_dim_ = 0;
    int32_t vocab_size_    = 0;
    std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> encoder_;
    std::shared_ptr<flow_matching::fmCausalConditionalCFM>             decoder_;
    ggml_tensor * token_embedding_weight_ = nullptr;
    ggml_tensor * spk_affine_weight_      = nullptr;
    ggml_tensor * spk_affine_bias_        = nullptr;
    ggml_tensor * encoder_proj_weight_    = nullptr;
    ggml_tensor * encoder_proj_bias_      = nullptr;
};
}  // namespace flow
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend *        ggml_backend_t;
namespace omni {
namespace flow {
class flowExtraModelLoaderGGUF {
  public:
    flowExtraModelLoaderGGUF();
    ~flowExtraModelLoaderGGUF();
    flowExtraModelLoaderGGUF(const flowExtraModelLoaderGGUF &)             = delete;
    flowExtraModelLoaderGGUF & operator=(const flowExtraModelLoaderGGUF &) = delete;
    bool load_from_file(const std::string & gguf_path, ggml_backend_t backend);
    ggml_tensor * get_tensor(const std::string & name) const;
    ggml_context * ctx_meta() const { return ctx_meta_; }
    ggml_context * ctx_data() const { return ctx_data_; }
    const std::string & path() const { return path_; }
  private:
    void reset();
    std::string path_;
    gguf_context * ctx_gguf_ = nullptr;
    ggml_context * ctx_meta_ = nullptr;
    ggml_context * ctx_data_ = nullptr;
    ggml_backend_t        backend_     = nullptr;
    ggml_backend_buffer_t buf_weights_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow_matching {
class fmAttention {
  public:
    fmAttention(int   dim,
                int   num_heads = 8,
                int   head_dim  = 64,
                bool  qkv_bias  = false,
                bool  qk_norm   = false,
                float attn_drop = 0.0f,
                float proj_drop = 0.0f,
                float norm_eps  = 1e-6f);
    void set_parameters(ggml_tensor * to_q_weight,
                        ggml_tensor * to_q_bias,
                        ggml_tensor * to_k_weight,
                        ggml_tensor * to_k_bias,
                        ggml_tensor * to_v_weight,
                        ggml_tensor * to_v_bias,
                        ggml_tensor * q_norm_weight,
                        ggml_tensor * q_norm_bias,
                        ggml_tensor * k_norm_weight,
                        ggml_tensor * k_norm_bias,
                        ggml_tensor * proj_weight,
                        ggml_tensor * proj_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * attn_mask) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  x,
                                            ggml_tensor *  att_cache,
                                            ggml_tensor *  attn_mask,
                                            ggml_tensor ** new_att_cache) const;
    int dim() const { return dim_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
  private:
    int   dim_       = 0;
    int   num_heads_ = 0;
    int   head_dim_  = 0;
    bool  qkv_bias_  = false;
    bool  qk_norm_   = false;
    float attn_drop_ = 0.0f;
    float proj_drop_ = 0.0f;
    float norm_eps_  = 1e-6f;
    ggml_tensor * to_q_weight_ = nullptr;
    ggml_tensor * to_q_bias_   = nullptr;
    ggml_tensor * to_k_weight_ = nullptr;
    ggml_tensor * to_k_bias_   = nullptr;
    ggml_tensor * to_v_weight_ = nullptr;
    ggml_tensor * to_v_bias_   = nullptr;
    ggml_tensor * q_norm_weight_ = nullptr;
    ggml_tensor * q_norm_bias_   = nullptr;
    ggml_tensor * k_norm_weight_ = nullptr;
    ggml_tensor * k_norm_bias_   = nullptr;
    ggml_tensor * proj_weight_ = nullptr;
    ggml_tensor * proj_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmDiT;
struct fmCFMCache {
    int n_time    = 0;
    int depth     = 0;
    int num_heads = 0;
    int head_dim  = 0;
    ggml_tensor * cnn_cache = nullptr;
    ggml_tensor * att_cache = nullptr;
    void clear() {
        n_time = depth = num_heads = head_dim = 0;
        cnn_cache                             = nullptr;
        att_cache                             = nullptr;
    }
};
class fmCausalConditionalCFM {
  public:
    fmCausalConditionalCFM(std::shared_ptr<fmDiT> estimator, float inference_cfg_rate = 0.7f);
    fmCausalConditionalCFM(const fmCausalConditionalCFM &)             = delete;
    fmCausalConditionalCFM & operator=(const fmCausalConditionalCFM &) = delete;
    ~fmCausalConditionalCFM() = default;
    int out_channels() const;
    void reset_stream_state() const;
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  mu,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  spks,
                                      ggml_tensor *  cond,
                                      int            n_timesteps,
                                      float          temperature) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context *     ctx,
                                            ggml_tensor *      mu,
                                            ggml_tensor *      spks,
                                            ggml_tensor *      cond,
                                            int                n_timesteps,
                                            float              temperature,
                                            const fmCFMCache * cache_in,
                                            fmCFMCache *       cache_out) const;
  private:
    std::shared_ptr<fmDiT> estimator_;
    float                  inference_cfg_rate_ = 0.7f;
    int                    out_channels_       = 0;
    mutable int chunk_call_id_ = 0;
    static ggml_tensor * deterministic_noise(ggml_context * ctx,
                                             int64_t        C,
                                             int64_t        T,
                                             int64_t        B,
                                             float          temperature,
                                             int64_t        offset_ct);
    static ggml_tensor * build_timestep_tensor(ggml_context * ctx, int64_t B_total, float t_value);
    static void build_cosine_t_span(int n_timesteps, std::vector<float> & t_span_out);
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmCausalConv1d {
  public:
    fmCausalConv1d(int in_channels, int out_channels, int kernel_size);
    void set_parameters(ggml_tensor * weight, ggml_tensor * bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  x,
                                            ggml_tensor *  cnn_cache,
                                            ggml_tensor ** new_cache) const;
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int kernel_size() const { return kernel_size_; }
  private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    ggml_tensor * weight_ = nullptr;
    ggml_tensor * bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
struct fmDebugContext {
    int call_id = -1;
    int step    = -1;
    int block   = -1;
};
void           fm_debug_set_context(int call_id, int step, int block = -1);
void           fm_debug_clear_context();
fmDebugContext fm_debug_get_context();
ggml_tensor * build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias);
ggml_tensor * build_layer_norm(ggml_context * ctx,
                               ggml_tensor *  x,
                               ggml_tensor *  weight,
                               ggml_tensor *  bias,
                               float          eps);
ggml_tensor * build_mish(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * build_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale);
ggml_tensor * build_silu(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * build_gelu(ggml_context * ctx, ggml_tensor * x);
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmTimestepEmbedder;
class fmDiTBlock;
class fmFinalLayer;
class fmDiT {
  public:
    fmDiT(int in_channels, int out_channels, float mlp_ratio, int depth, int num_heads, int head_dim, int hidden_size);
    fmDiT(const fmDiT &)             = delete;
    fmDiT & operator=(const fmDiT &) = delete;
    ~fmDiT();
    void set_parameters(ggml_tensor * in_proj_weight, ggml_tensor * in_proj_bias);
    fmTimestepEmbedder * timestep_embedder() const { return t_embedder_; }
    fmFinalLayer * final_layer() const { return final_layer_; }
    std::vector<fmDiTBlock *> & blocks() { return blocks_; }
    const std::vector<fmDiTBlock *> & blocks() const { return blocks_; }
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int depth() const { return depth_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
    int hidden_size() const { return hidden_size_; }
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  mu,
                                      ggml_tensor *  t,
                                      ggml_tensor *  spks,
                                      ggml_tensor *  cond) const;
    ggml_tensor * build_blocks_forward_graph(ggml_context * ctx,
                                             ggml_tensor *  x,
                                             ggml_tensor *  t_embed,
                                             ggml_tensor *  attn_mask) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context *                     ctx,
                                            ggml_tensor *                      x,
                                            ggml_tensor *                      mu,
                                            ggml_tensor *                      t,
                                            ggml_tensor *                      spks,
                                            ggml_tensor *                      cond,
                                            const std::vector<ggml_tensor *> & prev_cnn_cache,
                                            const std::vector<ggml_tensor *> & prev_att_cache,
                                            std::vector<ggml_tensor *> &       new_cnn_cache,
                                            std::vector<ggml_tensor *> &       new_att_cache) const;
    ggml_tensor * build_blocks_forward_chunk_graph(ggml_context *                     ctx,
                                                   ggml_tensor *                      x,
                                                   ggml_tensor *                      t_embed,
                                                   ggml_tensor *                      mask,
                                                   const std::vector<ggml_tensor *> & prev_cnn_cache,
                                                   const std::vector<ggml_tensor *> & prev_att_cache,
                                                   std::vector<ggml_tensor *> &       new_cnn_cache,
                                                   std::vector<ggml_tensor *> &       new_att_cache) const;
  private:
    int   in_channels_;
    int   out_channels_;
    float mlp_ratio_;
    int   depth_;
    int   num_heads_;
    int   head_dim_;
    int   hidden_size_;
    fmTimestepEmbedder *      t_embedder_ = nullptr;
    std::vector<fmDiTBlock *> blocks_;
    fmFinalLayer *            final_layer_ = nullptr;
    ggml_tensor * in_proj_weight_ = nullptr;
    ggml_tensor * in_proj_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmAttention;
class fmCausalConvBlock;
class fmMLP;
class fmDiTBlock {
  public:
    fmDiTBlock(int hidden_size, int num_heads, int head_dim, float mlp_ratio = 4.0f);
    fmDiTBlock(const fmDiTBlock &)             = delete;
    fmDiTBlock & operator=(const fmDiTBlock &) = delete;
    ~fmDiTBlock();
    void set_parameters(ggml_tensor * norm1_weight,
                        ggml_tensor * norm1_bias,
                        ggml_tensor * norm2_weight,
                        ggml_tensor * norm2_bias,
                        ggml_tensor * norm3_weight,
                        ggml_tensor * norm3_bias,
                        ggml_tensor * ada_weight,
                        ggml_tensor * ada_bias);
    void set_attention_parameters(ggml_tensor * to_q_weight,
                                  ggml_tensor * to_q_bias,
                                  ggml_tensor * to_k_weight,
                                  ggml_tensor * to_k_bias,
                                  ggml_tensor * to_v_weight,
                                  ggml_tensor * to_v_bias,
                                  ggml_tensor * q_norm_weight,
                                  ggml_tensor * q_norm_bias,
                                  ggml_tensor * k_norm_weight,
                                  ggml_tensor * k_norm_bias,
                                  ggml_tensor * proj_weight,
                                  ggml_tensor * proj_bias);
    void set_conv_parameters(ggml_tensor * conv1_weight,
                             ggml_tensor * conv1_bias,
                             ggml_tensor * conv2_weight,
                             ggml_tensor * conv2_bias,
                             ggml_tensor * ln_weight,
                             ggml_tensor * ln_bias);
    void set_mlp_parameters(ggml_tensor * fc1_weight,
                            ggml_tensor * fc1_bias,
                            ggml_tensor * fc2_weight,
                            ggml_tensor * fc2_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x,
                                      ggml_tensor *  c,
                                      ggml_tensor *  attn_mask) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  x,
                                            ggml_tensor *  c,
                                            ggml_tensor *  cnn_cache,
                                            ggml_tensor *  att_cache,
                                            ggml_tensor *  mask,
                                            ggml_tensor ** new_cnn_cache,
                                            ggml_tensor ** new_att_cache) const;
    int hidden_size() const { return hidden_size_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
  private:
    int   hidden_size_;
    int   num_heads_;
    int   head_dim_;
    float mlp_ratio_;
    fmAttention *       attn_ = nullptr;
    fmCausalConvBlock * conv_ = nullptr;
    fmMLP *             mlp_  = nullptr;
    ggml_tensor * norm1_weight_ = nullptr;
    ggml_tensor * norm1_bias_   = nullptr;
    ggml_tensor * norm2_weight_ = nullptr;
    ggml_tensor * norm2_bias_   = nullptr;
    ggml_tensor * norm3_weight_ = nullptr;
    ggml_tensor * norm3_bias_   = nullptr;
    ggml_tensor * ada_weight_ = nullptr;
    ggml_tensor * ada_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmFinalLayer {
  public:
    fmFinalLayer(int hidden_size, int out_channels);
    void set_parameters(ggml_tensor * ada_weight,
                        ggml_tensor * ada_bias,
                        ggml_tensor * ln_weight,
                        ggml_tensor * ln_bias,
                        ggml_tensor * linear_weight,
                        ggml_tensor * linear_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * c) const;
  private:
    int hidden_size_;
    int out_channels_;
    ggml_tensor * ada_weight_    = nullptr;
    ggml_tensor * ada_bias_      = nullptr;
    ggml_tensor * ln_weight_     = nullptr;
    ggml_tensor * ln_bias_       = nullptr;
    ggml_tensor * linear_weight_ = nullptr;
    ggml_tensor * linear_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
namespace omni {
namespace flow_matching {
class fmDiT;
class fmCausalConditionalCFM;
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend *        ggml_backend_t;
namespace omni {
namespace flow_matching {
class fmFlowMatchingModelLoaderGGUF {
  public:
    fmFlowMatchingModelLoaderGGUF();
    ~fmFlowMatchingModelLoaderGGUF();
    fmFlowMatchingModelLoaderGGUF(const fmFlowMatchingModelLoaderGGUF &)             = delete;
    fmFlowMatchingModelLoaderGGUF & operator=(const fmFlowMatchingModelLoaderGGUF &) = delete;
    bool load_from_file(const std::string & gguf_path, ggml_backend_t backend);
    ggml_tensor * get_tensor(const std::string & name) const;
    ggml_context * ctx_meta() const { return ctx_meta_; }
    ggml_context * ctx_data() const { return ctx_data_; }
    const std::string & path() const { return path_; }
  private:
    void reset();
    std::string path_;
    gguf_context * ctx_gguf_ = nullptr;
    ggml_context * ctx_meta_ = nullptr;
    ggml_context * ctx_data_ = nullptr;
    ggml_backend_t        backend_     = nullptr;
    ggml_backend_buffer_t buf_weights_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmFlowMatchingMaskUtils {
  public:
    static ggml_tensor * make_pad_mask(ggml_context * ctx, const std::vector<int> & lengths, int max_len = 0);
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmMLP {
  public:
    fmMLP(int          in_features,
          int          hidden_features = -1,
          int          out_features    = -1,
          const char * act_layer       = "gelu",
          const char * norm_layer      = nullptr,
          bool         bias            = true,
          float        drop            = 0.0f);
    void set_parameters(ggml_tensor * fc1_weight,
                        ggml_tensor * fc1_bias,
                        ggml_tensor * fc2_weight,
                        ggml_tensor * fc2_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x) const;
  private:
    int in_features_;
    int hidden_features_;
    int out_features_;
    const char * act_layer_;
    const char * norm_layer_;
    bool         bias_;
    float        drop_;
    ggml_tensor * fc1_weight_ = nullptr;
    ggml_tensor * fc1_bias_   = nullptr;
    ggml_tensor * fc2_weight_ = nullptr;
    ggml_tensor * fc2_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;
struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_gallocr *        ggml_gallocr_t;
namespace omni {
namespace flow_matching {
class fmDiT;
class fmCausalConditionalCFM;
class fmFlowMatchingModelLoaderGGUF;
class fmHostBufferF32 {
  public:
    fmHostBufferF32() = default;
    ~fmHostBufferF32() { std::free(ptr_); }
    fmHostBufferF32(const fmHostBufferF32 &)             = delete;
    fmHostBufferF32 & operator=(const fmHostBufferF32 &) = delete;
    fmHostBufferF32(fmHostBufferF32 && other) noexcept {
        ptr_        = other.ptr_;
        size_       = other.size_;
        cap_        = other.cap_;
        other.ptr_  = nullptr;
        other.size_ = 0;
        other.cap_  = 0;
    }
    fmHostBufferF32 & operator=(fmHostBufferF32 && other) noexcept {
        if (this == &other) {
            return *this;
        }
        std::free(ptr_);
        ptr_        = other.ptr_;
        size_       = other.size_;
        cap_        = other.cap_;
        other.ptr_  = nullptr;
        other.size_ = 0;
        other.cap_  = 0;
        return *this;
    }
    void clear() noexcept { size_ = 0; }
    bool empty() const noexcept { return size_ == 0; }
    size_t size() const noexcept { return size_; }
    float * data() noexcept { return ptr_; }
    const float * data() const noexcept { return ptr_; }
    void resize(size_t n) {
        if (n > cap_) {
            void * p = std::realloc(ptr_, n * sizeof(float));
            if (!p) {
                throw std::bad_alloc();
            }
            ptr_ = static_cast<float *>(p);
            cap_ = n;
        }
        size_ = n;
    }
  private:
    float * ptr_  = nullptr;
    size_t  size_ = 0;
    size_t  cap_  = 0;
};
struct fmCFMCacheHost {
    int n_time    = 0;
    int depth     = 0;
    int num_heads = 0;
    int head_dim  = 0;
    std::vector<int64_t> cnn_ne;
    std::vector<int64_t> att_ne;
    fmHostBufferF32 cnn;
    fmHostBufferF32 att;
    void clear() {
        n_time = depth = num_heads = head_dim = 0;
        cnn_ne.clear();
        att_ne.clear();
        cnn.clear();
        att.clear();
    }
    bool has_cache() const { return cnn_ne.size() == 4 && att_ne.size() == 4 && !cnn.empty() && !att.empty(); }
    int64_t att_t_cache() const { return att_ne.size() == 4 ? att_ne[1] : 0; }
};
class fmFlowMatchingGGUFModelLoader {
  public:
    fmFlowMatchingGGUFModelLoader();
    ~fmFlowMatchingGGUFModelLoader();
    fmFlowMatchingGGUFModelLoader(const fmFlowMatchingGGUFModelLoader &)             = delete;
    fmFlowMatchingGGUFModelLoader & operator=(const fmFlowMatchingGGUFModelLoader &) = delete;
    bool load_from_gguf(const std::string & gguf_path, const std::string & device = "cpu");
    void set_num_threads(int n_threads);
    void reset_stream();
    bool forward(const float *        mu_bct,
                 const float *        spks_bc,
                 const float *        cond_bct,
                 int64_t              B,
                 int64_t              C,
                 int64_t              T,
                 int                  n_timesteps,
                 float                temperature,
                 std::vector<float> & y_bct_out) const;
    bool forward_chunk(const float *        mu_bct,
                       const float *        spks_bc,
                       const float *        cond_bct,
                       int64_t              B,
                       int64_t              C,
                       int64_t              T_chunk,
                       int                  n_timesteps,
                       float                temperature,
                       fmCFMCacheHost *     cache_in_out,
                       std::vector<float> & y_bct_out);
    ggml_backend_t backend() const { return backend_; }
    const std::string & backend_name() const { return backend_name_; }
    const std::string & gguf_path() const { return gguf_path_; }
    std::shared_ptr<fmDiT> estimator() const { return estimator_; }
    std::shared_ptr<fmCausalConditionalCFM> cfm() const { return cfm_; }
  private:
    void reset();
    bool init_backend(const std::string & device);
    bool bind_parameters();
    static void build_cosine_t_span(int n_timesteps, std::vector<float> & t_span_out);
    static void fill_noise_ctb(std::vector<float> & noise_ctb,
                               int64_t              C,
                               int64_t              T,
                               int64_t              B,
                               float                temperature,
                               int64_t              offset_ct);
    static void fill_timestep_1d(std::vector<float> & t_host, int64_t B_total, float t_value);
    static void bct_to_ctb(const float * bct, int64_t B, int64_t C, int64_t T, std::vector<float> & ctb_out);
    static void bc_to_cb(const float * bc, int64_t B, int64_t C, std::vector<float> & cb_out);
    static void ctb_to_bct(const std::vector<float> & ctb,
                           int64_t                    C,
                           int64_t                    T,
                           int64_t                    B,
                           std::vector<float> &       bct_out);
    static bool read_tensor_3d_ctb_f32(ggml_backend_t backend, ggml_tensor * t, std::vector<float> & out_ctb);
    static bool read_tensor_4d_corder_f32(ggml_backend_t backend, ggml_tensor * t4, std::vector<float> & out_corder);
  private:
    int   in_channels_        = 320;
    int   out_channels_       = 80;
    float mlp_ratio_          = 4.0f;
    int   depth_              = 16;
    int   num_heads_          = 8;
    int   head_dim_           = 64;
    int   hidden_size_        = 512;
    float inference_cfg_rate_ = 0.7f;
    int num_threads_ = 1;
    std::string    gguf_path_;
    ggml_backend_t backend_ = nullptr;
    std::string    backend_name_;
    std::unique_ptr<fmFlowMatchingModelLoaderGGUF> weights_;
    std::shared_ptr<fmDiT>                         estimator_;
    std::shared_ptr<fmCausalConditionalCFM>        cfm_;
    ggml_gallocr_t galloc_ = nullptr;
    int chunk_call_id_ = 0;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmModulateUtils {
  public:
    static ggml_tensor * build_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale);
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmTimestepEmbedder {
  public:
    fmTimestepEmbedder(int hidden_size, int frequency_embedding_size = 256);
    void set_parameters(ggml_tensor * mlp1_weight,
                        ggml_tensor * mlp1_bias,
                        ggml_tensor * mlp2_weight,
                        ggml_tensor * mlp2_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * t) const;
  private:
    int hidden_size_;
    int frequency_embedding_size_;
    float scale_ = 1000.0f;
    ggml_tensor * mlp1_weight_ = nullptr;
    ggml_tensor * mlp1_bias_   = nullptr;
    ggml_tensor * mlp2_weight_ = nullptr;
    ggml_tensor * mlp2_bias_   = nullptr;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmTransposeModule {
  public:
    fmTransposeModule(int dim0, int dim1);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x) const;
  private:
    int dim0_;
    int dim1_;
};
}  // namespace flow_matching
}  // namespace omni
struct ggml_context;
struct ggml_tensor;
namespace omni {
namespace flow_matching {
class fmCausalConv1d;
class fmCausalConvBlock {
  public:
    fmCausalConvBlock(int in_channels, int out_channels, int kernel_size = 3);
    fmCausalConvBlock(const fmCausalConvBlock &)             = delete;
    fmCausalConvBlock & operator=(const fmCausalConvBlock &) = delete;
    ~fmCausalConvBlock();
    void set_parameters(ggml_tensor * conv1_weight,
                        ggml_tensor * conv1_bias,
                        ggml_tensor * conv2_weight,
                        ggml_tensor * conv2_bias,
                        ggml_tensor * ln_weight,
                        ggml_tensor * ln_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  x,
                                            ggml_tensor *  cnn_cache,
                                            ggml_tensor ** new_cnn_cache) const;
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int kernel_size() const { return kernel_size_; }
  private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    fmCausalConv1d * conv1_ = nullptr;
    fmCausalConv1d * conv2_ = nullptr;
    ggml_tensor * ln_weight_ = nullptr;
    ggml_tensor * ln_bias_   = nullptr;
    fmTransposeModule transpose_ctb_btc_;
    fmTransposeModule transpose_btc_ctb_;
};
}  // namespace flow_matching
}  // namespace omni
#if 0
namespace omni {
namespace upsample_encoder_v2 {
class ueMultiHeadedAttention {
  public:
    ueMultiHeadedAttention(int32_t n_head, int32_t n_feat, float dropout_rate, bool key_bias);
    void set_parameters(ggml_tensor * linear_q_weight,
                        ggml_tensor * linear_q_bias,
                        ggml_tensor * linear_k_weight,
                        ggml_tensor * linear_k_bias,
                        ggml_tensor * linear_v_weight,
                        ggml_tensor * linear_v_bias,
                        ggml_tensor * linear_out_weight,
                        ggml_tensor * linear_out_bias);
    void build_forward_qkv(ggml_context * ctx,
                           ggml_tensor *  query_ctb,
                           ggml_tensor *  key_ctb,
                           ggml_tensor *  value_ctb,
                           ggml_tensor ** q_dhtb_out,
                           ggml_tensor ** k_dhtb_out,
                           ggml_tensor ** v_dhtb_out) const;
    ggml_tensor * build_forward_attention(ggml_context * ctx,
                                          ggml_tensor *  value_dhtb,
                                          ggml_tensor *  scores_ktqh_b,
                                          ggml_tensor *  mask) const;
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  query_ctb,
                                      ggml_tensor *  key_ctb,
                                      ggml_tensor *  value_ctb,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  pos_emb,
                                      ggml_tensor *  cache,
                                      ggml_tensor ** new_cache_out) const;
    void forward_qkv(ggml_context * ctx,
                     ggml_tensor *  query_ctb,
                     ggml_tensor *  key_ctb,
                     ggml_tensor *  value_ctb,
                     ggml_tensor ** q_dhtb_out,
                     ggml_tensor ** k_dhtb_out,
                     ggml_tensor ** v_dhtb_out) const {
        build_forward_qkv(ctx, query_ctb, key_ctb, value_ctb, q_dhtb_out, k_dhtb_out, v_dhtb_out);
    }
    ggml_tensor * forward_attention(ggml_context * ctx,
                                    ggml_tensor *  value_dhtb,
                                    ggml_tensor *  scores_ktqh_b,
                                    ggml_tensor *  mask = nullptr) const {
        return build_forward_attention(ctx, value_dhtb, scores_ktqh_b, mask);
    }
    struct ueForwardOut {
        ggml_tensor * out_ctb   = nullptr;
        ggml_tensor * new_cache = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx,
                         ggml_tensor *  query_ctb,
                         ggml_tensor *  key_ctb,
                         ggml_tensor *  value_ctb,
                         ggml_tensor *  mask    = nullptr,
                         ggml_tensor *  pos_emb = nullptr,
                         ggml_tensor *  cache   = nullptr) const {
        ueForwardOut out;
        out.out_ctb = build_forward_graph(ctx, query_ctb, key_ctb, value_ctb, mask, pos_emb, cache, &out.new_cache);
        return out;
    }
    int32_t n_head() const { return n_head_; }
    int32_t n_feat() const { return n_feat_; }
    int32_t head_dim() const { return d_k_; }
  protected:
    int32_t n_head_       = 0;
    int32_t n_feat_       = 0;
    int32_t d_k_          = 0;
    float   dropout_rate_ = 0.0f;
    bool    key_bias_     = true;
    ggml_tensor * linear_q_weight_   = nullptr;
    ggml_tensor * linear_q_bias_     = nullptr;
    ggml_tensor * linear_k_weight_   = nullptr;
    ggml_tensor * linear_k_bias_     = nullptr;
    ggml_tensor * linear_v_weight_   = nullptr;
    ggml_tensor * linear_v_bias_     = nullptr;
    ggml_tensor * linear_out_weight_ = nullptr;
    ggml_tensor * linear_out_bias_   = nullptr;
};
class ueRelPositionMultiHeadedAttention final : public ueMultiHeadedAttention {
  public:
    ueRelPositionMultiHeadedAttention(int32_t n_head, int32_t n_feat, float dropout_rate, bool key_bias);
    void set_relpos_parameters(ggml_tensor * linear_pos_weight, ggml_tensor * pos_bias_u, ggml_tensor * pos_bias_v);
    ggml_tensor * build_rel_shift(ggml_context * ctx, ggml_tensor * x_tp_tq_hb) const;
    ggml_tensor * rel_shift(ggml_context * ctx, ggml_tensor * x_tp_tq_hb) const {
        return build_rel_shift(ctx, x_tp_tq_hb);
    }
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  query_ctb,
                                      ggml_tensor *  key_ctb,
                                      ggml_tensor *  value_ctb,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  pos_emb,
                                      ggml_tensor *  cache,
                                      ggml_tensor ** new_cache_out) const;
    ueForwardOut forward(ggml_context * ctx,
                         ggml_tensor *  query_ctb,
                         ggml_tensor *  key_ctb,
                         ggml_tensor *  value_ctb,
                         ggml_tensor *  mask    = nullptr,
                         ggml_tensor *  pos_emb = nullptr,
                         ggml_tensor *  cache   = nullptr) const {
        ueForwardOut out;
        out.out_ctb = build_forward_graph(ctx, query_ctb, key_ctb, value_ctb, mask, pos_emb, cache, &out.new_cache);
        return out;
    }
  private:
    ggml_tensor * linear_pos_weight_ = nullptr;
    ggml_tensor * pos_bias_u_        = nullptr;
    ggml_tensor * pos_bias_v_        = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
#endif
namespace omni {
namespace upsample_encoder_v2 {
class ueEspnetRelPositionalEncoding;
class ueBaseSubsampling {
  public:
    ueBaseSubsampling();
    virtual ~ueBaseSubsampling() = default;
    int32_t right_context    = 0;
    int32_t subsampling_rate = 1;
    void set_pos_enc(std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc) { pos_enc_ = std::move(pos_enc); }
    std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc() const { return pos_enc_; }
    ggml_tensor * position_encoding(ggml_context * ctx, int32_t offset, int32_t size) const;
    ggml_tensor * position_encoding(ggml_context * ctx, std::nullptr_t, int32_t size) const;
    ggml_tensor * position_encoding(ggml_context * ctx, ggml_tensor *, int32_t size) const;
  protected:
    std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
struct ueTensorLayout {
    static constexpr int kDimC = 0;
    static constexpr int kDimT = 1;
    static constexpr int kDimB = 2;
};
struct ueAttCacheSpec {
    int32_t num_heads = 0;
    int32_t head_dim  = 0;
};
struct ueCnnCacheSpec {
    int32_t channels              = 0;
    int32_t pre_lookahead_cache_t = 2;
    int32_t upsample_cache_t      = 0;
};
struct ueEncoderCacheHost {
    std::vector<float> cnn_cache_bct;
    int64_t            cnn_B = 0;
    int64_t            cnn_C = 0;
    int64_t            cnn_T = 0;
    std::vector<float> att_cache_lbhte;
    int64_t            att_L = 0;
    int64_t            att_B = 0;
    int64_t            att_H = 0;
    int64_t            att_T = 0;
    int64_t            att_E = 0;
    void clear() {
        cnn_cache_bct.clear();
        att_cache_lbhte.clear();
        cnn_B = cnn_C = cnn_T = 0;
        att_L = att_B = att_H = att_T = att_E = 0;
    }
    bool has_cnn_cache() const { return !cnn_cache_bct.empty(); }
    bool has_att_cache() const { return !att_cache_lbhte.empty(); }
};
ggml_tensor * ue_build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * ue_build_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps);
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueRelPositionMultiHeadedAttention;
class uePositionwiseFeedForward;
class ueConformerEncoderLayer {
  public:
    ueConformerEncoderLayer(int32_t                                            size,
                            std::shared_ptr<ueRelPositionMultiHeadedAttention> self_attn,
                            std::shared_ptr<uePositionwiseFeedForward>         feed_forward,
                            float                                              dropout_rate,
                            bool                                               normalize_before);
    void set_parameters(ggml_tensor * norm_ff_weight,
                        ggml_tensor * norm_ff_bias,
                        ggml_tensor * norm_mha_weight,
                        ggml_tensor * norm_mha_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x_ctb,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  pos_emb,
                                      ggml_tensor *  mask_pad,
                                      ggml_tensor *  att_cache,
                                      ggml_tensor *  cnn_cache,
                                      ggml_tensor ** new_att_cache_out,
                                      ggml_tensor ** new_cnn_cache_out) const;
    struct ueForwardOut {
        ggml_tensor * x_ctb         = nullptr;
        ggml_tensor * mask          = nullptr;
        ggml_tensor * new_att_cache = nullptr;
        ggml_tensor * new_cnn_cache = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx,
                         ggml_tensor *  x_ctb,
                         ggml_tensor *  mask,
                         ggml_tensor *  pos_emb,
                         ggml_tensor *  mask_pad  = nullptr,
                         ggml_tensor *  att_cache = nullptr,
                         ggml_tensor *  cnn_cache = nullptr) const {
        ueForwardOut out;
        out.mask  = mask;
        out.x_ctb = build_forward_graph(ctx, x_ctb, mask, pos_emb, mask_pad, att_cache, cnn_cache, &out.new_att_cache,
                                        &out.new_cnn_cache);
        return out;
    }
    int32_t size() const { return size_; }
    std::shared_ptr<ueRelPositionMultiHeadedAttention> self_attn() const { return self_attn_; }
    std::shared_ptr<uePositionwiseFeedForward> feed_forward() const { return feed_forward_; }
  private:
    int32_t size_             = 0;
    float   dropout_rate_     = 0.0f;
    bool    normalize_before_ = true;
    ggml_tensor * norm_ff_weight_  = nullptr;
    ggml_tensor * norm_ff_bias_    = nullptr;
    ggml_tensor * norm_mha_weight_ = nullptr;
    ggml_tensor * norm_mha_bias_   = nullptr;
    std::shared_ptr<ueRelPositionMultiHeadedAttention> self_attn_;
    std::shared_ptr<uePositionwiseFeedForward>         feed_forward_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueEspnetRelPositionalEncoding {
  public:
    ueEspnetRelPositionalEncoding(int32_t d_model, float dropout_rate, int32_t max_len);
    int32_t d_model() const { return d_model_; }
    int32_t max_len() const { return max_len_; }
    void extend_pe(ggml_tensor * x);
    void extend_pe(int32_t size);
    ggml_tensor * position_encoding(ggml_context * ctx, int32_t offset, int32_t size) const;
    ggml_tensor * position_encoding(ggml_context * ctx, std::nullptr_t, int32_t size) const;
    ggml_tensor * position_encoding(ggml_context * ctx, ggml_tensor *, int32_t size) const;
    void position_encoding_host(int32_t size, std::vector<float> & out_1td) const;
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x_ctb,
                                      int32_t        offset                  = 0,
                                      ggml_tensor ** pos_emb_placeholder_out = nullptr) const;
    struct ueForwardOut {
        ggml_tensor * x_ctb   = nullptr;
        ggml_tensor * pos_emb = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx, ggml_tensor * x_ctb, int32_t offset = 0) const {
        ueForwardOut out;
        out.x_ctb = build_forward_graph(ctx, x_ctb, offset, &out.pos_emb);
        return out;
    }
    ggml_tensor * build_position_encoding_placeholder(ggml_context * ctx, int32_t size, const char * name) const;
  private:
    int32_t d_model_      = 0;
    float   dropout_rate_ = 0.0f;
    int32_t max_len_      = 0;
    std::vector<float> pe_1td_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueMaskUtils {
  public:
    static ggml_tensor * make_pad_mask(ggml_context * ctx, ggml_tensor * lengths_b, int64_t max_len = 0);
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueUpsampleEncoderModelLoaderGGUF;
class ueUpsampleConformerEncoderV2;
class ueUpsampleEncoderGGUFModelRunner {
  public:
    ueUpsampleEncoderGGUFModelRunner();
    ~ueUpsampleEncoderGGUFModelRunner();
    ueUpsampleEncoderGGUFModelRunner(const ueUpsampleEncoderGGUFModelRunner &)             = delete;
    ueUpsampleEncoderGGUFModelRunner & operator=(const ueUpsampleEncoderGGUFModelRunner &) = delete;
    void reset();
    bool load_from_gguf(const std::string & gguf_path, const std::string & device);
    void set_num_threads(int n_threads);
    const std::string & backend_name() const { return backend_name_; }
    std::shared_ptr<ueUpsampleConformerEncoderV2> model() const { return model_; }
    int32_t output_size() const { return output_size_; }
    int32_t up_stride() const { return up_stride_; }
    int32_t pre_lookahead_len() const { return pre_lookahead_len_; }
    void reset_stream();
    bool forward(const float *          xs_btc,
                 const int32_t *        xs_lens_b,
                 int64_t                B,
                 int64_t                T,
                 int64_t                C,
                 std::vector<float> &   ys_btc_out,
                 std::vector<uint8_t> * masks_out) const;
    bool forward_chunk(const float *        xs_btc,
                       int64_t              B,
                       int64_t              dt,
                       int64_t              C,
                       bool                 last_chunk,
                       ueEncoderCacheHost * cache_in_out,
                       std::vector<float> & ys_btc_out);
  private:
    bool init_backend(const std::string & device);
    bool bind_parameters();
    int num_threads_ = 1;
    ggml_backend_t backend_ = nullptr;
    std::string    backend_name_;
    ggml_gallocr_t galloc_ = nullptr;
    std::string gguf_path_;
    std::unique_ptr<ueUpsampleEncoderModelLoaderGGUF> weights_;
    std::shared_ptr<ueUpsampleConformerEncoderV2>     model_;
    mutable int chunk_call_id_ = 0;
    int32_t output_size_       = 512;
    int32_t up_stride_         = 2;
    int32_t pre_lookahead_len_ = 3;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueMultiHeadedAttention {
  public:
    ueMultiHeadedAttention(int32_t n_head, int32_t n_feat, float dropout_rate, bool key_bias);
    void set_parameters(ggml_tensor * linear_q_weight,
                        ggml_tensor * linear_q_bias,
                        ggml_tensor * linear_k_weight,
                        ggml_tensor * linear_k_bias,
                        ggml_tensor * linear_v_weight,
                        ggml_tensor * linear_v_bias,
                        ggml_tensor * linear_out_weight,
                        ggml_tensor * linear_out_bias);
    void build_forward_qkv(ggml_context * ctx,
                           ggml_tensor *  query_ctb,
                           ggml_tensor *  key_ctb,
                           ggml_tensor *  value_ctb,
                           ggml_tensor ** q_dhtb_out,
                           ggml_tensor ** k_dhtb_out,
                           ggml_tensor ** v_dhtb_out) const;
    ggml_tensor * build_forward_attention(ggml_context * ctx,
                                          ggml_tensor *  value_dhtb,
                                          ggml_tensor *  scores_ktqh_b,
                                          ggml_tensor *  mask) const;
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  query_ctb,
                                      ggml_tensor *  key_ctb,
                                      ggml_tensor *  value_ctb,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  pos_emb,
                                      ggml_tensor *  cache,
                                      ggml_tensor ** new_cache_out) const;
    void forward_qkv(ggml_context * ctx,
                     ggml_tensor *  query_ctb,
                     ggml_tensor *  key_ctb,
                     ggml_tensor *  value_ctb,
                     ggml_tensor ** q_dhtb_out,
                     ggml_tensor ** k_dhtb_out,
                     ggml_tensor ** v_dhtb_out) const {
        build_forward_qkv(ctx, query_ctb, key_ctb, value_ctb, q_dhtb_out, k_dhtb_out, v_dhtb_out);
    }
    ggml_tensor * forward_attention(ggml_context * ctx,
                                    ggml_tensor *  value_dhtb,
                                    ggml_tensor *  scores_ktqh_b,
                                    ggml_tensor *  mask = nullptr) const {
        return build_forward_attention(ctx, value_dhtb, scores_ktqh_b, mask);
    }
    struct ueForwardOut {
        ggml_tensor * out_ctb   = nullptr;
        ggml_tensor * new_cache = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx,
                         ggml_tensor *  query_ctb,
                         ggml_tensor *  key_ctb,
                         ggml_tensor *  value_ctb,
                         ggml_tensor *  mask    = nullptr,
                         ggml_tensor *  pos_emb = nullptr,
                         ggml_tensor *  cache   = nullptr) const {
        ueForwardOut out;
        out.out_ctb = build_forward_graph(ctx, query_ctb, key_ctb, value_ctb, mask, pos_emb, cache, &out.new_cache);
        return out;
    }
    int32_t n_head() const { return n_head_; }
    int32_t n_feat() const { return n_feat_; }
    int32_t head_dim() const { return d_k_; }
  protected:
    int32_t n_head_       = 0;
    int32_t n_feat_       = 0;
    int32_t d_k_          = 0;
    float   dropout_rate_ = 0.0f;
    bool    key_bias_     = true;
    ggml_tensor * linear_q_weight_   = nullptr;
    ggml_tensor * linear_q_bias_     = nullptr;
    ggml_tensor * linear_k_weight_   = nullptr;
    ggml_tensor * linear_k_bias_     = nullptr;
    ggml_tensor * linear_v_weight_   = nullptr;
    ggml_tensor * linear_v_bias_     = nullptr;
    ggml_tensor * linear_out_weight_ = nullptr;
    ggml_tensor * linear_out_bias_   = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class uePositionwiseFeedForward {
  public:
    uePositionwiseFeedForward(int32_t idim, int32_t hidden_units, float dropout_rate, const char * activation = "relu");
    void set_parameters(
        ggml_tensor * w1_weight,
        ggml_tensor * w1_bias,
        ggml_tensor * w2_weight,
        ggml_tensor * w2_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * xs_ctb) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * xs_ctb) const { return build_forward_graph(ctx, xs_ctb); }
    int32_t idim() const { return idim_; }
    int32_t hidden_units() const { return hidden_units_; }
  private:
    int32_t      idim_         = 0;
    int32_t      hidden_units_ = 0;
    float        dropout_rate_ = 0.0f;
    const char * activation_   = "relu";
    ggml_tensor * w1_weight_ = nullptr;
    ggml_tensor * w1_bias_   = nullptr;
    ggml_tensor * w2_weight_ = nullptr;
    ggml_tensor * w2_bias_   = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class uePreLookaheadLayer {
  public:
    uePreLookaheadLayer(int32_t channels, int32_t pre_lookahead_len);
    void set_parameters(ggml_tensor * conv1_weight,
                        ggml_tensor * conv1_bias,
                        ggml_tensor * conv2_weight,
                        ggml_tensor * conv2_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * inputs_ctb) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  inputs_ctb,
                                            ggml_tensor *  cache_ctb,
                                            ggml_tensor ** new_cache_ctb_out) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * inputs_ctb) const {
        return build_forward_graph(ctx, inputs_ctb);
    }
    struct ueForwardChunkOut {
        ggml_tensor * outputs_ctb   = nullptr;
        ggml_tensor * new_cache_ctb = nullptr;
    };
    ueForwardChunkOut forward_chunk(ggml_context * ctx,
                                    ggml_tensor *  inputs_ctb,
                                    ggml_tensor *  cache_ctb = nullptr) const {
        ueForwardChunkOut out;
        out.outputs_ctb = build_forward_chunk_graph(ctx, inputs_ctb, cache_ctb, &out.new_cache_ctb);
        return out;
    }
    int32_t channels() const { return channels_; }
    int32_t pre_lookahead_len() const { return pre_lookahead_len_; }
    static constexpr int32_t cache_t() { return 2; }
  private:
    int32_t channels_          = 0;
    int32_t pre_lookahead_len_ = 0;
    ggml_tensor * conv1_weight_ = nullptr;
    ggml_tensor * conv1_bias_   = nullptr;
    ggml_tensor * conv2_weight_ = nullptr;
    ggml_tensor * conv2_bias_   = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueRelPositionMultiHeadedAttention final : public ueMultiHeadedAttention {
  public:
    ueRelPositionMultiHeadedAttention(int32_t n_head, int32_t n_feat, float dropout_rate, bool key_bias);
    void set_relpos_parameters(ggml_tensor * linear_pos_weight,
                               ggml_tensor * pos_bias_u,
                               ggml_tensor * pos_bias_v);
    ggml_tensor * build_rel_shift(ggml_context * ctx, ggml_tensor * x_tp_tq_hb) const;
    ggml_tensor * rel_shift(ggml_context * ctx, ggml_tensor * x_tp_tq_hb) const {
        return build_rel_shift(ctx, x_tp_tq_hb);
    }
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  query_ctb,
                                      ggml_tensor *  key_ctb,
                                      ggml_tensor *  value_ctb,
                                      ggml_tensor *  mask,
                                      ggml_tensor *  pos_emb,
                                      ggml_tensor *  cache,
                                      ggml_tensor ** new_cache_out) const;
    ueForwardOut forward(ggml_context * ctx,
                         ggml_tensor *  query_ctb,
                         ggml_tensor *  key_ctb,
                         ggml_tensor *  value_ctb,
                         ggml_tensor *  mask    = nullptr,
                         ggml_tensor *  pos_emb = nullptr,
                         ggml_tensor *  cache   = nullptr) const {
        ueForwardOut out;
        out.out_ctb = build_forward_graph(ctx, query_ctb, key_ctb, value_ctb, mask, pos_emb, cache, &out.new_cache);
        return out;
    }
  private:
    ggml_tensor * linear_pos_weight_ = nullptr;
    ggml_tensor * pos_bias_u_        = nullptr;
    ggml_tensor * pos_bias_v_        = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueEspnetRelPositionalEncoding;
class ueLinearNoSubsampling : public ueBaseSubsampling {
  public:
    ueLinearNoSubsampling(int32_t                                        idim,
                          int32_t                                        odim,
                          float                                          dropout_rate,
                          std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc);
    void set_parameters(ggml_tensor * linear_weight,
                        ggml_tensor * linear_bias,
                        ggml_tensor * ln_weight,
                        ggml_tensor * ln_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x_ctb,
                                      ggml_tensor *  x_mask,
                                      int32_t        offset,
                                      ggml_tensor ** pos_emb_placeholder_out,
                                      ggml_tensor ** out_mask_out) const;
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  x_ctb,
                                      std::nullptr_t,
                                      int32_t        offset,
                                      ggml_tensor ** pos_emb_placeholder_out) const;
    struct ueForwardOut {
        ggml_tensor * x_ctb   = nullptr;
        ggml_tensor * pos_emb = nullptr;
        ggml_tensor * x_mask  = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx, ggml_tensor * x_ctb, ggml_tensor * x_mask, int32_t offset = 0) const {
        ueForwardOut out;
        out.x_ctb = build_forward_graph(ctx, x_ctb, x_mask, offset, &out.pos_emb, &out.x_mask);
        return out;
    }
    std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc() const { return pos_enc_; }
    int32_t idim() const { return idim_; }
    int32_t odim() const { return odim_; }
  private:
    int32_t idim_         = 0;
    int32_t odim_         = 0;
    float   dropout_rate_ = 0.0f;
    ggml_tensor * linear_weight_ = nullptr;
    ggml_tensor * linear_bias_   = nullptr;
    ggml_tensor * ln_weight_     = nullptr;
    ggml_tensor * ln_bias_       = nullptr;
    std::shared_ptr<ueEspnetRelPositionalEncoding> pos_enc_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueUpsample1D {
  public:
    ueUpsample1D(int32_t channels, int32_t out_channels, int32_t stride, float scale_factor);
    void set_parameters(ggml_tensor * conv_weight, ggml_tensor * conv_bias);
    ggml_tensor * build_forward_graph(ggml_context * ctx, ggml_tensor * inputs_ctb) const;
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  inputs_ctb,
                                            ggml_tensor *  cache_ctb,
                                            ggml_tensor ** new_cache_ctb_out) const;
    struct ueForwardOut {
        ggml_tensor * outputs_ctb   = nullptr;
        ggml_tensor * out_lengths_b = nullptr;
    };
    struct ueForwardChunkOut {
        ggml_tensor * outputs_ctb   = nullptr;
        ggml_tensor * out_lengths_b = nullptr;
        ggml_tensor * new_cache_ctb = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx, ggml_tensor * inputs_ctb, ggml_tensor * input_lengths_b) const;
    ueForwardChunkOut forward_chunk(ggml_context * ctx,
                                    ggml_tensor *  inputs_ctb,
                                    ggml_tensor *  input_lengths_b = nullptr,
                                    ggml_tensor *  cache_ctb       = nullptr) const;
    int32_t channels() const { return channels_; }
    int32_t out_channels() const { return out_channels_; }
    int32_t stride() const { return stride_; }
    int32_t cache_t() const { return stride_ * 2; }
  private:
    int32_t channels_     = 0;
    int32_t out_channels_ = 0;
    int32_t stride_       = 2;
    float   scale_factor_ = 2.0f;
    ggml_tensor * conv_weight_ = nullptr;
    ggml_tensor * conv_bias_   = nullptr;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace upsample_encoder_v2 {
class ueLinearNoSubsampling;
class uePreLookaheadLayer;
class ueUpsample1D;
class ueConformerEncoderLayer;
class ueUpsampleConformerEncoderV2 {
  public:
    ueUpsampleConformerEncoderV2(int32_t input_size,
                                 int32_t output_size,
                                 int32_t pre_lookahead_len,
                                 int32_t num_blocks,
                                 int32_t num_up_blocks,
                                 int32_t up_stride,
                                 float   up_scale_factor,
                                 int32_t attention_heads,
                                 bool    key_bias,
                                 int32_t linear_units,
                                 float   dropout_rate,
                                 float   positional_dropout_rate,
                                 float   attention_dropout_rate,
                                 bool    normalize_before);
    void set_after_norm_parameters(ggml_tensor * after_norm_weight, ggml_tensor * after_norm_bias);
    std::shared_ptr<ueLinearNoSubsampling> embed() const { return embed_; }
    std::shared_ptr<ueLinearNoSubsampling> up_embed() const { return up_embed_; }
    std::shared_ptr<uePreLookaheadLayer> pre_lookahead_layer() const { return pre_lookahead_layer_; }
    std::shared_ptr<ueUpsample1D> up_layer() const { return up_layer_; }
    std::vector<std::shared_ptr<ueConformerEncoderLayer>> & encoders() { return encoders_; }
    std::vector<std::shared_ptr<ueConformerEncoderLayer>> & up_encoders() { return up_encoders_; }
    const std::vector<std::shared_ptr<ueConformerEncoderLayer>> & encoders() const { return encoders_; }
    const std::vector<std::shared_ptr<ueConformerEncoderLayer>> & up_encoders() const { return up_encoders_; }
    int32_t output_size() const { return output_size_; }
    bool normalize_before() const { return normalize_before_; }
    ggml_tensor * build_forward_graph(ggml_context * ctx,
                                      ggml_tensor *  xs_ctb,
                                      ggml_tensor *  masks,
                                      ggml_tensor ** out_masks_out) const;
    struct ueForwardOut {
        ggml_tensor * ys_ctb = nullptr;
        ggml_tensor * masks  = nullptr;
    };
    ueForwardOut forward(ggml_context * ctx, ggml_tensor * xs_ctb, ggml_tensor * xs_lens_b) const;
    ueForwardOut forward_with_masks(ggml_context * ctx, ggml_tensor * xs_ctb, ggml_tensor * masks) const {
        ueForwardOut out;
        out.ys_ctb = build_forward_graph(ctx, xs_ctb, masks, &out.masks);
        return out;
    }
    ggml_tensor * build_forward_chunk_graph(ggml_context * ctx,
                                            ggml_tensor *  xs_ctb,
                                            bool           last_chunk,
                                            ggml_tensor *  cnn_cache_ctb,
                                            ggml_tensor *  att_cache,
                                            ggml_tensor ** new_cnn_cache_ctb_out,
                                            ggml_tensor ** new_att_cache_out) const;
    struct ueForwardChunkOut {
        ggml_tensor * ys_ctb            = nullptr;
        ggml_tensor * new_cnn_cache_ctb = nullptr;
        ggml_tensor * new_att_cache     = nullptr;
    };
    ueForwardChunkOut forward_chunk(ggml_context * ctx,
                                    ggml_tensor *  xs_ctb,
                                    bool           last_chunk    = false,
                                    ggml_tensor *  cnn_cache_ctb = nullptr,
                                    ggml_tensor *  att_cache     = nullptr) const {
        ueForwardChunkOut out;
        out.ys_ctb = build_forward_chunk_graph(ctx, xs_ctb, last_chunk, cnn_cache_ctb, att_cache,
                                               &out.new_cnn_cache_ctb, &out.new_att_cache);
        return out;
    }
  private:
    int32_t input_size_              = 0;
    int32_t output_size_             = 0;
    int32_t pre_lookahead_len_       = 0;
    int32_t num_blocks_              = 0;
    int32_t num_up_blocks_           = 0;
    int32_t up_stride_               = 2;
    float   up_scale_factor_         = 2.0f;
    int32_t attention_heads_         = 0;
    bool    key_bias_                = true;
    int32_t linear_units_            = 0;
    float   dropout_rate_            = 0.0f;
    float   positional_dropout_rate_ = 0.0f;
    float   attention_dropout_rate_  = 0.0f;
    bool    normalize_before_        = true;
    ggml_tensor * after_norm_weight_ = nullptr;
    ggml_tensor * after_norm_bias_   = nullptr;
    std::shared_ptr<ueLinearNoSubsampling>                embed_;
    std::shared_ptr<uePreLookaheadLayer>                  pre_lookahead_layer_;
    std::vector<std::shared_ptr<ueConformerEncoderLayer>> encoders_;
    std::shared_ptr<ueUpsample1D>                         up_layer_;
    std::shared_ptr<ueLinearNoSubsampling>                up_embed_;
    std::vector<std::shared_ptr<ueConformerEncoderLayer>> up_encoders_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
struct gguf_context;
namespace omni {
namespace upsample_encoder_v2 {
class ueUpsampleEncoderModelLoaderGGUF {
  public:
    ueUpsampleEncoderModelLoaderGGUF();
    ~ueUpsampleEncoderModelLoaderGGUF();
    ueUpsampleEncoderModelLoaderGGUF(const ueUpsampleEncoderModelLoaderGGUF &)             = delete;
    ueUpsampleEncoderModelLoaderGGUF & operator=(const ueUpsampleEncoderModelLoaderGGUF &) = delete;
    void reset();
    bool load_from_file(const std::string & gguf_path, ggml_backend_t backend);
    ggml_tensor * get_tensor(const std::string & name) const;
    const std::string & path() const { return path_; }
  private:
    gguf_context *        ctx_gguf_    = nullptr;
    ggml_context *        ctx_meta_    = nullptr;
    ggml_context *        ctx_data_    = nullptr;
    ggml_backend_t        backend_     = nullptr;
    ggml_backend_buffer_t buf_weights_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    std::string                                    path_;
};
}  // namespace upsample_encoder_v2
}  // namespace omni
namespace omni {
namespace flow {
class flowGGUFModelLoader {
  public:
    flowGGUFModelLoader();
    ~flowGGUFModelLoader();
    flowGGUFModelLoader(const flowGGUFModelLoader &)             = delete;
    flowGGUFModelLoader & operator=(const flowGGUFModelLoader &) = delete;
    bool load_from_gguf(const std::string & encoder_gguf_path,
                        const std::string & flow_matching_gguf_path,
                        const std::string & flow_extra_gguf_path,
                        const std::string & device = "cpu");
    std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> encoder() const { return encoder_model_; }
    std::shared_ptr<flow_matching::fmDiT> estimator() const { return estimator_; }
    std::shared_ptr<flow_matching::fmCausalConditionalCFM> decoder() const { return decoder_; }
    std::shared_ptr<flowCausalMaskedDiffWithXvec> model() const { return flow_; }
    ggml_backend_t backend() const { return backend_; }
    const std::string & backend_name() const { return backend_name_; }
    const flowExtraModelLoaderGGUF * flow_extra_weights() const { return flow_extra_.get(); }
    const flow_matching::fmFlowMatchingModelLoaderGGUF * flow_matching_weights() const { return flow_matching_.get(); }
    const upsample_encoder_v2::ueUpsampleEncoderModelLoaderGGUF * encoder_weights() const {
        return encoder_weights_.get();
    }
  private:
    void reset();
    bool init_backend(const std::string & device);
    bool bind_all();
  private:
    ggml_backend_t backend_ = nullptr;
    std::string    backend_name_;
    std::unique_ptr<flowExtraModelLoaderGGUF>                          flow_extra_;
    std::unique_ptr<flow_matching::fmFlowMatchingModelLoaderGGUF>          flow_matching_;
    std::unique_ptr<upsample_encoder_v2::ueUpsampleEncoderModelLoaderGGUF> encoder_weights_;
    std::shared_ptr<upsample_encoder_v2::ueUpsampleConformerEncoderV2> encoder_model_;
    std::shared_ptr<flow_matching::fmDiT>                              estimator_;
    std::shared_ptr<flow_matching::fmCausalConditionalCFM>             decoder_;
    std::shared_ptr<flowCausalMaskedDiffWithXvec>                      flow_;
    int32_t encoder_pre_lookahead_len_ = 3;
};
}  // namespace flow
}  // namespace omni
namespace omni {
namespace flow {
struct flowStreamCacheHost {
    std::vector<uint8_t> conformer_cnn_cache;
    std::vector<int64_t> conformer_cnn_ne;
    std::vector<uint8_t> conformer_att_cache;
    std::vector<int64_t> conformer_att_ne;
    std::vector<uint8_t> estimator_cnn_cache;
    std::vector<int64_t> estimator_cnn_ne;
    std::vector<uint8_t> estimator_att_cache;
    std::vector<int64_t> estimator_att_ne;
    int n_timesteps = 0;
    void clear() {
        conformer_cnn_cache.clear();
        conformer_cnn_ne.clear();
        conformer_att_cache.clear();
        conformer_att_ne.clear();
        estimator_cnn_cache.clear();
        estimator_cnn_ne.clear();
        estimator_att_cache.clear();
        estimator_att_ne.clear();
        n_timesteps = 0;
    }
    bool empty() const {
        return conformer_cnn_cache.empty() && conformer_att_cache.empty() && estimator_cnn_cache.empty() &&
               estimator_att_cache.empty();
    }
};
class flowGGUFModelRunner {
  public:
    flowGGUFModelRunner();
    ~flowGGUFModelRunner();
    flowGGUFModelRunner(const flowGGUFModelRunner &)             = delete;
    flowGGUFModelRunner & operator=(const flowGGUFModelRunner &) = delete;
    void reset();
    bool load_from_gguf(const std::string & encoder_gguf_path,
                        const std::string & flow_matching_gguf_path,
                        const std::string & flow_extra_gguf_path,
                        const std::string & device);
    void set_num_threads(int n_threads);
    const std::string & backend_name() const { return loader_.backend_name(); }
    void set_export_caches_to_host(bool enable) { export_caches_to_host_ = enable; }
    void reset_stream();
    bool setup_cache(const int32_t *       token_bt,
                     int64_t               B,
                     int64_t               T_token,
                     const float *         mel_btc,
                     int64_t               T_mel,
                     int64_t               C_mel,
                     const float *         spk_bc,
                     int64_t               C_spk,
                     int                   n_timesteps,
                     float                 temperature,
                     flowStreamCacheHost & cache_out);
    bool inference_chunk(const int32_t *             token_bt,
                         int64_t                     B,
                         int64_t                     T_token,
                         const float *               spk_bc,
                         int64_t                     C_spk,
                         bool                        last_chunk,
                         const flowStreamCacheHost & cache_in,
                         int                         n_timesteps,
                         float                       temperature,
                         std::vector<float> &        mel_bct_out,
                         flowStreamCacheHost &       cache_out);
    bool init_from_host_caches(const flowStreamCacheHost & cache_host,
                               const float *               spk_bc,
                               int64_t                     B,
                               int                         n_timesteps,
                               float                       temperature);
  private:
    struct streamSession;
    int  num_threads_           = 1;
    bool export_caches_to_host_ = true;
    flowGGUFModelLoader            loader_;
    std::unique_ptr<streamSession> sess_;
};
}  // namespace flow
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_f0_predictor {
    ggml_tensor * conv0_weight = nullptr;
    ggml_tensor * conv0_bias   = nullptr;
    ggml_tensor * conv1_weight = nullptr;
    ggml_tensor * conv1_bias   = nullptr;
    ggml_tensor * conv2_weight = nullptr;
    ggml_tensor * conv2_bias   = nullptr;
    ggml_tensor * conv3_weight = nullptr;
    ggml_tensor * conv3_bias   = nullptr;
    ggml_tensor * conv4_weight = nullptr;
    ggml_tensor * conv4_bias   = nullptr;
    ggml_tensor * linear_weight = nullptr;
    ggml_tensor * linear_bias   = nullptr;
    ggml_tensor * hg_f0_predictor_build_graph(ggml_context * ctx, ggml_tensor * x_c80_t_b) const;
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
struct gguf_context;
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_gguf_model_loader {
    gguf_context *        ctx_gguf    = nullptr;
    ggml_context *        ctx_meta    = nullptr;
    ggml_context *        ctx_data    = nullptr;
    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::string                                    path;
    void          hg_gguf_model_loader_reset();
    bool          hg_gguf_model_loader_load_from_file(const std::string & gguf_path, ggml_backend_t backend_in);
    ggml_tensor * hg_gguf_model_loader_get_tensor(const std::string & name) const;
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_ops {
    static int32_t hg_ops_get_padding(int32_t kernel_size, int32_t dilation);
    static void hg_ops_init_weights_for_debug();
    static ggml_tensor * hg_ops_reflect_pad_left_1(ggml_context * ctx, ggml_tensor * x_tcb);
    static int32_t hg_ops_conv_transpose1d_out_len(int32_t in_len,
                                                    int32_t stride,
                                                    int32_t padding,
                                                    int32_t kernel_size,
                                                    int32_t dilation,
                                                    int32_t out_pad);
    static ggml_tensor * hg_ops_sample_phase_zeros(ggml_context * ctx,
                                                    int64_t        ne0,
                                                    int64_t        ne1,
                                                    int64_t        ne2,
                                                    int64_t        ne3);
    static ggml_tensor * hg_ops_sample_noise_zeros_like(ggml_context * ctx, ggml_tensor * ref);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_sine_gen2 {
    static constexpr int32_t HG2_SAMPLING_RATE  = 24000;
    static constexpr int32_t HG2_UPSAMPLE_SCALE = 480;
    int32_t harmonic_num     = 8;
    int32_t dim              = 9;
    float   sine_amp         = 0.1f;
    float   noise_std        = 0.003f;
    float   voiced_threshold = 10.0f;
    bool    flag_for_pulse   = false;
    ggml_tensor * harmonic_mul = nullptr;
    std::vector<float> host_harmonic_mul;
    ggml_tensor * dbg_rad_dn_tdb   = nullptr;
    ggml_tensor * dbg_phase_up_tdb = nullptr;
    bool hg_sine_gen2_init(ggml_context * ctx, int32_t harmonic_num_in);
    bool hg_sine_gen2_upload_consts(ggml_backend_t backend);
    bool hg_sine_gen2_build_graph(ggml_context * ctx,
                                   ggml_tensor *  f0_t1_b,
                                   ggml_tensor ** out_sine_tdb,
                                   ggml_tensor ** out_uv_t1_b,
                                   ggml_tensor ** out_noise_tdb);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_snake {
    static ggml_tensor * hg_snake_build_graph(ggml_context * ctx, ggml_tensor * x_tcb, ggml_tensor * alpha_c);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_resblock {
    struct hg2_resblock_conv {
        ggml_tensor * weight_kic_oc = nullptr;
        ggml_tensor * bias_oc       = nullptr;
        int32_t       kernel_size   = 0;
        int32_t       dilation      = 1;
        int32_t       padding       = 0;
    };
    std::vector<ggml_tensor *> activations1_alpha;
    std::vector<ggml_tensor *> activations2_alpha;
    std::vector<hg2_resblock_conv> convs1;
    std::vector<hg2_resblock_conv> convs2;
    ggml_tensor * hg_resblock_build_graph(ggml_context * ctx, ggml_tensor * x_tcb) const;
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_source_nsf2 {
    ggml_tensor * linear_weight = nullptr;
    ggml_tensor * linear_bias   = nullptr;
    hg2_sine_gen2 sine_gen;
    bool hg_source_nsf2_build_graph(ggml_context * ctx,
                                     ggml_tensor *  f0_t1_b,
                                     ggml_tensor ** out_sine_merge_t1_b,
                                     ggml_tensor ** out_noise_t1_b,
                                     ggml_tensor ** out_uv_t1_b);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_stft16_params {
    static constexpr int32_t HG2_N_FFT = 16;
    static constexpr int32_t HG2_HOP   = 4;
    static constexpr int32_t HG2_PAD   = 8;
    static constexpr int32_t HG2_F     = 9;
    ggml_tensor * window    = nullptr;
    ggml_tensor * window_sq = nullptr;
    ggml_tensor * dft_cos_t = nullptr;
    ggml_tensor * dft_sin_t = nullptr;
    ggml_tensor * nyq_sign  = nullptr;
    ggml_tensor * istft_ola_kernel = nullptr;
    std::vector<float> host_window;
    std::vector<float> host_window_sq;
    std::vector<float> host_dft_cos_t;
    std::vector<float> host_dft_sin_t;
    std::vector<float> host_nyq_sign;
    std::vector<float> host_istft_ola_kernel;
    bool hg_stft16_params_init(ggml_context * ctx);
    bool hg_stft16_params_build_consts(ggml_context * ctx);
    bool hg_stft16_params_upload_consts(ggml_backend_t backend);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_istft16 {
    static bool hg_istft16_build_graph(ggml_context *            ctx,
                                        const hg2_stft16_params & params,
                                        ggml_tensor *             real_ftb,
                                        ggml_tensor *             imag_ftb,
                                        ggml_tensor **            out_y_tb);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_stft16 {
    static bool hg_stft16_build_graph(ggml_context *            ctx,
                                       const hg2_stft16_params & params,
                                       ggml_tensor *             x_tb,
                                       ggml_tensor **            out_real_ftb,
                                       ggml_tensor **            out_imag_ftb);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_hift_generator {
    static constexpr int32_t HG2_SAMPLING_RATE   = 24000;
    static constexpr int32_t HG2_N_MEL           = 80;
    static constexpr int32_t HG2_N_FFT           = 16;
    static constexpr int32_t HG2_HOP             = 4;
    static constexpr int32_t HG2_F               = 9;
    static constexpr int32_t HG2_SAMPLES_PER_MEL = 480;
    static constexpr int32_t HG2_UPSAMPLE_0 = 8;
    static constexpr int32_t HG2_UPSAMPLE_1 = 5;
    static constexpr int32_t HG2_UPSAMPLE_2 = 3;
    float lrelu_slope = 0.1f;
    float audio_limit = 0.99f;
    hg2_stft16_params dsp;
    hg2_f0_predictor  f0_pred;
    hg2_source_nsf2   source_nsf;
    ggml_tensor * conv_pre_weight = nullptr;
    ggml_tensor * conv_pre_bias   = nullptr;
    ggml_tensor * up0_weight = nullptr;
    ggml_tensor * up0_bias   = nullptr;
    ggml_tensor * up1_weight = nullptr;
    ggml_tensor * up1_bias   = nullptr;
    ggml_tensor * up2_weight = nullptr;
    ggml_tensor * up2_bias   = nullptr;
    ggml_tensor * source_down0_weight = nullptr;
    ggml_tensor * source_down0_bias   = nullptr;
    ggml_tensor * source_down1_weight = nullptr;
    ggml_tensor * source_down1_bias   = nullptr;
    ggml_tensor * source_down2_weight = nullptr;
    ggml_tensor * source_down2_bias   = nullptr;
    hg2_resblock source_rb0;
    hg2_resblock source_rb1;
    hg2_resblock source_rb2;
    std::vector<hg2_resblock> resblocks;
    ggml_tensor * conv_post_weight = nullptr;
    ggml_tensor * conv_post_bias   = nullptr;
    bool build_graph_forward(ggml_context * ctx,
                             ggml_tensor *  speech_feat_c80_t_b,
                             ggml_tensor *  cache_source_t1_b,
                             ggml_tensor ** out_wave_t_b,
                             ggml_tensor ** out_source_t1_b);
    bool build_graph_decode(ggml_context * ctx,
                            ggml_tensor *  speech_feat_c80_t_b,
                            ggml_tensor *  source_t1_b,
                            ggml_tensor ** out_wave_t_b);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_model {
    hg2_hift_generator gen;
    bool hg_model_bind_from_loader(const hg2_gguf_model_loader & loader);
};
}  // namespace hifigan2
}  // namespace vocoder
}  // namespace omni
namespace omni {
namespace vocoder {
namespace hifigan2 {
struct hg2_model;
}
struct voc_hg2_model;
struct voc_hg2_runner;
struct voc_hg2_model {
    std::string gguf_path;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc  = nullptr;
    std::shared_ptr<void> weights_owner;
    std::shared_ptr<hifigan2::hg2_model> hg2;
    int32_t num_threads = 1;
    bool voc_hg2_model_init_from_gguf(const std::string & gguf_path_in,
                                      const std::string & device,
                                      int32_t             num_threads_in);
    void voc_hg2_model_free();
};
struct voc_hg2_runner {
    voc_hg2_model * model = nullptr;
    bool voc_hg2_runner_build_graph(ggml_context * ctx,
                                    ggml_cgraph *  gf,
                                    ggml_tensor *  speech_feat_c80_t_b,
                                    ggml_tensor *  cache_source_t1_b,
                                    ggml_tensor ** out_wave_t_b,
                                    ggml_tensor ** out_source_t1_b) const;
    bool voc_hg2_runner_eval(const std::vector<float> & speech_feat_bct,
                             int64_t                    T_mel,
                             std::vector<float> &       out_wave_bt,
                             int64_t &                  out_T_audio) const;
    bool voc_hg2_runner_eval_stream(const std::vector<float> & speech_feat_bct,
                                    int64_t                    T_mel,
                                    const std::vector<float> & cache_source_bt1,
                                    int64_t                    Tc,
                                    std::vector<float> &       out_wave_bt,
                                    int64_t &                  out_T_audio,
                                    std::vector<float> &       out_source_bt1,
                                    int64_t &                  out_T_source) const;
};
}  // namespace vocoder
}  // namespace omni

namespace omni {
namespace flow {

class Token2Mel {
  public:
    struct PromptBundle {
        std::vector<int32_t> prompt_tokens_bt;
        std::vector<float>   prompt_mel_btc;
        std::vector<float>   spk_bc;

        int64_t B              = 1;
        int64_t T_prompt_token = 0;
        int64_t T_prompt_mel   = 0;
    };

    Token2Mel()  = default;
    ~Token2Mel() = default;

    Token2Mel(const Token2Mel &)             = delete;
    Token2Mel & operator=(const Token2Mel &) = delete;

    bool load_model(const std::string & encoder_gguf,
                    const std::string & flow_matching_gguf,
                    const std::string & flow_extra_gguf,
                    const std::string & device  = "gpu",
                    int                 threads = 8);

    static bool load_prompt_bundle_dir(const std::string & dir, PromptBundle & out);

    bool start_stream_with_prompt(const PromptBundle & prompt, int n_timesteps = 10, float temperature = 1.0f);

    bool start_stream_with_prompt_cache_gguf(const std::string & prompt_cache_gguf_path,
                                             int                 n_timesteps = -1,
                                             float               temperature = -1.0f);

    bool push_tokens(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & mel_bct_out);

    bool push_tokens(const std::vector<int32_t> & tokens, bool is_final, std::vector<float> & mel_bct_out) {
        return push_tokens(tokens.data(), (int64_t) tokens.size(), is_final, mel_bct_out);
    }

    void reset_stream();

    static constexpr int32_t kMelChannels  = 80;
    static constexpr int32_t kSpkDim       = 192;
    static constexpr int32_t kPadToken     = 4218;
    static constexpr int32_t kPreLookahead = 3;
    static constexpr int32_t kChunkMain    = 25;
    static constexpr int32_t kDt           = kChunkMain + kPreLookahead;

    static void append_bct_along_time(const std::vector<float> & src_bct,
                                      int64_t                    B,
                                      int64_t                    C,
                                      std::vector<float> &       dst_bct_inout);

  private:
    bool ensure_ready_for_infer() const;
    bool infer_one_chunk(const std::vector<int32_t> & chunk_bt, bool last_chunk, std::vector<float> & mel_bct);

    flowGGUFModelRunner runner_;
    bool                model_loaded_   = false;
    bool                stream_started_ = false;

    int   n_timesteps_ = 10;
    float temperature_ = 1.0f;

    std::vector<float> spk_bc_;

    flowStreamCacheHost cache_in_;
};

struct Token2MelSession {
    bool init_from_prompt_bundle(const std::string & encoder_gguf,
                                 const std::string & flow_matching_gguf,
                                 const std::string & flow_extra_gguf,
                                 const std::string & device,
                                 int                 threads,
                                 const std::string & prompt_bundle_dir,
                                 int                 n_timesteps = 10,
                                 float               temperature = 1.0f);

    bool init_from_prompt_cache_gguf(const std::string & encoder_gguf,
                                     const std::string & flow_matching_gguf,
                                     const std::string & flow_extra_gguf,
                                     const std::string & device,
                                     int                 threads,
                                     const std::string & prompt_cache_gguf_path,
                                     int                 n_timesteps = -1,
                                     float               temperature = -1.0f);

    bool feed_tokens(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & mel_bct_out);

    bool feed_tokens(const std::vector<int32_t> & tokens, bool is_final, std::vector<float> & mel_bct_out) {
        return feed_tokens(tokens.data(), (int64_t) tokens.size(), is_final, mel_bct_out);
    }

    bool feed_window(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & mel_bct_out);

    bool feed_window(const std::vector<int32_t> & tokens, bool is_final, std::vector<float> & mel_bct_out) {
        return feed_window(tokens.data(), (int64_t) tokens.size(), is_final, mel_bct_out);
    }

    void reset();

    Token2Mel t2m;

  private:
    std::vector<int32_t> pending_;
};

namespace token2wav_utils {

void append_bt_along_time_b1(const std::vector<float> & src_bt, std::vector<float> & dst_bt_inout);

void crop_bct_tail_b1(const std::vector<float> & in_bct,
                      int64_t                    C,
                      int64_t                    T,
                      int64_t                    keep_T,
                      std::vector<float> &       out_bct);

void crop_t_tail_b1(const std::vector<float> & in_bt, int64_t keep_T, std::vector<float> & out_bt);

void ensure_hamming_window_2n(int64_t n, std::vector<float> & window_out);

void fade_in_out_b1(std::vector<float> &       wave_inout,
                    const std::vector<float> & prev_tail,
                    const std::vector<float> & window_2n,
                    int64_t                    n);

}  // namespace token2wav_utils

class Token2Wav {
  public:
    Token2Wav()  = default;
    ~Token2Wav() = default;

    Token2Wav(const Token2Wav &)             = delete;
    Token2Wav & operator=(const Token2Wav &) = delete;

    bool load_models(const std::string & encoder_gguf,
                     const std::string & flow_matching_gguf,
                     const std::string & flow_extra_gguf,
                     const std::string & vocoder_gguf,
                     const std::string & device_token2mel = "gpu",
                     const std::string & device_vocoder   = "gpu");

    bool start_stream_with_prompt_cache_gguf(const std::string & prompt_cache_gguf_path,
                                             int                 n_timesteps = -1,
                                             float               temperature = -1.0f);

    bool start_stream_with_prompt(const Token2Mel::PromptBundle & prompt,
                                  int                             n_timesteps = 10,
                                  float                           temperature = 1.0f);

    bool push_tokens_window(const int32_t *      tokens,
                            int64_t              n_tokens,
                            bool                 is_final,
                            std::vector<float> & wave_bt_out,
                            int64_t &            out_T_audio);

    bool push_tokens_window(const std::vector<int32_t> & tokens,
                            bool                         is_final,
                            std::vector<float> &         wave_bt_out,
                            int64_t &                    out_T_audio) {
        return push_tokens_window(tokens.data(), (int64_t) tokens.size(), is_final, wave_bt_out, out_T_audio);
    }

    void reset_stream();

    static constexpr int32_t kSampleRate = omni::vocoder::hifigan2::hg2_hift_generator::HG2_SAMPLING_RATE;

  private:
    Token2Mel t2m_;

    omni::vocoder::voc_hg2_model  voc_model_{};
    omni::vocoder::voc_hg2_runner voc_runner_{};

    static constexpr int32_t kMelCacheLen    = 8;
    static constexpr int32_t kSamplesPerMel  = omni::vocoder::hifigan2::hg2_hift_generator::HG2_SAMPLES_PER_MEL;
    static constexpr int32_t kSourceCacheLen = kMelCacheLen * kSamplesPerMel;

    std::vector<float> voc_mel_cache_bct_;

    std::vector<float> voc_cache_source_bt1_;
    int64_t            voc_Tc_ = 0;

    std::vector<float> voc_speech_cache_bt_;
    std::vector<float> voc_speech_window_;

    bool models_loaded_ = false;
};

struct Token2WavSession {
    bool init_from_prompt_cache_gguf(const std::string & encoder_gguf,
                                     const std::string & flow_matching_gguf,
                                     const std::string & flow_extra_gguf,
                                     const std::string & prompt_cache_gguf_path,
                                     const std::string & vocoder_gguf,
                                     const std::string & device_token2mel = "gpu",
                                     const std::string & device_vocoder   = "gpu",
                                     int                 n_timesteps      = 10,
                                     float               temperature      = 1.0f);

    bool init_from_prompt_bundle(const std::string & encoder_gguf,
                                 const std::string & flow_matching_gguf,
                                 const std::string & flow_extra_gguf,
                                 const std::string & prompt_bundle_dir,
                                 const std::string & vocoder_gguf,
                                 const std::string & device_token2mel = "gpu",
                                 const std::string & device_vocoder   = "gpu",
                                 int                 n_timesteps      = 10,
                                 float               temperature      = 1.0f);

    bool feed_tokens(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & wave_bt_out);

    bool feed_tokens(const std::vector<int32_t> & tokens, bool is_final, std::vector<float> & wave_bt_out) {
        return feed_tokens(tokens.data(), (int64_t) tokens.size(), is_final, wave_bt_out);
    }

    bool feed_window(const int32_t * tokens, int64_t n_tokens, bool is_final, std::vector<float> & wave_bt_out);

    bool feed_window(const std::vector<int32_t> & tokens, bool is_final, std::vector<float> & wave_bt_out) {
        return feed_window(tokens.data(), (int64_t) tokens.size(), is_final, wave_bt_out);
    }

    using audio_chunk_callback = std::function<void(const float * pcm, int64_t n_samples)>;

    bool feed_window(const int32_t *              tokens,
                     int64_t                      n_tokens,
                     bool                         is_final,
                     const audio_chunk_callback & on_audio_chunk);

    bool feed_window(const std::vector<int32_t> & tokens, bool is_final, const audio_chunk_callback & on_audio_chunk) {
        return feed_window(tokens.data(), (int64_t) tokens.size(), is_final, on_audio_chunk);
    }

    bool feed_tokens(const int32_t *              tokens,
                     int64_t                      n_tokens,
                     bool                         is_final,
                     const audio_chunk_callback & on_audio_chunk);

    bool feed_tokens(const std::vector<int32_t> & tokens, bool is_final, const audio_chunk_callback & on_audio_chunk) {
        return feed_tokens(tokens.data(), (int64_t) tokens.size(), is_final, on_audio_chunk);
    }

    void reset();

    Token2Wav t2w;

  private:
    std::vector<int32_t> pending_;
    std::vector<float>   wave_tmp_;
};

}  // namespace flow
}  // namespace omni
