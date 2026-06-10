#pragma once

#include "ggml-alloc.h"
#include "voxcpm2_audiovae.h"
#include "voxcpm2_cfm.h"
#include "voxcpm2_components.h"
#include "voxcpm2_fsq.h"
#include "voxcpm2_llm.h"
#include "voxcpm2_localenc.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using VoxCPM2AudioChunkCallback = std::function<void(const std::vector<float> & pcm, bool is_final)>;

struct VoxCPM2GenerateParams {
    int      max_steps             = 200;
    int      min_steps             = 2;
    int      inference_timesteps   = 10;
    float    cfg_value             = 2.0f;
    float    temperature           = 1.0f;
    int      target_sr             = 48000;
    int      reference_sample_rate = 0;  // 0 means AudioVAE input sample rate
    uint32_t seed                  = 0;
    bool     stop_on_predictor     = true;
    bool     append_audio_start    = true;
};

struct VoxCPM2PrefillInputs {
    // Token sequence already arranged on the VoxCPM2 timeline.
    std::vector<int32_t> token_ids;

    // Optional masks. Empty text_mask means every position is text.
    // Empty feat_mask means no audio feature positions.
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;

    // GGML layout [feat_dim, patch_size, seq_len]. Required if any feat_mask is non-zero.
    std::vector<float> audio_feat;
};

struct VoxCPM2DecodeStepResult {
    std::vector<float> latent_patch;   // [feat_dim, patch_size], GGML layout
    std::vector<float> current_embed;  // [lm_hidden_size]
    float              continue_logit = 0.0f;
    float              stop_logit     = 0.0f;
    bool               should_stop    = false;
    int                position       = 0;
};

struct VoxCPM2TokenEmbeddingTable {
    std::string path;
    size_t      data_offset = 0;
    size_t      row_bytes   = 0;
    int         n_embd      = 0;
    int         n_vocab     = 0;
    ggml_type   type        = GGML_TYPE_COUNT;

    bool load(const std::string & gguf_path);
    bool embedding_for_token(int32_t token_id, std::vector<float> & dst) const;
    void free();
};

struct VoxCPM2ResidualLM {
    VoxCPM2TransformerConfig  config;
    VoxCPM2TransformerWeights weights;
    VoxCPM2KVCache            kv_cache;

    ggml_backend_t                          backend = nullptr;  // not owned
    std::unique_ptr<VoxCPM2GGUFWeightStore> store;

    VoxCPM2ResidualLM() = default;
    ~VoxCPM2ResidualLM();

    VoxCPM2ResidualLM(const VoxCPM2ResidualLM &)             = delete;
    VoxCPM2ResidualLM & operator=(const VoxCPM2ResidualLM &) = delete;
    VoxCPM2ResidualLM(VoxCPM2ResidualLM &&)                  = delete;
    VoxCPM2ResidualLM & operator=(VoxCPM2ResidualLM &&)      = delete;

    bool               init_from_gguf(const std::string & path, ggml_backend_t backend);
    // Full-sequence forward (no KV cache)
    std::vector<float> forward(const std::vector<float> & input, int seq_len) const;
    std::vector<float> forward_last(const std::vector<float> & input, int seq_len) const;
    // Prefill with KV cache (populates cache positions 0..seq_len-1, returns last hidden)
    std::vector<float> prefill_kv(const std::vector<float> & input, int seq_len);
    // Single-step decode with KV cache (processes 1 token at `position`)
    std::vector<float> forward_step(const std::vector<float> & input_1d, int position);
    void               clear_kv();
    void               free();
};

struct VoxCPM2Runtime {
    MiniCPMLM                  base_lm;
    VoxCPM2ResidualLM          residual_lm;
    LocEncModel                loc_enc;
    LocDiTModel                loc_dit;
    UnifiedCFMSolver           cfm_solver;
    FSQModule                  fsq;
    AudioVAEModel              audio_vae;
    VoxCPM2Projections         projections;
    StopTokenPredictor         stop_predictor;
    VoxCPM2TokenEmbeddingTable token_embeddings;

    std::vector<float>              lm_hidden;
    std::vector<float>              residual_hidden;
    std::vector<float>              prefix_feat_cond;
    std::vector<float>              residual_input_history;
    std::vector<std::vector<float>> output_pool;

    int current_position  = 0;
    int audio_frame_count = 0;

    VoxCPM2Runtime();
    ~VoxCPM2Runtime();

    VoxCPM2Runtime(const VoxCPM2Runtime &)             = delete;
    VoxCPM2Runtime & operator=(const VoxCPM2Runtime &) = delete;
    VoxCPM2Runtime(VoxCPM2Runtime &&)                  = delete;
    VoxCPM2Runtime & operator=(VoxCPM2Runtime &&)      = delete;

    bool init(const std::string & base_lm_path,
              const std::string & acoustic_path,
              int                 n_gpu_layers    = -1,
              bool                use_gpu_backend = true);

    bool prefill(const VoxCPM2PrefillInputs & inputs);
    bool prefill_tokens(const std::vector<int32_t> & token_ids);

    VoxCPM2DecodeStepResult decode_step(const VoxCPM2GenerateParams & params = {});
    VoxCPM2DecodeStepResult decode_step(const std::vector<float> & noise, const VoxCPM2GenerateParams & params = {});

    void decode_loop(const VoxCPM2GenerateParams &                                params,
                     const std::function<void(const VoxCPM2DecodeStepResult &)> & callback = nullptr);

    std::vector<float>   decode_to_waveform(int target_sr = 0);
    std::vector<int32_t> tokenize_text(const std::string & text, bool add_special = true, bool parse_special = true);
    std::vector<float>   encode_reference_audio(const std::vector<float> & reference_wav, int sample_rate = 0);
    std::vector<float>   generate_tokens(const std::vector<int32_t> &  token_ids,
                                         const VoxCPM2GenerateParams & params = {});
    std::vector<float>   generate(const std::string & text, const VoxCPM2GenerateParams & params = {});
    std::vector<float>   generate_with_clone(const std::string &           text,
                                             const std::vector<float> &    reference_wav,
                                             const VoxCPM2GenerateParams & params = {});
    bool                 generate_tokens_streaming(const std::vector<int32_t> &      token_ids,
                                                   const VoxCPM2AudioChunkCallback & callback,
                                                   const VoxCPM2GenerateParams &     params = {});
    bool                 generate_streaming(const std::string &               text,
                                            const VoxCPM2AudioChunkCallback & callback,
                                            const VoxCPM2GenerateParams &     params = {});

    void reset_state();
    void free();

    bool initialized() const { return is_initialized; }

    bool ready() const { return state_ready; }

    bool text_tokenizer_available() const { return tokenizer_available; }

    const std::string & last_error() const { return last_error_msg; }

    int lm_hidden_size() const { return base_lm.n_embd; }

    int feat_dim() const { return loc_enc.config.feat_dim; }

    int patch_size() const { return loc_enc.config.patch_size; }

    int sample_rate() const { return audio_vae.config.output_sample_rate(); }

  private:
    ggml_backend_t                                    backend             = nullptr;  // owned
    bool                                              is_initialized      = false;
    bool                                              state_ready         = false;
    bool                                              tokenizer_available = false;
    float                                             embedding_scale     = 12.0f;
    std::string                                       last_error_msg;
    std::mt19937                                      rng;
    std::unordered_map<int32_t, std::vector<int32_t>> cjk_split_map;
    std::unordered_map<int32_t, std::vector<int32_t>> cjk_prefix_single_split_map;
    std::unordered_set<int32_t>                       cjk_token_ids;

    // Cached decode-front-half graph (CFM solve + LocEnc + stop predictor).
    // Built once per (timesteps, cfg_value) pair and reused across decode steps.
    struct CachedDecodeFrontHalfGraph {
        ggml_context * ctx    = nullptr;
        ggml_cgraph *  graph  = nullptr;
        ggml_gallocr_t galloc = nullptr;

        // Graph inputs (set before each compute)
        ggml_tensor * noise_input = nullptr;  // [feat_dim, patch_size]
        ggml_tensor * lm_input    = nullptr;  // [lm_hidden_size]
        ggml_tensor * res_input   = nullptr;  // [lm_hidden_size]
        ggml_tensor * cond_input  = nullptr;  // [feat_dim, patch_size]

        // Graph outputs (read after each compute)
        ggml_tensor * patch_output = nullptr;  // [feat_dim, patch_size]
        ggml_tensor * embed_output = nullptr;  // [lm_hidden_size]
        ggml_tensor * stop_output  = nullptr;  // [2]

        int   cached_timesteps = 0;
        float cached_cfg       = 0.0f;

        void free_graph();
    } cached_front_half;

    bool fail(const std::string & message);
    void clear_error();

    std::vector<float>   random_noise();
    std::vector<float>   run_locenc_sequence(const std::vector<float> & feat, int seq_len);
    std::vector<float>   run_enc_to_lm(const std::vector<float> & locenc_hidden, int seq_len);
    std::vector<float>   run_fsq(const std::vector<float> & input, int seq_len);
    std::vector<float>   run_residual_fusion(const std::vector<float> & blended,
                                             const std::vector<float> & feat_embed,
                                             int                        seq_len);
    bool                 run_decode_front_half(const std::vector<float> &    noise,
                                               const VoxCPM2GenerateParams & params,
                                               VoxCPM2DecodeStepResult &     result);
    bool                 ensure_decode_front_half_graph(int n_timesteps, float cfg_value);
    bool                 build_text_tokenizer_metadata(const std::string & base_lm_path);
    bool                 build_reference_prefill_inputs(const std::vector<int32_t> & text_token_ids,
                                                        const std::vector<float> &   reference_feat,
                                                        bool                         append_audio_start,
                                                        VoxCPM2PrefillInputs &       inputs);
    bool                 decode_streaming_from_ready_state(const VoxCPM2GenerateParams &     params,
                                                           const VoxCPM2AudioChunkCallback & callback);
    std::vector<int32_t> expand_multichar_cjk_tokens(const std::vector<int32_t> & ids) const;
};
