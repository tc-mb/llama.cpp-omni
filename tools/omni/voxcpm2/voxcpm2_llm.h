#pragma once

#include "llama.h"

#include <string>
#include <vector>

/**
 * MiniCPMLM: Wrapper around llama_model/llama_context for VoxCPM2 BaseLM.
 *
 * Provides embedding-based prefill and decode operations, hidden state
 * extraction, and KV cache management.
 *
 * BaseLM: 28-layer causal decoder (LLM_ARCH_MINICPM via llm_build_granite)
 *   n_embd=2048, n_head=16, n_kv_head=2, head_dim=128, n_ff=6144
 */
struct MiniCPMLM {
    struct llama_model   * model = nullptr;
    struct llama_context * ctx   = nullptr;
    const struct llama_vocab * vocab = nullptr;

    int n_embd    = 0;   // 2048
    int n_layer   = 0;   // 28
    int n_head    = 0;   // 16
    int n_kv_head = 0;   // 2
    int head_dim  = 0;   // 128
    int n_vocab   = 0;   // 73448

    // Reusable batch object
    struct llama_batch batch;
    int    batch_size_alloc = 0;   // allocated batch size
    float *batch_embd_data = nullptr;  // mutable embd buffer for batch

    // Context parameters
    struct llama_context_params cparams;

    // Number of output embeddings/logits produced by the last decode call.
    int last_n_outputs = 0;

    /**
     * Initialize from GGUF file.
     * @param path       GGUF file path (LLM_ARCH_MINICPM format)
     * @param n_gpu_layers  GPU offload layers (-1 = all)
     */
    bool init(const std::string & path, int n_gpu_layers = -1);

    /**
     * Prefill: feed a batch of embedding tokens into the model.
     * Builds the KV cache for all positions at once.
     *
     * @param embd      Float embeddings [n_embd * n_tokens], column-major
     * @param n_tokens  Number of tokens to prefill
     * @param n_past    Starting KV cache position (0 for initial prefill)
     * @return true on success
     */
    bool prefill(const float * embd, int n_tokens, int n_past = 0);

    /**
     * Decode step: feed a single embedding token into the model.
     * Appends one position to the KV cache.
     *
     * @param embd   Float embedding [n_embd]
     * @param n_past Current KV cache position
     * @return true on success
     */
    bool decode_step(const float * embd, int n_past);

    /**
     * Get the hidden state of the last output token.
     * Returns [n_embd] vector.
     */
    std::vector<float> get_last_hidden() const;

    /**
     * Get hidden states for all output tokens at the last layer.
     * Returns [n_embd * n_outputs] vector (column-major: hidden per token).
     */
    std::vector<float> get_all_hidden() const;

    /**
     * Get logits for the last output token.
     * Returns [n_vocab] vector.
     */
    std::vector<float> get_logits() const;

    /**
     * Clear KV cache to free memory and reset state.
     */
    void clear_kv_cache();

    /**
     * Get current KV cache token count.
     */
    int get_kv_cache_size() const;

    /**
     * Free all resources.
     */
    void free();

private:
    bool ensure_batch_size(int n_tokens, bool is_embd);
};
