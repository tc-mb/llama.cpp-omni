#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <vector>

struct omni_context;

// Fused graph for building TTS text condition embeddings:
// emb_text(token_ids) + normalize(projector_semantic(llm_hidden_states)).
struct tts_condition_graph_model {
    int32_t llm_hidden_dim = 0;
    int32_t tts_hidden_dim = 0;
    int32_t text_vocab_size = 0;

    // Non-owning backend borrowed from ctx_omni->projector.
    ggml_backend_t backend = nullptr;

    struct ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Reused across forward calls so we don't alloc/free a compute buffer per chunk.
    ggml_gallocr_t galloc = nullptr;

    struct ggml_tensor * emb_text_weight = nullptr; // [tts_hidden_dim, text_vocab_size]

    bool initialized = false;
    // Latched after a failed init so we don't retry (and re-spam errors) every chunk.
    bool init_failed = false;
};

bool tts_condition_graph_init(struct omni_context * ctx_omni);
void tts_condition_graph_free(struct omni_context * ctx_omni);

bool tts_condition_graph_forward(
    struct omni_context * ctx_omni,
    const llama_token * token_ids,
    const float * llm_hidden_states,
    int n_tokens,
    int llm_n_embd,
    std::vector<float> & merged_embeddings,
    int & tts_n_embd);
