#include "voxcpm2_llm.h"
#include "llama.h"

#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// MiniCPMLM implementation
// ============================================================================

bool MiniCPMLM::init(const std::string & path, int n_gpu_layers) {
    // Load model
    struct llama_model_params model_params = llama_model_default_params();
    if (n_gpu_layers >= 0) {
        model_params.n_gpu_layers = n_gpu_layers;
    }

    model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model from %s\n", path.c_str());
        return false;
    }

    // Extract model dimensions
    n_embd    = llama_model_n_embd(model);
    n_layer   = llama_model_n_layer(model);
    n_head    = llama_model_n_head(model);
    n_kv_head = llama_model_n_head_kv(model);
    head_dim  = n_embd / n_head;    // 2048 / 16 = 128
    vocab     = llama_model_get_vocab(model);
    n_vocab   = vocab ? llama_vocab_n_tokens(vocab) : 0;

    LOG_INF("MiniCPMLM: n_embd=%d, n_layer=%d, n_head=%d, n_kv_head=%d, head_dim=%d\n",
            n_embd, n_layer, n_head, n_kv_head, head_dim);

    // Create context
    struct llama_context_params cparams_local = llama_context_default_params();
    cparams_local.n_ctx       = 32768;
    cparams_local.n_batch     = 1024;
    cparams_local.embeddings  = true;
    cparams_local.offload_kqv = true;
    cparams = cparams_local;

    ctx = llama_init_from_model(model, cparams_local);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    LOG_INF("MiniCPMLM: initialized successfully\n");
    return true;
}

bool MiniCPMLM::ensure_batch_size(int n_tokens, bool is_embd) {
    if (n_tokens > batch_size_alloc) {
        if (batch_size_alloc > 0) {
            llama_batch_free(batch);
        }
        int32_t embd_bytes = is_embd ? n_embd : 0;
        batch = llama_batch_init(n_tokens, embd_bytes, 1);
        batch_size_alloc = n_tokens;
    }
    batch.n_tokens = n_tokens;
    return true;
}

bool MiniCPMLM::prefill(const float * embd, int n_tokens, int n_past) {
    if (!ctx || !embd || n_tokens <= 0) {
        LOG_ERR("prefill: invalid args (ctx=%p, embd=%p, n_tokens=%d)\n",
                (void *) ctx, (const void *) embd, n_tokens);
        return false;
    }

    if (!ensure_batch_size(n_tokens, true)) {
        return false;
    }

    // Set up batch with embedding input.
    // batch.embd is a flat float* of [n_tokens * n_embd].
    // Each token's embedding starts at batch.embd + i * n_embd.
    // Note: when using embeddings, batch.token is NULL (not allocated by llama_batch_init).
    // When embeddings=true in context, llama.cpp requires all tokens marked as outputs.
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i]      = n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;
    }

    // Copy embeddings (row-major: [n_tokens, n_embd])
    memcpy(batch.embd, embd, n_tokens * n_embd * sizeof(float));

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR("prefill: llama_decode failed with code %d\n", ret);
        return false;
    }

    last_n_outputs = n_tokens;
    return true;
}

bool MiniCPMLM::decode_step(const float * embd, int n_past) {
    if (!ctx || !embd) {
        LOG_ERR("decode_step: invalid args\n");
        return false;
    }

    if (!ensure_batch_size(1, true)) {
        return false;
    }

    batch.pos[0]       = n_past;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0]  = 0;
    batch.logits[0]    = 1;

    // Single token embedding
    memcpy(batch.embd, embd, n_embd * sizeof(float));

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR("decode_step: llama_decode failed with code %d\n", ret);
        return false;
    }

    last_n_outputs = 1;
    return true;
}

std::vector<float> MiniCPMLM::get_last_hidden() const {
    if (!ctx) {
        return {};
    }

    const float * embd_out = llama_get_embeddings_ith(ctx, -1);
    if (!embd_out) {
        return {};
    }

    std::vector<float> result(n_embd);
    memcpy(result.data(), embd_out, n_embd * sizeof(float));
    return result;
}

std::vector<float> MiniCPMLM::get_all_hidden() const {
    if (!ctx || last_n_outputs <= 0) {
        return {};
    }

    const float * embd_out = llama_get_embeddings(ctx);
    if (!embd_out) {
        return {};
    }

    std::vector<float> result(static_cast<size_t>(n_embd) * static_cast<size_t>(last_n_outputs));
    memcpy(result.data(), embd_out, result.size() * sizeof(float));
    return result;
}

std::vector<float> MiniCPMLM::get_logits() const {
    if (!ctx) {
        return {};
    }

    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        return {};
    }

    int vocab_size = n_vocab > 0 ? n_vocab : 73448;

    std::vector<float> result(vocab_size);
    memcpy(result.data(), logits, vocab_size * sizeof(float));
    return result;
}

void MiniCPMLM::clear_kv_cache() {
    if (ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    last_n_outputs = 0;
}

int MiniCPMLM::get_kv_cache_size() const {
    if (!ctx) {
        return 0;
    }
    const llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    return max_pos < 0 ? 0 : static_cast<int>(max_pos + 1);
}

void MiniCPMLM::free() {
    if (batch_size_alloc > 0) {
        llama_batch_free(batch);
        batch_size_alloc = 0;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    vocab = nullptr;
    n_vocab = 0;
    last_n_outputs = 0;
}
