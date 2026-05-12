#include "tts-condition-graph.h"

#include "omni.h"

#include "common/log.h"
#include "ggml-backend.h"

static struct ggml_tensor * tts_condition_graph_build_l2_normalize(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * eps) {
    struct ggml_tensor * sq       = ggml_sqr(ctx, x);
    struct ggml_tensor * sum      = ggml_sum_rows(ctx, sq);
    struct ggml_tensor * eps_2d   = ggml_reshape_2d(ctx, eps, 1, 1);
    struct ggml_tensor * eps_rep  = ggml_repeat(ctx, eps_2d, sum);
    struct ggml_tensor * denom    = ggml_sqrt(ctx, ggml_add(ctx, sum, eps_rep));
    struct ggml_tensor * denom_r  = ggml_repeat(ctx, denom, x);
    struct ggml_tensor * normed   = ggml_div(ctx, x, denom_r);
    return ggml_cont(ctx, normed);
}

static struct ggml_cgraph * tts_condition_graph_build(
        struct omni_context *  ctx_omni,
        struct ggml_context *  ctx,
        struct ggml_tensor  *  token_ids,
        struct ggml_tensor  *  llm_hidden,
        struct ggml_tensor  *  eps,
        struct ggml_tensor  ** out_merged) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * llm_embeds = ggml_get_rows(
        ctx,
        ctx_omni->tts_condition_graph.emb_text_weight,
        token_ids);

    struct ggml_tensor * projected = ggml_mul_mat(
        ctx,
        ctx_omni->projector.layer.linear1_weight,
        llm_hidden);
    projected = ggml_add(ctx, projected, ctx_omni->projector.layer.linear1_bias);
    projected = ggml_relu(ctx, projected);
    projected = ggml_mul_mat(ctx, ctx_omni->projector.layer.linear2_weight, projected);
    projected = ggml_add(ctx, projected, ctx_omni->projector.layer.linear2_bias);

    struct ggml_tensor * normalized = tts_condition_graph_build_l2_normalize(ctx, projected, eps);
    struct ggml_tensor * merged = ggml_add(ctx, llm_embeds, normalized);
    merged = ggml_cont(ctx, merged);
    ggml_set_name(merged, "tts_condition_merged");
    ggml_set_output(merged);

    ggml_build_forward_expand(gf, merged);
    if (out_merged) {
        *out_merged = merged;
    }
    return gf;
}

bool tts_condition_graph_init(struct omni_context * ctx_omni) {
    if (!ctx_omni) {
        return false;
    }
    tts_condition_graph_model & model = ctx_omni->tts_condition_graph;
    if (model.initialized) {
        return true;
    }
    if (model.init_failed) {
        // Already failed once; don't retry on every chunk.
        return false;
    }

    auto fail = [&](const char * msg) {
        LOG_ERR("TTS condition graph: %s\n", msg);
        model.init_failed = true;
    };

    if (!ctx_omni->projector.initialized || !ctx_omni->projector.backend) {
        fail("projector graph is not initialized");
        return false;
    }
    if (!ctx_omni->emb_text_weight || ctx_omni->emb_text_vocab_size <= 0 || ctx_omni->emb_text_hidden_size <= 0) {
        fail("emb_text weights are not loaded");
        return false;
    }
    if (!ctx_omni->projector.layer.linear1_weight || !ctx_omni->projector.layer.linear1_bias ||
        !ctx_omni->projector.layer.linear2_weight || !ctx_omni->projector.layer.linear2_bias) {
        fail("projector tensors are incomplete");
        return false;
    }

    model.llm_hidden_dim  = ctx_omni->projector.hparams.in_dim;
    model.tts_hidden_dim  = ctx_omni->emb_text_hidden_size;
    model.text_vocab_size = ctx_omni->emb_text_vocab_size;
    model.backend         = ctx_omni->projector.backend;

    const size_t ctx_size = ggml_tensor_overhead() * 1;
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    model.ctx_w = ggml_init(ctx_params);
    if (!model.ctx_w) {
        fail("failed to create weight context");
        return false;
    }

    model.emb_text_weight = ggml_new_tensor_2d(
        model.ctx_w,
        GGML_TYPE_F32,
        model.tts_hidden_dim,
        model.text_vocab_size);
    ggml_set_name(model.emb_text_weight, "tts_condition_emb_text_weight");

    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) {
        fail("failed to allocate emb_text backend buffer");
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
        model.emb_text_weight = nullptr;
        return false;
    }

    const size_t emb_text_bytes =
        (size_t) model.text_vocab_size * (size_t) model.tts_hidden_dim * sizeof(float);
    ggml_backend_tensor_set(model.emb_text_weight, ctx_omni->emb_text_weight, 0, emb_text_bytes);

    model.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!model.galloc) {
        fail("failed to create ggml_gallocr");
        ggml_backend_buffer_free(model.buf_w);
        ggml_free(model.ctx_w);
        model.buf_w = nullptr;
        model.ctx_w = nullptr;
        model.emb_text_weight = nullptr;
        return false;
    }

    model.initialized = true;
    LOG_INF("TTS condition graph: initialized (vocab=%d, hidden=%d)\n",
            model.text_vocab_size, model.tts_hidden_dim);
    return true;
}

void tts_condition_graph_free(struct omni_context * ctx_omni) {
    if (!ctx_omni) {
        return;
    }

    // NOTE: this borrows the backend and the linear1/linear2 tensors from
    // ctx_omni->projector, so it must be freed BEFORE the projector itself
    // (omni_free currently does this).
    tts_condition_graph_model & model = ctx_omni->tts_condition_graph;
    if (model.galloc) {
        ggml_gallocr_free(model.galloc);
    }
    if (model.buf_w) {
        ggml_backend_buffer_free(model.buf_w);
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
    }

    model.ctx_w           = nullptr;
    model.buf_w           = nullptr;
    model.galloc          = nullptr;
    model.emb_text_weight = nullptr;
    model.backend         = nullptr;
    model.initialized     = false;
    model.init_failed     = false;
}

bool tts_condition_graph_forward(
        struct omni_context * ctx_omni,
        const llama_token * token_ids,
        const float * llm_hidden_states,
        int n_tokens,
        int llm_n_embd,
        std::vector<float> & merged_embeddings,
        int & tts_n_embd) {
    merged_embeddings.clear();
    tts_n_embd = 0;

    if (!ctx_omni || !token_ids || !llm_hidden_states) {
        return false;
    }
    if (!ctx_omni->tts_condition_graph.initialized) {
        LOG_ERR("TTS condition graph: forward called before init\n");
        return false;
    }
    if (n_tokens <= 0 || n_tokens > 10000) {
        LOG_ERR("TTS condition graph: invalid n_tokens=%d\n", n_tokens);
        return false;
    }

    tts_condition_graph_model & model = ctx_omni->tts_condition_graph;
    if (llm_n_embd != model.llm_hidden_dim) {
        LOG_ERR("TTS condition graph: llm_n_embd (%d) != expected (%d)\n",
                llm_n_embd, model.llm_hidden_dim);
        return false;
    }

    for (int i = 0; i < n_tokens; ++i) {
        if (token_ids[i] < 0 || token_ids[i] >= model.text_vocab_size) {
            LOG_ERR("TTS condition graph: token_id %d out of range [0, %d)\n",
                    token_ids[i], model.text_vocab_size);
            return false;
        }
    }

    const int out_dim = model.tts_hidden_dim;
    const size_t output_size = (size_t) n_tokens * (size_t) out_dim;
    if (output_size > 100000000) {
        LOG_ERR("TTS condition graph: output too large (%zu floats)\n", output_size);
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERR("TTS condition graph: failed to create compute context\n");
        return false;
    }

    struct ggml_tensor * token_ids_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids_tensor, "tts_condition_token_ids");
    ggml_set_input(token_ids_tensor);

    struct ggml_tensor * hidden_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, llm_n_embd, n_tokens);
    ggml_set_name(hidden_tensor, "tts_condition_llm_hidden");
    ggml_set_input(hidden_tensor);

    struct ggml_tensor * eps_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(eps_tensor, "tts_condition_norm_eps");
    ggml_set_input(eps_tensor);

    struct ggml_tensor * output = nullptr;
    struct ggml_cgraph * gf = tts_condition_graph_build(
        ctx_omni,
        ctx,
        token_ids_tensor,
        hidden_tensor,
        eps_tensor,
        &output);
    if (!output) {
        LOG_ERR("TTS condition graph: build did not return an output tensor\n");
        ggml_free(ctx);
        return false;
    }

    // Reuse the backend buffer across forward calls; gallocr will resize as
    // needed when n_tokens grows.
    if (!ggml_gallocr_alloc_graph(model.galloc, gf)) {
        LOG_ERR("TTS condition graph: ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> token_ids_i32(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        token_ids_i32[i] = (int32_t) token_ids[i];
    }

    const float eps = 1e-8f;
    ggml_backend_tensor_set(token_ids_tensor, token_ids_i32.data(), 0, token_ids_i32.size() * sizeof(int32_t));
    ggml_backend_tensor_set(hidden_tensor, llm_hidden_states, 0, (size_t) n_tokens * (size_t) llm_n_embd * sizeof(float));
    ggml_backend_tensor_set(eps_tensor, &eps, 0, sizeof(float));

    enum ggml_status status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("TTS condition graph: graph compute failed with status %d\n", (int) status);
        ggml_free(ctx);
        return false;
    }

    merged_embeddings.resize(output_size);
    ggml_backend_tensor_get(output, merged_embeddings.data(), 0, output_size * sizeof(float));
    tts_n_embd = out_dim;

    ggml_free(ctx);
    return true;
}
