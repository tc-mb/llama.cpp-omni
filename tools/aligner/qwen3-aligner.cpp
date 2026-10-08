#if defined(_WIN32)
// miniaudio pulls in windows.h; without this its min/max macros collide with std::max
#   define NOMINMAX
#   define WIN32_LEAN_AND_MEAN
#endif

#include "qwen3-aligner.h"

#include "mtmd.h"

#include "llama.h"
#include "ggml.h"
#include "gguf.h"

// Our own copy of miniaudio: the one in mtmd-helper is compiled with MA_API static, so its
// symbols are not reachable from here. Two copies coexist without clashing.
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Model constants; used when the GGUF does not carry them
constexpr int SAMPLE_RATE     = 16000;
constexpr int DEF_SEGMENT_MS  = 80;
constexpr int DEF_TS_TOKEN_ID = 151705;
constexpr int DEF_N_BUCKETS   = 5000;

// Two timestamp slots follow every unit: one for its start, one for its end
constexpr const char * TS_PAIR = "<timestamp><timestamp>";

/** Timestamp head: score.weight, [n_embd, n_buckets] in ggml order, i.e. n_buckets rows of n_embd */
struct score_head {
    std::vector<float> w;
    int n_buckets = 0;
    int n_embd    = 0;
};

/** Read score.weight and the three aligner.* keys straight from the audio GGUF, since mtmd does
 *  not expose arbitrary tensors. */
bool load_score_head(const std::string & path,
                     score_head &        head,
                     int &               segment_ms,
                     int &               ts_token_id,
                     std::string &       err) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };

    gguf_context * gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) {
        err = "cannot read audio GGUF: " + path;
        return false;
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_guard(gguf, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> meta_guard(meta, ggml_free);

    auto read_u32 = [&](const char * key, int fallback) {
        const int64_t k = gguf_find_key(gguf, key);
        return k < 0 ? fallback : (int) gguf_get_val_u32(gguf, k);
    };
    segment_ms  = read_u32("aligner.timestamp_segment_ms", DEF_SEGMENT_MS);
    ts_token_id = read_u32("aligner.timestamp_token_id",   DEF_TS_TOKEN_ID);
    const int n_buckets_kv = read_u32("aligner.n_buckets", DEF_N_BUCKETS);

    const int64_t idx = gguf_find_tensor(gguf, "score.weight");
    if (idx < 0) {
        err = "audio GGUF has no score.weight (timestamp head)";
        return false;
    }
    ggml_tensor * t = ggml_get_tensor(meta, "score.weight");
    if (!t) {
        err = "cannot read the metadata of score.weight";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        err = "score.weight is not F32";
        return false;
    }
    head.n_embd    = (int) t->ne[0];
    head.n_buckets = (int) t->ne[1];
    if (head.n_buckets != n_buckets_kv) {
        err = "score.weight row count does not match aligner.n_buckets";
        return false;
    }

    const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, idx);
    const size_t nbytes = ggml_nbytes(t);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open audio GGUF: " + path;
        return false;
    }
    head.w.resize(nbytes / sizeof(float));
    const bool ok = fseek(f, (long) offset, SEEK_SET) == 0 && fread(head.w.data(), 1, nbytes, f) == nbytes;
    fclose(f);
    if (!ok) {
        err = "failed to read the score.weight data";
        return false;
    }
    return true;
}

/**
 * Timestamp repair, matching the reference implementation step for step.
 *
 * The longest non-decreasing subsequence marks the "sane" predictions. Runs of up to two
 * outliers take whichever neighbouring sane value is closer; longer runs are interpolated
 * between them; a run with sane values on only one side takes that side.
 */
std::vector<int> fix_timestamps(const std::vector<double> & data) {
    const int n = (int) data.size();
    if (n == 0) return {};

    std::vector<int> dp(n, 1), parent(n, -1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i]     = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    const int max_len = *std::max_element(dp.begin(), dp.end());
    const int max_idx = (int) (std::find(dp.begin(), dp.end(), max_len) - dp.begin());

    std::vector<bool> is_normal(n, false);
    for (int idx = max_idx; idx != -1; idx = parent[idx]) {
        is_normal[idx] = true;
    }

    std::vector<double> result = data;
    int i = 0;
    while (i < n) {
        if (is_normal[i]) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && !is_normal[j]) j++;
        const int count = j - i;

        bool   has_left = false, has_right = false;
        double left_val = 0.0, right_val = 0.0;
        for (int k = i - 1; k >= 0; k--) {
            if (is_normal[k]) { left_val = result[k]; has_left = true; break; }
        }
        for (int k = j; k < n; k++) {
            if (is_normal[k]) { right_val = result[k]; has_right = true; break; }
        }

        if (count <= 2) {
            for (int k = i; k < j; k++) {
                if (!has_left)       result[k] = right_val;
                else if (!has_right) result[k] = left_val;
                else                 result[k] = (k - (i - 1)) <= (j - k) ? left_val : right_val;
            }
        } else if (has_left && has_right) {
            const double step = (right_val - left_val) / (count + 1);
            for (int k = i; k < j; k++) {
                result[k] = left_val + step * (k - i + 1);
            }
        } else if (has_left) {
            for (int k = i; k < j; k++) result[k] = left_val;
        } else if (has_right) {
            for (int k = i; k < j; k++) result[k] = right_val;
        }
        i = j;
    }

    std::vector<int> out(n);
    for (int k = 0; k < n; k++) out[k] = (int) result[k];
    return out;
}

} // namespace

struct qwen3_aligner {
    llama_model *       model = nullptr;
    llama_context *     lctx  = nullptr;
    mtmd_context *      mctx  = nullptr;

    score_head head;
    int segment_ms  = DEF_SEGMENT_MS;
    int ts_token_id = DEF_TS_TOKEN_ID;
    int n_embd_inp  = 0;
    int n_ctx       = 0;
    int n_threads   = 4;

    // One model serves one request at a time
    std::mutex mutex;

    ~qwen3_aligner() {
        if (mctx)  mtmd_free(mctx);
        if (lctx)  llama_free(lctx);
        if (model) llama_model_free(model);
    }
};

qwen3_aligner * qwen3_aligner_init(const qwen3_aligner_params & params, std::string & err) {
    auto ctx = std::make_unique<qwen3_aligner>();
    ctx->n_threads = std::max(1, params.n_threads);
    ctx->n_ctx     = params.n_ctx > 0 ? params.n_ctx : 4096;

    if (!load_score_head(params.audio_path, ctx->head, ctx->segment_ms, ctx->ts_token_id, err)) {
        return nullptr;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    ctx->model = llama_model_load_from_file(params.lm_path.c_str(), mp);
    if (!ctx->model) {
        err = "failed to load the backbone: " + params.lm_path;
        return nullptr;
    }
    ctx->n_embd_inp = llama_model_n_embd_inp(ctx->model);
    if (llama_model_n_embd(ctx->model) != ctx->head.n_embd) {
        err = "backbone hidden size does not match the timestamp head";
        return nullptr;
    }

    llama_context_params lp = llama_context_default_params();
    lp.n_ctx           = ctx->n_ctx;
    lp.n_batch         = ctx->n_ctx;
    lp.n_ubatch        = ctx->n_ctx;
    lp.n_threads       = ctx->n_threads;
    lp.n_threads_batch = ctx->n_threads;
    lp.embeddings      = true;                    // we want per-position hidden states, not sampling
    lp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    ctx->lctx = llama_init_from_model(ctx->model, lp);
    if (!ctx->lctx) {
        err = "failed to create the backbone context";
        return nullptr;
    }

    mtmd_context_params tp = mtmd_context_params_default();
    tp.use_gpu       = params.n_gpu_layers != 0;
    tp.print_timings = false;
    tp.n_threads     = ctx->n_threads;
    tp.warmup        = false;
    ctx->mctx = mtmd_init_from_file(params.audio_path.c_str(), ctx->model, tp);
    if (!ctx->mctx) {
        err = "failed to load the audio tower: " + params.audio_path;
        return nullptr;
    }
    if (!mtmd_support_audio(ctx->mctx)) {
        err = "this GGUF has no audio encoder";
        return nullptr;
    }
    if (mtmd_decode_use_mrope(ctx->mctx)) {
        err = "the aligner does not support models using mrope positions";
        return nullptr;
    }

    return ctx.release();
}

void qwen3_aligner_free(qwen3_aligner * ctx) {
    delete ctx;
}

int qwen3_aligner_sample_rate(const qwen3_aligner * ctx) {
    if (ctx && ctx->mctx) {
        const int sr = mtmd_get_audio_sample_rate(ctx->mctx);
        if (sr > 0) return sr;
    }
    return SAMPLE_RATE;
}

bool qwen3_aligner_decode_audio(qwen3_aligner *      ctx,
                                const void *         data,
                                size_t               size,
                                std::vector<float> & pcm,
                                std::string &        err) {
    if (!data || size == 0) {
        err = "empty audio data";
        return false;
    }
    const int sr = qwen3_aligner_sample_rate(ctx);

    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, sr);
    ma_decoder decoder;
    if (ma_decoder_init_memory(data, size, &cfg, &decoder) != MA_SUCCESS) {
        err = "failed to decode the audio (wav / mp3 / flac are supported)";
        return false;
    }

    ma_uint64 n_frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &n_frames) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        err = "cannot determine the audio length";
        return false;
    }

    pcm.resize((size_t) n_frames);
    ma_uint64 got = 0;
    const bool ok = ma_decoder_read_pcm_frames(&decoder, pcm.data(), n_frames, &got) == MA_SUCCESS;
    ma_decoder_uninit(&decoder);
    if (!ok) {
        err = "failed to read the audio frames";
        return false;
    }
    pcm.resize((size_t) got);
    return true;
}

bool qwen3_aligner_align(qwen3_aligner *                   ctx,
                         const float *                     pcm,
                         size_t                            n_samples,
                         const std::vector<std::string> &  units,
                         std::vector<qwen3_aligner_span> & spans,
                         std::string &                     err) {
    if (!ctx) {
        err = "aligner not loaded";
        return false;
    }
    spans.clear();
    if (units.empty()) {
        return true;
    }
    if (!pcm || n_samples == 0) {
        err = "empty audio";
        return false;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);

    // ── chunks: the audio, then every unit followed by two timestamp slots ─────────────

    std::string text = mtmd_default_marker();
    for (const auto & unit : units) {
        text += unit;
        text += TS_PAIR;
    }

    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
        mtmd_bitmap_init_from_audio(n_samples, pcm), mtmd_bitmap_free);
    if (!bitmap) {
        err = "failed to wrap the audio";
        return false;
    }

    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free);

    mtmd_input_text input {};
    input.text          = text.c_str();
    input.add_special   = false;
    input.parse_special = true;   // <timestamp> and <|audio_start|> are vocabulary tokens

    const mtmd_bitmap * bitmaps[1] = { bitmap.get() };
    if (mtmd_tokenize(ctx->mctx, chunks.get(), &input, bitmaps, 1) != 0) {
        err = "tokenization failed";
        return false;
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    if (n_chunks == 0) {
        err = "tokenization produced no chunks";
        return false;
    }

    size_t n_total = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        n_total += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks.get(), i));
    }
    if ((int) n_total > ctx->n_ctx) {
        err = "audio and text exceed the aligner context (" + std::to_string(n_total) + " > " +
              std::to_string(ctx->n_ctx) + "); raise --aligner-ctx-size or use shorter audio";
        return false;
    }

    // ── decode chunk by chunk; only the last one needs hidden states ───────────────────

    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    llama_pos n_past = 0;
    std::vector<int> ts_positions; // indices inside the last chunk

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), i);
        const bool is_last = (i + 1 == n_chunks);
        const bool non_causal = mtmd_decode_use_non_causal(ctx->mctx, chunk);
        if (non_causal) {
            llama_set_causal_attn(ctx->lctx, false);
        }

        int rc = 0;
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tok = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n_tok);

            llama_batch batch = llama_batch_init((int32_t) n_tok, 0, 1);
            for (size_t k = 0; k < n_tok; k++) {
                batch.token[k]     = toks[k];
                batch.pos[k]       = n_past + (llama_pos) k;
                batch.n_seq_id[k]  = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k]    = is_last;
                if (is_last && toks[k] == ctx->ts_token_id) {
                    ts_positions.push_back((int) k);
                }
            }
            batch.n_tokens = (int32_t) n_tok;
            rc = llama_decode(ctx->lctx, batch);
            llama_batch_free(batch);
        } else {
            if (mtmd_encode_chunk(ctx->mctx, chunk) != 0) {
                err = "audio encoding failed";
                return false;
            }
            const float * embd  = mtmd_get_output_embd(ctx->mctx);
            const size_t  n_tok = mtmd_input_chunk_get_n_tokens(chunk);

            llama_batch batch = llama_batch_init((int32_t) n_tok, ctx->n_embd_inp, 1);
            std::memcpy(batch.embd, embd, n_tok * ctx->n_embd_inp * sizeof(float));
            for (size_t k = 0; k < n_tok; k++) {
                batch.pos[k]       = n_past + (llama_pos) k;
                batch.n_seq_id[k]  = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k]    = false;
            }
            batch.n_tokens = (int32_t) n_tok;
            rc = llama_decode(ctx->lctx, batch);
            llama_batch_free(batch);
        }

        if (non_causal) {
            llama_set_causal_attn(ctx->lctx, true);
        }
        if (rc != 0) {
            err = "decode failed (chunk " + std::to_string(i) + ")";
            return false;
        }
        n_past += mtmd_input_chunk_get_n_pos(chunk);
    }

    if (ts_positions.size() != units.size() * 2) {
        err = "timestamp slot count does not match the unit count (" + std::to_string(ts_positions.size()) +
              " vs " + std::to_string(units.size() * 2) + "); check the units for special markers";
        return false;
    }

    // ── timestamp head ─────────────────────────────────────────────────────────────────

    std::vector<double> ts_ms;
    ts_ms.reserve(ts_positions.size());
    for (int idx : ts_positions) {
        const float * h = llama_get_embeddings_ith(ctx->lctx, idx);
        if (!h) {
            err = "no hidden state at a timestamp position";
            return false;
        }
        int   best_bucket = 0;
        float best_logit  = -INFINITY;
        for (int b = 0; b < ctx->head.n_buckets; b++) {
            const float * w = ctx->head.w.data() + (size_t) b * ctx->head.n_embd;
            float acc = 0.0f;
            for (int k = 0; k < ctx->head.n_embd; k++) acc += w[k] * h[k];
            if (acc > best_logit) {
                best_logit  = acc;
                best_bucket = b;
            }
        }
        ts_ms.push_back((double) best_bucket * ctx->segment_ms);
    }

    const std::vector<int> fixed = fix_timestamps(ts_ms);

    spans.resize(units.size());
    for (size_t i = 0; i < units.size(); i++) {
        spans[i].text  = units[i];
        spans[i].start = fixed[i * 2]     / 1000.0;
        spans[i].end   = fixed[i * 2 + 1] / 1000.0;
    }
    return true;
}
