#include "models.h"
#include "llama-moe-experts.h"
#include "llama-moe-slot.h"

#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>

void llama_model_minicpm5moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false)) {
        hparams.n_layer_dense_lead = 1;
    }

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);

    if (!ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false)) {
        hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
    }

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer) {
        case 28: type = LLM_TYPE_15B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_minicpm5moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp   = hparams.n_ff_exp;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;
    const int     n_dense    = (int) hparams.n_layer_dense_lead;

    // ── LES：路由专家内联在模型文件里，decode 阶段按层窗口按需读入 ──
    //
    // decode 窗口物理布局：每个 (family, 量化档位) 组一个 base，
    //   base: ne[2] = max(prefill 容量, Σ(组内各层窗口宽度))
    //     ├─ prefill view（组内共享）: offset 0
    //     └─ decode view（每层一份） : offset 前面各层宽度之和
    // prefill bank 与 decode 窗口共用同一段内存：decode 阶段开始时窗口总是先清空，prefill
    // 也用不到窗口，所以不必相加（qwen35moe 仍是相加的布局，它的 prewarm 依赖这一点）。
    // 保证安全的条件见 llama_model::moe_bank_window_overlap。
    // 组内所有层必须同档位，否则视图按自己的步长寻址、跨层静默互相覆盖。
    const int64_t dec_c = moe_decode_window();
    const int64_t k_pf  = moe_prefetch_slots();
    const std::vector<int32_t> layer_c = moe_layer_windows();
    const bool dec_enabled = dec_c > 0 && !layer_c.empty();
    const int64_t prefill_cap = moe_slot_bank_size();

    auto moe_idx_of = [&](int il) { return il - n_dense; };

    auto width_of = [&](int il) -> int64_t {
        const int mi = moe_idx_of(il);
        if (!dec_enabled || mi < 0 || mi >= (int) layer_c.size()) {
            return 0;
        }
        return (int64_t) layer_c[(size_t) mi] + k_pf;
    };

    const llm_tensor fam_ids[4] = {
        LLM_TENSOR_FFN_GATE_UP_EXPS, LLM_TENSOR_FFN_GATE_EXPS,
        LLM_TENSOR_FFN_UP_EXPS,      LLM_TENSOR_FFN_DOWN_EXPS,
    };

    // family -> (bytes_per_expert -> 层号升序)
    std::map<size_t, std::vector<int>> groups[4];
    if (moe_slot_bank_enabled()) {
        for (int il = n_dense; il < n_layer; ++il) {
            for (int fi = 0; fi < 4; ++fi) {
                const auto * e = moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str());
                if (e) {
                    groups[fi][e->bytes_per_expert].push_back(il);
                }
            }
        }
    }

    std::map<std::string, int64_t> group_slots;
    for (int fi = 0; fi < 4; ++fi) {
        for (const auto & kv : groups[fi]) {
            int64_t total = 0;
            for (int il : kv.second) {
                total += width_of(il);
            }
            group_slots[tn(fam_ids[fi], "weight", kv.second.front()).str()] =
                dec_enabled ? std::max(prefill_cap, total) : prefill_cap;
        }
    }
    auto slots_for = [&](const LLM_TN_IMPL & name) -> int64_t {
        const auto it = group_slots.find(name.str());
        return it == group_slots.end() ? prefill_cap : it->second;
    };

    auto create_routed_tensor = [&](const LLM_TN_IMPL & tensor_name,
                                    const std::initializer_list<int64_t> & dims,
                                    int flags) -> ggml_tensor * {
        // slot bank 关闭时专家就是普通权重，走常规加载（能进 mmap 映射区）。
        // 走 own-alloc 会让同一份专家权重再单独映射一遍，常驻内存翻倍。
        if (!moe_slot_bank_enabled()) {
            // an LES file without a slot bank: a strided view into the mapped carrier
            if (const auto * packed = moe_packed_entry(tensor_name.str().c_str())) {
                return create_packed_expert_view(ml, tensor_name, dims, *packed);
            }
            return create_tensor(tensor_name, dims, flags);
        }

        // 专家内容来自模型文件里的专家区，运行时 preadv 写入。所以按「非必需」创建，
        // 由后续的布局绑定决定它的位置。
        const bool required = !(flags & TENSOR_NOT_REQUIRED);
        ggml_tensor * tensor = create_tensor_own_alloc(ml, tensor_name, dims, flags | TENSOR_NOT_REQUIRED);

        const auto * entry = moe_experts_entry(tensor_name.str().c_str());

        if (!tensor && entry) {
            tensor = create_expert_tensor_from_manifest(ml, tensor_name, dims, entry->type);
        }
        if (!tensor) {
            if (required) {
                throw std::runtime_error(format(
                    "routed tensor '%s' is in neither the GGUF nor the MoE experts",
                    tensor_name.str().c_str()));
            }
            return tensor;
        }
        if (!entry) {
            throw std::runtime_error(format(
                "MoE experts is missing routed tensor '%s'", tensor_name.str().c_str()));
        }
        if (ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) n_expert) {
            throw std::runtime_error(format(
                "expert tensor '%s' byte size mismatch: tensor=%zu experts=%zu",
                tensor_name.str().c_str(), ggml_nbytes(tensor),
                entry->bytes_per_expert * (size_t) n_expert));
        }
        ml.virtualize_expert_tensor(tensor, n_expert, slots_for(tensor_name),
                                    entry->type, entry->bytes_per_expert);
        return tensor;
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // attn_q holds the query and the attention gate concatenated per head
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, 2 * n_head * n_embd_head_k}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_head_kv * n_embd_head_k}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_head_kv * n_embd_head_v}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < n_dense) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0 for MINICPM5MOE");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0 for MINICPM5MOE");
            }

            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            layer.ffn_down_exps = create_routed_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            layer.ffn_gate_up_exps = create_routed_tensor(
                tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);
            if (!layer.ffn_gate_up_exps) {
                layer.ffn_gate_exps = create_routed_tensor(
                    tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_up_exps = create_routed_tensor(
                    tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            }

            if (n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
            }
        }
    }

    // 每组一个物理 base，组内其它层是它的 view。prefill 视图沿用原名，
    // decode 视图加 `.dec` 后缀（moe_experts_entry() 会剥掉再查表）。
    if (!moe_slot_bank_enabled()) {
        return;
    }

    struct fam_t {
        ggml_tensor * llama_layer::* pf;
        ggml_tensor * llama_layer::* pf_dec;
    };
    const fam_t fams[4] = {
        { &llama_layer::ffn_gate_up_exps, &llama_layer::ffn_gate_up_exps_dec },
        { &llama_layer::ffn_gate_exps,    &llama_layer::ffn_gate_exps_dec    },
        { &llama_layer::ffn_up_exps,      &llama_layer::ffn_up_exps_dec      },
        { &llama_layer::ffn_down_exps,    &llama_layer::ffn_down_exps_dec    },
    };

    for (int fi = 0; fi < 4; ++fi) {
        for (const auto & kv : groups[fi]) {
            const std::vector<int> & members = kv.second;
            if (members.empty()) {
                continue;
            }
            // 组长必须是真正分配内存的那一层，其余层先别名过去
            ggml_tensor * base = layers[members.front()].*fams[fi].pf;
            if (!base) {
                continue;
            }
            const auto * e = moe_experts_entry(ggml_get_name(base));
            if (!e || e->bytes_per_expert == 0) {
                throw std::runtime_error(format(
                    "MoE base '%s' has no usable experts entry", ggml_get_name(base)));
            }

            for (size_t j = 1; j < members.size(); ++j) {
                ggml_tensor * t = layers[members[j]].*fams[fi].pf;
                if (t) {
                    ml.alias_expert_slot_tensor(t, base);
                }
            }

            // base 的 ne[2] 可能超过 Metal mul_mat_id 的上限，必须换成 view 才能进图
            for (size_t j = 0; j < members.size(); ++j) {
                auto & L = layers[members[j]];
                if (j == 0) {
                    const std::string orig = ggml_get_name(base);
                    ggml_set_name(base, (orig + ".base").c_str());
                    L.*fams[fi].pf = ml.window_expert_slot_tensor(
                        orig.c_str(), base, 0, prefill_cap, e->bytes_per_expert);
                }
                if (dec_enabled) {
                    moe_bank_window_overlap = true;
                    const std::string n2 = format("%s.dec", ggml_get_name(L.*fams[fi].pf));
                    int64_t off = 0;
                    for (size_t k = 0; k < j; ++k) {
                        off += width_of(members[k]);
                    }
                    L.*fams[fi].pf_dec = ml.window_expert_slot_tensor(
                        n2.c_str(), base, off, width_of(members[j]), e->bytes_per_expert);
                }
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minicpm5moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// ── Trained expert predictor (LLAMA_MOE_PREDICTOR=<path to gguf>) ────────────────
// Replaces the LES guess for block b (router of b applied to ffn_norm of b-1) with a
// trained linear predictor. predictor-b predicts block b (0-based; 1..27 = the MoE blocks,
// block 1's input comes from the dense block 0). Tensors per target block b:
//   blk.b.ffn_gate_inp_pred.weight       {n_embd, n_expert}  applied to ffn_norm of block b-1 (same token)
//   blk.b.ffn_gate_inp_pred_prev.weight  {n_embd, n_expert}  optional (current_prev): applied to
//                                                             ffn_norm of block b at t-1
// score_b = sigmoid(W_pred·x_{b-1}(t) [+ W_prev·x_b(t-1)]) + exp_probs_b[b].
// The exported blk.b.exp_probs_b_pred.bias equals the model's blk.b.exp_probs_b.bias bit for
// bit, so the model's bias is used and the predictor file need not carry it.
// The previous token's ffn_norm lives in a persistent F32 tensor with two banks of n_layer rows,
// owned by the context's slot runtime (llama_moe_slot_runtime::predictor_state), so two
// contexts on one model keep separate state; the weights here are read-only and shared. Each
// ubatch reads one bank and writes the other (rows picked by a graph input, see
// minicpm5moe_pred_state_input), so a read never shares a row with a write in the same graph
// and the result does not depend on the order in which the backend runs the nodes.
// Only used when lookahead prefetch is on; it changes what is prefetched, never the output.
struct llama_model_minicpm5moe::predictor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> w, wprev;   // [n_layer], nullptr where absent
    bool has_prev = false;                 // current_prev: needs the per-context _prev state
    ggml_backend_buffer_type_t buft = nullptr;   // where the weights (and the _prev state) live

    ~predictor() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

llama_model_minicpm5moe::llama_model_minicpm5moe(const struct llama_model_params & params) : llama_model_base(params) {}
llama_model_minicpm5moe::~llama_model_minicpm5moe() = default;

static std::unique_ptr<llama_model_minicpm5moe::predictor> minicpm5moe_load_predictor(const llama_model & model, const char * path) {
    gguf_context_ptr g;
    ggml_context_ptr meta;
    {
        ggml_context * meta_raw = nullptr;
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_raw };
        g.reset(gguf_init_from_file(path, gp));
        meta.reset(meta_raw);
    }
    if (!g) {
        throw std::runtime_error(format("cannot open %s", path));
    }

    const int n_layer = (int) model.layers.size();
    const int64_t n_embd = model.hparams.n_embd;

    auto P = std::make_unique<llama_model_minicpm5moe::predictor>();
    P->w.assign(n_layer, nullptr);
    P->wprev.assign(n_layer, nullptr);
    ggml_init_params ip = { ggml_tensor_overhead() * (2 * n_layer), nullptr, true };
    P->ctx = ggml_init(ip);

    struct item { ggml_tensor * dst; size_t off; };
    std::vector<item> items;
    const size_t data_off = gguf_get_data_offset(g.get());
    bool any_prev = false;
    for (int b = 0; b < n_layer; ++b) {
        for (int kind = 0; kind < 2; ++kind) {
            const std::string name = format(kind ? "blk.%d.ffn_gate_inp_pred_prev.weight" : "blk.%d.ffn_gate_inp_pred.weight", b);
            const int64_t id = gguf_find_tensor(g.get(), name.c_str());
            if (id < 0) {
                continue;
            }
            const ggml_tensor * src = ggml_get_tensor(meta.get(), name.c_str());
            if (!model.layers[b].ffn_gate_inp || src->ne[0] != n_embd ||
                src->ne[1] != model.layers[b].ffn_gate_inp->ne[1]) {
                throw std::runtime_error(format("%s: shape mismatch", name.c_str()));
            }
            ggml_tensor * t = ggml_dup_tensor(P->ctx, src);
            ggml_set_name(t, name.c_str());
            (kind ? P->wprev : P->w)[b] = t;
            any_prev |= kind == 1;
            items.push_back({ t, data_off + gguf_get_tensor_offset(g.get(), id) });
        }
    }
    if (items.empty()) {
        throw std::runtime_error("no blk.*.ffn_gate_inp_pred* tensors");
    }
    P->has_prev = any_prev;

    // same device as the routers
    const ggml_tensor * ref = nullptr;
    for (const auto & L : model.layers) {
        if (L.ffn_gate_inp) {
            ref = L.ffn_gate_inp;
            break;
        }
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(ref->buffer)));
    P->buft = buft;
    P->buf = ggml_backend_alloc_ctx_tensors_from_buft(P->ctx, buft);
    if (!P->buf) {
        throw std::runtime_error("buffer allocation failed");
    }
    ggml_backend_buffer_clear(P->buf, 0);

    std::unique_ptr<FILE, decltype(&fclose)> fp(fopen(path, "rb"), &fclose);
    std::vector<uint8_t> tmp;
    size_t total = 0;
    for (const auto & it : items) {
        const size_t n = ggml_nbytes(it.dst);
        tmp.resize(n);
        if (!fp || fseeko(fp.get(), (off_t) it.off, SEEK_SET) != 0 || fread(tmp.data(), 1, n, fp.get()) != n) {
            throw std::runtime_error(format("read failed for %s", ggml_get_name(it.dst)));
        }
        ggml_backend_tensor_set(it.dst, tmp.data(), 0, n);
        total += n;
    }

    int n_w = 0;
    for (auto * t : P->w) {
        n_w += t != nullptr;
    }
    LLAMA_LOG_INFO("[les] predictor: %s: %d target layers, prev=%s, type=%s, %.1f MiB on %s\n",
            path, n_w, any_prev ? "yes" : "no", ggml_type_name(items[0].dst->type),
            total / 1048576.0, ggml_backend_buft_name(buft));
    return P;
}

const llama_model_minicpm5moe::predictor * llama_model_minicpm5moe::get_predictor() const {
    std::call_once(pred_once, [this] {
        const char * path = std::getenv("LLAMA_MOE_PREDICTOR");
        if (!path || !*path) {
            LLAMA_LOG_INFO("[les] predictor: off (LLAMA_MOE_PREDICTOR unset), lookahead uses the LES router\n");
            return;
        }
        try {
            pred = minicpm5moe_load_predictor(*this, path);
        } catch (const std::exception & err) {
            LLAMA_LOG_WARN("[les] predictor: disabled, lookahead uses the LES router: %s\n", err.what());
            pred.reset();
        }
    });
    return pred.get();
}

// Row indices into the _prev state for one ubatch: read the bank the previous ubatch wrote,
// write the other one. Set on the host before every compute, so a reused graph stays correct.
class minicpm5moe_pred_state_input : public llm_graph_input_i {
public:
    minicpm5moe_pred_state_input(uint64_t & step, int n_layer) : step(step), n_layer(n_layer) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        const int64_t wr_bank = (int64_t) (step & 1);
        const int64_t rd_bank = 1 - wr_bank;
        std::vector<int32_t> rd(n_layer);
        std::vector<int64_t> wr(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            rd[il] = (int32_t) (rd_bank * n_layer + il);
            wr[il] = wr_bank * n_layer + il;
        }
        // A prefill graph makes no prediction, so rd_rows is not in it and has no buffer.
        if (rd_rows->buffer) {
            ggml_backend_tensor_set(rd_rows, rd.data(), 0, rd.size() * sizeof(int32_t));
        }
        if (wr_rows->buffer) {
            ggml_backend_tensor_set(wr_rows, wr.data(), 0, wr.size() * sizeof(int64_t));
        }
        step++;
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return true;   // fixed shape [n_layer]; the values are set every step
    }

    ggml_tensor * rd_rows = nullptr;   // I32 [n_layer], for ggml_get_rows
    ggml_tensor * wr_rows = nullptr;   // I64 [n_layer], for ggml_set_rows

private:
    uint64_t & step;   // the context's slot runtime counter (predictor_step)
    const int n_layer;
};

llama_model_minicpm5moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // lookahead prefetch only exists with the LES expert layout; it predicts on single-token
    // steps. The predictor's _prev state is also refreshed during prefill.
    const bool lookahead_on = moe_slots && moe_slots->lookahead_prefetch_enabled();
    const bool lookahead    = lookahead_on && ubatch.n_tokens == 1;
    const llama_model_minicpm5moe::predictor * pred = lookahead_on
        ? static_cast<const llama_model_minicpm5moe &>(model).get_predictor() : nullptr;

    // current_prev: rows of predictor::state this ubatch reads / writes (see minicpm5moe_pred_state_input)
    ggml_tensor * prev_rd_rows = nullptr;
    ggml_tensor * prev_wr_rows = nullptr;
    // current_prev: the previous token's ffn_norm, per context (in the slot runtime)
    ggml_tensor * prev_state = pred && pred->has_prev
        ? moe_slots->predictor_state(n_embd, 2 * n_layer, pred->buft) : nullptr;
    if (prev_state) {
        auto inp = std::make_unique<minicpm5moe_pred_state_input>(moe_slots->predictor_step(), n_layer);
        inp->rd_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_layer);
        inp->wr_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_layer);
        ggml_set_input(inp->rd_rows);
        ggml_set_input(inp->wr_rows);
        prev_rd_rows = inp->rd_rows;
        prev_wr_rows = inp->wr_rows;
        res->add_input(std::move(inp));
    }

    // Lookahead: predict block `target`'s experts from the ffn_norm of block target-1 (`x`).
    // With a trained predictor, predictor-`target` scores
    //   sigmoid(W_pred[target]·x [+ W_prev[target]·ffn_norm_target(t-1)]) + exp_probs_b[target];
    // without one, block target's own router is applied to x (LES). Top-`depth` ids; outputs
    // are never affected, only what gets prefetched.
    auto build_prediction = [&](ggml_tensor * x, int target) -> ggml_tensor * {
        const auto & tgt = model.layers[target];
        ggml_tensor * logits;
        if (pred && pred->w[target]) {
            logits = ggml_mul_mat(ctx0, pred->w[target], x);
            if (pred->wprev[target]) {
                // ffn_norm of block `target` at t-1, from the bank the previous ubatch wrote
                ggml_tensor * xprev = ggml_get_rows(ctx0, prev_state,
                    ggml_view_1d(ctx0, prev_rd_rows, 1, (size_t) target * prev_rd_rows->nb[0]));
                logits = ggml_add(ctx0, logits, ggml_mul_mat(ctx0, pred->wprev[target], xprev));
            }
        } else {
            logits = build_lora_mm(tgt.ffn_gate_inp, x);
        }
        ggml_tensor * scores = ggml_sigmoid(ctx0, logits);
        if (tgt.ffn_exp_probs_b) {
            scores = ggml_add(ctx0, scores, tgt.ffn_exp_probs_b);
        }
        ggml_tensor * ids = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, scores, moe_slots->lookahead_prefetch_depth()));
        cb(ids, "moe_lookahead_ids", target);
        return ids;
    };

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q_full = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(q_full, "q_full", il);

            // the q_proj output of each head is [query, gate]
            const int64_t nb1 = ggml_row_size(q_full->type, 2*n_embd_head);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, q_full, n_embd_head, n_head, n_tokens, nb1, nb1*n_head, 0);
            ggml_tensor * Qgate = ggml_view_3d(ctx0, q_full, n_embd_head, n_head, n_tokens, nb1, nb1*n_head,
                    ggml_row_size(q_full->type, n_embd_head));

            Qcur  = ggml_cont(ctx0, Qcur);
            Qgate = ggml_cont(ctx0, Qgate);

            cb(Qcur,  "Qcur",  il);
            cb(Qgate, "Qgate", il);

            ggml_tensor * Kcur = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model.layers[il].wk, cur), n_embd_head, n_head_kv, n_tokens);
            ggml_tensor * Vcur = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model.layers[il].wv, cur), n_embd_head, n_head_kv, n_tokens);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            Qgate = ggml_reshape_2d(ctx0, Qgate, n_embd_head * n_head, n_tokens);

            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, Qgate));
            cb(cur, "attn_out_gated", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_out_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (prev_state && pred->wprev[il] && cur->ne[1] > 0) {
            // keep the last token's ffn_norm for the next token's _prev term, in the bank
            // this ubatch writes (no prediction in this graph reads it)
            ggml_tensor * last = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (cur->ne[1] - 1) * cur->nb[1]);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, prev_state, last,
                ggml_view_1d(ctx0, prev_wr_rows, 1, (size_t) il * prev_wr_rows->nb[0])));
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // a dense layer has no per-layer CPU op to carry the prediction, so the next
            // (MoE) layer's prediction gets a prediction-only op of its own
            if (lookahead && il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp) {
                ggml_tensor * hand_off = moe_slots->build_lookahead_only(ctx0, build_prediction(cur, il + 1), il + 1);
                if (hand_off) {
                    ggml_build_forward_expand(gf, hand_off);
                }
            }
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // the prediction for layer il+1 rides on this layer's CPU op (build_slot_ids_tensor)
            ggml_tensor * prediction = nullptr;
            if (lookahead && il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp) {
                prediction = build_prediction(cur, il + 1);
                ggml_build_forward_expand(gf, prediction);
            }
            if (moe_slots) {
                moe_slots->set_lookahead_pred(prediction);
            }

            // LES：decode 用每层独立的窗口张量（base 上的 view），prefill 用共享窗口。
            // 两者 ne[2] 不同，必须按阶段选。
            const auto & L = model.layers[il];
            ggml_tensor * p_up   = moe_decode_phase && L.ffn_up_exps_dec      ? L.ffn_up_exps_dec      : L.ffn_up_exps;
            ggml_tensor * p_gate = moe_decode_phase && L.ffn_gate_exps_dec    ? L.ffn_gate_exps_dec    : L.ffn_gate_exps;
            ggml_tensor * p_down = moe_decode_phase && L.ffn_down_exps_dec    ? L.ffn_down_exps_dec    : L.ffn_down_exps;
            ggml_tensor * p_gu   = moe_decode_phase && L.ffn_gate_up_exps_dec ? L.ffn_gate_up_exps_dec : L.ffn_gate_up_exps;

            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    p_up,
                    p_gate,
                    p_down,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    p_gu);
            cb(moe_out, "ffn_moe_out", il);

            if (model.layers[il].ffn_gate_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                moe_out = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(moe_out, "ffn_out", il);
            }

            cur = moe_out;
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
