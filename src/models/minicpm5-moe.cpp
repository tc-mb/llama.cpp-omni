#include "models.h"
#include "llama-moe-experts.h"
#include "llama-moe-slot.h"

void llama_model_minicpm5_moe::load_arch_hparams(llama_model_loader & ml) {
    if (hparams.n_embd_head_k_full != hparams.n_embd_head_v_full) {
        throw std::runtime_error("MiniCPM5 MoE requires equal key and value head sizes");
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    // the routed expert FFN size may be a single value or one value per layer
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer);
    hparams.n_ff_exp = hparams.n_ff_exp_arr[0];
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    // the reference routing is sigmoid + renormalized top-k + fixed scale; see modeling_minicpm.py
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // modeling_minicpm.py casts the router inputs and weights to F32 before the matmul
    hparams.expert_router_f32 = true;

    hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_minicpm5_moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int     n_dense    = (int) hparams.n_layer_dense_lead;

    // ── LES：路由专家内联在模型文件里，decode 阶段按层窗口按需读入 ──
    //
    // decode 窗口物理布局（与 qwen35moe 一致）：每个 (family, 量化档位) 组一个 base，
    //   base: ne[2] = prefill 容量 + Σ(组内各层窗口宽度)
    //     ├─ prefill view（组内共享）: offset 0
    //     └─ decode view（每层一份） : offset prefill 容量 + 前面各层宽度之和
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
                dec_enabled ? (prefill_cap + total) : prefill_cap;
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

        // attention projections
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // attention gating
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < n_dense) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0 for minicpm5_moe");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0 for minicpm5_moe");
            }

            const int64_t n_ff_exp_layer   = hparams.n_ff_exp_arr[i];
            const int64_t n_ff_shexp_layer = n_ff_exp_layer * hparams.n_expert_shared;

            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            // Trained expert predictor (optional): an approximate router for the *next* block,
            // used only to prefetch. Kept f32; it is a small file next to the experts.
            layer.ffn_gate_inp_pred = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_INP_PRED, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_inp_pred_prev = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_INP_PRED_PREV, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);

            layer.ffn_down_exps = create_routed_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp_layer, n_embd, n_expert}, 0);

            layer.ffn_gate_up_exps = create_routed_tensor(
                tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, n_ff_exp_layer * 2, n_expert}, TENSOR_NOT_REQUIRED);
            if (!layer.ffn_gate_up_exps) {
                layer.ffn_gate_exps = create_routed_tensor(
                    tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp_layer, n_expert}, 0);
                layer.ffn_up_exps = create_routed_tensor(
                    tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp_layer, n_expert}, 0);
            }

            if (n_ff_shexp_layer > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp_layer}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp_layer, n_embd}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp_layer}, 0);
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
                    const std::string n2 = format("%s.dec", ggml_get_name(L.*fams[fi].pf));
                    int64_t off = prefill_cap;
                    for (size_t k = 0; k < j; ++k) {
                        off += width_of(members[k]);
                    }
                    L.*fams[fi].pf_dec = ml.window_expert_slot_tensor(
                        n2.c_str(), base, off, width_of(members[j]), e->bytes_per_expert);
                }
            }
        }
    }

    bool file_has_pred = false;
    for (uint32_t il = hparams.n_layer_dense_lead; il < hparams.n_layer; ++il) {
        file_has_pred = file_has_pred || layers[il].ffn_gate_inp_pred != nullptr;
    }
    if (params.moe_expert_predictor) {
        for (uint32_t il = hparams.n_layer_dense_lead; il < hparams.n_layer; ++il) {
            if (!layers[il].ffn_gate_inp_pred) {
                throw std::runtime_error(format(
                        "moe_expert_predictor is on but the model file has no %s",
                        tn(LLM_TENSOR_FFN_GATE_INP_PRED, "weight", il).str().c_str()));
            }
        }
        LLAMA_LOG_INFO("%s: lookahead uses the trained expert predictor\n", __func__);
    } else if (file_has_pred && moe_slot_bank_enabled()) {
        LLAMA_LOG_INFO("%s: the file carries a trained expert predictor; pass --moe-predictor to use it, "
                "otherwise each block's own router predicts what to prefetch\n", __func__);
    }
}

std::unique_ptr<llm_graph_context> llama_model_minicpm5_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Picks the predictor state bank for this ubatch: read the one the previous ubatch wrote,
// write the other. The row lists are graph inputs, so the choice costs no graph rebuild.
class minicpm5_moe_pred_state_input : public llm_graph_input_i {
public:
    minicpm5_moe_pred_state_input(uint64_t & step, int n_layer) : step(step), n_layer(n_layer) {}

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

llama_model_minicpm5_moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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

    // Trained expert predictor: an approximate router for the *next* block's experts.
    // It only decides what to prefetch; the real router still picks the experts, so the
    // output does not depend on whether it is on. Without the trained weights the next
    // block's own router answers the same question, which is the plain LES behaviour.
    const bool lookahead   = moe_slots && moe_slots->lookahead_prefetch_enabled();
    const bool trained     = model.moe_expert_predictor_enabled();
    // `_prev` state is only read by the trained head; keep it for every ubatch (prefill
    // included) so the first decode step already has the last prefill token.
    const bool keep_prev   = lookahead && trained && !cparams.warmup;
    const bool predict     = lookahead && ubatch.n_tokens == 1;

    // current_prev: the previous token's ffn_norm per block. Two banks, one per ubatch parity:
    // each ubatch reads the bank the previous ubatch wrote and writes the other, so no row is
    // read and written by the same graph regardless of the order the backend runs the nodes.
    ggml_tensor * prev_state   = nullptr;
    ggml_tensor * prev_rd_rows = nullptr;
    ggml_tensor * prev_wr_rows = nullptr;
    if (keep_prev) {
        ggml_backend_buffer_type_t buft = model.moe_expert_predictor_buft();
        if (buft) {
            prev_state = moe_slots->predictor_state(n_embd, 2 * n_layer, buft);
            auto inp = std::make_unique<minicpm5_moe_pred_state_input>(moe_slots->predictor_step(), n_layer);
            inp->rd_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_layer);
            inp->wr_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_layer);
            ggml_set_input(inp->rd_rows);
            ggml_set_input(inp->wr_rows);
            prev_rd_rows = inp->rd_rows;
            prev_wr_rows = inp->wr_rows;
            res->add_input(std::move(inp));
        }
    }

    // Predict block `target`'s experts from the ffn_norm of block target-1 (`x`).
    // With the trained head:
    //   sigmoid(W_pred[target] * x [+ W_prev[target] * ffn_norm_target(t-1)]) + exp_probs_b[target]
    // Without it, block `target`'s own router scores `x`, which is what the plain LES
    // does. Both mirror build_moe_ffn: the transform matches `gating_op` and the bias is
    // added after it, because the real router ranks on the biased scores.
    auto build_prediction = [&](ggml_tensor * x, int target) -> ggml_tensor * {
        const auto & tgt = model.layers[target];
        ggml_tensor * logits;
        if (trained) {
            logits = ggml_mul_mat(ctx0, tgt.ffn_gate_inp_pred, x);
            if (prev_state && tgt.ffn_gate_inp_pred_prev) {
                ggml_tensor * xprev = ggml_get_rows(ctx0, prev_state,
                        ggml_view_1d(ctx0, prev_rd_rows, 1, (size_t) target * prev_rd_rows->nb[0]));
                logits = ggml_add(ctx0, logits, ggml_mul_mat(ctx0, tgt.ffn_gate_inp_pred_prev, xprev));
            }
        } else {
            logits = build_lora_mm(tgt.ffn_gate_inp, x);
            if (hparams.expert_router_f32) {
                ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
            }
        }
        ggml_tensor * scores;
        switch ((llama_expert_gating_func_type) hparams.expert_gating_func) {
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
                scores = ggml_soft_max(ctx0, logits);
                break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
                scores = logits;
                break;
            default:
                scores = ggml_sigmoid(ctx0, logits);
                break;
        }
        if (tgt.ffn_exp_probs_b) {
            scores = ggml_add(ctx0, scores, tgt.ffn_exp_probs_b);
        }
        return ggml_cont(ctx0, ggml_argsort_top_k(ctx0, scores, moe_slots->lookahead_prefetch_depth()));
    };

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * attn_inp = cur;  // save input for gate computation

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // compute gate from input
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp, model.layers[il].wqkv_gate_s);
            cb(gate, "attn_gate_proj", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,  // wo will be applied after gating
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // attention gating: attn_out * sigmoid(gate) BEFORE o_proj
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sig", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_out_gated", il);

            // now apply output projection
            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_o_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (prev_state && model.layers[il].ffn_gate_inp_pred_prev && cur->ne[1] > 0) {
            // keep this block's ffn_norm for the next token's _prev term, in the bank this
            // ubatch writes (no prediction in this graph reads it)
            ggml_tensor * last = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (cur->ne[1] - 1) * cur->nb[1]);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, prev_state, last,
                ggml_view_1d(ctx0, prev_wr_rows, 1, (size_t) il * prev_wr_rows->nb[0])));
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // a dense block has no per-layer CPU op to carry the prediction, so the next
            // (MoE) block's prediction gets an op of its own
            if (predict && il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp) {
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
            // the prediction for block il+1 rides on this block's slot-mapping CPU op
            ggml_tensor * prediction = nullptr;
            if (predict && il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp) {
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
