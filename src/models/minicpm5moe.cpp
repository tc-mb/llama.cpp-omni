#include "models.h"
#include "llama-moe-experts.h"
#include "llama-moe-slot.h"

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
}

std::unique_ptr<llm_graph_context> llama_model_minicpm5moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

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

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
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
