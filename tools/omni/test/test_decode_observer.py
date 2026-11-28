"""Exercise the production sampling helper with deterministic backend stubs.

This verifies the observation contract, not model accuracy or sampler quality.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DecodeObserverTests(unittest.TestCase):
    def test_observer_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "omni.cpp").read_text(encoding="utf-8")
        header = (root / "omni.h").read_text(encoding="utf-8")
        start = source.index("static const char * sample_with_hidden_and_token(")
        helper = source[start:source.index("\nstatic const char * llama_loop_with_hidden(", start)]
        start = header.index("enum class omni_decode_phase")
        observation = header[start:header.index("\nstruct omni_context {", start)]
        prefix = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
using llama_token = int32_t;
'''
        stubs = r'''
void require(bool ok) { if (!ok) std::abort(); }
struct common_params {};
struct common_sampler { int accepts = 0; int samples = 0; llama_token accepted = -1; };
struct llama_context {
    std::array<float, 5> logits{1, 3, 10, 4, 2};
    bool null_logits = false;
    bool eval_ok = true;
    int eval_calls = 0;
};
struct omni_context {
    omni_decode_observer_t decode_observer;
    llama_context * ctx_llama;
    bool duplex_mode = true;
    int special_token_listen = 0;
    int special_token_tts_pad = 2;
    int special_token_turn_eos = 3;
    int special_token_tts_eos = 4;
    float listen_prob_scale = 2;
    float length_penalty = 2;
};
float * llama_get_logits_ith(llama_context * c, int) { return c->null_logits ? nullptr : c->logits.data(); }
int llama_get_model(llama_context *) { return 0; }
int llama_model_get_vocab(int) { return 0; }
int llama_vocab_n_tokens(int) { return 5; }
bool llama_vocab_is_eog(int, llama_token t) { return t == 4; }
std::string common_token_to_piece(llama_context *, llama_token t) { return std::to_string(t); }
llama_token common_sampler_sample(common_sampler * s, llama_context * c, int) {
    ++s->samples;
    return std::max_element(c->logits.begin(), c->logits.end()) - c->logits.begin();
}
void common_sampler_accept(common_sampler * s, llama_token t, bool) { ++s->accepts; s->accepted = t; }
bool eval_id_with_hidden(omni_context * c, common_params *, int, int * pos, float *& hidden) {
    ++c->ctx_llama->eval_calls;
    if (!c->ctx_llama->eval_ok) { hidden = nullptr; return false; }
    ++*pos;
    c->ctx_llama->logits.fill(99); // previous logits cease to describe this decision
    static float embeddings[2] = {0.25f, 0.75f};
    hidden = embeddings;
    return true;
}
'''
        harness = r'''
struct Event { omni_decode_phase phase; int pos; std::vector<float> logits; llama_token token; };
struct Run { std::vector<Event> events; llama_context backend; common_sampler sampler;
             llama_token token = -1; int pos = 41; std::string piece; float * hidden = nullptr; };
Run run(bool observe, int scenario) {
    Run r;
    omni_context c{}; c.ctx_llama = &r.backend;
    if (scenario == 1) { r.backend.logits[3] = -4; }
    if (scenario == 2) { c.duplex_mode = false; }
    if (scenario == 3) { c.duplex_mode = false; r.backend.logits[4] = -2; }
    if (scenario == 4) { c.special_token_listen = c.special_token_tts_pad = c.special_token_turn_eos = -1; }
    if (scenario == 5) { r.backend.eval_ok = false; }
    if (scenario == 6) { r.backend.null_logits = true; }
    if (scenario == 7) { c.duplex_mode = false; r.backend.logits[4] = 100; }
    if (observe) {
        c.decode_observer = [&](const omni_decode_observation & e) {
            require(r.backend.eval_calls == 0); // observation precedes next decode
            require(e.n_past == 41);
            Event copy{e.phase, e.n_past, {}, e.token};
            if (e.phase == omni_decode_phase::sampled) {
                require(e.logits == nullptr && e.n_vocab == 0);
                require(r.sampler.accepts == 1 && r.sampler.accepted == e.token);
            } else {
                require(e.logits != nullptr && e.n_vocab == 5 && e.token == -1);
                require(r.sampler.samples == 0);
                copy.logits.assign(e.logits, e.logits + e.n_vocab);
            }
            r.events.push_back(copy);
        };
    }
    common_params p;
    r.piece = sample_with_hidden_and_token(&r.sampler, &c, &p, &r.pos, r.hidden, r.token);
    return r;
}
int main() {
    for (int scenario = 0; scenario < 8; ++scenario) {
        auto off = run(false, scenario);
        auto on = run(true, scenario);
        require(off.events.empty());
        require(on.token == off.token && on.piece == off.piece && on.pos == off.pos);
        require(on.hidden == off.hidden && on.backend.logits == off.backend.logits);
        require(on.sampler.samples == 1 && on.sampler.accepts == 1 && on.backend.eval_calls == 1);
        require(on.events.size() == (scenario == 6 ? 1u : 3u));
        require(on.events.back().phase == omni_decode_phase::sampled);
        require(on.events.back().token == on.token);
        if (scenario != 6) {
            require(on.events[0].phase == omni_decode_phase::raw_logits);
            require(on.events[1].phase == omni_decode_phase::adjusted_logits);
            require(on.events[0].logits[2] == 10); // copied before mask/decode
        }
        if (scenario == 0) {
            require(on.events[1].logits == std::vector<float>({3, 3, -INFINITY, 2, 2}));
            require(on.token == 0 && on.pos == 42);
        }
        if (scenario == 1) { require(on.events[1].logits[3] == -8); }
        if (scenario == 2) { require(on.events[1].logits == std::vector<float>({1, 3, 10, 4, 1})); }
        if (scenario == 3) { require(on.events[1].logits[4] == -4); }
        if (scenario == 4) { require(on.events[0].logits == on.events[1].logits); }
        if (scenario == 5) { require(on.hidden == nullptr && on.pos == 41); }
        if (scenario == 7) { require(on.token == 4 && on.piece == "</s>"); }
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "observer.cpp").write_text(prefix + observation + stubs + helper + harness, encoding="utf-8")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-g",
                            "-fsanitize=address,undefined", str(path / "observer.cpp"),
                            "-o", str(path / "observer")], check=True)
            subprocess.run([str(path / "observer")], check=True)


if __name__ == "__main__":
    unittest.main()
