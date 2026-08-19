#include "omni-token-policy.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    OmniTokenPolicy policy;
    policy.speak         = 128266;
    policy.listen        = 128267;
    policy.chunk_eos     = 128261;
    policy.chunk_tts_eos = 128268;
    policy.turn_eos      = 128260;
    policy.tts_eos       = 151704;
    policy.im_end        = 151645;
    policy.slash_s       = 128247;
    policy.eos           = 128249;

    assert(policy.is_tts_condition_token(1000));
    assert(!policy.is_tts_condition_token(policy.speak));
    assert(!policy.is_tts_condition_token(policy.listen));
    assert(!policy.is_tts_condition_token(policy.chunk_eos));
    assert(!policy.is_tts_condition_token(policy.chunk_tts_eos));
    assert(!policy.is_tts_condition_token(policy.turn_eos));
    assert(!policy.is_tts_condition_token(policy.tts_eos));
    assert(!policy.is_tts_condition_token(151667)); // <think>
    assert(!policy.is_tts_condition_token(151668)); // </think>
    assert(!policy.is_tts_condition_token(271));    // reserved newline
    assert(!policy.is_tts_condition_token(150000));
    assert(!policy.is_tts_condition_token(-1));

    // These terminators are below the old numeric cutoff and must still never
    // be forwarded as text conditions to TTS.
    assert(policy.slash_s < 150000);
    assert(policy.eos < 150000);
    assert(!policy.is_tts_condition_token(policy.im_end));
    assert(!policy.is_tts_condition_token(policy.slash_s));
    assert(!policy.is_tts_condition_token(policy.eos));

    assert(policy.is_simplex_terminator(policy.tts_eos));
    assert(policy.is_simplex_terminator(policy.im_end));
    assert(policy.is_simplex_terminator(policy.slash_s));
    assert(policy.is_simplex_terminator(policy.eos));

    // A vocabulary may alias a control token to EOS. The token must remain a
    // valid simplex terminator even though classification uses the control role.
    policy.speak = policy.eos;
    assert(policy.classify(policy.eos) == OmniTokenType::SPEAK);
    assert(policy.is_simplex_terminator(policy.eos));
    assert(!policy.is_tts_condition_token(policy.eos));
}
