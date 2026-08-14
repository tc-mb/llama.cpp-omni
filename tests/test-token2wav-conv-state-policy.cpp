#include "token2wav-conv-state-policy.h"

using omni::flow::token2wav_legacy_conv_state_requested;
using omni::flow::token2wav_use_current_tail_for_conv_state;

static_assert(!token2wav_legacy_conv_state_requested(nullptr));
static_assert(!token2wav_legacy_conv_state_requested(""));
static_assert(!token2wav_legacy_conv_state_requested("0"));
static_assert(!token2wav_legacy_conv_state_requested("true"));
static_assert(!token2wav_legacy_conv_state_requested("01"));
static_assert(token2wav_legacy_conv_state_requested("1"));

// Production shape: the last two current frames fully determine the next state.
static_assert(token2wav_use_current_tail_for_conv_state(56, 2, true, nullptr));
static_assert(token2wav_use_current_tail_for_conv_state(56, 2, true, "0"));
static_assert(!token2wav_use_current_tail_for_conv_state(56, 2, true, "1"));

// Boundary and fallback cases.
static_assert(token2wav_use_current_tail_for_conv_state(2, 2, true, "0"));
static_assert(!token2wav_use_current_tail_for_conv_state(1, 2, true, "0"));
static_assert(!token2wav_use_current_tail_for_conv_state(56, 0, true, "0"));
static_assert(!token2wav_use_current_tail_for_conv_state(56, 2, false, "0"));

int main() {
    return 0;
}
