#include "token2wav-adaln-cache-policy.h"

using omni::flow::token2wav_adaln_cache_graph_structure;
using omni::flow::token2wav_legacy_adaln_cache_requested;
using omni::flow::token2wav_should_cache_adaln;

static_assert(!token2wav_legacy_adaln_cache_requested(nullptr));
static_assert(!token2wav_legacy_adaln_cache_requested(""));
static_assert(!token2wav_legacy_adaln_cache_requested("0"));
static_assert(!token2wav_legacy_adaln_cache_requested("01"));
static_assert(!token2wav_legacy_adaln_cache_requested("true"));
static_assert(token2wav_legacy_adaln_cache_requested("1"));

// Canonical MiniCPM-o 4.5 CFM: 16 DiT blocks, hidden=512, five ODE steps, CFG batch=2, F32 timestep.
static_assert(token2wav_should_cache_adaln(nullptr, 16, 512, 5, 2, true));
static_assert(token2wav_should_cache_adaln("0", 16, 512, 5, 2, true));
static_assert(token2wav_should_cache_adaln("01", 16, 512, 5, 2, true));
static_assert(!token2wav_should_cache_adaln("1", 16, 512, 5, 2, true));

// Any topology, step count, CFG shape, or type mismatch keeps the original per-call graph.
static_assert(!token2wav_should_cache_adaln("0", 15, 512, 5, 2, true));
static_assert(!token2wav_should_cache_adaln("0", 16, 1024, 5, 2, true));
static_assert(!token2wav_should_cache_adaln("0", 16, 512, 4, 2, true));
static_assert(!token2wav_should_cache_adaln("0", 16, 512, 5, 1, true));
static_assert(!token2wav_should_cache_adaln("0", 16, 512, 5, 2, false));

constexpr auto cached = token2wav_adaln_cache_graph_structure(16, 5, true);
static_assert(cached.modulation_linears_on_first_use == 85);
static_assert(cached.modulation_linears_on_reuse == 0);
static_assert(cached.persistent_cache_tensors_per_call_id == 2);
static_assert(cached.separately_cached_call_ids == 2);

constexpr auto legacy = token2wav_adaln_cache_graph_structure(16, 5, false);
static_assert(legacy.modulation_linears_on_first_use == 85);
static_assert(legacy.modulation_linears_on_reuse == 85);
static_assert(legacy.persistent_cache_tensors_per_call_id == 0);
static_assert(legacy.separately_cached_call_ids == 0);

int main() {
    return 0;
}
