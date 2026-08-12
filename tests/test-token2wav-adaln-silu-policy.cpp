#include "token2wav-adaln-silu-policy.h"

using omni::flow::token2wav_adaln_silu_graph_structure;
using omni::flow::token2wav_legacy_adaln_silu_requested;
using omni::flow::token2wav_should_share_adaln_silu;

static_assert(token2wav_legacy_adaln_silu_requested(nullptr));
static_assert(token2wav_legacy_adaln_silu_requested(""));
static_assert(!token2wav_legacy_adaln_silu_requested("0"));
static_assert(token2wav_legacy_adaln_silu_requested("01"));
static_assert(token2wav_legacy_adaln_silu_requested("true"));
static_assert(token2wav_legacy_adaln_silu_requested("1"));

// Canonical MiniCPM-o 4.5 DiT: 16 blocks, hidden=512, CFG batch=2.
static_assert(!token2wav_should_share_adaln_silu(nullptr, 16, 512, 3, 512, 1, 2, true));
static_assert(token2wav_should_share_adaln_silu("0", 16, 512, 3, 512, 1, 2, true));
static_assert(!token2wav_should_share_adaln_silu("01", 16, 512, 3, 512, 1, 2, true));
static_assert(!token2wav_should_share_adaln_silu("true", 16, 512, 3, 512, 1, 2, true));
static_assert(!token2wav_should_share_adaln_silu("1", 16, 512, 3, 512, 1, 2, true));

// Unknown topology, layout, batch, or type keeps the legacy per-consumer graph.
static_assert(!token2wav_should_share_adaln_silu("0", 15, 512, 3, 512, 1, 2, true));
static_assert(!token2wav_should_share_adaln_silu("0", 16, 1024, 3, 1024, 1, 2, true));
static_assert(!token2wav_should_share_adaln_silu("0", 16, 512, 2, 512, 2, 1, true));
static_assert(!token2wav_should_share_adaln_silu("0", 16, 512, 3, 512, 2, 1, true));
static_assert(!token2wav_should_share_adaln_silu("0", 16, 512, 3, 512, 1, 4, true));
static_assert(!token2wav_should_share_adaln_silu("0", 16, 512, 3, 512, 1, 2, false));

constexpr auto canonical_shared = token2wav_adaln_silu_graph_structure(16, true);
static_assert(canonical_shared.legacy_nodes_per_step == 17);
static_assert(canonical_shared.selected_nodes_per_step == 1);
static_assert(canonical_shared.removed_nodes_per_step == 16);
static_assert(canonical_shared.removed_nodes_per_step * 5 == 80);

constexpr auto canonical_legacy = token2wav_adaln_silu_graph_structure(16, false);
static_assert(canonical_legacy.legacy_nodes_per_step == 17);
static_assert(canonical_legacy.selected_nodes_per_step == 17);
static_assert(canonical_legacy.removed_nodes_per_step == 0);

int main() {
    return 0;
}
