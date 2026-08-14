#include "token2wav-est-att-writeback-policy.h"

using omni::flow::token2wav_est_att_shape;
using omni::flow::token2wav_est_att_writeback_graph_structure;
using omni::flow::token2wav_legacy_est_att_cpy_requested;
using omni::flow::token2wav_should_elide_est_att_writeback;

static_assert(!token2wav_legacy_est_att_cpy_requested(nullptr));
static_assert(!token2wav_legacy_est_att_cpy_requested(""));
static_assert(!token2wav_legacy_est_att_cpy_requested("0"));
static_assert(!token2wav_legacy_est_att_cpy_requested("01"));
static_assert(!token2wav_legacy_est_att_cpy_requested("true"));
static_assert(token2wav_legacy_est_att_cpy_requested("1"));

constexpr token2wav_est_att_shape persistent{ 4, 128, 302, 640, 2, true };
constexpr token2wav_est_att_shape nonlast{ 4, 128, 352, 640, 2, true };
constexpr token2wav_est_att_shape last{ 4, 128, 358, 640, 2, true };

// Canonical non-last and last graphs prove [current(delta), old] -> slice(delta, L) == old.
static_assert(token2wav_should_elide_est_att_writeback(nullptr, nonlast, persistent, 50));
static_assert(token2wav_should_elide_est_att_writeback("0", last, persistent, 56));
static_assert(!token2wav_should_elide_est_att_writeback("1", nonlast, persistent, 50));

// Max-cache truncation, type/layout changes, and inconsistent deltas retain the legacy root.
constexpr token2wav_est_att_shape max_crossing_persistent{ 4, 128, 551, 640, 2, true };
constexpr token2wav_est_att_shape max_crossing_output{ 4, 128, 600, 640, 2, true };
constexpr token2wav_est_att_shape max_full_persistent{ 4, 128, 600, 640, 2, true };
constexpr token2wav_est_att_shape wrong_type{ 4, 128, 352, 640, 2, false };
constexpr token2wav_est_att_shape wrong_slots{ 4, 128, 352, 512, 2, true };
constexpr token2wav_est_att_shape wrong_batch{ 4, 128, 352, 640, 4, true };
constexpr token2wav_est_att_shape wrong_rank{ 3, 128, 352, 640, 2, true };
static_assert(!token2wav_should_elide_est_att_writeback("0", max_crossing_output,
                                                        max_crossing_persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", max_full_persistent,
                                                        max_full_persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", wrong_type, persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", wrong_slots, persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", wrong_batch, persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", wrong_rank, persistent, 50));
static_assert(!token2wav_should_elide_est_att_writeback("0", nonlast, persistent, 49));
static_assert(!token2wav_should_elide_est_att_writeback("0", persistent, persistent, 0));

constexpr auto canonical_elided = token2wav_est_att_writeback_graph_structure(16, 5, true, true);
static_assert(canonical_elided.cache_value_concat_nodes == 80);
static_assert(canonical_elided.block_pack_concat_nodes == 75);
static_assert(canonical_elided.step_pack_concat_nodes == 4);
static_assert(canonical_elided.trim_cont_nodes == 1);
static_assert(canonical_elided.writeback_cpy_nodes == 1);
static_assert(canonical_elided.selected_compute_nodes == 0);
static_assert(canonical_elided.removed_compute_nodes == 161);

constexpr auto canonical_legacy = token2wav_est_att_writeback_graph_structure(16, 5, true, false);
static_assert(canonical_legacy.selected_compute_nodes == 161);
static_assert(canonical_legacy.removed_compute_nodes == 0);

int main() {
    return 0;
}
