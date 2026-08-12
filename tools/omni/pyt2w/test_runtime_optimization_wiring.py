#!/usr/bin/env python3
"""Static gates for opt-in runtime optimizations and their legacy defaults."""

import unittest
from pathlib import Path


OMNI_CPP = Path(__file__).resolve().parents[1] / "omni.cpp"
OMNI_H = Path(__file__).resolve().parents[1] / "omni.h"


class RuntimeOptimizationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = OMNI_CPP.read_text(encoding="utf-8")
        cls.header = OMNI_H.read_text(encoding="utf-8")

    def test_sampler_reuses_workspace_and_partial_topk(self):
        self.assertIn("thread_local tts_sampling_workspace workspace;", self.source)
        self.assertIn("std::partial_sort(", self.source)
        self.assertIn('std::getenv("OMNI_TTS_LEGACY_SAMPLER_FULL_SORT")', self.source)
        self.assertIn("workspace.repetition_tokens", self.source)
        self.assertNotIn("std::vector<int> freq(num_tokens", self.source)

    def test_sampler_fixed_seed_and_tie_order_are_deterministic(self):
        self.assertIn(
            "lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second",
            self.source,
        )
        self.assertIn("local_rng.seed(seed);", self.source)
        self.assertNotIn("std::mt19937(std::random_device{}())", self.source)

    def test_same_turn_kv_candidate_is_explicit_opt_in(self):
        start = self.source.index("static bool omni_tts_legacy_duplex_chunk_reset_enabled()")
        body = self.source[start:self.source.index("static bool omni_tts_debug_artifacts_enabled()", start)]
        self.assertIn('std::getenv("OMNI_TTS_LEGACY_DUPLEX_CHUNK_RESET")', body)
        self.assertIn("value == nullptr", body)
        self.assertIn("value[0] == '0'", body)
        self.assertIn("preserving KV cache at chunk_idx", self.source)

    def test_stage_devices_and_apm_api_are_wired(self):
        self.assertIn('std::getenv("OMNI_TTS_DEVICE")', self.source)
        self.assertIn('std::getenv("OMNI_PROJECTOR_DEVICE")', self.source)
        for symbol in (
            "omni_duplex_set_apm_ab_mode",
            "omni_duplex_prepare_apm_replay",
            "omni_duplex_get_apm_ab_stats",
        ):
            self.assertIn(symbol, self.source)
            self.assertIn(symbol, self.header)

    def test_generation_limit_is_capped_by_remaining_context(self):
        self.assertGreaterEqual(
            self.source.count("params->n_ctx - ctx_omni->n_past"), 2
        )
        self.assertGreaterEqual(
            self.source.count("generation_hit_limit.store(true)"), 2
        )


if __name__ == "__main__":
    unittest.main()
