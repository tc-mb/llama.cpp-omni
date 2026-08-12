#!/usr/bin/env python3
"""Static gates for opt-in runtime optimizations and their legacy defaults."""

import unittest
from pathlib import Path


OMNI_CPP = Path(__file__).resolve().parents[1] / "omni.cpp"
OMNI_H = Path(__file__).resolve().parents[1] / "omni.h"
META_BACKEND_CPP = Path(__file__).resolve().parents[3] / "ggml" / "src" / "ggml-backend-meta.cpp"


class RuntimeOptimizationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = OMNI_CPP.read_text(encoding="utf-8")
        cls.header = OMNI_H.read_text(encoding="utf-8")
        cls.meta_backend_source = META_BACKEND_CPP.read_text(encoding="utf-8")

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

    def test_main_llm_tensor_parallel_configuration_is_validated(self):
        for variable in (
            "OMNI_LLM_DEVICES",
            "OMNI_LLM_SPLIT_MODE",
            "OMNI_LLM_TENSOR_SPLIT",
        ):
            self.assertIn(f'std::getenv("{variable}")', self.source)
        for mode in (
            "LLAMA_SPLIT_MODE_NONE",
            "LLAMA_SPLIT_MODE_ROW",
            "LLAMA_SPLIT_MODE_LAYER",
            "LLAMA_SPLIT_MODE_TENSOR",
        ):
            self.assertIn(mode, self.source)
        self.assertIn('std::strcmp(split_mode, "tensor")', self.source)
        self.assertIn("Meta backend path", self.source)
        self.assertIn("tensor parallel, experimental", self.source)
        self.assertIn("GGML_BACKEND_DEVICE_TYPE_CPU", self.source)
        self.assertIn('std::strcmp(device_list, "auto")', self.source)
        self.assertIn("ggml_backend_dev_count()", self.source)
        self.assertIn('std::strncmp(name, "CANN", 4)', self.source)
        self.assertIn("only one CANN device is available", self.source)
        self.assertIn("index >= tensor_split.size()", self.source)
        self.assertIn("!std::isfinite(parsed)", self.source)
        self.assertIn("parsed < 0.0f", self.source)
        self.assertIn("model_params.tensor_split = tensor_split.data();", self.source)

    def test_generation_limit_is_capped_by_remaining_context(self):
        self.assertGreaterEqual(
            self.source.count("params->n_ctx - ctx_omni->n_past"), 2
        )
        self.assertGreaterEqual(
            self.source.count("generation_hit_limit.store(true)"), 2
        )

    def test_meta_split_state_address_reuse_evicts_only_stale_entry(self):
        mismatch_start = self.meta_backend_source.index(
            "if (it != buf_ctx->split_state_cache.end() && memcmp("
        )
        mismatch_body = self.meta_backend_source[
            mismatch_start:self.meta_backend_source.index(
                "if (it == buf_ctx->split_state_cache.end())", mismatch_start
            )
        ]
        self.assertIn("buf_ctx->split_state_cache.erase(it);", mismatch_body)
        self.assertNotIn("buf_ctx->split_state_cache.clear();", mismatch_body)


if __name__ == "__main__":
    unittest.main()
