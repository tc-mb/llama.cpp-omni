#!/usr/bin/env python3
"""Static gates for runtime optimizations and their explicit rollback paths."""

import unittest
from pathlib import Path


OMNI_CPP = Path(__file__).resolve().parents[1] / "omni.cpp"
OMNI_H = Path(__file__).resolve().parents[1] / "omni.h"
TOKEN2WAV_CPP = Path(__file__).resolve().parents[1] / "token2wav" / "token2wav.cpp"
TOKEN2WAV_IMPL_H = Path(__file__).resolve().parents[1] / "token2wav" / "token2wav-impl.h"
TOKEN2WAV_IMPL_CPP = Path(__file__).resolve().parents[1] / "token2wav" / "token2wav-impl.cpp"


class RuntimeOptimizationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = OMNI_CPP.read_text(encoding="utf-8")
        cls.header = OMNI_H.read_text(encoding="utf-8")
        cls.token2wav_source = TOKEN2WAV_CPP.read_text(encoding="utf-8")
        cls.token2wav_header = TOKEN2WAV_IMPL_H.read_text(encoding="utf-8")
        cls.token2wav_impl_source = TOKEN2WAV_IMPL_CPP.read_text(encoding="utf-8")

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

    def test_same_turn_kv_is_default_on_with_exact_legacy_rollback(self):
        start = self.source.index("static bool omni_tts_legacy_duplex_chunk_reset_enabled()")
        body = self.source[start:self.source.index("static bool omni_tts_debug_artifacts_enabled()", start)]
        self.assertIn('std::getenv("OMNI_TTS_LEGACY_DUPLEX_CHUNK_RESET")', body)
        self.assertIn("value != nullptr", body)
        self.assertIn("value[0] == '1'", body)
        self.assertIn("preserving KV cache at chunk_idx", self.source)

    def test_head_projection_has_scalar_ablation_and_reuses_executor(self):
        self.assertIn(
            'std::getenv("OMNI_TTS_LEGACY_HEAD_PROJECTION")', self.source
        )
        self.assertIn('std::getenv("OMNI_TTS_HEAD_THREADS")', self.source)
        self.assertIn("TtsHeadCodeExecutor::compute_scalar(", self.source)
        self.assertEqual(self.source.count("tts_head_code_project(ctx_omni,"), 2)

    def test_debug_artifact_work_is_disabled_by_default(self):
        self.assertEqual(
            self.source.count("if (omni_tts_debug_artifacts_enabled())"), 5
        )

    def test_apm_api_is_wired(self):
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

    def test_tts_eos_skips_dead_embedding_forward(self):
        simplex_start = self.source.index("llama_token sample_tts_token_simplex(")
        duplex_start = self.source.index("llama_token sample_tts_token(", simplex_start)
        simplex = self.source[simplex_start:duplex_start]
        duplex_end = self.source.index("static bool is_audio_token(", duplex_start)
        duplex = self.source[duplex_start:duplex_end]

        self.assertLess(
            simplex.index("if (is_eos) {"),
            simplex.index("audio_token_embedding"),
        )
        self.assertLess(
            duplex.index("if (ctx_omni->duplex_mode && is_eos) {"),
            duplex.index("audio_token_embedding"),
        )

    def test_duplex_path_does_not_merge_nonexistent_chunk_wavs(self):
        duplex_start = self.source.index("void tts_thread_func_duplex(")
        duplex_end = self.source.index("void tts_thread_func(", duplex_start)
        self.assertNotIn("merge_wav_files(", self.source[duplex_start:duplex_end])

    def test_duplex_handoff_avoids_redundant_hidden_state_copies(self):
        self.assertGreaterEqual(
            self.source.count("std::move(chunk_hidden_states)"), 2
        )
        self.assertIn(
            "current_chunk_hidden_states = std::move(llm_out->hidden_states)",
            self.source,
        )
        self.assertIn("std::memmove(hidden_states.data()", self.source)
        self.assertIn(
            "tts_condition_embeddings = std::move(merged_embeddings)",
            self.source,
        )

    def test_token2wav_init_warmup_restores_stream_and_rng_state(self):
        self.assertIn('std::getenv("OMNI_T2W_INIT_WARMUP")', self.token2wav_source)
        self.assertIn("const std::mt19937 initial_noise_state", self.token2wav_source)
        self.assertEqual(self.token2wav_source.count("t2w.restore_noise_state(initial_noise_state);"), 2)
        self.assertGreaterEqual(
            self.token2wav_source.count("token2wav_warmup_fixed_window(t2w)"),
            2,
        )
        self.assertIn("value && value[0] == '1'", self.token2wav_source)
        self.assertIn("class flowRunnerNoiseState", self.token2wav_header)
        self.assertIn("flowRunnerNoiseState                  noise_state_;", self.token2wav_header)
        self.assertEqual(
            self.token2wav_impl_source.count("noise_state_.generator()"), 3
        )
        self.assertNotIn("runner_noise_generator()", self.token2wav_impl_source)

    def test_duplex_terminal_suffix_has_batched_ab_path(self):
        start = self.source.index("static bool omni_llm_terminal_suffix_batch_enabled()")
        body = self.source[start:self.source.index("static bool omni_llm_turn_suffix_speculation_enabled()", start)]
        self.assertIn('std::getenv("OMNI_LLM_BATCH_TERMINAL_SUFFIX")', body)
        self.assertIn("value == nullptr", body)
        self.assertIn("value[0] == '0'", body)
        self.assertIn("bool * evaluation_deferred", self.source)
        self.assertIn("ctx_omni->n_past + 2 < params->n_ctx - 512", self.source)
        self.assertIn(
            "std::vector<llama_token> suffix_tokens = {deferred_terminal_token}",
            self.source,
        )
        self.assertIn("suffix_tokens.push_back(ctx_omni->special_token_unit_end)", self.source)

    def test_duplex_turn_suffix_speculation_verifies_and_rolls_back(self):
        start = self.source.index("static bool omni_llm_turn_suffix_speculation_enabled()")
        body = self.source[start:self.source.index("static bool resolve_python_t2w_seed", start)]
        self.assertIn('std::getenv("OMNI_LLM_SPECULATE_TURN_SUFFIX")', body)
        self.assertIn("value == nullptr", body)
        self.assertIn("value[0] == '0'", body)
        self.assertIn("speculate_turn_eos_suffix(", self.source)
        self.assertIn("actual == ctx_omni->special_token_chunk_eos", self.source)
        self.assertIn("llama_memory_seq_rm(memory, 0, start_pos + 1, -1)", self.source)
        self.assertIn("ctx_omni->n_past + 3 < params->n_ctx - 512", self.source)
        self.assertIn("suffix_fully_evaluated = true", self.source)

    def test_tts_final_speculation_is_default_on_and_transactional(self):
        start = self.source.index("static bool omni_tts_speculate_final_enabled()")
        body = self.source[start:self.source.index("static bool omni_t2w_speculate_final_wav_enabled()", start)]
        self.assertIn('std::getenv("OMNI_TTS_SPECULATE_FINAL")', body)
        self.assertIn("value == nullptr", body)
        self.assertIn("value[0] == '0'", body)
        self.assertIn("[tts-spec-final] attempt", self.source)
        self.assertIn("[tts-spec-final] ready", self.source)
        self.assertIn("[tts-spec-final] hit", self.source)
        self.assertIn("[tts-spec-final] miss", self.source)
        self.assertIn("[tts-spec-final] abort", self.source)
        self.assertIn("enum class TtsGenerationOutcome", self.source)
        self.assertIn("TtsGenerationOutcome::COMPLETE", self.source)
        self.assertIn("TtsGenerationOutcome::ABORTED", self.source)
        self.assertIn("TtsGenerationOutcome::ERROR", self.source)
        self.assertIn("const TtsGenerationOutcome generation_outcome = generate_audio_tokens_local(", self.source)
        self.assertIn(
            "generation_outcome == TtsGenerationOutcome::COMPLETE",
            self.source,
        )
        self.assertIn("const bool rollback_success = mem != nullptr &&", self.source)
        self.assertIn("llama_memory_seq_rm(mem, 0, base_n_past, -1)", self.source)
        self.assertIn("llama_memory_clear(mem, false)", self.source)
        self.assertIn("!generation_success || !rollback_success", self.source)
        self.assertIn("speculation_aborted || !generation_unchanged", self.source)
        self.assertIn("!break_inactive || !still_empty", self.source)
        self.assertIn("speculative_tokens.empty()", self.source)
        self.assertIn("ctx_omni->tts_n_past_accumulated = 0", self.source)
        self.assertIn("ctx_omni->tts_all_generated_tokens.clear()", self.source)
        self.assertIn("ctx_omni->tts_condition_embeddings.clear()", self.source)
        self.assertIn("ctx_omni->tts_condition_length = 0", self.source)
        self.assertIn("ctx_omni->tts_condition_n_embd = 0", self.source)
        self.assertIn("ctx_omni->tts_condition_saved = false", self.source)
        self.assertIn("[tts-spec-final] failure=rollback", self.source)
        self.assertIn("tts_failure_break_generation =", self.source)
        self.assertIn("omni_request_break(ctx_omni)", self.source)
        self.assertIn("speculative_rollback", self.source)
        self.assertIn(
            "DuplexBreakStage::TTS,\n                    "
            "tts_failure_break_generation",
            self.source,
        )
        self.assertIn(
            "ctx_omni->tts_all_generated_tokens.resize(generated_size)",
            self.source,
        )
        self.assertIn(
            "std::move(saved_condition_embeddings)",
            self.source,
        )

    def test_tts_final_speculation_never_streams_during_generation(self):
        self.assertIn("bool suppress_t2w = false", self.source)
        self.assertIn("!suppress_t2w && ctx_omni->t2w_thread_info", self.source)
        self.assertIn("true, cancel_speculation, &speculation_aborted", self.source)
        self.assertIn(
            'current_chunk_idx + 1, speculative_tokens, true, "",',
            self.source,
        )
        self.assertIn("return !ctx_omni->tts_thread_info->queue.empty()", self.source)
        self.assertIn("t2w_out->audio_tokens = speculative_final.relative_tokens", self.source)
        self.assertIn("t2w_out->is_final = true", self.source)
        self.assertIn("t2w_out->is_chunk_end = false", self.source)

    def test_tts_final_speculation_waits_for_cpp_t2w_idle(self):
        self.assertIn("bool busy = false", self.header)
        self.assertIn("ctx_omni->t2w_thread_info->busy = true", self.source)
        self.assertIn("info.busy = false", self.source)
        self.assertIn("info.cv.notify_all()", self.source)
        self.assertIn("T2W_IDLE_WAIT_TIMEOUT{300}", self.source)
        self.assertIn("T2W_IDLE_WAIT_SLICE{5}", self.source)
        self.assertIn("lock, T2W_IDLE_WAIT_SLICE", self.source)
        self.assertIn("!ctx_omni->t2w_thread_info->busy", self.source)
        self.assertIn("[tts-spec-final] skip=t2w-timeout", self.source)
        self.assertIn('break_invalidated() ? "break" : "new-input"', self.source)

    def test_tts_final_speculation_is_bound_to_break_generation(self):
        self.assertIn(
            "const uint64_t speculation_break_generation =", self.source
        )
        self.assertIn("const auto break_invalidated = [&]", self.source)
        self.assertIn(
            "ctx_omni->break_event.generation() !=\n"
            "                                    speculation_break_generation",
            self.source,
        )
        self.assertIn("const bool generation_unchanged =", self.source)
        self.assertIn("const bool break_inactive =", self.source)

    def test_duplex_break_stages_acknowledge_their_generation(self):
        self.assertIn(
            "DuplexBreakStage::TTS, break_generation", self.source
        )
        self.assertIn(
            "DuplexBreakStage::T2W, break_generation", self.source
        )
        self.assertIn(
            "DuplexBreakStage::LLM, break_generation", self.source
        )
        self.assertNotIn("ctx_omni->break_event = false", self.source)
        self.assertNotIn(
            'print_with_timestamp("Duplex decode: reset break_event at start',
            self.source,
        )
        self.assertNotIn(
            'print_with_timestamp("📍 stream_decode: reset break_event',
            self.source,
        )

    def test_duplex_encoder_discards_stale_break_generations(self):
        self.assertIn("uint64_t    break_generation = 0", self.source)
        self.assertIn("uint64_t                        break_generation = 0", self.source)
        self.assertIn("packet->break_generation = req->break_generation", self.source)
        self.assertIn("Duplex encoder: discard stale packet", self.source)
        self.assertIn(
            "ctx_omni->break_event.generation() != request_break_generation",
            self.source,
        )
        self.assertIn("dup->in_flight_cv.wait(barrier_lock", self.source)
        self.assertIn("dup->in_flight_prefill.load() == 0", self.source)
        self.assertIn("while (!dup->encoder_queue.empty())", self.source)

    def test_duplex_prefill_publishes_count_before_request(self):
        start = self.source.rindex("static bool duplex_prefill(")
        end = self.source.index("static bool duplex_decode(", start)
        prefill = self.source[start:end]
        count_pos = prefill.index(
            "dup->in_flight_prefill.fetch_add(1, std::memory_order_release)"
        )
        push_pos = prefill.index("dup->encoder_queue.push(req.release())")
        unlock_pos = prefill.index("dup->encoder_cv.notify_all()")
        self.assertLess(count_pos, push_pos)
        self.assertLess(push_pos, unlock_pos)
        self.assertIn("std::unique_lock<std::mutex> lk(dup->encoder_mtx)", prefill)

    def test_duplex_prefill_count_never_underflows(self):
        self.assertIn("static bool duplex_finish_in_flight_prefill(", self.source)
        self.assertIn("while (current >= count)", self.source)
        self.assertIn("current, current - count", self.source)
        self.assertIn("invalid in_flight_prefill decrement", self.source)
        self.assertNotIn("in_flight_prefill.fetch_sub", self.source)

    def test_duplex_stop_wakes_in_flight_barrier(self):
        stop_sites = [
            "ctx_omni->duplex->running.store(false);",
            "dup->running.store(false);",
        ]
        for site in stop_sites:
            start = self.source.index(site)
            tail = self.source[start : start + 400]
            self.assertIn("in_flight_cv.notify_all()", tail)
        stop_start = self.source.rindex("static void duplex_stop_threads(")
        stop_end = self.source.index("static void duplex_encoder_thread_func(", stop_start)
        stop_body = self.source[stop_start:stop_end]
        self.assertGreaterEqual(
            stop_body.count("duplex_finish_in_flight_prefill(dup)"), 2
        )
        self.assertIn("GGML_ASSERT(remaining_in_flight == 0)", stop_body)

    def test_duplex_encoder_stop_returns_current_packet_count(self):
        needle = "if (!dup->running.load()) {\n                delete packet;"
        start = self.source.index(needle)
        tail = self.source[start : start + 180]
        self.assertIn("duplex_finish_in_flight_prefill(dup)", tail)

    def test_tts_final_speculation_skips_python_token2wav(self):
        self.assertIn("if (ctx_omni->use_python_token2wav)", self.source)
        self.assertIn("[tts-spec-final] skip=python-t2w", self.source)

    def test_tts_final_speculation_requires_fixed_seed(self):
        self.assertIn(
            "params->sampling.seed == LLAMA_DEFAULT_SEED", self.source
        )
        self.assertIn("[tts-spec-final] skip=random-seed", self.source)

    def test_tts_final_speculation_rejects_short_conditions(self):
        self.assertIn(
            "MIN_SPECULATIVE_FINAL_CONDITION_TOKENS = 5", self.source
        )
        self.assertIn(
            "n_tokens_filtered <\n                        "
            "MIN_SPECULATIVE_FINAL_CONDITION_TOKENS",
            self.source,
        )
        self.assertIn("[tts-spec-final] skip=short-condition", self.source)

    def test_final_wav_forced_miss_is_exact_validation_only(self):
        start = self.source.index(
            "static bool omni_t2w_force_final_wav_miss_for_validation()"
        )
        end = self.source.index("static bool omni_tts_profile_steps_enabled()", start)
        body = self.source[start:end]
        self.assertIn('std::getenv("OMNI_T2W_FORCE_FINAL_WAV_MISS")', body)
        self.assertIn("value && value[0] == '1' && value[1] == '\\0'", body)
        self.assertIn(
            "!omni_t2w_force_final_wav_miss_for_validation()", self.source
        )

    def test_final_wav_speculation_is_default_on_with_exact_rollback(self):
        start = self.source.index("static bool omni_t2w_speculate_final_wav_enabled()")
        end = self.source.index("static bool omni_t2w_force_final_wav_miss_for_validation()", start)
        body = self.source[start:end]
        self.assertIn('std::getenv("OMNI_T2W_SPECULATE_FINAL_WAV")', body)
        self.assertIn("value == nullptr", body)
        self.assertIn("value[0] == '0'", body)

    def test_final_wav_render_serializes_next_single_device_encoder(self):
        self.assertIn("final_wav_admission_state final_wav_admission", self.header)
        self.assertIn(
            "command == omni::flow::final_wav_command::spec_render", self.source
        )
        self.assertIn(
            "t2w->final_wav_admission.try_admit_encoder()", self.source
        )
        self.assertIn(
            "final_wav_admission.try_begin_render()", self.source
        )
        self.assertIn("final_wav_admission.release_encoder()", self.source)
        self.assertIn("t2w->cv.wait", self.source)
        busy_start = self.source.index("struct T2WBusyScope")
        busy_end = self.source.index("// Token2Wav sliding window parameters", busy_start)
        self.assertNotIn(
            "final_wav_admission.end_render()",
            self.source[busy_start:busy_end],
        )
        finalizer_start = self.source.index(
            "auto finalize_final_wav_transaction ="
        )
        finalizer_end = self.source.index("while (t2w_thread_running)", finalizer_start)
        finalizer = self.source[finalizer_start:finalizer_end]
        self.assertIn("final_wav_admission.rendering()", finalizer)
        self.assertIn("render_reserved && ctx_omni->token2wav_session", finalizer)
        self.assertIn("release_final_wav_rendering();", finalizer)

    def test_final_wav_failed_rollback_converges_to_recovery_or_invalidation(self):
        self.assertIn("bool recover_committed_replay();", self.token2wav_header)
        self.assertIn("return recover_committed_replay();", self.token2wav_source)
        self.assertIn("replay_calls_.size() < replay_max_calls_", self.token2wav_source)
        recover_start = self.token2wav_source.index(
            "bool Token2WavSession::recover_committed_replay()"
        )
        recover_end = self.token2wav_source.index(
            "bool Token2WavSession::commit_tentative()", recover_start
        )
        recover_body = self.token2wav_source[recover_start:recover_end]
        self.assertNotIn(
            "replay_enabled_ = false;\n        replay_calls_.clear();",
            recover_body,
        )
        self.assertIn("if (final_wav_tx.needs_recovery())", self.source)
        self.assertIn(
            "ctx_omni->token2wav_session->recover_committed_replay()",
            self.source,
        )
        self.assertIn("final_wav_tx.abort(key, rollback_ok)", self.source)
        self.assertIn(
            "Token2WavLifecycleState::invalidated", self.source
        )
        self.assertIn(
            "token2wav_lifecycle.load() != Token2WavLifecycleState::ready ||",
            self.source,
        )
        self.assertIn("a full context initialization is required", self.source)
        self.assertIn("recover_committed_replay();", self.source)
        self.assertIn("refusing to start inference", self.source)
        self.assertIn(
            "std::atomic<Token2WavLifecycleState> token2wav_lifecycle",
            self.header,
        )
        self.assertNotIn("token2wav_initialized", self.header)
        self.assertNotIn("token2wav_invalidated", self.header)

if __name__ == "__main__":
    unittest.main()
