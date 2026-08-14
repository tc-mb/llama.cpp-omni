from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "tools/omni/omni.cpp").read_text()


def test_profile_is_exact_opt_in() -> None:
    assert 'std::getenv("OMNI_TTS_PROFILE_STEPS")' in SOURCE
    assert "value[0] == '1' && value[1] == '\\0'" in SOURCE


def test_chunk_report_contains_requested_stages_and_counts() -> None:
    report = SOURCE[SOURCE.index('"[tts-step-profile]'):]
    for field in (
        "first_reforward_ms",
        "reforward_tokens",
        "head_projection_ms",
        "sampling_ms",
        "embedding_gather_ms",
        "token_decode_ms",
        "condition_prefill_ms",
        "prefill_tokens",
        "projector_ms",
        "projector_tokens",
        "tokens",
        "decoded",
    ):
        assert field in report


def test_profile_scope_only_activates_when_enabled() -> None:
    constructor = SOURCE[SOURCE.index("OmniTtsStageTimer(struct omni_context"):]
    constructor = constructor[: constructor.index("~OmniTtsStageTimer")]
    assert "profile_steps(omni_tts_profile_steps_enabled())" in constructor
    assert "if (profile_steps)" in constructor
    assert "g_omni_tts_step_profile = &step_profile" in constructor
