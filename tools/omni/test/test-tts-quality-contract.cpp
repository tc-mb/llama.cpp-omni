#include "omni.h"

#undef NDEBUG
#include <cassert>
#include <cstddef>
#include <vector>

static void test_head_code_uses_row_major_audio_token_rows() {
    constexpr int hidden_size = 2;
    constexpr int audio_tokens = 3;
    const float source[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    std::vector<float> destination(audio_tokens * hidden_size, 0.0f);

    assert(omni_copy_head_code_row_major(
        source, /*dim0=*/hidden_size, /*dim1=*/audio_tokens,
        hidden_size, audio_tokens, destination.data()));
    assert(destination == std::vector<float>({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    }));
}

static void test_python_tts_budget_excludes_exhausted_loop_slot() {
    assert(omni_duplex_tts_output_token_count_for_budget(26) == 25);
    assert(omni_duplex_tts_output_token_count_for_budget(1) == 0);
    assert(omni_duplex_tts_output_token_count_for_budget(0) == 0);
}

static void test_python_text_chunk_plan_reserves_one_lookahead_token() {
    const OmniTextChunkPlan first = omni_text_chunk_plan(10, false);
    assert(first.prefix_tokens == 0);
    assert(first.condition_tokens == 10);
    assert(first.generated_tokens == 11);

    const OmniTextChunkPlan next = omni_text_chunk_plan(10, true);
    assert(next.prefix_tokens == 1);
    assert(next.condition_tokens == 9);
    assert(next.generated_tokens == 10);
}

int main() {
    test_head_code_uses_row_major_audio_token_rows();
    test_python_tts_budget_excludes_exhausted_loop_slot();
    test_python_text_chunk_plan_reserves_one_lookahead_token();
    return 0;
}
