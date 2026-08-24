#include "tts-weight-layout.h"

#include <cstdio>
#include <vector>

namespace {

bool expect_values(const char * name, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: unexpected values\n", name);
    for (size_t i = 0; i < actual.size(); ++i) {
        std::fprintf(stderr, "  [%zu] actual=%.1f expected=%.1f\n", i, actual[i], expected[i]);
    }
    return false;
}

bool test_standard_gguf_layout_is_copied_without_transpose() {
    constexpr int64_t hidden_size      = 3;
    constexpr int64_t num_audio_tokens = 2;
    const std::vector<float> source    = {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f};
    const std::vector<float> expected  = source;
    std::vector<float>       actual(source.size(), -1.0f);

    if (!omni::copy_head_code_weight_to_output_major(
            source.data(), hidden_size, num_audio_tokens, hidden_size, num_audio_tokens, actual.data())) {
        std::fprintf(stderr, "standard GGUF layout was rejected\n");
        return false;
    }
    return expect_values("standard GGUF layout", actual, expected);
}

bool test_legacy_gguf_layout_is_transposed() {
    constexpr int64_t hidden_size      = 3;
    constexpr int64_t num_audio_tokens = 2;
    const std::vector<float> source    = {10.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f};
    const std::vector<float> expected  = {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f};
    std::vector<float>       actual(source.size(), -1.0f);

    if (!omni::copy_head_code_weight_to_output_major(
            source.data(), num_audio_tokens, hidden_size, hidden_size, num_audio_tokens, actual.data())) {
        std::fprintf(stderr, "legacy GGUF layout was rejected\n");
        return false;
    }
    return expect_values("legacy GGUF layout", actual, expected);
}

bool test_unexpected_shape_is_rejected() {
    const std::vector<float> source(6, 1.0f);
    std::vector<float>       actual(source.size(), -1.0f);

    return !omni::copy_head_code_weight_to_output_major(source.data(), 1, 6, 3, 2, actual.data());
}

bool test_higher_rank_shape_is_rejected() {
    return !omni::has_supported_head_code_weight_layout(3, 12, 3, 2, 3, 2);
}

bool test_wrong_element_count_is_rejected() {
    return !omni::has_supported_head_code_weight_layout(2, 12, 3, 2, 3, 2);
}

} // namespace

int main() {
    if (!test_standard_gguf_layout_is_copied_without_transpose()) {
        return 1;
    }
    if (!test_legacy_gguf_layout_is_transposed()) {
        return 1;
    }
    if (!test_unexpected_shape_is_rejected()) {
        std::fprintf(stderr, "unexpected GGUF layout was accepted\n");
        return 1;
    }
    if (!test_higher_rank_shape_is_rejected()) {
        std::fprintf(stderr, "higher-rank GGUF layout was accepted\n");
        return 1;
    }
    if (!test_wrong_element_count_is_rejected()) {
        std::fprintf(stderr, "GGUF layout with wrong element count was accepted\n");
        return 1;
    }
    return 0;
}
