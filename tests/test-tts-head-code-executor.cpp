#include "tts-head-code-executor.h"

#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <vector>

static void verify_shape(std::size_t rows, std::size_t cols) {
    std::vector<float> hidden(cols);
    std::vector<float> weights(rows * cols);
    for (std::size_t index = 0; index < cols; ++index) {
        hidden[index] = std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = std::cos(static_cast<float>(index % 4093) * 0.013f) * 0.125f;
    }

    std::vector<float> reference(rows);
    TtsHeadCodeExecutor::compute_scalar(
        hidden.data(), weights.data(), reference.data(), rows, cols);

    for (std::size_t thread_count : { 1U, 2U, 4U, 8U }) {
        TtsHeadCodeExecutor executor(thread_count);
        std::vector<float> candidate(rows, 123.0f);
        executor.compute(
            hidden.data(), weights.data(), candidate.data(), rows, cols);
        assert(std::memcmp(
            reference.data(), candidate.data(), rows * sizeof(float)) == 0);

        std::fill(candidate.begin(), candidate.end(), -321.0f);
        executor.compute(
            hidden.data(), weights.data(), candidate.data(), rows, cols);
        assert(std::memcmp(
            reference.data(), candidate.data(), rows * sizeof(float)) == 0);
    }
}

int main() {
    verify_shape(1, 1);
    verify_shape(17, 31);
    verify_shape(6562, 768);
    const int original_rounding = std::fegetround();
    assert(std::fesetround(FE_DOWNWARD) == 0);
    verify_shape(257, 129);
    assert(std::fesetround(FE_UPWARD) == 0);
    verify_shape(257, 129);
    assert(std::fesetround(original_rounding) == 0);
    return 0;
}
