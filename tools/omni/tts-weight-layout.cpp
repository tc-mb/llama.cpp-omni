#include "tts-weight-layout.h"

#include <algorithm>

namespace omni {

bool has_supported_head_code_weight_layout(int64_t n_dims,
                                           int64_t n_elements,
                                           int64_t dim0,
                                           int64_t dim1,
                                           int64_t hidden_size,
                                           int64_t num_audio_tokens) {
    if (n_dims != 2 || hidden_size <= 0 || num_audio_tokens <= 0 ||
        n_elements != hidden_size * num_audio_tokens) {
        return false;
    }

    return (dim0 == hidden_size && dim1 == num_audio_tokens) ||
           (dim0 == num_audio_tokens && dim1 == hidden_size);
}

bool copy_head_code_weight_to_output_major(const float * source,
                                           int64_t       dim0,
                                           int64_t       dim1,
                                           int64_t       hidden_size,
                                           int64_t       num_audio_tokens,
                                           float *       destination) {
    if (source == nullptr || destination == nullptr ||
        !has_supported_head_code_weight_layout(
            2, hidden_size * num_audio_tokens, dim0, dim1, hidden_size, num_audio_tokens)) {
        return false;
    }

    const bool has_standard_layout = dim0 == hidden_size;
    if (has_standard_layout) {
        std::copy_n(source, hidden_size * num_audio_tokens, destination);
        return true;
    }

    for (int64_t token = 0; token < num_audio_tokens; ++token) {
        for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
            destination[token * hidden_size + hidden] = source[hidden * num_audio_tokens + token];
        }
    }
    return true;
}

} // namespace omni
