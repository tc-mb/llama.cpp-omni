#pragma once

#include <cstdint>

namespace omni {

bool has_supported_head_code_weight_layout(int64_t n_dims,
                                           int64_t n_elements,
                                           int64_t dim0,
                                           int64_t dim1,
                                           int64_t hidden_size,
                                           int64_t num_audio_tokens);

bool copy_head_code_weight_to_output_major(const float * source,
                                           int64_t       dim0,
                                           int64_t       dim1,
                                           int64_t       hidden_size,
                                           int64_t       num_audio_tokens,
                                           float *       destination);

} // namespace omni
