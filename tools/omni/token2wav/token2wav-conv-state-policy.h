#pragma once

#include <cstdint>

namespace omni {
namespace flow {

constexpr bool token2wav_legacy_conv_state_requested(const char * value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

constexpr bool token2wav_use_current_tail_for_conv_state(int64_t      dt,
                                                         int64_t      pad,
                                                         bool         compatible_shape,
                                                         const char * legacy_env) {
    return compatible_shape && pad > 0 && dt >= pad &&
           !token2wav_legacy_conv_state_requested(legacy_env);
}

}  // namespace flow
}  // namespace omni
