#pragma once

#include <cstdint>

struct vocoder_gather_mask_pattern {
    uint32_t word0 = 0;
    uint32_t word1 = 0;
    uint32_t selected = 0;
};

#if defined(__NPU_ARCH__)
#define GGML_CANN_VOCODER_AICORE __aicore__
#else
#define GGML_CANN_VOCODER_AICORE
#endif

GGML_CANN_VOCODER_AICORE constexpr vocoder_gather_mask_pattern
make_vocoder_gather_mask_pattern(
        uint32_t kernel,
        uint32_t dilation) {
    vocoder_gather_mask_pattern result;
    for (uint32_t tap = 0; tap < kernel; ++tap) {
        const uint32_t lane = tap * dilation;
        if (lane < 32) {
            result.word0 |= UINT32_C(1) << lane;
        } else {
            result.word1 |= UINT32_C(1) << (lane - 32);
        }
    }
    result.selected = kernel;
    return result;
}

#undef GGML_CANN_VOCODER_AICORE
