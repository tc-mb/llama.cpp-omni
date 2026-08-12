#pragma once

#include "set-rows-f32-f16-policy.h"

#include <array>
#include <cstddef>
#include <cstdint>

struct ggml_backend_cann_context;
struct ggml_tensor;

struct ggml_cann_set_rows_f32_f16_stats_snapshot {
    uint64_t attempts = 0;
    uint64_t hits = 0;
    std::array<uint64_t,
               static_cast<size_t>(ggml_cann_set_rows_f32_f16_fallback::COUNT)>
        fallback{};
};

ggml_cann_set_rows_f32_f16_fallback ggml_cann_set_rows_f32_f16_validate(
        const ggml_tensor * dst);
bool ggml_cann_set_rows_f32_f16_try(
        ggml_backend_cann_context & ctx,
        ggml_tensor * dst);
const char * ggml_cann_set_rows_f32_f16_fallback_name(
        ggml_cann_set_rows_f32_f16_fallback reason);
void ggml_cann_set_rows_f32_f16_stats_reset();
ggml_cann_set_rows_f32_f16_stats_snapshot
ggml_cann_set_rows_f32_f16_stats_get();
