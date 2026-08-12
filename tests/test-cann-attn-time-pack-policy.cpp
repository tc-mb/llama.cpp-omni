#include "attn-time-pack-policy.h"

#include <cassert>

int main() {
    ggml_cann_attn_time_pack_policy_input input;
    assert(ggml_cann_attn_time_pack_validate_policy(input) ==
           ggml_cann_attn_time_pack_fallback::NOT_EXACT_CHAIN);
    input.exact_chain = true;
    input.all_f32 = true;
    input.canonical_shape = true;
    input.supported_layout = true;
    input.buffers_disjoint = true;
    input.buffers_aligned = true;
    assert(ggml_cann_attn_time_pack_validate_policy(input) ==
           ggml_cann_attn_time_pack_fallback::NONE);
    assert(ggml_cann_attn_time_pack_launches_removed_per_hit == 1);

    input.buffers_aligned = false;
    assert(ggml_cann_attn_time_pack_validate_policy(input) ==
           ggml_cann_attn_time_pack_fallback::ALIGNMENT);
    input.buffers_aligned = true;
    input.supported_layout = false;
    assert(ggml_cann_attn_time_pack_validate_policy(input) ==
           ggml_cann_attn_time_pack_fallback::LAYOUT);
    return 0;
}
