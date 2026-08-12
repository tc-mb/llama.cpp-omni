#pragma once

#include <cstdint>

namespace omni {
namespace flow {

constexpr bool token2wav_legacy_est_att_cpy_requested(const char * value) {
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

struct token2wav_est_att_shape {
    int     n_dims;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    bool    is_f32;
};

constexpr bool token2wav_est_att_identity_shape_is_canonical(
    const token2wav_est_att_shape & produced,
    const token2wav_est_att_shape & persistent,
    int64_t                         delta) {
    // Canonical MiniCPM-o 4.5 packed estimator attention cache:
    //   [2 * head_dim=128, time, heads * depth * steps=640, CFG batch=2].
    // The current flow attention graph packs [current(delta), persistent].
    // Only the exact L+delta case proves that slice(delta, L) is byte-for-byte
    // the persistent input. Max-cache truncation and unknown layouts fall back.
    return produced.is_f32 && persistent.is_f32 &&
           produced.n_dims == 4 && persistent.n_dims == 4 &&
           persistent.ne0 == 128 && persistent.ne1 > 0 &&
           persistent.ne2 == 640 && persistent.ne3 == 2 &&
           delta > 0 &&
           produced.ne0 == persistent.ne0 &&
           produced.ne2 == persistent.ne2 &&
           produced.ne3 == persistent.ne3 &&
           produced.ne1 > persistent.ne1 &&
           produced.ne1 - persistent.ne1 == delta;
}

constexpr bool token2wav_should_elide_est_att_writeback(
    const char *                    legacy_env,
    const token2wav_est_att_shape & produced,
    const token2wav_est_att_shape & persistent,
    int64_t                         delta) {
    return !token2wav_legacy_est_att_cpy_requested(legacy_env) &&
           token2wav_est_att_identity_shape_is_canonical(produced, persistent, delta);
}

struct token2wav_est_att_graph_structure {
    int cache_value_concat_nodes;
    int block_pack_concat_nodes;
    int step_pack_concat_nodes;
    int trim_cont_nodes;
    int writeback_cpy_nodes;
    int selected_compute_nodes;
    int removed_compute_nodes;
};

constexpr token2wav_est_att_graph_structure token2wav_est_att_writeback_graph_structure(
    int  depth,
    int  n_timesteps,
    bool has_trim,
    bool elided) {
    const int cache_values = depth > 0 && n_timesteps > 0 ? depth * n_timesteps : 0;
    const int block_packs  = depth > 0 && n_timesteps > 0 ? (depth - 1) * n_timesteps : 0;
    const int step_packs   = depth > 0 && n_timesteps > 0 ? n_timesteps - 1 : 0;
    const int trim         = cache_values > 0 && has_trim ? 1 : 0;
    const int cpy          = cache_values > 0 ? 1 : 0;
    const int legacy       = cache_values + block_packs + step_packs + trim + cpy;
    return {
        cache_values,
        block_packs,
        step_packs,
        trim,
        cpy,
        elided ? 0 : legacy,
        elided ? legacy : 0,
    };
}

}  // namespace flow
}  // namespace omni
