#pragma once

#include <cstdint>

namespace omni {
namespace flow {

constexpr bool token2wav_legacy_adaln_cache_requested(const char * value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

constexpr bool token2wav_adaln_cache_is_canonical(int     depth,
                                                   int     hidden_size,
                                                   int     n_timesteps,
                                                   int64_t cfg_batch,
                                                   bool    timestep_is_f32) {
    return depth == 16 && hidden_size == 512 && n_timesteps == 5 && cfg_batch == 2 && timestep_is_f32;
}

constexpr bool token2wav_should_cache_adaln(const char * legacy_env,
                                             int          depth,
                                             int          hidden_size,
                                             int          n_timesteps,
                                             int64_t      cfg_batch,
                                             bool         timestep_is_f32) {
    return !token2wav_legacy_adaln_cache_requested(legacy_env) &&
           token2wav_adaln_cache_is_canonical(depth, hidden_size, n_timesteps, cfg_batch, timestep_is_f32);
}

struct token2wav_adaln_cache_structure {
    int modulation_linears_on_first_use;
    int modulation_linears_on_reuse;
    int persistent_cache_tensors_per_call_id;
    int separately_cached_call_ids;
};

constexpr token2wav_adaln_cache_structure token2wav_adaln_cache_graph_structure(int depth,
                                                                                 int n_timesteps,
                                                                                 bool enabled) {
    const int linears = depth > 0 && n_timesteps > 0 ? (depth + 1) * n_timesteps : 0;
    return {
        linears,
        enabled ? 0 : linears,
        enabled ? 2 : 0, // packed block modulation plus packed final modulation
        enabled ? 2 : 0, // non-last and last call-id stay isolated
    };
}

}  // namespace flow
}  // namespace omni
