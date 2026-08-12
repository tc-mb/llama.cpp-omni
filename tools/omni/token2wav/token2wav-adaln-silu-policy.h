#pragma once

#include <cstdint>

namespace omni {
namespace flow {

constexpr bool token2wav_legacy_adaln_silu_requested(const char * value) {
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

constexpr bool token2wav_adaln_silu_is_canonical(int     depth,
                                                  int     hidden_size,
                                                  int     n_dims,
                                                  int64_t ne0,
                                                  int64_t ne1,
                                                  int64_t ne2,
                                                  bool    compatible_type) {
    return compatible_type && depth == 16 && hidden_size == 512 && n_dims == 3 &&
           ne0 == hidden_size && ne1 == 1 && ne2 == 2;
}

constexpr bool token2wav_should_share_adaln_silu(const char * legacy_env,
                                                  int          depth,
                                                  int          hidden_size,
                                                  int          n_dims,
                                                  int64_t      ne0,
                                                  int64_t      ne1,
                                                  int64_t      ne2,
                                                  bool         compatible_type) {
    return !token2wav_legacy_adaln_silu_requested(legacy_env) &&
           token2wav_adaln_silu_is_canonical(depth, hidden_size, n_dims, ne0, ne1, ne2, compatible_type);
}

struct token2wav_adaln_silu_graph_counts {
    int legacy_nodes_per_step;
    int selected_nodes_per_step;
    int removed_nodes_per_step;
};

constexpr token2wav_adaln_silu_graph_counts token2wav_adaln_silu_graph_structure(int  depth,
                                                                                  bool shared) {
    const int legacy = depth > 0 ? depth + 1 : 0; // one per block plus final layer
    const int selected = shared && legacy > 0 ? 1 : legacy;
    return { legacy, selected, legacy - selected };
}

}  // namespace flow
}  // namespace omni
