#pragma once

#include <cstdint>

namespace omni {
namespace flow {

struct token2wav_final_graph_bypass {
    bool cuda;
    bool cann;
};

constexpr token2wav_final_graph_bypass token2wav_final_graph_bypass_for(
    bool last_chunk,
    bool disable_cuda_last_graph) {
    return { last_chunk && disable_cuda_last_graph, last_chunk };
}

struct token2wav_vocoder_graph_bucket {
    int64_t t_mel;
    int64_t tc;
};

constexpr token2wav_vocoder_graph_bucket token2wav_vocoder_cann_graph_bucket(
    int64_t mel_cache_len,
    int64_t mel_main_len,
    int64_t source_cache_len) {
    return { mel_cache_len + mel_main_len, source_cache_len };
}

constexpr bool token2wav_vocoder_uses_cann_graph(bool    is_final,
                                                  int64_t t_mel,
                                                  int64_t tc,
                                                  int64_t mel_cache_len,
                                                  int64_t mel_main_len,
                                                  int64_t source_cache_len) {
    return !is_final &&
           t_mel == mel_cache_len + mel_main_len &&
           tc == source_cache_len;
}

}  // namespace flow
}  // namespace omni
