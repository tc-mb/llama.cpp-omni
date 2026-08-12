#include "token2wav-graph-policy.h"

#include <cstdint>

int main() {
    constexpr int64_t mel_cache_len    = 8;
    constexpr int64_t mel_main_len     = 50;
    constexpr int64_t source_cache_len = 3840;
    constexpr int64_t steady_mel_len   = mel_cache_len + mel_main_len;

    using omni::flow::token2wav_vocoder_uses_cann_graph;

    // The debug opt-out only affects CUDA. CANN final chunks must remain eager
    // so one-shot shapes never enter the ACL Graph cache.
    constexpr auto final_with_opt_out = omni::flow::token2wav_final_graph_bypass_for(
        true, false);
    static_assert(!final_with_opt_out.cuda);
    static_assert(final_with_opt_out.cann);
    constexpr auto non_final = omni::flow::token2wav_final_graph_bypass_for(
        false, true);
    static_assert(!non_final.cuda);
    static_assert(!non_final.cann);

    constexpr auto prewarm_bucket = omni::flow::token2wav_vocoder_cann_graph_bucket(
        mel_cache_len, mel_main_len, source_cache_len);
    static_assert(prewarm_bucket.t_mel == steady_mel_len);
    static_assert(prewarm_bucket.tc == source_cache_len);
    static_assert(token2wav_vocoder_uses_cann_graph(false, prewarm_bucket.t_mel, prewarm_bucket.tc,
                                                    mel_cache_len, mel_main_len, source_cache_len));

    // Startup has no source cache and must not capture a one-shot graph.
    if (token2wav_vocoder_uses_cann_graph(false, mel_main_len, 0,
                                          mel_cache_len, mel_main_len, source_cache_len)) {
        return 1;
    }
    // Only the repeatable steady bucket is Graph eligible.
    if (!token2wav_vocoder_uses_cann_graph(false, steady_mel_len, source_cache_len,
                                           mel_cache_len, mel_main_len, source_cache_len)) {
        return 2;
    }
    // Final and irregular shapes are one-shot eager calls.
    if (token2wav_vocoder_uses_cann_graph(true, steady_mel_len, source_cache_len,
                                          mel_cache_len, mel_main_len, source_cache_len)) {
        return 3;
    }
    if (token2wav_vocoder_uses_cann_graph(false, steady_mel_len - 2, source_cache_len,
                                          mel_cache_len, mel_main_len, source_cache_len)) {
        return 4;
    }
    if (token2wav_vocoder_uses_cann_graph(false, steady_mel_len, source_cache_len - 1,
                                          mel_cache_len, mel_main_len, source_cache_len)) {
        return 5;
    }

    return 0;
}
