#include "token2wav.h"

#include <cstdio>

namespace omni {
namespace flow {

bool Token2WavSession::init_from_prompt_cache_gguf(const std::string & encoder_gguf,
                                                   const std::string & flow_matching_gguf,
                                                   const std::string & flow_extra_gguf,
                                                   const std::string & prompt_cache_gguf_path,
                                                   const std::string & vocoder_gguf,
                                                   const std::string & device_token2mel,
                                                   const std::string & device_vocoder,
                                                   int                 n_timesteps,
                                                   float               temperature) {
    // 初始化方式第一种，仅需使用此方式即可，加载模型所有的gguf内容
    reset();
    if (!t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf, device_token2mel,
                         device_vocoder)) {
        return false;
    }
    if (!t2w.start_stream_with_prompt_cache_gguf(prompt_cache_gguf_path, n_timesteps, temperature)) {
        return false;
    }

    return true;
}

bool Token2WavSession::init_from_prompt_bundle(const std::string & encoder_gguf,
                                               const std::string & flow_matching_gguf,
                                               const std::string & flow_extra_gguf,
                                               const std::string & prompt_bundle_dir,
                                               const std::string & vocoder_gguf,
                                               const std::string & device_token2mel,
                                               const std::string & device_vocoder,
                                               int                 n_timesteps,
                                               float               temperature) {
    // 初始化方式第二种（如要更换示例音频使用此接口作为备用）
    reset();
    if (!t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf, device_token2mel,
                         device_vocoder)) {
        return false;
    }
    Token2Mel::PromptBundle pb;
    if (!Token2Mel::load_prompt_bundle_dir(prompt_bundle_dir, pb)) {
        return false;
    }
    if (!t2w.start_stream_with_prompt(pb, n_timesteps, temperature)) {
        return false;
    }
    return true;
}

bool Token2WavSession::feed_window(const int32_t *      tokens,
                                   int64_t              n_tokens,
                                   bool                 is_final,
                                   std::vector<float> & wave_bt_out) {
    // 推理送入token第一种（vector返回，持续写到vector中。调用方持有wave_bt_out，返回后数据还在）
    // 在外部做好25token+下一个chunk的3个token，送入28token
    wave_bt_out.clear();
    int64_t T_audio = 0;
    if (!t2w.push_tokens_window(tokens, n_tokens, is_final, wave_bt_out, T_audio)) {
        return false;
    }
    return true;
}

bool Token2WavSession::feed_window(const int32_t *              tokens,
                                   int64_t                      n_tokens,
                                   bool                         is_final,
                                   const audio_chunk_callback & on_audio_chunk) {
    // 推理送入token第一种（callback返回，等于调一下就推出来一个chunk，不会有队列存。先把结果写到成员中，再通过回调拿到wave_tmp_.data() ）
    // 在外部做好25token+下一个chunk的3个token，送入28token（example中使用此回调方式，适合边生成边推出）
    wave_tmp_.clear();
    int64_t T_audio = 0;
    if (!t2w.push_tokens_window(tokens, n_tokens, is_final, wave_tmp_, T_audio)) {
        return false;
    }
    if (on_audio_chunk && !wave_tmp_.empty()) {
        on_audio_chunk(wave_tmp_.data(), (int64_t) wave_tmp_.size());
    }
    return true;
}

bool Token2WavSession::feed_tokens(const int32_t *      tokens,
                                   int64_t              n_tokens,
                                   bool                 is_final,
                                   std::vector<float> & wave_bt_out) {
    // 推理送入token第二种（vector返回）（不推荐使用）
    // 上层没切好没切好，此处内部自动累积25+3开始推理
    wave_bt_out.clear();
    if (tokens && n_tokens > 0) {
        pending_.insert(pending_.end(), tokens, tokens + n_tokens);
    }

    while ((int64_t) pending_.size() >= Token2Mel::kDt) {
        std::vector<int32_t> window(pending_.begin(), pending_.begin() + Token2Mel::kDt);
        std::vector<float>   wave_call;
        int64_t              T_audio = 0;
        if (!t2w.push_tokens_window(window.data(), (int64_t) window.size(), false, wave_call, T_audio)) {
            return false;
        }
        token2wav_utils::append_bt_along_time_b1(wave_call, wave_bt_out);
        pending_.erase(pending_.begin(), pending_.begin() + Token2Mel::kChunkMain);
    }

    if (is_final) {
        std::vector<float> wave_call;
        int64_t            T_audio   = 0;
        const int64_t      remaining = (int64_t) pending_.size();
        const int32_t *    tail      = remaining > 0 ? pending_.data() : nullptr;
        if (!t2w.push_tokens_window(tail, remaining, true, wave_call, T_audio)) {
            return false;
        }
        token2wav_utils::append_bt_along_time_b1(wave_call, wave_bt_out);
        pending_.clear();
    }

    return true;
}

bool Token2WavSession::feed_tokens(const int32_t *              tokens,
                                   int64_t                      n_tokens,
                                   bool                         is_final,
                                   const audio_chunk_callback & on_audio_chunk) {
    // 推理送入token第二种（callback返回）（不推荐使用）
    // 上层没切好没切好，此处内部自动累积25+3开始推理
    if (tokens && n_tokens > 0) {
        pending_.insert(pending_.end(), tokens, tokens + n_tokens);
    }

    while ((int64_t) pending_.size() >= Token2Mel::kDt) {
        std::vector<int32_t> window(pending_.begin(), pending_.begin() + Token2Mel::kDt);
        int64_t              T_audio = 0;
        wave_tmp_.clear();
        if (!t2w.push_tokens_window(window.data(), (int64_t) window.size(), false, wave_tmp_, T_audio)) {
            return false;
        }
        if (on_audio_chunk && !wave_tmp_.empty()) {
            on_audio_chunk(wave_tmp_.data(), (int64_t) wave_tmp_.size());
        }
        pending_.erase(pending_.begin(), pending_.begin() + Token2Mel::kChunkMain);
    }

    if (is_final) {
        int64_t T_audio = 0;
        wave_tmp_.clear();
        const int64_t   remaining = (int64_t) pending_.size();
        const int32_t * tail      = remaining > 0 ? pending_.data() : nullptr;
        if (!t2w.push_tokens_window(tail, remaining, true, wave_tmp_, T_audio)) {
            return false;
        }
        if (on_audio_chunk && !wave_tmp_.empty()) {
            on_audio_chunk(wave_tmp_.data(), (int64_t) wave_tmp_.size());
        }
        pending_.clear();
    }

    return true;
}

void Token2WavSession::reset() {
    // 清空 pending，并重置内部流式状态
    pending_.clear();
    t2w.reset_stream();
}

}  // namespace flow
}  // namespace omni
