#include "token2wav.h"
#include "token2wav-profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace omni {
namespace flow {

namespace {

bool token2wav_init_warmup_enabled() {
    const char * value = std::getenv("OMNI_T2W_INIT_WARMUP");
    return value && value[0] == '1' && value[1] == '\0';
}

bool token2wav_warmup_fixed_window(Token2Wav & t2w) {
    std::vector<int32_t> tokens(Token2Mel::kDt, 4218);
    std::vector<float> wave;
    int64_t n_audio = 0;
    return t2w.push_tokens_window(tokens, /*is_final=*/false, wave, n_audio);
}

}  // namespace

bool Token2WavSession::init_from_prompt_cache_gguf(const std::string & encoder_gguf,
                                                   const std::string & flow_matching_gguf,
                                                   const std::string & flow_extra_gguf,
                                                   const std::string & prompt_cache_gguf_path,
                                                   const std::string & vocoder_gguf,
                                                   const std::string & device_token2mel,
                                                   const std::string & device_vocoder,
                                                   int                 n_timesteps,
                                                   float               temperature,
                                                   const std::string & coreml_model_path) {
    // 初始化方式第一种，仅需使用此方式即可，加载模型所有的gguf内容
    const auto t0 = std::chrono::steady_clock::now();
    reset();
    if (!t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf, device_token2mel,
                         device_vocoder, coreml_model_path)) {
        return false;
    }
    const bool do_warmup = token2wav_init_warmup_enabled() && coreml_model_path.empty();
    const std::mt19937 initial_noise_state = t2w.capture_noise_state();
    if (!t2w.start_stream_with_prompt_cache_gguf(prompt_cache_gguf_path, n_timesteps, temperature)) {
        return false;
    }
    if (do_warmup) {
        if (!token2wav_warmup_fixed_window(t2w)) {
            return false;
        }
        // The warmup mutates all streaming caches. Reload the prompt cache so
        // the first user token sees exactly the same pristine state as before.
        // Restoring the process-stream RNG before that reload also preserves
        // the baseline noise sequence bit-for-bit.
        t2w.restore_noise_state(initial_noise_state);
        if (!t2w.start_stream_with_prompt_cache_gguf(prompt_cache_gguf_path, n_timesteps, temperature)) {
            return false;
        }
    }
    replay_prompt_kind_ = PromptKind::cache_gguf;
    replay_prompt_cache_path_ = prompt_cache_gguf_path;
    replay_prompt_bundle_ = {};
    replay_n_timesteps_ = n_timesteps;
    replay_temperature_ = temperature;
    replay_initial_noise_ = initial_noise_state;
    replay_calls_.clear();
    const auto   t1     = std::chrono::steady_clock::now();
    const double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    omni::flow::profile::record_init_ms(init_ms);
    if (omni::flow::profile::verbose()) {
        std::fprintf(stderr, "[timing] init_from_prompt_cache_gguf=%.3fms\n", init_ms);
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
                                               float               temperature,
                                               const std::string & coreml_model_path) {
    // 初始化方式第二种（如要更换示例音频使用此接口作为备用）
    reset();
    if (!t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf, device_token2mel,
                         device_vocoder, coreml_model_path)) {
        return false;
    }
    const bool do_warmup = token2wav_init_warmup_enabled() && coreml_model_path.empty();
    const std::mt19937 initial_noise_state = t2w.capture_noise_state();
    Token2Mel::PromptBundle pb;
    if (!Token2Mel::load_prompt_bundle_dir(prompt_bundle_dir, pb)) {
        return false;
    }
    if (!t2w.start_stream_with_prompt(pb, n_timesteps, temperature)) {
        return false;
    }
    if (do_warmup) {
        if (!token2wav_warmup_fixed_window(t2w)) {
            return false;
        }
        t2w.restore_noise_state(initial_noise_state);
        if (!t2w.start_stream_with_prompt(pb, n_timesteps, temperature)) {
            return false;
        }
    }
    replay_prompt_kind_ = PromptKind::bundle;
    replay_prompt_cache_path_.clear();
    replay_prompt_bundle_ = pb;
    replay_n_timesteps_ = n_timesteps;
    replay_temperature_ = temperature;
    replay_initial_noise_ = initial_noise_state;
    replay_calls_.clear();
    return true;
}

bool Token2WavSession::switch_prompt_bundle(const std::string & prompt_bundle_dir,
                                            int n_timesteps, float temperature) {
    // 仅更换示例音频：先 reset 流式状态（清 pending + flow-matching/vocoder 缓存），
    // 再用新的 prompt bundle 重新 start_stream。不重新加载模型。
    reset();
    Token2Mel::PromptBundle pb;
    if (!Token2Mel::load_prompt_bundle_dir(prompt_bundle_dir, pb)) {
        fprintf(stderr, "Token2WavSession: failed to load prompt bundle from %s\n", prompt_bundle_dir.c_str());
        return false;
    }
    const std::mt19937 initial_noise_state = t2w.capture_noise_state();
    if (!t2w.start_stream_with_prompt(pb, n_timesteps, temperature)) {
        fprintf(stderr, "Token2WavSession: failed to start stream with new prompt bundle\n");
        return false;
    }
    replay_prompt_kind_ = PromptKind::bundle;
    replay_prompt_cache_path_.clear();
    replay_prompt_bundle_ = pb;
    replay_n_timesteps_ = n_timesteps;
    replay_temperature_ = temperature;
    replay_initial_noise_ = initial_noise_state;
    replay_calls_.clear();
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
    if (replay_enabled_ && !replay_tentative_ && !replay_replaying_) {
        if (replay_calls_.size() >= replay_max_calls_) {
            replay_enabled_ = false;
            replay_calls_.clear();
        } else {
            ReplayCall call;
            if (tokens && n_tokens > 0) call.tokens.assign(tokens, tokens + n_tokens);
            call.is_final = is_final;
            replay_calls_.push_back(std::move(call));
        }
    }
    return true;
}

void Token2WavSession::enable_bounded_replay(bool enable, std::size_t max_calls) {
    replay_enabled_ = enable && replay_prompt_kind_ != PromptKind::none && max_calls > 0;
    replay_max_calls_ = max_calls;
    replay_tentative_ = false;
    replay_replaying_ = false;
    replay_calls_.clear();
    replay_tentative_calls_.clear();
    replay_checkpoint_valid_ = false;
}

bool Token2WavSession::bounded_replay_available() const {
    return replay_enabled_ && !replay_tentative_ && replay_prompt_kind_ != PromptKind::none &&
           replay_calls_.size() < replay_max_calls_;
}

bool Token2WavSession::begin_tentative() {
    if (!bounded_replay_available()) return false;
    replay_checkpoint_checksum_ = t2w.debug_stream_cache_checksum();
    replay_checkpoint_noise_ = t2w.capture_noise_state();
    replay_checkpoint_valid_ = true;
    replay_tentative_calls_.clear();
    replay_tentative_ = true;
    return true;
}

bool Token2WavSession::feed_window_tentative(const std::vector<int32_t> & tokens, bool is_final,
                                             std::vector<float> & wave_bt_out) {
    if (!replay_tentative_) return false;
    int64_t n_audio = 0;
    wave_bt_out.clear();
    if (!t2w.push_tokens_window(tokens.data(), (int64_t) tokens.size(), is_final, wave_bt_out, n_audio)) {
        return false;
    }
    replay_tentative_calls_.push_back(ReplayCall{tokens, is_final});
    return true;
}

bool Token2WavSession::restart_replay_base() {
    t2w.restore_noise_state(replay_initial_noise_);
    if (replay_prompt_kind_ == PromptKind::cache_gguf) {
        return t2w.start_stream_with_prompt_cache_gguf(
            replay_prompt_cache_path_, replay_n_timesteps_, replay_temperature_);
    }
    if (replay_prompt_kind_ == PromptKind::bundle) {
        return t2w.start_stream_with_prompt(
            replay_prompt_bundle_, replay_n_timesteps_, replay_temperature_);
    }
    return false;
}

bool Token2WavSession::abort_tentative_replay() {
    if (!replay_tentative_) return false;
    replay_tentative_ = false;
    replay_tentative_calls_.clear();
    return recover_committed_replay();
}

bool Token2WavSession::recover_committed_replay() {
    replay_tentative_ = false;
    replay_tentative_calls_.clear();
    replay_replaying_ = true;
    bool ok = restart_replay_base();
    std::vector<float> suppressed;
    for (const ReplayCall & call : replay_calls_) {
        if (!ok) break;
        int64_t n_audio = 0;
        const int32_t * data = call.tokens.empty() ? nullptr : call.tokens.data();
        ok = t2w.push_tokens_window(data, (int64_t) call.tokens.size(), call.is_final, suppressed, n_audio);
    }
    replay_replaying_ = false;
    if (ok && replay_checkpoint_valid_) {
        ok = t2w.debug_stream_cache_checksum() == replay_checkpoint_checksum_ &&
             t2w.capture_noise_state() == replay_checkpoint_noise_;
    }
    if (!ok) {
        replay_enabled_ = false;
        // Keep the exact committed calls and checkpoint so the transaction
        // finalizer can retry from the saved prompt/RNG base. Clearing them
        // here would make a failed rollback unrecoverable.
        return false;
    }
    replay_checkpoint_valid_ = false;
    replay_enabled_ = replay_prompt_kind_ != PromptKind::none &&
                      replay_max_calls_ > 0 && replay_calls_.size() < replay_max_calls_;
    return true;
}

bool Token2WavSession::commit_tentative() {
    if (!replay_tentative_) return false;
    replay_tentative_ = false;
    replay_checkpoint_valid_ = false;
    if (replay_calls_.size() + replay_tentative_calls_.size() >= replay_max_calls_) {
        replay_enabled_ = false;
        replay_calls_.clear();
        replay_tentative_calls_.clear();
        return true;
    }
    // Commit keeps the exact post-final streaming state. The tentative calls
    // become part of the replay history so a later miss can rebuild it.
    replay_calls_.insert(replay_calls_.end(),
                         std::make_move_iterator(replay_tentative_calls_.begin()),
                         std::make_move_iterator(replay_tentative_calls_.end()));
    replay_tentative_calls_.clear();
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
        const auto t_cb0 = std::chrono::steady_clock::now();
        on_audio_chunk(wave_tmp_.data(), (int64_t) wave_tmp_.size());
        const auto   t_cb1 = std::chrono::steady_clock::now();
        const double cb_ms = std::chrono::duration<double, std::milli>(t_cb1 - t_cb0).count();
        omni::flow::profile::record_ms("callback", cb_ms);
        if (omni::flow::profile::verbose()) {
            std::fprintf(stderr, "[timing] callback=%.3fms samples=%lld\n",
                         cb_ms, (long long) wave_tmp_.size());
        }
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
    replay_tentative_ = false;
    replay_replaying_ = false;
    replay_calls_.clear();
    replay_tentative_calls_.clear();
    replay_checkpoint_valid_ = false;
}

}  // namespace flow
}  // namespace omni
