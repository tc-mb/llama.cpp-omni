#pragma once

#include <cstddef>
#include <cstdint>

namespace omni {
namespace flow {

enum class final_wav_command : uint8_t {
    data,
    spec_render,
    commit,
    abort,
};

struct final_wav_key {
    int      turn  = -1;
    uint64_t epoch = 0;

    constexpr bool operator==(const final_wav_key & other) const {
        return turn == other.turn && epoch == other.epoch;
    }
};

class final_wav_transaction_state {
  public:
    enum class phase : uint8_t { idle, rendering, tentative, damaged, disabled };

    explicit constexpr final_wav_transaction_state(std::size_t max_replay_calls = 64)
        : max_replay_calls_(max_replay_calls) {}

    constexpr bool note_committed_call() {
        if (phase_ != phase::idle || committed_calls_ >= max_replay_calls_) {
            phase_ = phase::disabled;
            return false;
        }
        ++committed_calls_;
        return true;
    }

    constexpr bool begin(final_wav_key key) {
        if (phase_ != phase::idle || committed_calls_ >= max_replay_calls_ ||
            key.turn < 0 || key.epoch == 0) {
            if (phase_ == phase::idle && committed_calls_ >= max_replay_calls_) {
                phase_ = phase::disabled;
            }
            return false;
        }
        key_ = key;
        phase_ = phase::rendering;
        return true;
    }

    constexpr bool rendered(final_wav_key key, bool ok) {
        if (phase_ != phase::rendering || !(key == key_)) {
            return false;
        }
        phase_ = ok ? phase::tentative : phase::damaged;
        return ok;
    }

    constexpr bool commit(final_wav_key key, std::size_t tentative_calls = 0) {
        if (!can_commit(key, tentative_calls)) {
            return false;
        }
        if (tentative_calls >= max_replay_calls_ - committed_calls_) {
            committed_calls_ = max_replay_calls_;
            phase_ = phase::disabled;
            key_ = {};
            return true;
        }
        committed_calls_ += tentative_calls;
        reset_turn();
        return true;
    }

    constexpr bool can_commit(final_wav_key key, std::size_t tentative_calls = 0) const {
        return phase_ == phase::tentative && key == key_ &&
               tentative_calls <= max_replay_calls_ - committed_calls_;
    }

    constexpr bool abort(final_wav_key key, bool replay_equivalent) {
        if ((phase_ != phase::rendering && phase_ != phase::tentative &&
             phase_ != phase::damaged) || !(key == key_)) {
            return false;
        }
        if (!replay_equivalent) {
            phase_ = phase::damaged;
            return false;
        }
        phase_ = phase::idle;
        key_ = {};
        return true;
    }

    constexpr void break_reset(bool replay_equivalent) {
        if (phase_ == phase::rendering || phase_ == phase::tentative || phase_ == phase::damaged) {
            phase_ = replay_equivalent ? phase::idle : phase::damaged;
        }
        if (replay_equivalent) key_ = {};
    }

    constexpr void reset_turn() {
        phase_ = phase::idle;
        key_ = {};
    }

    constexpr void discard_reset() {
        committed_calls_ = 0;
        phase_ = phase::idle;
        key_ = {};
    }

    constexpr phase current_phase() const { return phase_; }
    constexpr bool has_active_key(final_wav_key key) const {
        return (phase_ == phase::rendering || phase_ == phase::tentative ||
                phase_ == phase::damaged) &&
               key == key_;
    }
    constexpr bool has_tentative_key(final_wav_key key) const {
        return phase_ == phase::tentative && key == key_;
    }
    constexpr bool busy_or_tentative() const {
        return phase_ == phase::rendering || phase_ == phase::tentative;
    }
    constexpr bool needs_recovery() const {
        return phase_ == phase::rendering || phase_ == phase::tentative ||
               phase_ == phase::damaged;
    }
    constexpr std::size_t committed_calls() const { return committed_calls_; }

  private:
    std::size_t max_replay_calls_ = 64;
    std::size_t committed_calls_ = 0;
    phase       phase_ = phase::idle;
    final_wav_key key_{};
};

// Pure state guarded by T2WThreadInfo::mtx. Keeping admission transitions in
// one testable object prevents check/unlock races without introducing a global
// or cross-device lock.
class final_wav_admission_state {
  public:
    constexpr bool try_admit_encoder() {
        if (rendering_) return false;
        ++encoder_in_flight_;
        return true;
    }
    constexpr void release_encoder() {
        if (encoder_in_flight_ > 0) --encoder_in_flight_;
    }
    constexpr bool try_begin_render() {
        if (rendering_ || encoder_in_flight_ != 0) return false;
        rendering_ = true;
        return true;
    }
    constexpr void end_render() { rendering_ = false; }
    constexpr bool rendering() const { return rendering_; }
    constexpr int encoder_in_flight() const { return encoder_in_flight_; }
    constexpr bool idle() const { return !rendering_ && encoder_in_flight_ == 0; }

  private:
    int  encoder_in_flight_ = 0;
    bool rendering_ = false;
};

} // namespace flow
} // namespace omni
