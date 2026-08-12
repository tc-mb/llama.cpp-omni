#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>

enum class DuplexTerminalStatus {
    UNKNOWN,
    ENQUEUED,
    COMPLETED,
    CANCELED,
    FAILED,
};

enum class DuplexBreakStage {
    LLM,
    TTS,
    T2W,
};

class DuplexSpeakSegmentTracker {
public:
    bool begin(int & segment_id) {
        int current = active_.load(std::memory_order_acquire);
        if (current >= 0) {
            segment_id = current;
            return false;
        }

        const int candidate = next_.fetch_add(1, std::memory_order_relaxed);
        current = -1;
        if (active_.compare_exchange_strong(
                current, candidate,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            segment_id = candidate;
            return true;
        }
        segment_id = current;
        return false;
    }

    int active() const {
        return active_.load(std::memory_order_acquire);
    }

    bool end(int segment_id) {
        if (segment_id < 0) {
            return false;
        }
        int expected = segment_id;
        return active_.compare_exchange_strong(
            expected, -1,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    int cancel() {
        return active_.exchange(-1, std::memory_order_acq_rel);
    }

    void start_session() {
        active_.store(-1, std::memory_order_release);
    }

private:
    std::atomic<int> active_{-1};
    std::atomic<int> next_{0};
};

class DuplexBreakCoordinator {
public:
    bool load(std::memory_order order = std::memory_order_seq_cst) const {
        return active_.load(order);
    }

    void store(bool value, std::memory_order order = std::memory_order_seq_cst) {
        if (!value) {
            reset();
            return;
        }
        active_.store(true, order);
        cv_.notify_all();
    }

    DuplexBreakCoordinator & operator=(bool value) {
        store(value);
        return *this;
    }

    uint64_t request(bool require_llm, bool require_tts, bool require_t2w) {
        std::lock_guard<std::mutex> lock(mtx_);
        ++generation_;
        llm_ack_ = require_llm ? 0 : generation_;
        tts_ack_ = require_tts ? 0 : generation_;
        t2w_ack_ = require_t2w ? 0 : generation_;
        active_.store(true, std::memory_order_release);
        resolve_if_fully_acked_locked();
        cv_.notify_all();
        return generation_;
    }

    uint64_t generation() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return generation_;
    }

    bool acknowledge(DuplexBreakStage stage, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (generation == 0 || generation != generation_) {
            return false;
        }
        uint64_t * ack = nullptr;
        switch (stage) {
            case DuplexBreakStage::LLM: ack = &llm_ack_; break;
            case DuplexBreakStage::TTS: ack = &tts_ack_; break;
            case DuplexBreakStage::T2W: ack = &t2w_ack_; break;
        }
        *ack = generation;
        resolve_if_fully_acked_locked();
        cv_.notify_all();
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        llm_ack_ = generation_;
        tts_ack_ = generation_;
        t2w_ack_ = generation_;
        active_.store(false, std::memory_order_release);
        cv_.notify_all();
    }

    bool wait_until_inactive(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx_);
        return cv_.wait_for(lock, timeout, [&]() {
            return !active_.load(std::memory_order_acquire);
        });
    }

private:
    void resolve_if_fully_acked_locked() {
        if (llm_ack_ == generation_ && tts_ack_ == generation_
            && t2w_ack_ == generation_) {
            active_.store(false, std::memory_order_release);
        }
    }

    std::atomic<bool> active_{false};
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    uint64_t generation_ = 0;
    uint64_t llm_ack_ = 0;
    uint64_t tts_ack_ = 0;
    uint64_t t2w_ack_ = 0;
};

class DuplexTerminalTracker {
public:
    uint64_t begin() {
        std::lock_guard<std::mutex> lock(mtx_);
        const uint64_t seq = next_seq_++;
        states_.emplace(seq, DuplexTerminalStatus::ENQUEUED);
        latest_seq_ = seq;
        cv_.notify_all();
        return seq;
    }

    bool mark_completed(uint64_t seq) {
        return mark_terminal(seq, DuplexTerminalStatus::COMPLETED);
    }

    bool mark_canceled(uint64_t seq) {
        return mark_terminal(seq, DuplexTerminalStatus::CANCELED);
    }

    bool mark_failed(uint64_t seq) {
        return mark_terminal(seq, DuplexTerminalStatus::FAILED);
    }

    void cancel_all_pending() {
        resolve_all_pending(DuplexTerminalStatus::CANCELED);
    }

    void fail_all_pending() {
        resolve_all_pending(DuplexTerminalStatus::FAILED);
    }

    DuplexTerminalStatus status(uint64_t seq) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return status_locked(seq);
    }

    uint64_t latest_sequence() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return latest_seq_;
    }

    bool has_pending() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return has_pending_locked();
    }

    bool has_status(DuplexTerminalStatus status) const {
        std::lock_guard<std::mutex> lock(mtx_);
        for (const auto & entry : states_) {
            if (entry.second == status) {
                return true;
            }
        }
        return false;
    }

    bool start_session() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (const auto & entry : states_) {
            if (entry.second == DuplexTerminalStatus::ENQUEUED) {
                return false;
            }
        }
        latest_seq_ = 0;
        states_.clear();
        cv_.notify_all();
        return true;
    }

    template <typename StopPredicate>
    bool wait_until_quiescent(
            std::chrono::milliseconds timeout,
            StopPredicate stop_predicate) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait_for(lock, timeout, [&]() {
            return !has_pending_locked() || stop_predicate();
        });
        return !has_pending_locked();
    }

    template <typename StopPredicate>
    DuplexTerminalStatus wait_until_resolved(
            uint64_t seq,
            std::chrono::milliseconds timeout,
            StopPredicate stop_predicate) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait_for(lock, timeout, [&]() {
            return status_locked(seq) != DuplexTerminalStatus::ENQUEUED
                || stop_predicate();
        });
        return status_locked(seq);
    }

    void notify_waiters() {
        cv_.notify_all();
    }

private:
    static bool is_final(DuplexTerminalStatus status) {
        return status == DuplexTerminalStatus::COMPLETED
            || status == DuplexTerminalStatus::CANCELED
            || status == DuplexTerminalStatus::FAILED;
    }

    DuplexTerminalStatus status_locked(uint64_t seq) const {
        const auto it = states_.find(seq);
        return it == states_.end() ? DuplexTerminalStatus::UNKNOWN : it->second;
    }

    bool mark_terminal(uint64_t seq, DuplexTerminalStatus status) {
        if (seq == 0 || !is_final(status)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        const auto it = states_.find(seq);
        if (it == states_.end() || it->second != DuplexTerminalStatus::ENQUEUED) {
            return false;
        }
        it->second = status;
        cv_.notify_all();
        return true;
    }

    void resolve_all_pending(DuplexTerminalStatus status) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto & entry : states_) {
            if (entry.second == DuplexTerminalStatus::ENQUEUED) {
                entry.second = status;
            }
        }
        cv_.notify_all();
    }

    bool has_pending_locked() const {
        for (const auto & entry : states_) {
            if (entry.second == DuplexTerminalStatus::ENQUEUED) {
                return true;
            }
        }
        return false;
    }

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    uint64_t next_seq_ = 1;
    uint64_t latest_seq_ = 0;
    std::map<uint64_t, DuplexTerminalStatus> states_;
};
