#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace omni {
namespace flow {

struct MelTask {
    std::vector<float> mel_bct;
    bool is_final = false;
    double flow_ms = 0.0;
    double queue_wait_ms = 0.0;
    std::string output_dir;
    int turn_base = 0;
};

// One Flow producer and one Vocoder consumer. All producer-side operations
// (push/drain/cancel) are serialized by the T2W thread. The consumer must not
// call these operations. A failed consumer closes the queue and wakes waiters.
class VocoderPipeline {
  public:
    using consumer = std::function<bool(const MelTask &)>;

    VocoderPipeline(size_t capacity, std::atomic<size_t> & pending, consumer consume) :
        capacity_(capacity > 0 ? capacity : 1), pending_(pending), consume_(std::move(consume)) {
        pending_.store(0);
        worker_ = std::thread([this] { run(); });
    }

    ~VocoderPipeline() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
        worker_.join(); // Finish accepted tasks before the session can be freed.
    }

    VocoderPipeline(const VocoderPipeline &) = delete;
    VocoderPipeline & operator=(const VocoderPipeline &) = delete;

    bool push(MelTask task) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        queue_.push(std::move(task));
        publish_pending();
        cv_.notify_all();
        return true;
    }

    bool drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return queue_.empty() && !active_; });
        return !failed_;
    }

    // Drop queued mel and wait for the active task, including its output
    // callback, before the producer changes turn state or reuses the session.
    bool cancel() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        publish_pending();
        cv_.notify_all();
        cv_.wait(lock, [&] { return !active_; });
        return !failed_;
    }

  private:
    void publish_pending() {
        pending_.store(queue_.size() + (active_ ? 1 : 0), std::memory_order_release);
    }

    void run() {
        for (;;) {
            MelTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
                if (queue_.empty()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop();
                active_ = true;
                publish_pending();
            }
            cv_.notify_all();
            bool ok = false;
            try {
                ok = consume_(task);
            } catch (...) {
                // Never allow an inference/output exception to terminate a
                // joinable worker or strand a producer blocked on capacity.
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
                if (!ok) {
                    failed_ = true;
                    closed_ = true;
                    while (!queue_.empty()) {
                        queue_.pop();
                    }
                }
                publish_pending();
            }
            cv_.notify_all();
        }
    }

    const size_t capacity_;
    std::atomic<size_t> & pending_;
    consumer consume_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<MelTask> queue_;
    bool active_ = false;
    bool closed_ = false;
    bool failed_ = false;
    std::thread worker_;
};

} // namespace flow
} // namespace omni
