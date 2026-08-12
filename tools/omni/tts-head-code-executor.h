#ifndef OMNI_TTS_HEAD_CODE_EXECUTOR_H
#define OMNI_TTS_HEAD_CODE_EXECUTOR_H

#include <condition_variable>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Persistent row-parallel executor for the CosyVoice2 head_code projection.
// Each output row retains the legacy left-to-right float accumulation order;
// only independent output rows are distributed across threads.
class TtsHeadCodeExecutor {
public:
    explicit TtsHeadCodeExecutor(std::size_t total_threads);
    ~TtsHeadCodeExecutor();

    TtsHeadCodeExecutor(const TtsHeadCodeExecutor &) = delete;
    TtsHeadCodeExecutor & operator=(const TtsHeadCodeExecutor &) = delete;

    void compute(
        const float * hidden,
        const float * weights,
        float * logits,
        std::size_t rows,
        std::size_t cols);

    static void compute_scalar(
        const float * hidden,
        const float * weights,
        float * logits,
        std::size_t rows,
        std::size_t cols);

    std::size_t thread_count() const noexcept;

private:
    struct Job {
        const float * hidden = nullptr;
        const float * weights = nullptr;
        float * logits = nullptr;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::fenv_t floating_point_environment = {};
    };

    void worker_loop(std::size_t worker_index);
    static void compute_range(
        const Job & job,
        std::size_t shard_index,
        std::size_t shard_count);

    const std::size_t total_threads_;
    std::vector<std::thread> workers_;
    std::mutex compute_mutex_;
    std::mutex job_mutex_;
    std::condition_variable job_cv_;
    std::condition_variable done_cv_;
    Job job_;
    std::uint64_t generation_ = 0;
    std::size_t remaining_workers_ = 0;
    bool stop_ = false;
};

#endif
