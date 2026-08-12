#include "tts-head-code-executor.h"

#include <algorithm>
#include <stdexcept>

TtsHeadCodeExecutor::TtsHeadCodeExecutor(std::size_t total_threads)
    : total_threads_(std::max<std::size_t>(1, total_threads)) {
    workers_.reserve(total_threads_ - 1);
    try {
        for (std::size_t index = 0; index + 1 < total_threads_; ++index) {
            workers_.emplace_back(&TtsHeadCodeExecutor::worker_loop, this, index);
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(job_mutex_);
            stop_ = true;
        }
        job_cv_.notify_all();
        for (std::thread & worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

TtsHeadCodeExecutor::~TtsHeadCodeExecutor() {
    std::unique_lock<std::mutex> compute_lock(compute_mutex_);
    {
        std::lock_guard<std::mutex> job_lock(job_mutex_);
        stop_ = true;
    }
    job_cv_.notify_all();
    compute_lock.unlock();
    for (std::thread & worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void TtsHeadCodeExecutor::compute_range(
        const Job & job,
        std::size_t shard_index,
        std::size_t shard_count) {
    const std::size_t begin = job.rows * shard_index / shard_count;
    const std::size_t end = job.rows * (shard_index + 1) / shard_count;
    for (std::size_t row_index = begin; row_index < end; ++row_index) {
        const float * row = job.weights + row_index * job.cols;
        float sum = 0.0f;
        for (std::size_t column = 0; column < job.cols; ++column) {
            sum += job.hidden[column] * row[column];
        }
        job.logits[row_index] = sum;
    }
}

void TtsHeadCodeExecutor::compute_scalar(
        const float * hidden,
        const float * weights,
        float * logits,
        std::size_t rows,
        std::size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    if (hidden == nullptr || weights == nullptr || logits == nullptr) {
        throw std::invalid_argument("head_code projection received a null buffer");
    }
    Job job;
    job.hidden = hidden;
    job.weights = weights;
    job.logits = logits;
    job.rows = rows;
    job.cols = cols;
    compute_range(job, 0, 1);
}

void TtsHeadCodeExecutor::compute(
        const float * hidden,
        const float * weights,
        float * logits,
        std::size_t rows,
        std::size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    if (hidden == nullptr || weights == nullptr || logits == nullptr) {
        throw std::invalid_argument("head_code projection received a null buffer");
    }
    if (total_threads_ == 1) {
        compute_scalar(hidden, weights, logits, rows, cols);
        return;
    }

    std::lock_guard<std::mutex> compute_lock(compute_mutex_);
    {
        std::lock_guard<std::mutex> job_lock(job_mutex_);
        job_.hidden = hidden;
        job_.weights = weights;
        job_.logits = logits;
        job_.rows = rows;
        job_.cols = cols;
        if (std::fegetenv(&job_.floating_point_environment) != 0) {
            throw std::runtime_error("failed to capture head_code floating-point environment");
        }
        remaining_workers_ = workers_.size();
        ++generation_;
    }
    job_cv_.notify_all();

    compute_range(job_, total_threads_ - 1, total_threads_);

    std::unique_lock<std::mutex> job_lock(job_mutex_);
    done_cv_.wait(job_lock, [this] { return remaining_workers_ == 0; });
}

void TtsHeadCodeExecutor::worker_loop(std::size_t worker_index) {
    std::uint64_t seen_generation = 0;
    for (;;) {
        Job job;
        std::uint64_t job_generation = 0;
        {
            std::unique_lock<std::mutex> lock(job_mutex_);
            job_cv_.wait(lock, [this, &seen_generation] {
                return stop_ || generation_ != seen_generation;
            });
            if (stop_) {
                return;
            }
            job = job_;
            job_generation = generation_;
            seen_generation = job_generation;
        }

        std::fenv_t previous_environment;
        const bool restore_environment = std::fegetenv(&previous_environment) == 0;
        std::fesetenv(&job.floating_point_environment);
        compute_range(job, worker_index, total_threads_);
        if (restore_environment) {
            std::fesetenv(&previous_environment);
        }

        {
            std::lock_guard<std::mutex> lock(job_mutex_);
            if (generation_ == job_generation && remaining_workers_ > 0) {
                --remaining_workers_;
                if (remaining_workers_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }
}

std::size_t TtsHeadCodeExecutor::thread_count() const noexcept {
    return total_threads_;
}
