#pragma once

#include <mutex>
#include <shared_mutex>

class ggml_cann_graph_device_gate {
public:
    std::shared_lock<std::shared_mutex> lock_replay() {
        return std::shared_lock<std::shared_mutex>(mutex_);
    }

    std::unique_lock<std::shared_mutex> lock_exclusive() {
        return std::unique_lock<std::shared_mutex>(mutex_);
    }

private:
    std::shared_mutex mutex_;
};

class ggml_cann_graph_context_gate {
public:
    std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(mutex_);
    }

private:
    std::mutex mutex_;
};

class ggml_cann_graph_transaction {
public:
    ggml_cann_graph_transaction(
            ggml_cann_graph_context_gate & context_gate,
            ggml_cann_graph_device_gate & device_gate) :
        context_guard_(context_gate.lock()),
        device_gate_(&device_gate),
        replay_guard_(device_gate.lock_replay()) {}

    void upgrade_to_exclusive() {
        replay_guard_.unlock();
        exclusive_guard_ = device_gate_->lock_exclusive();
    }

private:
    std::unique_lock<std::mutex> context_guard_;
    ggml_cann_graph_device_gate * device_gate_;
    std::shared_lock<std::shared_mutex> replay_guard_;
    std::unique_lock<std::shared_mutex> exclusive_guard_;
};
