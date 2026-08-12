#pragma once

#include <atomic>
#include <cstdint>

class ggml_cann_graph_bypass {
public:
    void set_disabled(bool disable) noexcept {
        if (disable) {
            depth_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        uint32_t depth = depth_.load(std::memory_order_relaxed);
        while (depth != 0 &&
               !depth_.compare_exchange_weak(
                   depth, depth - 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    bool disabled() const noexcept {
        return depth_.load(std::memory_order_relaxed) != 0;
    }

private:
    std::atomic<uint32_t> depth_{0};
};
