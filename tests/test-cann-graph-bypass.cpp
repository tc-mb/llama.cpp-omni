#include "graph-bypass.h"

#include <atomic>
#include <thread>

static void wait_until(const std::atomic<int> & value, int expected) {
    while (value.load(std::memory_order_acquire) != expected) {
        std::this_thread::yield();
    }
}

int main() {
    ggml_cann_graph_bypass bypass;

    if (bypass.disabled()) {
        return 1;
    }
    bypass.set_disabled(true);
    bypass.set_disabled(true);
    if (!bypass.disabled()) {
        return 2;
    }
    bypass.set_disabled(false);
    if (!bypass.disabled()) {
        return 3;
    }
    bypass.set_disabled(false);
    if (bypass.disabled()) {
        return 4;
    }

    // An unmatched release must not underflow and permanently disable Graph.
    bypass.set_disabled(false);
    if (bypass.disabled()) {
        return 5;
    }
    bypass.set_disabled(true);
    bypass.set_disabled(false);
    if (bypass.disabled()) {
        return 6;
    }

    std::atomic<int>  entered{0};
    std::atomic<bool> release_first{false};
    std::atomic<bool> release_second{false};

    std::thread first([&]() {
        bypass.set_disabled(true);
        entered.fetch_add(1, std::memory_order_release);
        while (!release_first.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        bypass.set_disabled(false);
    });
    std::thread second([&]() {
        bypass.set_disabled(true);
        entered.fetch_add(1, std::memory_order_release);
        while (!release_second.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        bypass.set_disabled(false);
    });

    wait_until(entered, 2);
    int concurrent_error = 0;
    if (!bypass.disabled()) {
        concurrent_error = 7;
    }
    release_first.store(true, std::memory_order_release);
    first.join();
    if (!bypass.disabled() && concurrent_error == 0) {
        concurrent_error = 8;
    }
    release_second.store(true, std::memory_order_release);
    second.join();
    if (bypass.disabled() && concurrent_error == 0) {
        concurrent_error = 9;
    }

    return concurrent_error;
}
