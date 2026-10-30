#include "../token2wav/token2wav-pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>

using omni::flow::MelTask;
using omni::flow::VocoderPipeline;
using namespace std::chrono_literals;

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

static MelTask task(int id, bool final = false) {
    MelTask result;
    result.mel_bct = { (float) id };
    result.is_final = final;
    return result;
}

static void ready(std::future<void> & future) {
    check(future.wait_for(5s) == std::future_status::ready, "worker synchronization timed out");
    future.get();
}

static void wait_pending(std::atomic<size_t> & pending, size_t expected) {
    const auto end = std::chrono::steady_clock::now() + 5s;
    while (pending.load() != expected && std::chrono::steady_clock::now() < end) {
        std::this_thread::yield();
    }
    check(pending.load() == expected, "pending count did not reach expected value");
}

static void test_fifo_and_empty_final() {
    std::atomic<size_t> pending{0};
    std::vector<int> seen;
    int finals = 0;
    VocoderPipeline pipeline(2, pending, [&](const MelTask & item) {
        if (!item.mel_bct.empty()) {
            seen.push_back((int) item.mel_bct[0]);
        }
        finals += item.is_final;
        return true;
    });
    for (int i = 0; i < 128; ++i) {
        check(pipeline.push(task(i)), "FIFO enqueue failed");
    }
    MelTask final;
    final.is_final = true;
    check(pipeline.push(std::move(final)), "empty final enqueue failed");
    check(pipeline.drain(), "FIFO drain failed");
    check(pending.load() == 0 && seen.size() == 128 && finals == 1, "FIFO/final delivery incomplete");
    for (int i = 0; i < 128; ++i) {
        check(seen[i] == i, "audio order changed");
    }
}

static void test_backpressure_and_overlap() {
    std::atomic<size_t> pending{0};
    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto gate = release.get_future().share();
    VocoderPipeline pipeline(1, pending, [&](const MelTask & item) {
        if (item.mel_bct[0] == 0) {
            entered.set_value();
            gate.wait();
        }
        return true;
    });
    check(pipeline.push(task(0)), "initial enqueue failed");
    ready(started);
    // Flow can hand off its next result while Vocoder is still active.
    check(pipeline.push(task(1)), "overlapped handoff failed");
    check(pending.load() == 2, "queue capacity does not include one waiting plus one active task");
    std::promise<void> pushing;
    auto pushing_started = pushing.get_future();
    auto blocked = std::async(std::launch::async, [&] {
        pushing.set_value();
        return pipeline.push(task(2));
    });
    ready(pushing_started);
    check(blocked.wait_for(30ms) == std::future_status::timeout, "bounded queue did not apply backpressure");
    release.set_value();
    check(blocked.wait_for(5s) == std::future_status::ready && blocked.get(), "blocked producer not released");
    check(pipeline.drain() && pending.load() == 0, "backpressure drain failed");
}

static void test_final_drain_waits_for_callback() {
    std::atomic<size_t> pending{0};
    std::promise<void> callback, release;
    auto callback_started = callback.get_future();
    auto gate = release.get_future().share();
    bool delivered = false;
    VocoderPipeline pipeline(2, pending, [&](const MelTask & item) {
        check(item.is_final, "final marker lost");
        callback.set_value();
        gate.wait();
        delivered = true;
        return true;
    });
    check(pipeline.push(task(0, true)), "final enqueue failed");
    ready(callback_started);
    auto drained = std::async(std::launch::async, [&] { return pipeline.drain(); });
    check(drained.wait_for(30ms) == std::future_status::timeout, "drain finished before output callback");
    release.set_value();
    check(drained.wait_for(5s) == std::future_status::ready && drained.get(), "final drain timed out");
    check(delivered && pending.load() == 0, "final callback not complete after drain");
}

static void test_cancel_and_reuse() {
    std::atomic<size_t> pending{0};
    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto gate = release.get_future().share();
    std::vector<int> seen;
    VocoderPipeline pipeline(2, pending, [&](const MelTask & item) {
        const int id = (int) item.mel_bct[0];
        if (id == 0) {
            entered.set_value();
            gate.wait();
        }
        seen.push_back(id);
        return true;
    });
    check(pipeline.push(task(0)), "active enqueue failed");
    ready(started);
    check(pipeline.push(task(1)) && pipeline.push(task(2)), "queued enqueue failed");
    auto cancelled = std::async(std::launch::async, [&] { return pipeline.cancel(); });
    wait_pending(pending, 1);
    check(cancelled.wait_for(30ms) == std::future_status::timeout, "cancel did not wait for active callback");
    release.set_value();
    check(cancelled.wait_for(5s) == std::future_status::ready && cancelled.get(), "cancel timed out");
    check(seen == std::vector<int>{0}, "cancel delivered queued old-turn output");
    check(pipeline.push(task(3, true)) && pipeline.drain(), "new turn after cancel failed");
    check(seen == std::vector<int>({0, 3}), "turn state crossed cancellation barrier");
}

static void test_error_releases_waiters(bool throws) {
    std::atomic<size_t> pending{0};
    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto gate = release.get_future().share();
    VocoderPipeline pipeline(1, pending, [&](const MelTask &) -> bool {
        entered.set_value();
        gate.wait();
        if (throws) {
            throw std::runtime_error("inference failure");
        }
        return false;
    });
    check(pipeline.push(task(0)), "error task enqueue failed");
    ready(started);
    check(pipeline.push(task(1)), "pending task enqueue failed");
    auto blocked = std::async(std::launch::async, [&] { return pipeline.push(task(2)); });
    release.set_value();
    check(blocked.wait_for(5s) == std::future_status::ready && !blocked.get(), "failure did not release producer");
    check(!pipeline.drain() && !pipeline.cancel() && pending.load() == 0, "failure not reported or pending leaked");
}

static void test_shutdown_drains() {
    std::atomic<size_t> pending{0};
    int delivered = 0;
    {
        VocoderPipeline pipeline(2, pending, [&](const MelTask &) { ++delivered; return true; });
        for (int i = 0; i < 64; ++i) {
            check(pipeline.push(task(i)), "shutdown enqueue failed");
        }
        // Destructor joins before the session and captured output state expire.
    }
    check(delivered == 64 && pending.load() == 0, "shutdown lost accepted tasks");
}

int main() {
    test_fifo_and_empty_final();
    test_backpressure_and_overlap();
    test_final_drain_waits_for_callback();
    test_cancel_and_reuse();
    test_error_releases_waiters(false);
    test_error_releases_waiters(true);
    test_shutdown_drains();
    std::puts("T2W pipeline: 7 regression scenarios passed");
}
