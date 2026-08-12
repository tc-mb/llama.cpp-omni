#include "duplex-terminal-tracker.h"

#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

int main() {
    DuplexSpeakSegmentTracker speak;
    assert(speak.active() == -1);
    int speak_id = -1;
    assert(speak.begin(speak_id));
    assert(speak_id == 0);
    int same_speak_id = -1;
    assert(!speak.begin(same_speak_id));
    assert(same_speak_id == speak_id);
    assert(speak.end(speak_id));
    assert(!speak.end(speak_id));

    int second_speak_id = -1;
    assert(speak.begin(second_speak_id));
    assert(second_speak_id == 1);
    assert(speak.cancel() == second_speak_id);
    speak.start_session();
    int reused_session_id = -1;
    assert(speak.begin(reused_session_id));
    assert(reused_session_id == 2);
    assert(speak.end(reused_session_id));

    DuplexTerminalTracker tracker;
    assert(!tracker.has_pending());
    assert(tracker.latest_sequence() == 0);

    const uint64_t first = tracker.begin();
    const uint64_t second = tracker.begin();
    assert(first == 1);
    assert(second == 2);
    assert(tracker.latest_sequence() == second);
    assert(tracker.has_pending());

    assert(tracker.mark_completed(first));
    assert(!tracker.mark_completed(first));
    assert(tracker.status(first) == DuplexTerminalStatus::COMPLETED);

    // A later cancellation must not overwrite the earlier sequence's result.
    assert(tracker.mark_canceled(second));
    assert(tracker.status(first) == DuplexTerminalStatus::COMPLETED);
    assert(tracker.status(second) == DuplexTerminalStatus::CANCELED);
    assert(!tracker.has_pending());

    assert(tracker.start_session());
    const uint64_t async_seq = tracker.begin();
    assert(async_seq == 3);
    std::thread resolver([&]() {
        std::this_thread::sleep_for(10ms);
        assert(tracker.mark_completed(async_seq));
    });
    const auto completed = tracker.wait_until_resolved(async_seq, 1s, []() {
        return false;
    });
    resolver.join();
    assert(completed == DuplexTerminalStatus::COMPLETED);

    assert(tracker.start_session());
    const uint64_t stopped_seq = tracker.begin();
    assert(stopped_seq == 4);
    std::atomic<bool> stopped{false};
    std::thread stopper([&]() {
        std::this_thread::sleep_for(10ms);
        stopped.store(true);
        tracker.notify_waiters();
    });
    const auto still_pending = tracker.wait_until_resolved(stopped_seq, 1s, [&]() {
        return stopped.load();
    });
    stopper.join();
    assert(still_pending == DuplexTerminalStatus::ENQUEUED);
    assert(tracker.mark_failed(stopped_seq));

    assert(tracker.start_session());
    const uint64_t pending_a = tracker.begin();
    const uint64_t pending_b = tracker.begin();
    assert(!tracker.start_session());
    tracker.cancel_all_pending();
    assert(tracker.status(pending_a) == DuplexTerminalStatus::CANCELED);
    assert(tracker.status(pending_b) == DuplexTerminalStatus::CANCELED);
    assert(!tracker.has_pending());
    assert(tracker.start_session());

    const uint64_t quiescent_seq = tracker.begin();
    std::thread quiescer([&]() {
        std::this_thread::sleep_for(10ms);
        assert(tracker.mark_completed(quiescent_seq));
    });
    assert(tracker.wait_until_quiescent(1s, []() { return false; }));
    quiescer.join();

    DuplexBreakCoordinator break_state;
    assert(!break_state.load());
    const uint64_t break_1 = break_state.request(true, true, true);
    assert(break_1 == 1);
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::T2W, break_1));
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::LLM, break_1));
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::TTS, break_1));
    assert(!break_state.load());

    // A stale worker acknowledgement must not resolve a newer request.
    const uint64_t break_2 = break_state.request(true, true, true);
    const uint64_t break_3 = break_state.request(true, true, true);
    assert(break_2 == 2);
    assert(break_3 == 3);
    assert(!break_state.acknowledge(DuplexBreakStage::LLM, break_2));
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::LLM, break_3));
    assert(break_state.acknowledge(DuplexBreakStage::TTS, break_3));
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::T2W, break_3));
    assert(!break_state.load());

    // Missing stages are pre-acknowledged at request time.
    const uint64_t break_4 = break_state.request(false, true, false);
    assert(break_state.load());
    assert(break_state.acknowledge(DuplexBreakStage::TTS, break_4));
    assert(break_state.wait_until_inactive(10ms));

    const uint64_t break_5 = break_state.request(true, true, true);
    assert(break_state.load());
    break_state.reset();
    assert(!break_state.load());
    assert(!break_state.acknowledge(DuplexBreakStage::LLM, break_5 + 1));

    return 0;
}
