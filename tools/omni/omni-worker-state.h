#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

struct OmniWorkerState {
    std::condition_variable decode_cv;
    bool prefill_done = true;

    std::mutex speek_mtx;
    std::condition_variable speek_cv;
    bool last_speek_done_flag = false;

    std::atomic<bool> llm_thread_running{true};
    std::atomic<bool> tts_thread_running{true};
    std::atomic<bool> t2w_thread_running{true};

    std::mutex buffer_mutex;
};
