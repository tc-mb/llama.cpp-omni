#pragma once

#include "llama.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

struct LLMOut {
    std::string text;
    int n_past = 0;
    bool llm_finish = false;
    std::string debug_dir;
    std::vector<llama_token> token_ids;
    std::vector<float> hidden_states;
    int n_embd = 0;
    bool is_end_of_turn = false;
    int duplex_chunk_idx = -1;
};

struct TTSThreadInfo {
    const int MAX_QUEUE_SIZE;
    std::queue<LLMOut *> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    int n_past = 0;

    explicit TTSThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};
