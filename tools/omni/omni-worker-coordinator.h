#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

// Worker thread synchronization state shared across the LLM / TTS / T2W pipeline.
// Owns the running-flags that gate each worker loop, the prefill-done condition
// variable used to hand off between prefill and decode, and the speek (speech-ready)
// condition variable used by the LLM decode thread to signal readiness.
struct OmniWorkerState {
    std::condition_variable decode_cv;
    bool                    prefill_done = true;

    std::mutex              speek_mtx;
    std::condition_variable speek_cv;
    bool                    last_speek_done_flag = false;

    std::atomic<bool> llm_thread_running{ true };
    std::atomic<bool> tts_thread_running{ true };
    std::atomic<bool> t2w_thread_running{ true };

    std::mutex buffer_mutex;
};

struct common_params;
struct omni_context;

using OmniWorkerThreadFn = void (*)(struct omni_context *, struct common_params *);

struct OmniWorkerThreadFns {
    OmniWorkerThreadFn llm         = nullptr;
    OmniWorkerThreadFn tts_simplex = nullptr;
    OmniWorkerThreadFn tts_duplex  = nullptr;
    OmniWorkerThreadFn t2w         = nullptr;
};

void omni_clear_tts_queue(struct omni_context * ctx_omni, const char * log_reason = nullptr);
void omni_ensure_prefill_workers_started(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns);
void omni_ensure_decode_workers_started(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns);
void omni_request_prefill(struct omni_context * ctx_omni);
void omni_wait_for_prefill_completion(struct omni_context * ctx_omni);
void omni_reset_prefill_completion(struct omni_context * ctx_omni);
void omni_mark_prefill_started(struct omni_context * ctx_omni);
void omni_mark_prefill_completed(struct omni_context * ctx_omni);
void omni_request_worker_shutdown(struct omni_context * ctx_omni);
