#include "omni-worker-coordinator.h"

#include "omni-impl.h"
#include "omni-runtime-messages.h"
#include "omni.h"

void print_with_timestamp(const char * format, ...);

void omni_clear_tts_queue(struct omni_context * ctx_omni, const char * log_reason) {
    if (ctx_omni == nullptr || ctx_omni->tts_thread_info == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> tts_lock(ctx_omni->tts_thread_info->mtx);
    auto & tts_queue = ctx_omni->tts_thread_info->queue;
    while (!tts_queue.empty()) {
        LLMOut * old_out = tts_queue.front();
        tts_queue.pop();
        delete old_out;
    }

    if (log_reason != nullptr) {
        print_with_timestamp("%s\n", log_reason);
    }
}

static void omni_start_tts_worker_if_needed(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns, const char * prefix) {
    if (!ctx_omni->use_tts || ctx_omni->tts_thread.joinable()) {
        return;
    }

    ctx_omni->workers.tts_thread_running = true;
    if (ctx_omni->duplex_mode) {
        ctx_omni->tts_thread = std::thread(worker_fns.tts_duplex, ctx_omni, ctx_omni->params);
        print_with_timestamp("%screate tts thread (duplex mode)%s\n", prefix, prefix[0] == '\0' ? " success" : "");
    } else {
        ctx_omni->tts_thread = std::thread(worker_fns.tts_simplex, ctx_omni, ctx_omni->params);
        print_with_timestamp("%screate tts thread (simplex mode)%s\n", prefix, prefix[0] == '\0' ? " success" : "");
    }
}

static void omni_start_t2w_worker_if_needed(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns, const char * prefix) {
    if (!ctx_omni->use_tts || ctx_omni->t2w_thread_info == nullptr || ctx_omni->t2w_thread.joinable()) {
        return;
    }

    ctx_omni->workers.t2w_thread_running = true;
    ctx_omni->t2w_thread = std::thread(worker_fns.t2w, ctx_omni, ctx_omni->params);
    print_with_timestamp("%screate t2w thread%s\n", prefix, prefix[0] == '\0' ? " success" : "");
}

void omni_ensure_prefill_workers_started(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns) {
    if (ctx_omni == nullptr || !ctx_omni->async) {
        return;
    }

    print_with_timestamp("create llm & tts thread\n");
    if (!ctx_omni->llm_thread.joinable()) {
        ctx_omni->workers.llm_thread_running = true;
        ctx_omni->llm_thread = std::thread(worker_fns.llm, ctx_omni, ctx_omni->params);
        print_with_timestamp("create llm thread success\n");
    }

    omni_start_tts_worker_if_needed(ctx_omni, worker_fns, "");
    omni_start_t2w_worker_if_needed(ctx_omni, worker_fns, "");
}

void omni_ensure_decode_workers_started(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns) {
    if (ctx_omni == nullptr || !ctx_omni->async) {
        return;
    }

    omni_start_tts_worker_if_needed(ctx_omni, worker_fns, "stream_decode: ");
    omni_start_t2w_worker_if_needed(ctx_omni, worker_fns, "stream_decode: ");
}

void omni_request_prefill(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return;
    }

    ctx_omni->need_speek = true;
    ctx_omni->llm_thread_info->cv.notify_all();
}

void omni_wait_for_prefill_completion(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return;
    }

    std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    ctx_omni->workers.decode_cv.wait(lock, [&] { return ctx_omni->workers.prefill_done; });
}

void omni_reset_prefill_completion(struct omni_context * ctx_omni) {
    if (ctx_omni != nullptr) {
        ctx_omni->workers.prefill_done = false;
    }
}

void omni_mark_prefill_started(struct omni_context * ctx_omni) {
    if (ctx_omni != nullptr) {
        ctx_omni->workers.prefill_done = false;
    }
}

void omni_mark_prefill_completed(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->workers.prefill_done = true;
    ctx_omni->workers.decode_cv.notify_all();
}

void omni_request_worker_shutdown(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->workers.llm_thread_running = false;
    ctx_omni->workers.tts_thread_running = false;
    ctx_omni->workers.t2w_thread_running = false;

    if (ctx_omni->llm_thread_info != nullptr) {
        ctx_omni->llm_thread_info->cv.notify_all();
    }
    if (ctx_omni->tts_thread_info != nullptr) {
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    if (ctx_omni->t2w_thread_info != nullptr) {
        ctx_omni->t2w_thread_info->cv.notify_all();
    }
    ctx_omni->workers.speek_cv.notify_all();
    ctx_omni->workers.decode_cv.notify_all();
}
