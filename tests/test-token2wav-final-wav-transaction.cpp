#include "token2wav-final-wav-transaction.h"

// This state-machine test must remain effective in Release builds, where the
// project otherwise defines NDEBUG globally.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>

using omni::flow::final_wav_key;
using omni::flow::final_wav_transaction_state;
using omni::flow::final_wav_admission_state;

int main() {
    using phase = final_wav_transaction_state::phase;
    constexpr final_wav_key a{7, 11};
    constexpr final_wav_key b{7, 12};

    final_wav_transaction_state tx(2);
    assert(tx.note_committed_call());
    assert(tx.begin(a));
    assert(tx.busy_or_tentative());
    assert(!tx.commit(a));
    assert(tx.rendered(a, true));
    assert(tx.has_tentative_key(a));
    assert(tx.can_commit(a, 1));
    assert(!tx.can_commit(b, 1));
    assert(!tx.has_active_key(b));
    assert(!tx.commit(b));
    assert(tx.abort(a, true));
    assert(tx.current_phase() == phase::idle);
    assert(tx.committed_calls() == 1);

    assert(tx.begin(a));
    assert(tx.rendered(a, true));
    assert(tx.commit(a, 1));
    assert(tx.current_phase() == phase::disabled);
    assert(tx.committed_calls() == 2);

    assert(!tx.note_committed_call());
    assert(tx.current_phase() == phase::disabled);

    final_wav_transaction_state exact_cap(2);
    assert(exact_cap.note_committed_call());
    assert(exact_cap.begin(a));
    assert(exact_cap.rendered(a, true));
    assert(exact_cap.commit(a, 1));
    assert(exact_cap.current_phase() == phase::disabled);
    assert(exact_cap.committed_calls() == 2);
    assert(!exact_cap.begin(b));

    final_wav_transaction_state miss;
    assert(miss.begin(a));
    assert(miss.rendered(a, true));
    assert(!miss.abort(a, false));
    assert(miss.current_phase() == phase::damaged);
    assert(miss.needs_recovery());
    assert(miss.has_active_key(a));
    assert(!miss.has_active_key(b));
    miss.break_reset(false);
    assert(miss.current_phase() == phase::damaged);
    assert(miss.needs_recovery());
    miss.discard_reset();
    assert(!miss.needs_recovery());

    final_wav_transaction_state render_failure;
    assert(render_failure.begin(a));
    assert(!render_failure.rendered(a, false));
    assert(render_failure.current_phase() == phase::damaged);
    assert(render_failure.needs_recovery());
    assert(render_failure.abort(a, true));
    assert(render_failure.current_phase() == phase::idle);

    final_wav_transaction_state interrupted;
    assert(interrupted.begin(a));
    interrupted.break_reset(true);
    assert(interrupted.current_phase() == phase::idle);
    assert(!interrupted.busy_or_tentative());

    final_wav_transaction_state stopped;
    assert(stopped.note_committed_call());
    assert(stopped.begin(a));
    assert(stopped.rendered(a, true));
    stopped.discard_reset();
    assert(stopped.current_phase() == phase::idle);
    assert(stopped.committed_calls() == 0);
    assert(!stopped.busy_or_tentative());

    final_wav_admission_state admission;
    assert(admission.try_admit_encoder());
    assert(!admission.try_begin_render());
    admission.release_encoder();
    assert(admission.try_begin_render());
    assert(!admission.try_admit_encoder());
    admission.end_render();
    assert(admission.idle());

    // Host concurrency proof: the reservation remains held for the entire
    // tentative transaction, so an encoder cannot register before rollback.
    std::mutex admission_mtx;
    std::condition_variable admission_cv;
    bool encoder_admitted = false;
    final_wav_transaction_state lease_tx;
    {
        std::lock_guard<std::mutex> lock(admission_mtx);
        assert(admission.try_begin_render());
        assert(lease_tx.begin(a));
        assert(lease_tx.rendered(a, true));
    }
    std::thread encoder([&] {
        std::unique_lock<std::mutex> lock(admission_mtx);
        admission_cv.wait(lock, [&] { return !admission.rendering(); });
        encoder_admitted = admission.try_admit_encoder();
    });
    {
        std::lock_guard<std::mutex> lock(admission_mtx);
        assert(!encoder_admitted);
        assert(lease_tx.abort(a, true));
        admission.end_render();
    }
    admission_cv.notify_all();
    encoder.join();
    assert(encoder_admitted);
    admission.release_encoder();
    assert(admission.idle());
}
