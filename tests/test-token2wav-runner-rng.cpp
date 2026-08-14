#include "token2wav-impl.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using omni::flow::flowGGUFModelRunner;

int main() {
    flowGGUFModelRunner runner_a;
    flowGGUFModelRunner runner_b;
    std::mt19937 initial(42);

    // Every session starts from the legacy deterministic seed.
    assert(runner_a.capture_noise_state() == initial);
    assert(runner_a.capture_noise_state() == runner_b.capture_noise_state());

    // Restore is local to the owning production runner. The old process-static
    // implementation made restoring A change B as well.
    std::mt19937 advanced_a = initial;
    (void) advanced_a();
    runner_a.restore_noise_state(advanced_a);
    assert(runner_a.capture_noise_state() == advanced_a);
    assert(runner_b.capture_noise_state() == initial);

    std::mt19937 advanced_b = initial;
    (void) advanced_b();
    (void) advanced_b();
    runner_b.restore_noise_state(advanced_b);
    assert(runner_b.capture_noise_state() == advanced_b);
    assert(runner_a.capture_noise_state() == advanced_a);

    runner_a.restore_noise_state(initial);
    assert(runner_a.capture_noise_state() == initial);
    assert(runner_b.capture_noise_state() == advanced_b);
}
