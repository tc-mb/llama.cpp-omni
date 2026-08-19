#include "token2wav-impl.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using omni::flow::flowRunnerNoiseState;

int main() {
    flowRunnerNoiseState first;
    flowRunnerNoiseState second;
    const std::mt19937 initial(42);

    assert(first.capture() == initial);
    assert(second.capture() == initial);

    (void) first.generator()();
    const std::mt19937 first_advanced = first.capture();
    assert(first_advanced != initial);
    assert(second.capture() == initial);

    (void) second.generator()();
    (void) second.generator()();
    const std::mt19937 second_advanced = second.capture();
    assert(second_advanced != first_advanced);

    first.restore(initial);
    assert(first.capture() == initial);
    assert(second.capture() == second_advanced);

    second.reset();
    assert(second.capture() == initial);
}
