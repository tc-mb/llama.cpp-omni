#include "omni.h"

#undef NDEBUG
#include <cassert>

int main() {
    assert(omni_tts_native_frontend_failure_is_fatal(true));
    assert(!omni_tts_native_frontend_failure_is_fatal(false));
    return 0;
}
