#include "ws_handler.h"

#undef NDEBUG
#include <cassert>
#include <functional>
#include <mutex>
#include <string>

static int release_count = 0;

static void record_release(omni_context *) {
    ++release_count;
}

int main() {
    assert(!should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/voice.wav", true));
    assert(should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/tts-voice.wav", true));
    assert(should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/voice.wav", false));
    assert(!should_remove_prefill_audio("", "", false));
    assert(!should_materialize_tts_ref_audio(false, "encoded-audio"));
    assert(should_materialize_tts_ref_audio(true, "encoded-audio"));
    assert(!should_materialize_tts_ref_audio(true, ""));

    const std::string default_ref_audio = omni_default_ref_audio_path();
    assert(std::filesystem::path(default_ref_audio).is_absolute());
    assert(default_ref_audio.find("default_ref_audio/default_ref_audio.wav") != std::string::npos);

    std::mutex octx_mutex;
    omni_context * shared_octx = nullptr;
    omni_context * pending_octx = reinterpret_cast<omni_context *>(0x1);
    omni_context * other_octx = reinterpret_cast<omni_context *>(0x2);

    assert(!release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, false, record_release));

    shared_octx = pending_octx;
    assert(release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, false, record_release));
    assert(shared_octx == nullptr);
    assert(release_count == 1);

    shared_octx = pending_octx;
    assert(!release_unactivated_octx_if_owned(
        other_octx, shared_octx, octx_mutex, false, record_release));
    assert(shared_octx == pending_octx);
    assert(release_count == 1);

    assert(!release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, true, record_release));
    assert(shared_octx == pending_octx);
    assert(release_count == 1);

    shared_octx = pending_octx;
    assert(release_failed_shared_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, record_release));
    assert(shared_octx == nullptr);
    assert(release_count == 2);
    return 0;
}
