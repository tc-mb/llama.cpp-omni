#include "protocol.h"
#include "ws_handler.h"
#include "common/base64.hpp"

#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <limits>
#include <string>

int main() {
    const float finite_sample = 0.25f;
    const std::string finite_audio =
        float32_pcm_to_b64(&finite_sample, 1);
    assert(!finite_audio.empty());
    assert(ws_audio_base64_within_limits(finite_audio));

    assert(ws_audio_base64_within_limits(
        std::string(kWsMaxAudioBase64Chars, 'A')));
    assert(!ws_audio_base64_within_limits(
        std::string(kWsMaxAudioBase64Chars + 1, 'A')));
    assert(!ws_audio_base64_within_limits("!!!!"));
    assert(!ws_audio_base64_within_limits(base64::encode("abc", 3)));

    const float nan_sample = std::numeric_limits<float>::quiet_NaN();
    const float infinity_sample = std::numeric_limits<float>::infinity();
    assert(!ws_audio_base64_within_limits(
        float32_pcm_to_b64(&nan_sample, 1)));
    assert(!ws_audio_base64_within_limits(
        float32_pcm_to_b64(&infinity_sample, 1)));

    const float second_sample = -0.5f;
    const std::string second_audio =
        float32_pcm_to_b64(&second_sample, 1);
    const auto parsed_init = parse_session_init({
        {"type", "session.init"},
        {"payload", {
            {"mode", "turn_based"},
            {"voice", {
                {"ref_audio", "data:audio/wav;base64," + finite_audio},
                {"tts_ref_audio", "data:audio/wav;base64," + second_audio},
            }},
        }},
    });
    assert(parsed_init.ok);
    assert(parsed_init.ref_audio_b64 == finite_audio);
    assert(parsed_init.tts_ref_audio_b64 == second_audio);

    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "omni_audio_limits_test";
    std::filesystem::create_directories(temp_dir);
    const std::string rejected = TempMediaFiles::write_audio_wav(
        std::string(kWsMaxAudioBase64Chars + 1, 'A'), temp_dir.string(), 1);
    assert(rejected.empty());

    std::filesystem::remove_all(temp_dir);
    return 0;
}
