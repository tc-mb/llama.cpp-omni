#include "token2wav-frontend.h"
#include "token2wav-frontend-gguf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

namespace omni {
namespace flow {
namespace {

constexpr float  kPi = 3.14159265358979323846f;
constexpr size_t kMaxResampleKernelElements = 4ULL * 1024ULL * 1024ULL;

static float hz_to_mel_slaney(float hz) {
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    constexpr float logstep = 0.06875177742094912f;  // log(6.4) / 27
    if (hz < min_log_hz) {
        return hz * (3.0f / 200.0f);
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

static float mel_to_hz_slaney(float mel) {
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    constexpr float logstep = 0.06875177742094912f;
    if (mel < min_log_mel) {
        return mel * (200.0f / 3.0f);
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

static std::vector<float> make_librosa_mel_filterbank(int sample_rate,
                                                       int n_fft,
                                                       int n_mels,
                                                       float fmin,
                                                       float fmax) {
    const int n_freqs = n_fft / 2 + 1;
    std::vector<float> filters(static_cast<size_t>(n_mels) * static_cast<size_t>(n_freqs), 0.0f);

    const float mel_min = hz_to_mel_slaney(fmin);
    const float mel_max = hz_to_mel_slaney(fmax);
    std::vector<float> points(static_cast<size_t>(n_mels + 2));
    for (int i = 0; i < n_mels + 2; ++i) {
        const float mel = mel_min + (mel_max - mel_min) * static_cast<float>(i) /
                                           static_cast<float>(n_mels + 1);
        points[static_cast<size_t>(i)] = mel_to_hz_slaney(mel);
    }

    for (int m = 0; m < n_mels; ++m) {
        const float left   = points[static_cast<size_t>(m)];
        const float center = points[static_cast<size_t>(m + 1)];
        const float right  = points[static_cast<size_t>(m + 2)];
        const int left_i   = std::max(0, static_cast<int>(std::floor(
                                      left * static_cast<float>(n_fft) / static_cast<float>(sample_rate))));
        const int right_i  = std::min(n_freqs - 1, static_cast<int>(std::ceil(
                                       right * static_cast<float>(n_fft) / static_cast<float>(sample_rate))));

        for (int k = left_i; k <= right_i; ++k) {
            const float frequency = static_cast<float>(k) * static_cast<float>(sample_rate) /
                                    static_cast<float>(n_fft);
            const float up = (frequency - left) / (center - left);
            const float down = (right - frequency) / (right - center);
            filters[static_cast<size_t>(m) * n_freqs + k] = std::max(0.0f, std::min(up, down));
        }

        // librosa's default Slaney normalization makes the area of every
        // triangular filter equal to one in the frequency domain.
        const float enorm = 2.0f / std::max(1e-12f, right - left);
        for (int k = left_i; k <= right_i; ++k) {
            filters[static_cast<size_t>(m) * n_freqs + k] *= enorm;
        }
    }
    return filters;
}

struct DftPlan {
    int                n_fft = 0;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};

static DftPlan make_dft_plan(int n_fft) {
    DftPlan plan;
    plan.n_fft = n_fft;
    plan.cos_table.resize(static_cast<size_t>(n_fft / 2 + 1) * static_cast<size_t>(n_fft));
    plan.sin_table.resize(static_cast<size_t>(n_fft / 2 + 1) * static_cast<size_t>(n_fft));
    for (int k = 0; k <= n_fft / 2; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            const float angle = 2.0f * kPi * static_cast<float>(k * n) / static_cast<float>(n_fft);
            plan.cos_table[static_cast<size_t>(k) * n_fft + n] = std::cos(angle);
            plan.sin_table[static_cast<size_t>(k) * n_fft + n] = std::sin(angle);
        }
    }
    return plan;
}

static std::vector<float> dft_power(const std::vector<float> & frame, const DftPlan & plan) {
    const int n_fft   = plan.n_fft;
    const int n_freqs = n_fft / 2 + 1;
    std::vector<float> power(static_cast<size_t>(n_freqs), 0.0f);
    for (int k = 0; k < n_freqs; ++k) {
        float real = 0.0f;
        float imag = 0.0f;
        const size_t base = static_cast<size_t>(k) * n_fft;
        for (int n = 0; n < n_fft; ++n) {
            real += frame[static_cast<size_t>(n)] * plan.cos_table[base + n];
            imag -= frame[static_cast<size_t>(n)] * plan.sin_table[base + n];
        }
        power[static_cast<size_t>(k)] = real * real + imag * imag;
    }
    return power;
}

static std::vector<float> dft_magnitude(const std::vector<float> & frame, const DftPlan & plan) {
    const int n_fft   = plan.n_fft;
    const int n_freqs = n_fft / 2 + 1;
    std::vector<float> magnitude(static_cast<size_t>(n_freqs), 0.0f);
    for (int k = 0; k < n_freqs; ++k) {
        float real = 0.0f;
        float imag = 0.0f;
        const size_t base = static_cast<size_t>(k) * n_fft;
        for (int n = 0; n < n_fft; ++n) {
            real += frame[static_cast<size_t>(n)] * plan.cos_table[base + n];
            imag -= frame[static_cast<size_t>(n)] * plan.sin_table[base + n];
        }
        magnitude[static_cast<size_t>(k)] = std::sqrt(real * real + imag * imag + 1e-9f);
    }
    return magnitude;
}

static float reflect_sample(const std::vector<float> & input, int index) {
    const int n = static_cast<int>(input.size());
    if (n <= 1) {
        return input.empty() ? 0.0f : input[0];
    }
    while (index < 0 || index >= n) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * n - 2 - index;
        }
    }
    return input[static_cast<size_t>(index)];
}

static std::vector<float> hann_window(int size) {
    std::vector<float> window(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(size));
    }
    return window;
}

static std::vector<float> povey_window(int size) {
    std::vector<float> window(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        const float phase = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(size - 1);
        window[static_cast<size_t>(i)] = std::pow(0.5f - 0.5f * std::cos(phase), 0.85f);
    }
    return window;
}

static void set_features(AudioFeatures & out, int channels, int frames, std::vector<float> values) {
    out.channels = channels;
    out.frames   = frames;
    out.values   = std::move(values);
}

static uint16_t read_u16_le(const uint8_t * data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

static uint32_t read_u32_le(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

static int32_t sign_extend_24(uint32_t value) {
    if (value & 0x00800000U) {
        value |= 0xff000000U;
    }
    return static_cast<int32_t>(value);
}

static bool valid_audio(const std::vector<float> & audio) {
    if (audio.empty()) {
        return false;
    }
    for (const float value : audio) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

static bool replicate_pad_features(AudioFeatures & features, int target_frames, std::string * error) {
    if (features.channels <= 0 || features.frames <= 0 ||
        features.values.size() != static_cast<size_t>(features.channels) * features.frames) {
        return set_error(error, "cannot pad invalid audio features");
    }
    if (target_frames < features.frames) {
        return set_error(error,
                         "prompt Mel has more frames than speech token target: target " +
                             std::to_string(target_frames) + ", got " + std::to_string(features.frames));
    }
    if (target_frames == features.frames) {
        return true;
    }

    std::vector<float> padded(static_cast<size_t>(features.channels) * target_frames);
    for (int c = 0; c < features.channels; ++c) {
        const size_t source_base = static_cast<size_t>(c) * features.frames;
        const size_t target_base = static_cast<size_t>(c) * target_frames;
        std::copy(features.values.begin() + source_base,
                  features.values.begin() + source_base + features.frames,
                  padded.begin() + target_base);
        const float last = features.values[source_base + features.frames - 1];
        std::fill(padded.begin() + target_base + features.frames,
                  padded.begin() + target_base + target_frames,
                  last);
    }
    features.frames = target_frames;
    features.values = std::move(padded);
    return true;
}

static bool subtract_feature_column_mean(AudioFeatures & features, std::string * error) {
    if (features.channels <= 0 || features.frames <= 0 ||
        features.values.size() != static_cast<size_t>(features.channels) * features.frames) {
        return set_error(error, "cannot normalize invalid audio features");
    }
    for (int c = 0; c < features.channels; ++c) {
        float mean = 0.0f;
        for (int t = 0; t < features.frames; ++t) {
            mean += features.values[static_cast<size_t>(t) * features.channels + c];
        }
        mean /= static_cast<float>(features.frames);
        for (int t = 0; t < features.frames; ++t) {
            features.values[static_cast<size_t>(t) * features.channels + c] -= mean;
        }
    }
    return true;
}

}  // namespace

bool Token2WavFrontend::validate_speech_tokenizer_output_shape(const std::vector<int64_t> & shape,
                                                               std::string *                  error) {
    const bool valid = (shape.size() == 1 && shape[0] > 0) ||
                       (shape.size() == 2 && shape[0] == 1 && shape[1] > 0);
    if (!valid) {
        return set_error(error, "speech tokenizer output must have shape [T] or [1, T] with B=1");
    }
    return true;
}

bool Token2WavFrontend::validate_campplus_output_shape(const std::vector<int64_t> & shape,
                                                       std::string *                  error) {
    const bool valid = (shape.size() == 1 && shape[0] == 192) ||
                       (shape.size() == 2 && shape[0] == 1 && shape[1] == 192);
    if (!valid) {
        return set_error(error, "CampPlus output must have shape [192] or [1, 192] with B=1");
    }
    return true;
}

bool Token2WavFrontend::load_wav_mono(const std::string & path, AudioBuffer & out) {
    out = AudioBuffer();

    std::error_code file_error;
    const std::filesystem::file_status status = std::filesystem::status(path, file_error);
    if (file_error || !std::filesystem::is_regular_file(status)) {
        return false;
    }
    const uintmax_t file_size = std::filesystem::file_size(path, file_error);
    if (file_error || file_size < 12 || file_size > kMaxReferenceWavBytes ||
        file_size > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return false;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (!bytes.empty() &&
        (!file.read(reinterpret_cast<char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())) ||
         file.gcount() != static_cast<std::streamsize>(bytes.size()))) {
        return false;
    }
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        return false;
    }

    uint16_t format_tag = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    uint16_t block_align = 0;
    size_t data_offset = 0;
    size_t data_size = 0;
    size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const uint8_t * chunk = bytes.data() + offset;
        const uint32_t chunk_size = read_u32_le(chunk + 4);
        const size_t payload = offset + 8;
        if (payload > bytes.size() || chunk_size > bytes.size() - payload) {
            return false;
        }
        if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16) {
            format_tag       = read_u16_le(bytes.data() + payload);
            channels         = read_u16_le(bytes.data() + payload + 2);
            sample_rate      = read_u32_le(bytes.data() + payload + 4);
            block_align      = read_u16_le(bytes.data() + payload + 12);
            bits_per_sample  = read_u16_le(bytes.data() + payload + 14);
            if (format_tag == 0xfffe && chunk_size >= 40) {
                // WAVE_FORMAT_EXTENSIBLE stores the PCM/IEEE subtype at
                // the beginning of the 16-byte subformat GUID.
                const uint16_t subtype = read_u16_le(bytes.data() + payload + 24);
                if (subtype == 1 || subtype == 3) {
                    format_tag = subtype;
                }
            }
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_offset = payload;
            data_size   = chunk_size;
        }
        offset = payload + chunk_size + (chunk_size & 1U);
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    const size_t frame_bytes = static_cast<size_t>(channels) * bytes_per_sample;
    if ((format_tag != 1 && format_tag != 3) || channels == 0 || sample_rate == 0 ||
        bytes_per_sample == 0 || block_align == 0 || frame_bytes > block_align ||
        data_size == 0 || data_offset > bytes.size() ||
        data_size > bytes.size() - data_offset || data_size % block_align != 0) {
        return false;
    }
    if (format_tag == 3 && bits_per_sample != 32) {
        return false;
    }
    if (format_tag == 1 && bits_per_sample != 8 && bits_per_sample != 16 &&
        bits_per_sample != 24 && bits_per_sample != 32) {
        return false;
    }

    const size_t frame_count = data_size / block_align;
    const uint64_t max_frame_count =
        static_cast<uint64_t>(sample_rate) * kMaxReferenceAudioDurationSeconds;
    if (frame_count > max_frame_count) {
        return false;
    }

    out.sample_rate = static_cast<int32_t>(sample_rate);
    out.samples.resize(frame_count);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        const size_t frame_offset = data_offset + frame * block_align;
        float mono = 0.0f;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            const size_t channel_offset = static_cast<size_t>(channel) * bytes_per_sample;
            if (channel_offset > block_align ||
                bytes_per_sample > block_align - channel_offset ||
                frame_offset > bytes.size() ||
                channel_offset > bytes.size() - frame_offset ||
                bytes_per_sample > bytes.size() - frame_offset - channel_offset) {
                return false;
            }
            const size_t sample_offset = frame_offset + channel_offset;
            const uint8_t * sample = bytes.data() + sample_offset;
            float value = 0.0f;
            if (format_tag == 3) {
                std::memcpy(&value, sample, sizeof(float));
            } else if (bits_per_sample == 8) {
                value = (static_cast<int>(sample[0]) - 128) / 128.0f;
            } else if (bits_per_sample == 16) {
                const int16_t integer = static_cast<int16_t>(read_u16_le(sample));
                value = static_cast<float>(integer) / 32768.0f;
            } else if (bits_per_sample == 24) {
                const uint32_t packed = static_cast<uint32_t>(sample[0]) |
                                         (static_cast<uint32_t>(sample[1]) << 8) |
                                         (static_cast<uint32_t>(sample[2]) << 16);
                value = static_cast<float>(sign_extend_24(packed)) / 8388608.0f;
            } else {
                const int32_t integer = static_cast<int32_t>(read_u32_le(sample));
                value = static_cast<float>(static_cast<double>(integer) / 2147483648.0);
            }
            mono += value;
        }
        out.samples[frame] = mono / static_cast<float>(channels);
    }
    return valid_audio(out.samples);
}

bool Token2WavFrontend::resample_mono(const AudioBuffer & input, int32_t target_sample_rate, AudioBuffer & out) {
    out = AudioBuffer();
    if (input.sample_rate <= 0 || target_sample_rate <= 0 || !valid_audio(input.samples)) {
        return false;
    }
    if (input.sample_rate == target_sample_rate) {
        out = input;
        return true;
    }

    const int rate_gcd = std::gcd(input.sample_rate, target_sample_rate);
    const int64_t orig_freq_64 = input.sample_rate / rate_gcd;
    const int64_t new_freq_64 = target_sample_rate / rate_gcd;
    constexpr int lowpass_filter_width = 6;
    constexpr double rolloff = 0.99;
    const double base_freq = static_cast<double>(std::min(orig_freq_64, new_freq_64)) * rolloff;
    const int64_t width_64 = static_cast<int64_t>(std::ceil(
        static_cast<double>(lowpass_filter_width) * static_cast<double>(orig_freq_64) / base_freq));
    const int64_t kernel_width_64 = 2 * width_64 + orig_freq_64;
    if (orig_freq_64 > std::numeric_limits<int>::max() ||
        new_freq_64 > std::numeric_limits<int>::max() ||
        width_64 > std::numeric_limits<int>::max() ||
        kernel_width_64 > std::numeric_limits<int>::max() ||
        static_cast<uint64_t>(new_freq_64) >
            kMaxResampleKernelElements / static_cast<uint64_t>(kernel_width_64)) {
        return false;
    }
    const int orig_freq = static_cast<int>(orig_freq_64);
    const int new_freq = static_cast<int>(new_freq_64);
    const int width = static_cast<int>(width_64);
    const int kernel_width = static_cast<int>(kernel_width_64);
    const int group_count = static_cast<int>(
        (input.samples.size() + static_cast<size_t>(orig_freq) - 1) / static_cast<size_t>(orig_freq));
    const size_t output_size = static_cast<size_t>(
        std::max<int64_t>(
            1,
            (static_cast<int64_t>(input.samples.size()) * new_freq + orig_freq - 1) / orig_freq));

    out.sample_rate = target_sample_rate;
    out.samples.resize(output_size);

    // Match torchaudio.functional._get_sinc_resample_kernel and
    // _apply_sinc_resample_kernel. The kernel is stored as float32 because
    // torchaudio.transforms.Resample caches a float32 kernel by default.
    std::vector<float> kernels(static_cast<size_t>(new_freq) * kernel_width, 0.0f);
    for (int phase = 0; phase < new_freq; ++phase) {
        for (int k = 0; k < kernel_width; ++k) {
            const double idx = static_cast<double>(k - width) / static_cast<double>(orig_freq);
            double t = (idx - static_cast<double>(phase) / static_cast<double>(new_freq)) * base_freq;
            t = std::max(-static_cast<double>(lowpass_filter_width),
                         std::min(static_cast<double>(lowpass_filter_width), t));
            const double window =
                std::cos(t * kPi / static_cast<double>(lowpass_filter_width) / 2.0);
            const double t_pi = t * kPi;
            const double sinc = std::fabs(t_pi) < 1e-15 ? 1.0 : std::sin(t_pi) / t_pi;
            kernels[static_cast<size_t>(phase) * kernel_width + k] =
                static_cast<float>(sinc * window * window * base_freq / static_cast<double>(orig_freq));
        }
    }

    size_t output_index = 0;
    for (int group = 0; group < group_count && output_index < output_size; ++group) {
        for (int phase = 0; phase < new_freq && output_index < output_size; ++phase) {
            float value = 0.0f;
            const size_t kernel_base = static_cast<size_t>(phase) * kernel_width;
            for (int k = 0; k < kernel_width; ++k) {
                const int64_t source_index =
                    static_cast<int64_t>(group) * orig_freq + k - width;
                if (source_index >= 0 && source_index < static_cast<int64_t>(input.samples.size())) {
                    value += input.samples[static_cast<size_t>(source_index)] *
                             kernels[kernel_base + k];
                }
            }
            out.samples[output_index++] = value;
        }
    }
    if (output_index != output_size) {
        return false;
    }
    return valid_audio(out.samples);
}

bool Token2WavFrontend::assemble_prompt_bundle(const std::vector<int32_t> & speech_tokens,
                                               const AudioFeatures &          prompt_mel_ct,
                                               const std::vector<float> &     speaker_embedding,
                                               FrontendPromptBundle &         out,
                                               std::string *                  error) {
    out = FrontendPromptBundle();
    if (speech_tokens.empty()) {
        return set_error(error, "speech tokenizer returned no tokens");
    }
    if (prompt_mel_ct.channels != 80 || prompt_mel_ct.frames <= 0 ||
        prompt_mel_ct.values.size() != static_cast<size_t>(prompt_mel_ct.channels) * prompt_mel_ct.frames) {
        return set_error(error, "prompt Mel must have shape [80, T]");
    }
    if (speaker_embedding.size() != 192) {
        return set_error(error, "speaker embedding must have 192 dimensions");
    }

    const int64_t expected_frames = static_cast<int64_t>(speech_tokens.size()) * 2;
    if (prompt_mel_ct.frames != expected_frames) {
        return set_error(error,
                         "prompt Mel frame count does not match speech token count: expected " +
                             std::to_string(expected_frames) + ", got " +
                             std::to_string(prompt_mel_ct.frames));
    }
    for (const float value : prompt_mel_ct.values) {
        if (!std::isfinite(value)) {
            return set_error(error, "prompt Mel contains a non-finite value");
        }
    }
    for (const float value : speaker_embedding) {
        if (!std::isfinite(value)) {
            return set_error(error, "speaker embedding contains a non-finite value");
        }
    }

    out.B = 1;
    out.prompt_tokens_bt = speech_tokens;
    out.prompt_tokens_bt.insert(out.prompt_tokens_bt.end(), 3, 4218);
    out.T_prompt_token = static_cast<int64_t>(out.prompt_tokens_bt.size());
    out.T_prompt_mel   = prompt_mel_ct.frames;
    out.prompt_mel_btc.resize(prompt_mel_ct.values.size());
    for (int64_t t = 0; t < prompt_mel_ct.frames; ++t) {
        for (int64_t c = 0; c < prompt_mel_ct.channels; ++c) {
            out.prompt_mel_btc[static_cast<size_t>(t) * prompt_mel_ct.channels + c] =
                prompt_mel_ct.values[static_cast<size_t>(c) * prompt_mel_ct.frames + t];
        }
    }
    out.spk_bc = speaker_embedding;
    return true;
}

bool Token2WavFrontend::prepare_prompt_bundle(const std::string & ref_wav_path,
                                              const std::string & speech_tokenizer_gguf,
                                              const std::string & campplus_gguf,
                                              FrontendPromptBundle & out,
                                              std::string *            error,
                                              int                      num_threads) {
    out = FrontendPromptBundle();
    AudioBuffer input;
    if (!load_wav_mono(ref_wav_path, input)) {
        return set_error(error, "failed to decode reference WAV: " + ref_wav_path);
    }

    AudioBuffer audio_16k;
    AudioBuffer audio_24k;
    if (!resample_mono(input, 16000, audio_16k)) {
        return set_error(error, "failed to resample reference WAV to 16 kHz");
    }
    if (!resample_mono(input, 24000, audio_24k)) {
        return set_error(error, "failed to resample reference WAV to 24 kHz");
    }

    AudioFeatures speech_features;
    if (!compute_s3_log_mel(audio_16k.samples, speech_features)) {
        return set_error(error, "failed to compute speech tokenizer log-Mel features");
    }

    AudioFeatures campplus_features;
    if (!compute_campplus_fbank(audio_16k.samples, campplus_features)) {
        return set_error(error, "failed to compute CampPlus fbank features");
    }
    if (!subtract_feature_column_mean(campplus_features, error)) {
        return false;
    }
    std::vector<int32_t> speech_tokens;
    std::vector<float> speaker_embedding;
    if (!prepare_prompt_bundle_gguf(speech_features, campplus_features, speech_tokenizer_gguf,
                                    campplus_gguf, num_threads, speech_tokens, speaker_embedding, error)) {
        return false;
    }

    AudioFeatures prompt_mel_ct;
    if (!compute_token2wav_prompt_mel(audio_24k.samples, prompt_mel_ct)) {
        return set_error(error, "failed to compute Token2Wav prompt Mel features");
    }
    const int64_t expected_prompt_mel_frames = static_cast<int64_t>(speech_tokens.size()) * 2;
    if (!replicate_pad_features(prompt_mel_ct, static_cast<int>(expected_prompt_mel_frames), error)) {
        return false;
    }
    return assemble_prompt_bundle(speech_tokens, prompt_mel_ct, speaker_embedding, out, error);
}

bool Token2WavFrontend::compute_s3_log_mel(const std::vector<float> & audio_16k, AudioFeatures & out) {
    out = AudioFeatures();
    if (!valid_audio(audio_16k)) {
        return false;
    }

    constexpr int sample_rate = 16000;
    constexpr int n_fft = 400;
    constexpr int hop = 160;
    constexpr int n_mels = 128;
    constexpr int center_padding = n_fft / 2;

    const int frame_count = 1 + static_cast<int>(
                                      (audio_16k.size() + 2 * center_padding - n_fft) /
                                      static_cast<size_t>(hop));
    if (frame_count <= 1) {
        return false;
    }

    const auto window = hann_window(n_fft);
    const auto plan = make_dft_plan(n_fft);
    const auto filters = make_librosa_mel_filterbank(sample_rate, n_fft, n_mels, 0.0f, 8000.0f);
    const int output_frames = frame_count - 1;
    std::vector<float> values(static_cast<size_t>(n_mels) * output_frames, 0.0f);
    std::vector<float> frame(static_cast<size_t>(n_fft));
    float global_max_log = -std::numeric_limits<float>::infinity();

    for (int t = 0; t < output_frames; ++t) {
        const int offset = t * hop - center_padding;
        for (int n = 0; n < n_fft; ++n) {
            frame[static_cast<size_t>(n)] = reflect_sample(audio_16k, offset + n) * window[static_cast<size_t>(n)];
        }
        const auto power = dft_power(frame, plan);
        for (int m = 0; m < n_mels; ++m) {
            float mel = 0.0f;
            const size_t filter_base = static_cast<size_t>(m) * (n_fft / 2 + 1);
            for (int k = 0; k < n_fft / 2; ++k) {
                mel += filters[filter_base + k] * power[static_cast<size_t>(k)];
            }
            const float log_mel = std::log10(std::max(mel, 1e-10f));
            values[static_cast<size_t>(m) * output_frames + t] = log_mel;
            global_max_log = std::max(global_max_log, log_mel);
        }
    }
    for (float & value : values) {
        value = (std::max(value, global_max_log - 8.0f) + 4.0f) / 4.0f;
    }

    set_features(out, n_mels, output_frames, std::move(values));
    return true;
}

bool Token2WavFrontend::compute_token2wav_prompt_mel(const std::vector<float> & audio_24k,
                                                      AudioFeatures & out) {
    out = AudioFeatures();
    if (!valid_audio(audio_24k)) {
        return false;
    }

    constexpr int sample_rate = 24000;
    constexpr int n_fft = 1920;
    constexpr int hop = 480;
    constexpr int n_mels = 80;
    constexpr int pad = (n_fft - hop) / 2;

    const int padded_size = static_cast<int>(audio_24k.size()) + 2 * pad;
    if (padded_size < n_fft) {
        return false;
    }
    const int frames = 1 + (padded_size - n_fft) / hop;
    const auto window = hann_window(n_fft);
    const auto plan = make_dft_plan(n_fft);
    const auto filters = make_librosa_mel_filterbank(sample_rate, n_fft, n_mels, 0.0f, 8000.0f);
    std::vector<float> values(static_cast<size_t>(n_mels) * frames, 0.0f);
    std::vector<float> frame(static_cast<size_t>(n_fft));

    for (int t = 0; t < frames; ++t) {
        const int offset = t * hop - pad;
        for (int n = 0; n < n_fft; ++n) {
            frame[static_cast<size_t>(n)] = reflect_sample(audio_24k, offset + n) * window[static_cast<size_t>(n)];
        }
        const auto magnitude = dft_magnitude(frame, plan);
        for (int m = 0; m < n_mels; ++m) {
            float mel = 0.0f;
            const size_t filter_base = static_cast<size_t>(m) * (n_fft / 2 + 1);
            for (int k = 0; k <= n_fft / 2; ++k) {
                mel += filters[filter_base + k] * magnitude[static_cast<size_t>(k)];
            }
            values[static_cast<size_t>(m) * frames + t] = std::log(std::max(mel, 1e-5f));
        }
    }

    set_features(out, n_mels, frames, std::move(values));
    return true;
}

bool Token2WavFrontend::compute_campplus_fbank(const std::vector<float> & audio_16k,
                                                AudioFeatures & out) {
    out = AudioFeatures();
    if (!valid_audio(audio_16k)) {
        return false;
    }

    constexpr int sample_rate = 16000;
    constexpr int window_size = 400;
    constexpr int window_shift = 160;
    constexpr int padded_window_size = 512;
    constexpr int n_mels = 80;
    constexpr float low_freq = 20.0f;
    constexpr float high_freq = 8000.0f;

    if (audio_16k.size() < static_cast<size_t>(window_size)) {
        return false;
    }
    const int frames = 1 + static_cast<int>((audio_16k.size() - window_size) / window_shift);
    const auto window = povey_window(window_size);
    const auto plan = make_dft_plan(padded_window_size);
    const float mel_min = 1127.0f * std::log(1.0f + low_freq / 700.0f);
    const float mel_max = 1127.0f * std::log(1.0f + high_freq / 700.0f);
    const float mel_delta = (mel_max - mel_min) / static_cast<float>(n_mels + 1);
    const float fft_bin_width = static_cast<float>(sample_rate) / padded_window_size;
    std::vector<float> filters(static_cast<size_t>(n_mels) * (padded_window_size / 2 + 1), 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const float left_mel = mel_min + static_cast<float>(m) * mel_delta;
        const float center_mel = left_mel + mel_delta;
        const float right_mel = center_mel + mel_delta;
        for (int k = 0; k <= padded_window_size / 2; ++k) {
            const float hz = fft_bin_width * static_cast<float>(k);
            const float mel = 1127.0f * std::log(1.0f + hz / 700.0f);
            const float up = (mel - left_mel) / (center_mel - left_mel);
            const float down = (right_mel - mel) / (right_mel - center_mel);
            filters[static_cast<size_t>(m) * (padded_window_size / 2 + 1) + k] =
                std::max(0.0f, std::min(up, down));
        }
    }

    std::vector<float> values(static_cast<size_t>(frames) * n_mels, 0.0f);
    std::vector<float> frame(static_cast<size_t>(padded_window_size), 0.0f);
    for (int t = 0; t < frames; ++t) {
        const size_t offset = static_cast<size_t>(t * window_shift);
        float mean = 0.0f;
        for (int n = 0; n < window_size; ++n) {
            mean += audio_16k[offset + static_cast<size_t>(n)];
        }
        mean /= static_cast<float>(window_size);
        float previous = 0.0f;
        for (int n = 0; n < window_size; ++n) {
            const float centered = audio_16k[offset + static_cast<size_t>(n)] - mean;
            if (n == 0) {
                frame[static_cast<size_t>(n)] = (centered - 0.97f * centered) * window[static_cast<size_t>(n)];
            } else {
                frame[static_cast<size_t>(n)] = (centered - 0.97f * previous) * window[static_cast<size_t>(n)];
            }
            previous = centered;
        }
        std::fill(frame.begin() + window_size, frame.end(), 0.0f);
        const auto power = dft_power(frame, plan);
        for (int m = 0; m < n_mels; ++m) {
            float energy = 0.0f;
            const size_t filter_base = static_cast<size_t>(m) * (padded_window_size / 2 + 1);
            for (int k = 0; k < padded_window_size / 2; ++k) {
                energy += filters[filter_base + k] * power[static_cast<size_t>(k)];
            }
            values[static_cast<size_t>(t) * n_mels + m] =
                std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
        }
    }

    set_features(out, n_mels, frames, std::move(values));
    return true;
}

}  // namespace flow
}  // namespace omni
