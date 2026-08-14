#include "audition.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static int mel_frames_for(size_t n_samples) {
    whisper_preprocessor::whisper_filters filters;
    filters.n_mel = 80;
    filters.n_fft = WHISPER_N_FFT / 2 + 1;
    filters.data.assign(
        static_cast<size_t>(filters.n_mel) * filters.n_fft,
        0.0f);

    std::vector<float> samples(n_samples, 0.0f);
    std::vector<whisper_preprocessor::whisper_mel> output;
    assert(whisper_preprocessor::preprocess_audio(
        samples.data(), samples.size(), filters, output));
    assert(output.size() == 1);
    assert(output[0].n_len == output[0].n_len_org);
    return output[0].n_len;
}

int main() {
    assert(mel_frames_for(62400) == 390);
    assert(mel_frames_for(62401) == 391);
    assert(mel_frames_for(62464) == 391);
}
