#pragma once

// Qwen3-ForcedAligner: given audio and a list of alignment units, return the start and end
// time of every unit.
//
// The model ships as two GGUF files:
//   - LM     the qwen3 backbone (0.6B) plus its vocabulary
//   - Audio  the qwen3a audio tower and multimodal projector, carrying the timestamp head
//            (`score.weight`) as an extra tensor
//
// How the units are segmented is up to the caller; they are taken as given, so per-character,
// per-word or any other granularity all work.

#include <cstddef>
#include <string>
#include <vector>

struct qwen3_aligner;

struct qwen3_aligner_params {
    std::string lm_path;
    std::string audio_path;
    int         n_gpu_layers = -1;    // -1 offloads everything
    int         n_threads    = 4;
    int         n_ctx        = 4096;  // about 5 minutes of audio
};

// Where a unit sits in the audio, in seconds
struct qwen3_aligner_span {
    std::string text;
    double      start = 0.0;
    double      end   = 0.0;
};

// Returns nullptr and fills err on failure
qwen3_aligner * qwen3_aligner_init(const qwen3_aligner_params & params, std::string & err);

// Accepts nullptr
void qwen3_aligner_free(qwen3_aligner * ctx);

// Sample rate the model expects (16000)
int qwen3_aligner_sample_rate(const qwen3_aligner * ctx);

// Decode an audio file (wav/mp3/flac) into mono float PCM at the model's sample rate
bool qwen3_aligner_decode_audio(qwen3_aligner *      ctx,
                                const void *         data,
                                size_t               size,
                                std::vector<float> & pcm,
                                std::string &        err);

// pcm must be mono at qwen3_aligner_sample_rate().
// spans comes back one per unit, in the same order.
bool qwen3_aligner_align(qwen3_aligner *                        ctx,
                         const float *                          pcm,
                         size_t                                 n_samples,
                         const std::vector<std::string> &       units,
                         std::vector<qwen3_aligner_span> &      spans,
                         std::string &                          err);
