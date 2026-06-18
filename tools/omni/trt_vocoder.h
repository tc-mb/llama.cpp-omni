#pragma once
// TRT-accelerated HiFi-GAN vocoder for MiniCPM-o.
// Replaces ggml voc_hg2_runner_eval_stream with TensorRT engine inference.

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

// Forward-declare TRT types to avoid header dependency
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}
typedef struct CUstream_st *cudaStream_t;

namespace omni {
namespace vocoder {

struct TRTVocoderConfig {
    std::string engine_path;  // path to .plan file
    int         T_mel  = 100; // fixed input mel frames
};

class TRTVocoder {
public:
    TRTVocoder();
    ~TRTVocoder();

    bool init(const TRTVocoderConfig & cfg);
    bool is_ready() const { return ready_; }

    /// Infer vocoder: mel[80 * T_mel] (row-major BCT, B=1) → wave_bt_out
    /// Returns true on success.
    bool infer(const float * mel_bct, int T_mel,
               std::vector<float> & wave_bt_out, int64_t & out_T_audio);

private:
    bool ready_ = false;
    TRTVocoderConfig cfg_;

    // TRT objects
    nvinfer1::IRuntime           * runtime_  = nullptr;
    nvinfer1::ICudaEngine        * engine_   = nullptr;
    nvinfer1::IExecutionContext  * ctx_      = nullptr;
    cudaStream_t                   stream_   = 0;

    // GPU buffers (mel_in, stft_out)
    void * d_mel_  = nullptr;
    void * d_stft_ = nullptr;
    size_t mel_bytes_  = 0;
    size_t stft_bytes_ = 0;

    // iSTFT constants
    std::vector<float> window_;       // Hann window [N_FFT]
    std::vector<float> idft_matrix_;  // [N_FFT × 18] pre-computed IDFT
    int T_frame_out_ = 0;  // computed at first infer
};

} // namespace vocoder
} // namespace omni
