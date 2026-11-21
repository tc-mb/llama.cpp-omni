#include "ggml-cuda/quantize.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

// Check the encoded RHS contract, not a second invocation of the quantizer.
// Deliberately biased, sub-half-step values distinguish sum(x) from d*sum(q).
static bool test_q8_1(int64_t width, ggml_type type, bool mmq, bool gather, bool varied = false) {
    constexpr int rows = 3;
    constexpr int channels = 2;
    constexpr int padded = 512;
    const int64_t stride = width + 4; // keep float4 loads aligned, with row padding
    std::vector<float> input(channels * rows * stride, 42.0f);
    const int32_t ids[rows] = {2, 0, 1};
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < rows; ++row) {
            const float scale = (channel + 1) * (row + 1) * 0.25f;
            for (int64_t col = 0; col < width; ++col) {
                const float peaks[] = {12.7f, 6.1f, 25.4f, -9.3f, 3.7f, -4.9f, 18.2f, 7.5f};
                const float value = varied ?
                    (col % 16 == 0 ? peaks[(col / 16) % 8] : float(col % 13 - 6) * 0.113f) :
                    (col % 32 == 0 ? 127.0f : (row == 1 ? -0.49f : 0.49f));
                input[(channel * rows + row) * stride + col] = row == 0 ? 0.0f : value * scale;
            }
        }
    }

    float * x = nullptr;
    void * y = nullptr;
    int32_t * device_ids = nullptr;
    const size_t bytes = channels * rows * padded / QK8_1 * sizeof(block_q8_1);
    CUDA_CHECK(cudaMalloc(&x, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, bytes));
    CUDA_CHECK(cudaMemset(y, 0xa5, bytes));
    CUDA_CHECK(cudaMemcpy(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (gather) {
        CUDA_CHECK(cudaMalloc(&device_ids, sizeof(ids)));
        CUDA_CHECK(cudaMemcpy(device_ids, ids, sizeof(ids), cudaMemcpyHostToDevice));
    }
    quantize_cuda_t quantize = mmq ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda;
    quantize(x, device_ids, y, type, width, stride, rows * stride, channels * rows * stride,
             padded, rows, channels, 1, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<block_q8_1> regular(bytes / sizeof(block_q8_1));
    std::vector<block_q8_1_mmq> tiled(bytes / sizeof(block_q8_1_mmq));
    if (mmq) {
        CUDA_CHECK(cudaMemcpy(tiled.data(), y, bytes, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(regular.data(), y, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(device_ids));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(x));

    bool ok = true;
    const auto check = [&](const char * field, float actual, float expected, int channel, int row, int col,
                           float tolerance = 0.0f) {
        if (!std::isfinite(actual) || fabsf(actual - expected) > tolerance) {
            if (ok) {
                fprintf(stderr, "Q8_1 %s width=%lld type=%s mmq=%d gather=%d varied=%d channel=%d row=%d col=%d: got %g expected %g\n",
                        field, (long long) width, ggml_type_name(type), mmq, gather, varied, channel, row, col, actual, expected);
            }
            ok = false;
        }
    };
    const mmq_q8_1_ds_layout layout = type == GGML_TYPE_Q2_K ? MMQ_Q8_1_DS_LAYOUT_D2S6 :
                                     type == GGML_TYPE_Q4_1 ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
    GGML_ASSERT(mmq_get_q8_1_ds_layout(type) == layout);
    const int block_width = mmq ? 128 : 32;
    const int scale_width = mmq && layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    const int sum_width = mmq && layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < rows; ++row) {
            const int source_row = gather ? ids[row] : row;
            for (int col = 0; col < padded; col += block_width) {
                const int index = mmq ? (channel * (padded / 128) + col / 128) * rows + row :
                                        ((channel * rows + row) * padded + col) / 32;
                const int8_t * qs = mmq ? tiled[index].qs : regular[index].qs;
                int expected_q[128] = {};
                for (int group = 0; group < block_width; group += scale_width) {
                    float amax = 0.0f;
                    for (int i = 0; i < scale_width && col + group + i < width; ++i) {
                        amax = std::max(amax, fabsf(input[(channel * rows + source_row) * stride + col + group + i]));
                    }
                    const float d = amax / 127.0f;
                    const float stored_d = !mmq ? __low2float(regular[index].ds) :
                        layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? float(tiled[index].d2s6[group / 64]) :
                        layout == MMQ_Q8_1_DS_LAYOUT_DS4 ? __low2float(tiled[index].ds4[group / 32]) :
                        tiled[index].d4[group / 32];
                    const bool full_precision = mmq && layout == MMQ_Q8_1_DS_LAYOUT_D4;
                    check("scale", stored_d, full_precision ? d : float(__float2half(d)), channel, row, col + group,
                          full_precision ? 2e-7f * d : 0.0f);
                    for (int i = 0; i < scale_width; ++i) {
                        const float value = col + group + i < width ?
                            input[(channel * rows + source_row) * stride + col + group + i] : 0.0f;
                        expected_q[group + i] = d == 0.0f ? 0 : int(roundf(value / d));
                        check("payload", qs[group + i], expected_q[group + i], channel, row, col + group + i);
                    }
                    if (mmq && layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                        continue; // D4 has no correction sums.
                    }
                    for (int sub = 0; sub < scale_width; sub += sum_width) {
                        const int start = group + sub;
                        if (mmq && layout == MMQ_Q8_1_DS_LAYOUT_D2S6 && start >= 96) {
                            continue; // Only the first six 16-element sums are stored.
                        }
                        int sum_q = 0;
                        for (int i = 0; i < sum_width; ++i) {
                            sum_q += expected_q[start + i];
                        }
                        const float expected = float(__float2half(d * sum_q));
                        const float actual = !mmq ? __high2float(regular[index].ds) :
                            layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? float(tiled[index].d2s6[2 + start / 16]) :
                            __high2float(tiled[index].ds4[start / 32]);
                        check("correction", actual, expected, channel, row, col + start);
                    }
                }
            }
        }
    }
    printf("Q8_1 width=%lld type=%s mmq=%d gather=%d varied=%d: %s\n",
           (long long) width, ggml_type_name(type), mmq, gather, varied, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
#ifdef cudaErrorStubLibrary
        status == cudaErrorStubLibrary ||
#endif
        (status == cudaSuccess && devices == 0)) {
        printf("No usable CUDA device (%s); skipping quantization tests\n", cudaGetErrorString(status));
        return 77;
    }
    CUDA_CHECK(status);
    CUDA_CHECK(cudaSetDevice(0));
    int passed = 0;
    for (int64_t width : {32, 132, 512}) {
        passed += test_q8_1(width, GGML_TYPE_Q4_1, false, false);
        if (width == 132) {
            passed += test_q8_1(width, GGML_TYPE_Q4_1, false, false, true);
        }
        for (ggml_type type : {GGML_TYPE_Q4_1, GGML_TYPE_Q2_K, GGML_TYPE_Q8_0}) {
            for (bool gather : {false, true}) {
                passed += test_q8_1(width, type, true, gather);
                if (width == 132) {
                    passed += test_q8_1(width, type, true, gather, true);
                }
            }
        }
    }
    printf("%d/28 quantization tests passed\n", passed);
    return passed == 28 ? 0 : 1;
}
