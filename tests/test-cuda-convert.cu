#include "ggml-cuda/convert.cuh"

// Use exact allocations: the backend pool's padding can hide short-row overruns.
template<typename T>
static void test_iq4_nl(int64_t k, to_t_cuda_t<T> convert, const char * type) {
    std::vector<float> input(k);
    for (int64_t i = 0; i < k; i++) {
        input[i] = float(i % 31 - 15) / 16.0f;
    }
    std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_IQ4_NL, k));
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_IQ4_NL, input.data(), quantized.data(), 0, 1, k, nullptr);
    GGML_ASSERT(bytes == quantized.size());
    std::vector<float> expected(k);
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(quantized.data(), expected.data(), k);

    void * src = nullptr;
    T * dst = nullptr;
    CUDA_CHECK(cudaMalloc(&src, quantized.size()));
    CUDA_CHECK(cudaMalloc(&dst, k*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(src, quantized.data(), quantized.size(), cudaMemcpyHostToDevice));
    GGML_ASSERT(convert != nullptr);
    convert(src, dst, k, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<T> actual(k);
    CUDA_CHECK(cudaMemcpy(actual.data(), dst, k*sizeof(T), cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < k; i++) {
        // Both F16 and F32 must match dequantization of the stored quantized input.
        const float want = float(T(expected[i]));
        if (float(actual[i]) != want) {
            fprintf(stderr, "IQ4_NL -> %s k=%lld index=%lld: got %g, expected %g\n",
                    type, (long long) k, (long long) i, float(actual[i]), want);
            GGML_ABORT("dequantization mismatch");
        }
    }
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(src));
    printf("IQ4_NL -> %s k=%lld: OK\n", type, (long long) k);
}

int main() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
#ifdef cudaErrorStubLibrary
        status == cudaErrorStubLibrary ||
#endif
        (status == cudaSuccess && devices == 0)) {
        printf("No usable CUDA device (%s); skipping conversion tests\n", cudaGetErrorString(status));
        return 77;
    }
    CUDA_CHECK(status);
    CUDA_CHECK(cudaSetDevice(0));
    ggml_quantize_init(GGML_TYPE_IQ4_NL);
    for (int64_t k : {32, 64, 96, 256, 288}) {
        test_iq4_nl(k, ggml_get_to_fp32_cuda(GGML_TYPE_IQ4_NL), "F32");
        test_iq4_nl(k, ggml_get_to_fp16_cuda(GGML_TYPE_IQ4_NL), "F16");
    }
    ggml_quantize_free();
    printf("10/10 conversion tests passed\n");
    return 0;
}
