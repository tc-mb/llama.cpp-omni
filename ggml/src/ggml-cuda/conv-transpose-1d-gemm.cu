#include "conv-transpose-1d-gemm.cuh"

static __global__ void conv_transpose_1d_col2im_kernel(
        const float * __restrict__ columns,
        float * __restrict__ dst,
        const int s0, const int K, const int C_out,
        const int T_in, const int T_out, const int output_size) {
    const int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    const int c_out = global_index / T_out;
    const int p     = global_index % T_out;

    const int t_max = min(T_in - 1, p / s0);
    const int numer = p - K + 1;
    const int t_min = max(0, (numer >= 0) ? ((numer + s0 - 1) / s0) : (numer / s0));

    float acc = 0.0f;
    for (int t = t_min; t <= t_max; ++t) {
        const int k = p - t * s0;
        acc += columns[t + T_in * (k + K * c_out)];
    }
    dst[global_index] = acc;
}

bool ggml_cuda_conv_transpose_1d_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int K     = (int)src0->ne[0];
    const int C_out = (int)src0->ne[1];
    const int C_in  = (int)src0->ne[2];
    const int T_in  = (int)src1->ne[0];

    if (C_in < 8 || C_out < 8 || T_in < 4) {
        return false;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float       * dst_d  = (float *)dst->data;
    cudaStream_t  stream = ctx.stream();

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s0    = opts[0];
    const int T_out = (int)dst->ne[0];
    const int64_t output_size = ggml_nelements(dst);

    // GEMM: columns[T_in, K*C_out] = input[T_in, C_in] × kernel^T[C_in, K*C_out]
    //
    // ggml column-major layout (ne[0] contiguous):
    //   src0 (kernel): [K, C_out, C_in] → viewed as column-major [K*C_out, C_in]
    //   src1 (input):  [T_in, C_in]     → column-major [T_in, C_in]
    //
    // cublasSgemm(N, T): C = A × B^T
    //   A = src1 [T_in × C_in],   lda = T_in
    //   B = src0 [K*C_out × C_in], ldb = K*C_out
    //   C = columns [T_in × K*C_out], ldc = T_in

    const int64_t col_size = (int64_t)T_in * K * C_out;
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> columns_alloc(pool, col_size);
    float * columns = columns_alloc.ptr;

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        T_in,          // m
        K * C_out,     // n
        C_in,          // k
        &alpha,
        src1_d, T_in,       // A, lda
        src0_d, K * C_out,  // B, ldb
        &beta,
        columns, T_in));    // C, ldc

    // col2im: scatter-add columns → output
    const int block_size = 256;
    const int num_blocks = ((int)output_size + block_size - 1) / block_size;
    conv_transpose_1d_col2im_kernel<<<num_blocks, block_size, 0, stream>>>(
        columns, dst_d, s0, K, C_out, T_in, T_out, (int)output_size);

    return true;
}
