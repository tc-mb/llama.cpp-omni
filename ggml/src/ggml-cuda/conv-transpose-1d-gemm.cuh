#pragma once

#include "common.cuh"

// cuBLAS GEMM + col2im implementation of conv_transpose_1d.
// Returns true if the GEMM path was used, false if caller should fall back to the direct kernel.
bool ggml_cuda_conv_transpose_1d_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
