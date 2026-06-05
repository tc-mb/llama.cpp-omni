// coreml_t2w_dit.h
//
// Token2Mel DiT estimator -> CoreML 桥接（C ABI），独立于 ggml。
//
// 用途：MiniCPM-o-4.5 token2wav 的 DiT estimator 跑在 Apple Neural Engine 上，
//   彻底绕开 Metal command queue 在双工模式下被 LLM 抢资源的瓶颈。
//
// CoreML 模型由 tools/convert/o45_tts/ane_export_dit.py 生成，输入/输出 schema:
//
//   inputs:
//     x            (B=2, 80,                                T=56)        fp32
//     mu           (B=2, 80,                                T=56)        fp32
//     t            (B=2,)                                                 fp32
//     spks         (B=2, 80)                                              fp32
//     cond         (B=2, 80,                                T=56)        fp32
//     cnn_cache_in (depth*B=32,                  1024,      2)            fp32
//     att_cache_in (depth*B=32, num_heads=8, max_cache_len=600, head_dim*2=128) fp32
//     attn_mask    (B=2, T=56, T + max_cache_len=656)                     fp32 additive
//                                                                         (0 = valid, -1e4 = masked)
//
//   outputs:
//     feat            (B=2, 80, T=56)
//     cnn_cache_out   (depth*B=32, 1024, 2)
//     att_cache_out   (depth*B=32, 8, 600, 128)
//
// 一次 predict 完成一个 timestep；调用方需要在外部循环 n_timesteps（cosine
// schedule + Euler ODE + CFG），并为每个 timestep 维护一份独立的 cnn/att cache。

#pragma once

#include <stdint.h>
#include <stddef.h>

#if __cplusplus
extern "C" {
#endif

typedef void * t2w_dit_handle_t;

// CoreML 模型几何信息（加载时从 metadata 读取，便于调用方校验形状）
typedef struct {
    int32_t depth;             // 16
    int32_t batch;             // 2  (CFG cond + uncond batched)
    int32_t chunk_size;        // 56  mel frames per call
    int32_t max_cache_len;     // 600 attention cache padding length
    int32_t cnn_cache_channels; // 1024 (= in_channels + out_channels of conv block)
    int32_t cnn_cache_t;       // 2
    int32_t num_heads;         // 8
    int32_t head_dim;          // 64
    int32_t mel_channels;      // 80
    int32_t spk_dim;           // 80
} t2w_dit_dims_t;

// 加载 CoreML 模型（路径），默认 compute_units = CPUAndNeuralEngine。
// 环境变量 OMNI_T2W_DIT_COMPUTE_UNITS 可覆盖：
//   "ane"     -> CPUAndNeuralEngine（默认，目标场景）
//   "all"     -> All
//   "cpu_gpu" -> CPUAndGPU
//   "cpu"     -> CPUOnly
// 失败返回 nullptr。
t2w_dit_handle_t t2w_dit_load(const char * model_path);

// 读取模型几何。
// 失败返回非 0。
int t2w_dit_get_dims(t2w_dit_handle_t h, t2w_dit_dims_t * out);

// 单次 forward。所有指针指向 contiguous fp32 buffer，layout 见文件顶部 schema。
// 输出 buffer 由调用方分配；分配大小见 t2w_dit_predict_buffer_sizes。
// 失败返回非 0。
int t2w_dit_predict(t2w_dit_handle_t h,
                    const float * x,            // (B*80*T)
                    const float * mu,           // (B*80*T)
                    const float * t,            // (B,)
                    const float * spks,         // (B*80)
                    const float * cond,         // (B*80*T)
                    const float * cnn_cache_in, // (depth*B*1024*2)
                    const float * att_cache_in, // (depth*B*8*max_cache_len*128)
                    const float * attn_mask,    // (B*T*(T+max_cache_len))
                    float * feat_out,           // (B*80*T)
                    float * cnn_cache_out,      // (depth*B*1024*2)
                    float * att_cache_out);     // (depth*B*8*max_cache_len*128)

// 计算各 buffer 需要的元素数量（fp32 个数），方便调用方分配。
// nelem_* 任一可为 nullptr（不需要时跳过）。
int t2w_dit_predict_buffer_sizes(t2w_dit_handle_t h,
                                  size_t * nelem_x_mu_cond_feat,
                                  size_t * nelem_t,
                                  size_t * nelem_spks,
                                  size_t * nelem_cnn_cache,
                                  size_t * nelem_att_cache,
                                  size_t * nelem_attn_mask);

// 释放。
void t2w_dit_free(t2w_dit_handle_t h);

#if __cplusplus
}  // extern "C"
#endif
