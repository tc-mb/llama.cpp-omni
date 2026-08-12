#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr uint32_t hidden = 512;
constexpr uint32_t tile_bytes = hidden * sizeof(float);

}  // namespace

extern "C" __global__ __aicore__ void modulate_fusion(
        GM_ADDR x,
        GM_ADDR scale,
        GM_ADDR shift,
        GM_ADDR dst,
        uint64_t time,
        uint64_t batch,
        uint64_t scale_batch_stride,
        uint64_t shift_batch_stride,
        uint32_t mode) {
    AscendC::GlobalTensor<float> x_gm;
    AscendC::GlobalTensor<float> scale_gm;
    AscendC::GlobalTensor<float> shift_gm;
    AscendC::GlobalTensor<float> dst_gm;
    x_gm.SetGlobalBuffer((__gm__ float *) x);
    scale_gm.SetGlobalBuffer((__gm__ float *) scale);
    shift_gm.SetGlobalBuffer((__gm__ float *) shift);
    dst_gm.SetGlobalBuffer((__gm__ float *) dst);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shift_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaled_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_buffer;
    pipe.InitBuffer(x_buffer, tile_bytes);
    pipe.InitBuffer(scale_buffer, tile_bytes);
    pipe.InitBuffer(shift_buffer, tile_bytes);
    pipe.InitBuffer(scaled_buffer, tile_bytes);
    pipe.InitBuffer(acc_buffer, tile_bytes);

    AscendC::LocalTensor<float> x_ub = x_buffer.Get<float>();
    AscendC::LocalTensor<float> scale_ub = scale_buffer.Get<float>();
    AscendC::LocalTensor<float> shift_ub = shift_buffer.Get<float>();
    AscendC::LocalTensor<float> scaled_ub = scaled_buffer.Get<float>();
    AscendC::LocalTensor<float> acc_ub = acc_buffer.Get<float>();

    const uint64_t block = AscendC::GetBlockIdx();
    const uint64_t blocks = AscendC::GetBlockNum();
    const uint64_t blocks_per_batch = blocks / batch;
    const uint64_t batch_index = block / blocks_per_batch;
    const uint64_t time_lane = block - batch_index * blocks_per_batch;
    if (batch_index >= batch) {
        return;
    }

    AscendC::DataCopy(
        scale_ub, scale_gm[batch_index * scale_batch_stride], hidden);
    if (mode == 0) {
        AscendC::DataCopy(
            shift_ub, shift_gm[batch_index * shift_batch_stride], hidden);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    for (uint64_t t = time_lane; t < time; t += blocks_per_batch) {
        const uint64_t x_offset = (batch_index * time + t) * hidden;
        AscendC::DataCopy(x_ub, x_gm[x_offset], hidden);
        if (mode != 0) {
            AscendC::DataCopy(shift_ub, shift_gm[x_offset], hidden);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Mul(scaled_ub, x_ub, scale_ub, hidden);
        AscendC::PipeBarrier<PIPE_V>();
        if (mode == 0) {
            AscendC::Add(acc_ub, x_ub, scaled_ub, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(scaled_ub, acc_ub, shift_ub, hidden);
        } else {
            AscendC::Add(scaled_ub, shift_ub, scaled_ub, hidden);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopy(dst_gm[x_offset], scaled_ub, hidden);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
