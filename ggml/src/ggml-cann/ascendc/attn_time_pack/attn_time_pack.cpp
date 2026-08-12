#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr uint32_t head_dim = 64;
constexpr uint32_t row_bytes = head_dim * sizeof(uint32_t);
constexpr uint32_t max_time = 358;
constexpr uint32_t packed_bytes = max_time * row_bytes;

}  // namespace

extern "C" __global__ __aicore__ void attn_time_pack(
        GM_ADDR src0,
        GM_ADDR src1,
        GM_ADDR dst,
        uint64_t time0,
        uint64_t time1,
        uint64_t heads,
        uint64_t batch,
        uint64_t src0_head_stride,
        uint64_t src0_time_stride,
        uint64_t src0_batch_stride,
        uint64_t src1_head_stride,
        uint64_t src1_time_stride,
        uint64_t src1_batch_stride) {
    AscendC::GlobalTensor<uint32_t> src0_gm;
    AscendC::GlobalTensor<uint32_t> src1_gm;
    AscendC::GlobalTensor<uint32_t> dst_gm;
    src0_gm.SetGlobalBuffer((__gm__ uint32_t *) src0);
    src1_gm.SetGlobalBuffer((__gm__ uint32_t *) src1);
    dst_gm.SetGlobalBuffer((__gm__ uint32_t *) dst);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packed_buffer;
    pipe.InitBuffer(packed_buffer, packed_bytes);
    AscendC::LocalTensor<uint32_t> packed = packed_buffer.Get<uint32_t>();

    const uint64_t total_time = time0 + time1;
    const uint64_t block = AscendC::GetBlockIdx();
    const uint64_t head_batches = batch * heads;
    if (block >= head_batches || total_time > max_time) {
        return;
    }

    const uint64_t head_batch = block;
    const uint64_t head = head_batch % heads;
    const uint64_t batch_index = head_batch / heads;
    const uint64_t src0_base = batch_index * src0_batch_stride +
        head * src0_head_stride;
    const uint64_t src1_base = batch_index * src1_batch_stride +
        head * src1_head_stride;
    constexpr uint16_t row_blocks = row_bytes / 32;
    const AscendC::DataCopyParams src0_params(
        static_cast<uint16_t>(time0), row_blocks,
        static_cast<uint16_t>(src0_time_stride * sizeof(uint32_t) / 32 -
                              row_blocks),
        0);
    const AscendC::DataCopyParams src1_params(
        static_cast<uint16_t>(time1), row_blocks,
        static_cast<uint16_t>(src1_time_stride * sizeof(uint32_t) / 32 -
                              row_blocks),
        0);
    AscendC::DataCopy(packed, src0_gm[src0_base], src0_params);
    AscendC::DataCopy(packed[time0 * head_dim], src1_gm[src1_base],
                      src1_params);
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint64_t dst_base = head_batch * total_time * head_dim;
    AscendC::DataCopy(dst_gm[dst_base], packed, total_time * head_dim);
    AscendC::PipeBarrier<PIPE_ALL>();
}
