#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr uint32_t max_row_width = 1024;
constexpr uint32_t src_ub_bytes = max_row_width * sizeof(float);
constexpr uint32_t dst_ub_bytes = max_row_width * sizeof(half);

}  // namespace

extern "C" __global__ __aicore__ void set_rows_f32_f16(
        GM_ADDR src,
        GM_ADDR index,
        GM_ADDR dst,
        uint64_t row_width,
        uint64_t cache_rows) {
    AscendC::GlobalTensor<float> src_gm;
    AscendC::GlobalTensor<int64_t> index_gm;
    AscendC::GlobalTensor<half> dst_gm;
    src_gm.SetGlobalBuffer((__gm__ float *) src);
    index_gm.SetGlobalBuffer((__gm__ int64_t *) index);
    dst_gm.SetGlobalBuffer((__gm__ half *) dst);

    const uint64_t row = static_cast<uint64_t>(index_gm.GetValue(0));
    if (row >= cache_rows || row_width > max_row_width) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> src_ub_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dst_ub_buffer;
    pipe.InitBuffer(src_ub_buffer, src_ub_bytes);
    pipe.InitBuffer(dst_ub_buffer, dst_ub_bytes);
    AscendC::LocalTensor<float> src_ub = src_ub_buffer.Get<float>();
    AscendC::LocalTensor<half> dst_ub = dst_ub_buffer.Get<half>();

    const uint32_t elements = static_cast<uint32_t>(row_width);
    AscendC::DataCopy(src_ub, src_gm, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    // CANN Cast's F32->F16 policy is CAST_RINT (round-to-nearest-even).
    AscendC::Cast(dst_ub, src_ub, AscendC::RoundMode::CAST_RINT, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(dst_gm[row * row_width], dst_ub, elements);
}
