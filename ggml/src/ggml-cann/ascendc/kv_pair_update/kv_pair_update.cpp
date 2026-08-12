#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr uint32_t max_row_width = 1024;
constexpr uint32_t src_ub_bytes = max_row_width * sizeof(float);
constexpr uint32_t dst_ub_bytes = max_row_width * sizeof(half);

}  // namespace

extern "C" __global__ __aicore__ void kv_pair_update(
        GM_ADDR first_src,
        GM_ADDR first_index,
        GM_ADDR first_dst,
        GM_ADDR second_src,
        GM_ADDR second_index,
        GM_ADDR second_dst,
        uint64_t row_width,
        uint64_t cache_rows) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::GlobalTensor<float> first_src_gm;
    AscendC::GlobalTensor<int64_t> first_index_gm;
    AscendC::GlobalTensor<half> first_dst_gm;
    AscendC::GlobalTensor<float> second_src_gm;
    AscendC::GlobalTensor<int64_t> second_index_gm;
    AscendC::GlobalTensor<half> second_dst_gm;
    first_src_gm.SetGlobalBuffer((__gm__ float *) first_src);
    first_index_gm.SetGlobalBuffer((__gm__ int64_t *) first_index);
    first_dst_gm.SetGlobalBuffer((__gm__ half *) first_dst);
    second_src_gm.SetGlobalBuffer((__gm__ float *) second_src);
    second_index_gm.SetGlobalBuffer((__gm__ int64_t *) second_index);
    second_dst_gm.SetGlobalBuffer((__gm__ half *) second_dst);

    const uint64_t first_row =
        static_cast<uint64_t>(first_index_gm.GetValue(0));
    const uint64_t second_row =
        static_cast<uint64_t>(second_index_gm.GetValue(0));
    if (first_row >= cache_rows || second_row >= cache_rows ||
        row_width > max_row_width) {
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
    AscendC::DataCopy(src_ub, first_src_gm, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Cast(dst_ub, src_ub, AscendC::RoundMode::CAST_RINT, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(first_dst_gm[first_row * row_width], dst_ub, elements);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopy(src_ub, second_src_gm, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Cast(dst_ub, src_ub, AscendC::RoundMode::CAST_RINT, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(second_dst_gm[second_row * row_width], dst_ub, elements);
}
