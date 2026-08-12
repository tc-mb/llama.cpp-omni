#include "kernel_operator.h"
#include "im2col1d_ctb_offsets_512.h"
#include "im2col1d_vocoder_gather_mask.h"

#include <cstdint>

namespace {

constexpr uint32_t channel_tile_max = 512;
constexpr uint32_t tap_words = 3;
constexpr uint32_t ub_row_words = 8;
constexpr uint32_t ub_bytes = channel_tile_max * ub_row_words * sizeof(uint32_t);

constexpr uint32_t ctb_input_ub_bytes =
    channel_tile_max * tap_words * sizeof(uint32_t);
constexpr uint32_t ctb_output_ub_bytes =
    channel_tile_max * tap_words * sizeof(uint32_t);
constexpr uint32_t ctb_offset_ub_bytes = ctb_output_ub_bytes;

constexpr uint32_t vocoder_channel_tile_max = 256;
constexpr uint32_t vocoder_span_words_max = 51;
constexpr uint32_t vocoder_span_row_words = 56;
constexpr uint32_t vocoder_output_time_tile = 4;
constexpr uint32_t vocoder_packed_row_words = 16;
constexpr uint32_t vocoder_input_ub_bytes =
    vocoder_channel_tile_max * vocoder_span_row_words * sizeof(uint32_t);
constexpr uint32_t vocoder_output_ub_bytes =
    vocoder_channel_tile_max * vocoder_packed_row_words * sizeof(uint32_t);
constexpr uint32_t vocoder_pattern_ub_bytes =
    vocoder_output_time_tile * 32;
constexpr uint32_t vocoder_input_row_blocks =
    vocoder_span_row_words * sizeof(uint32_t) / 32U;

__aicore__ inline uint32_t normalize_legacy_f32_word(uint32_t value) {
    const uint32_t magnitude = value & UINT32_C(0x7fffffff);
    if (magnitude > UINT32_C(0x7f800000)) {
        return UINT32_C(0x7fffffff);
    }
    return value == UINT32_C(0x80000000) ? 0U : value;
}

}  // namespace

extern "C" __global__ __aicore__ void im2col1d_f32(
        GM_ADDR src,
        GM_ADDR src_aux,
        GM_ADDR dst,
        uint64_t t_size,
        uint64_t channels,
        uint64_t batch,
        uint64_t kernel,
        uint64_t out_width,
        uint64_t stride,
        int64_t  padding,
        uint64_t dilation,
        uint64_t causal_ctb) {
    AscendC::GlobalTensor<uint32_t> src_gm;
    AscendC::GlobalTensor<uint32_t> src_aux_gm;
    AscendC::GlobalTensor<uint32_t> dst_gm;
    src_gm.SetGlobalBuffer((__gm__ uint32_t *) src);
    src_aux_gm.SetGlobalBuffer((__gm__ uint32_t *) src_aux);
    dst_gm.SetGlobalBuffer((__gm__ uint32_t *) dst);

    constexpr uint64_t gm_cache_line_words = 64 / sizeof(uint32_t);
    const uint64_t dst_words = batch * out_width * channels * kernel;
    const uint64_t line_count = (dst_words - 1) / gm_cache_line_words + 1;
    const uint64_t block = AscendC::GetBlockIdx();
    const uint64_t blocks = AscendC::GetBlockNum();

    const bool use_ctb_gather =
        causal_ctb != 0 && causal_ctb <= 4 &&
        kernel == tap_words && stride == 1 &&
        padding == 0 && dilation == 1 && t_size >= tap_words &&
        channels <= channel_tile_max;
    if (use_ctb_gather) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_ub_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_ub_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_ub_buffer;
        pipe.InitBuffer(input_ub_buffer, ctb_input_ub_bytes);
        pipe.InitBuffer(output_ub_buffer, ctb_output_ub_bytes);
        pipe.InitBuffer(offset_ub_buffer, ctb_offset_ub_bytes);
        AscendC::LocalTensor<uint32_t> input_ub = input_ub_buffer.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> output_ub = output_ub_buffer.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> offset_ub = offset_ub_buffer.Get<uint32_t>();

        const bool split_ctb = causal_ctb == 2 || causal_ctb == 4;
        const bool static_offsets = causal_ctb == 3 || causal_ctb == 4;
        const uint32_t channel_count = static_cast<uint32_t>(channels);
        const uint32_t compact_words = channel_count * tap_words;
        if (static_offsets) {
            AscendC::GlobalTensor<uint32_t> static_offset_gm;
            static_offset_gm.SetGlobalBuffer(
                (__gm__ uint32_t *) ctb_gather_offsets_512);
            const AscendC::DataCopyExtParams offset_copy{
                1,
                static_cast<uint32_t>(compact_words * sizeof(uint32_t)),
                0,
                0,
                0,
            };
            const AscendC::DataCopyPadExtParams<uint32_t> no_pad{
                false, 0, 0, 0U,
            };
            AscendC::DataCopyPad(
                offset_ub, static_offset_gm, offset_copy, no_pad);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            for (uint32_t c = 0; c < channel_count; ++c) {
                for (uint32_t tap = 0; tap < tap_words; ++tap) {
                    const uint32_t dst_index = c * tap_words + tap;
                    const uint32_t src_index = tap * channel_count + c;
                    offset_ub.SetValue(dst_index, src_index * sizeof(uint32_t));
                }
            }
            auto offset_event = GetTPipePtr()->FetchEventID(
                AscendC::HardEvent::S_V);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(offset_event);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(offset_event);
        }

        const uint64_t output_units = batch * out_width;
        for (uint64_t bo = block; bo < output_units; bo += blocks) {
            const uint64_t b = bo / out_width;
            const uint64_t ow = bo - b * out_width;

            const AscendC::DataCopyPadExtParams<uint32_t> no_pad{
                false, 0, 0, 0U,
            };
            const uint64_t cache_time = t_size - out_width;
            if (split_ctb && ow < cache_time) {
                const uint32_t cache_words = static_cast<uint32_t>(
                    (cache_time - ow) * channels);
                const uint32_t x_words = compact_words - cache_words;
                const AscendC::DataCopyExtParams cache_copy{
                    1,
                    static_cast<uint32_t>(cache_words * sizeof(uint32_t)),
                    0,
                    0,
                    0,
                };
                const AscendC::DataCopyExtParams x_copy{
                    1,
                    static_cast<uint32_t>(x_words * sizeof(uint32_t)),
                    0,
                    0,
                    0,
                };
                AscendC::DataCopyPad(
                    input_ub,
                    src_aux_gm[(b * cache_time + ow) * channels],
                    cache_copy,
                    no_pad);
                AscendC::DataCopyPad(
                    input_ub[cache_words],
                    src_gm[b * out_width * channels],
                    x_copy,
                    no_pad);
            } else {
                const uint64_t src_time =
                    split_ctb ? ow - cache_time : ow;
                const uint64_t src_width =
                    split_ctb ? out_width : t_size;
                const uint64_t src_offset =
                    (b * src_width + src_time) * channels;
                const AscendC::DataCopyExtParams input_copy{
                    1,
                    static_cast<uint32_t>(compact_words * sizeof(uint32_t)),
                    0,
                    0,
                    0,
                };
                AscendC::DataCopyPad(
                    input_ub, src_gm[src_offset], input_copy, no_pad);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Gather<uint32_t>(
                output_ub, input_ub, offset_ub, 0U, compact_words);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> output_f32 =
                output_ub.ReinterpretCast<float>();
            AscendC::Adds(
                output_f32,
                output_f32,
                0.0f,
                static_cast<int32_t>(compact_words));
            AscendC::PipeBarrier<PIPE_ALL>();

            const uint64_t dst_offset = bo * channels * tap_words;
            const AscendC::DataCopyExtParams output_copy{
                1,
                static_cast<uint32_t>(compact_words * sizeof(uint32_t)),
                0,
                0,
                0,
            };
            AscendC::DataCopyPad(dst_gm[dst_offset], output_ub, output_copy);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        return;
    }

    const bool use_hift_strided =
        causal_ctb == 5 && batch == 1 && kernel == 30 && stride == 15 &&
        padding == 7 && dilation == 1 && channels <= vocoder_channel_tile_max;
    if (use_hift_strided) {
        constexpr uint32_t row_words = 32;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_ub_buffer;
        pipe.InitBuffer(input_ub_buffer, vocoder_input_ub_bytes);
        AscendC::LocalTensor<uint32_t> input_ub =
            input_ub_buffer.Get<uint32_t>();

        for (uint64_t ow = block; ow < out_width; ow += blocks) {
            const int64_t first_t =
                static_cast<int64_t>(ow * stride) - padding;
            const bool interior = first_t >= 0 &&
                static_cast<uint64_t>(first_t) + kernel <= t_size;

            for (uint64_t c0 = 0; c0 < channels;
                 c0 += vocoder_channel_tile_max) {
                const uint64_t remaining = channels - c0;
                const uint32_t tile_channels = static_cast<uint32_t>(
                    remaining < vocoder_channel_tile_max
                        ? remaining
                        : vocoder_channel_tile_max);

                if (interior) {
                    const uint64_t src_offset =
                        c0 * t_size + static_cast<uint64_t>(first_t);
                    const AscendC::DataCopyExtParams input_copy{
                        static_cast<uint16_t>(tile_channels),
                        static_cast<uint32_t>(kernel * sizeof(uint32_t)),
                        static_cast<uint32_t>((t_size - kernel) * sizeof(uint32_t)),
                        0,
                        0,
                    };
                    const AscendC::DataCopyPadExtParams<uint32_t> input_pad{
                        true, 0, 2, 0U,
                    };
                    AscendC::DataCopyPad(
                        input_ub, src_gm[src_offset], input_copy, input_pad);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::LocalTensor<float> input_f32 =
                        input_ub.ReinterpretCast<float>();
                    AscendC::Adds(
                        input_f32,
                        input_f32,
                        0.0f,
                        static_cast<int32_t>(tile_channels * row_words));
                    AscendC::PipeBarrier<PIPE_ALL>();
                } else {
                    for (uint32_t c = 0; c < tile_channels; ++c) {
                        const uint64_t src_base = (c0 + c) * t_size;
                        const uint32_t output_row = c * row_words;
                        for (uint64_t tap = 0; tap < kernel; ++tap) {
                            const int64_t input_t =
                                first_t + static_cast<int64_t>(tap);
                            uint32_t value = 0U;
                            if (input_t >= 0 &&
                                static_cast<uint64_t>(input_t) < t_size) {
                                value = src_gm.GetValue(
                                    src_base + static_cast<uint64_t>(input_t));
                                value = normalize_legacy_f32_word(value);
                            }
                            input_ub.SetValue(output_row + tap, value);
                        }
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                const uint64_t dst_offset =
                    (ow * channels + c0) * kernel;
                const AscendC::DataCopyExtParams output_copy{
                    static_cast<uint16_t>(tile_channels),
                    static_cast<uint32_t>(kernel * sizeof(uint32_t)),
                    0,
                    0,
                    0,
                };
                AscendC::DataCopyPad(
                    dst_gm[dst_offset], input_ub, output_copy);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        return;
    }

    const bool use_dma_ub =
        kernel == tap_words && stride == 1 && padding == 0 && dilation == 1 &&
        t_size >= tap_words &&
        t_size - tap_words <= UINT32_MAX / sizeof(uint32_t);
    if (use_dma_ub) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buffer;
        pipe.InitBuffer(ub_buffer, ub_bytes);
        AscendC::LocalTensor<uint32_t> ub = ub_buffer.Get<uint32_t>();

        const uint64_t output_units = batch * out_width;
        for (uint64_t bo = block; bo < output_units; bo += blocks) {
            const uint64_t b = bo / out_width;
            const uint64_t ow = bo - b * out_width;

            for (uint64_t c0 = 0; c0 < channels; c0 += channel_tile_max) {
                const uint64_t remaining = channels - c0;
                const uint32_t tile_channels = static_cast<uint32_t>(
                    remaining < channel_tile_max ? remaining : channel_tile_max);
                const uint64_t src_offset = (b * channels + c0) * t_size + ow;
                const uint64_t dst_offset = (bo * channels + c0) * tap_words;

                const AscendC::DataCopyExtParams input_copy{
                    static_cast<uint16_t>(tile_channels),
                    tap_words * sizeof(uint32_t),
                    static_cast<uint32_t>((t_size - tap_words) * sizeof(uint32_t)),
                    0,
                    0,
                };
                const AscendC::DataCopyPadExtParams<uint32_t> input_pad{
                    true,
                    0,
                    static_cast<uint8_t>(ub_row_words - tap_words),
                    0U,
                };
                AscendC::DataCopyPad(ub, src_gm[src_offset], input_copy, input_pad);
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::LocalTensor<float> ub_f32 = ub.ReinterpretCast<float>();
                AscendC::Adds(ub_f32, ub_f32, 0.0f,
                    static_cast<int32_t>(tile_channels * ub_row_words));
                AscendC::PipeBarrier<PIPE_ALL>();
                const AscendC::DataCopyExtParams output_copy{
                    static_cast<uint16_t>(tile_channels),
                    tap_words * sizeof(uint32_t),
                    0,
                    0,
                    0,
                };
                AscendC::DataCopyPad(dst_gm[dst_offset], ub, output_copy);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        return;
    }

    const bool use_vocoder_ub_shape =
        batch == 1 && stride == 1 && out_width == t_size &&
        (kernel == 3 || kernel == 7 || kernel == 11) &&
        (dilation == 1 || dilation == 3 || dilation == 5) &&
        padding == static_cast<int64_t>(dilation * (kernel - 1) / 2);
    const uint64_t span_words = use_vocoder_ub_shape
        ? (kernel - 1) * dilation + 1
        : 0;
    const bool use_vocoder_ub =
        use_vocoder_ub_shape && span_words <= vocoder_span_words_max &&
        (span_words > t_size ||
         t_size - span_words <= UINT32_MAX / sizeof(uint32_t));
    if (use_vocoder_ub) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_ub_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_ub_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> pattern_ub_buffer;
        pipe.InitBuffer(input_ub_buffer, vocoder_input_ub_bytes);
        pipe.InitBuffer(output_ub_buffer, vocoder_output_ub_bytes);
        pipe.InitBuffer(pattern_ub_buffer, vocoder_pattern_ub_bytes);
        AscendC::LocalTensor<uint32_t> input_ub = input_ub_buffer.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> output_ub = output_ub_buffer.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> pattern_ub = pattern_ub_buffer.Get<uint32_t>();

        const vocoder_gather_mask_pattern pattern =
            make_vocoder_gather_mask_pattern(
                static_cast<uint32_t>(kernel),
                static_cast<uint32_t>(dilation));
        const uint64_t base_pattern =
            static_cast<uint64_t>(pattern.word0) |
            (static_cast<uint64_t>(pattern.word1) << 32U);
        for (uint32_t local_ow = 0;
             local_ow < vocoder_output_time_tile;
             ++local_ow) {
            const uint64_t shifted_pattern = base_pattern << local_ow;
            const uint32_t pattern_offset = local_ow * 8;
            pattern_ub.SetValue(
                pattern_offset,
                static_cast<uint32_t>(shifted_pattern));
            pattern_ub.SetValue(
                pattern_offset + 1,
                static_cast<uint32_t>(shifted_pattern >> 32U));
        }
        auto pattern_event = GetTPipePtr()->FetchEventID(
            AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(pattern_event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(pattern_event);
        // Long vocoder sequences have heavily overlapping convolution
        // windows. Reuse one aligned GM->UB window for four adjacent output
        // positions; shifted, 32-byte-aligned masks preserve the legacy
        // GatherMask + Adds(+0) byte semantics while reducing MTE2 traffic.
        const bool use_time_tile = causal_ctb == 6 &&
            out_width >= vocoder_output_time_tile * blocks;
        const uint64_t time_tile =
            use_time_tile ? vocoder_output_time_tile : 1;
        const uint64_t work_count =
            (out_width + time_tile - 1) / time_tile;

        for (uint64_t work = block; work < work_count; work += blocks) {
            const uint64_t first_ow = work * time_tile;
            const uint32_t tile_outputs = static_cast<uint32_t>(
                out_width - first_ow < time_tile
                    ? out_width - first_ow
                    : time_tile);
            const int64_t tile_first_t =
                static_cast<int64_t>(first_ow) - padding;
            const uint64_t tile_span_words =
                span_words + tile_outputs - 1;
            const bool tile_interior = tile_first_t >= 0 &&
                static_cast<uint64_t>(tile_first_t) + tile_span_words <=
                    t_size;

            for (uint64_t c0 = 0; c0 < channels;
                 c0 += vocoder_channel_tile_max) {
                const uint64_t remaining = channels - c0;
                const uint32_t tile_channels = static_cast<uint32_t>(
                    remaining < vocoder_channel_tile_max
                        ? remaining
                        : vocoder_channel_tile_max);

                if (tile_interior) {
                    const uint64_t src_offset = c0 * t_size +
                        static_cast<uint64_t>(tile_first_t);
                    const uint32_t padding_words = static_cast<uint32_t>(
                        vocoder_span_row_words - tile_span_words);
                    const AscendC::DataCopyExtParams input_copy{
                        static_cast<uint16_t>(tile_channels),
                        static_cast<uint32_t>(
                            tile_span_words * sizeof(uint32_t)),
                        static_cast<uint32_t>(
                            (t_size - tile_span_words) * sizeof(uint32_t)),
                        static_cast<uint32_t>(
                            padding_words * sizeof(uint32_t) / 32U),
                        0,
                    };
                    const AscendC::DataCopyPadExtParams<uint32_t> input_pad{
                        true,
                        0,
                        static_cast<uint8_t>(padding_words % 8U),
                        0U,
                    };
                    AscendC::DataCopyPad(
                        input_ub, src_gm[src_offset], input_copy, input_pad);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                for (uint32_t local_ow = 0; local_ow < tile_outputs;
                     ++local_ow) {
                    const uint64_t ow = first_ow + local_ow;
                    const int64_t first_t =
                        static_cast<int64_t>(ow) - padding;
                    const bool interior = first_t >= 0 &&
                        static_cast<uint64_t>(first_t) + span_words <=
                            t_size;
                    bool compact_output = false;
                    const uint32_t compact_words =
                        tile_channels * static_cast<uint32_t>(kernel);

                    if (interior) {
                        if (!tile_interior) {
                            const uint64_t src_offset = c0 * t_size +
                                static_cast<uint64_t>(first_t);
                            const uint32_t padding_words =
                                static_cast<uint32_t>(
                                    vocoder_span_row_words - span_words);
                            const AscendC::DataCopyExtParams input_copy{
                                static_cast<uint16_t>(tile_channels),
                                static_cast<uint32_t>(
                                    span_words * sizeof(uint32_t)),
                                static_cast<uint32_t>(
                                    (t_size - span_words) * sizeof(uint32_t)),
                                static_cast<uint32_t>(
                                    padding_words * sizeof(uint32_t) / 32U),
                                0,
                            };
                            const AscendC::DataCopyPadExtParams<uint32_t>
                                input_pad{
                                    true,
                                    0,
                                    static_cast<uint8_t>(
                                        padding_words % 8U),
                                    0U,
                                };
                            AscendC::DataCopyPad(
                                input_ub,
                                src_gm[src_offset],
                                input_copy,
                                input_pad);
                            AscendC::PipeBarrier<PIPE_ALL>();
                        }

                        const AscendC::GatherMaskParams gather_params{
                            1,
                            static_cast<uint16_t>(tile_channels),
                            static_cast<uint16_t>(
                                vocoder_input_row_blocks),
                            0,
                        };
                        uint64_t gathered_words = 0;
                        AscendC::GatherMask<uint32_t, uint32_t>(
                            output_ub,
                            input_ub,
                            pattern_ub[(tile_interior ? local_ow : 0) * 8],
                            true,
                            64,
                            gather_params,
                            gathered_words);
                        AscendC::PipeBarrier<PIPE_V>();

                        AscendC::LocalTensor<float> output_f32 =
                            output_ub.ReinterpretCast<float>();
                        AscendC::Adds(
                            output_f32,
                            output_f32,
                            0.0f,
                            static_cast<int32_t>(compact_words));
                        compact_output = true;
                    } else {
                        for (uint32_t c = 0; c < tile_channels; ++c) {
                            const uint64_t src_base =
                                (c0 + c) * t_size;
                            const uint32_t output_row =
                                c * vocoder_packed_row_words;
                            for (uint64_t tap = 0; tap < kernel; ++tap) {
                                const int64_t input_t = first_t +
                                    static_cast<int64_t>(tap * dilation);
                                uint32_t value = 0U;
                                if (input_t >= 0 &&
                                    static_cast<uint64_t>(input_t) < t_size) {
                                    value = src_gm.GetValue(
                                        src_base +
                                        static_cast<uint64_t>(input_t));
                                    value = normalize_legacy_f32_word(value);
                                }
                                output_ub.SetValue(
                                    output_row +
                                        static_cast<uint32_t>(tap),
                                    value);
                            }
                        }
                    }

                    AscendC::PipeBarrier<PIPE_ALL>();
                    const uint64_t dst_offset =
                        (ow * channels + c0) * kernel;
                    if (compact_output) {
                        const AscendC::DataCopyExtParams output_copy{
                            1,
                            static_cast<uint32_t>(
                                compact_words * sizeof(uint32_t)),
                            0,
                            0,
                            0,
                        };
                        AscendC::DataCopyPad(
                            dst_gm[dst_offset], output_ub, output_copy);
                    } else {
                        const uint32_t block_bytes = static_cast<uint32_t>(
                            kernel * sizeof(uint32_t));
                        const uint32_t aligned_block_bytes =
                            (block_bytes + 31U) & ~UINT32_C(31);
                        const AscendC::DataCopyExtParams output_copy{
                            static_cast<uint16_t>(tile_channels),
                            block_bytes,
                            static_cast<uint32_t>(
                                (vocoder_packed_row_words *
                                     sizeof(uint32_t) -
                                 aligned_block_bytes) /
                                32U),
                            0,
                            0,
                        };
                        AscendC::DataCopyPad(
                            dst_gm[dst_offset], output_ub, output_copy);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        }
        return;
    }

    for (uint64_t line = block; line < line_count; line += blocks) {
        const uint64_t line_begin = line * gm_cache_line_words;
        const uint64_t line_words =
            (dst_words - line_begin < gm_cache_line_words)
                ? dst_words - line_begin
                : gm_cache_line_words;

        for (uint64_t offset = 0; offset < line_words; ++offset) {
            const uint64_t dst_index = line_begin + offset;
            const uint64_t tap = dst_index % kernel;
            const uint64_t unit = dst_index / kernel;
            const uint64_t channel = unit % channels;
            const uint64_t q = unit / channels;
            const uint64_t ow = q % out_width;
            const uint64_t b = q / out_width;
            const int64_t t = static_cast<int64_t>(ow * stride) - padding +
                              static_cast<int64_t>(tap * dilation);
            const uint64_t src_base = (b * channels + channel) * t_size;
            uint32_t value = 0U;
            if (t >= 0 && static_cast<uint64_t>(t) < t_size) {
                value = src_gm.GetValue(src_base + static_cast<uint64_t>(t));
                value = normalize_legacy_f32_word(value);
            }
            dst_gm.SetValue(dst_index, value);
        }
    }
}
