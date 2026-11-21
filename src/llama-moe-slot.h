#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

#include <memory>

struct llama_model;

class llama_moe_slot_runtime {
public:
    explicit llama_moe_slot_runtime(const llama_model & model);
    ~llama_moe_slot_runtime();

    bool uses_layer(int layer) const;
    bool lookahead_prefetch_enabled() const;
    int  lookahead_prefetch_depth() const;
    /// 每层 decode 窗口宽度（slot 数）。0 = 未启用按层窗口。
    int  decode_window() const;
    /// 窗口覆盖的最大批宽（行数）。1 = 只服务单 token decode；>1 时投机验证
    /// （DSpark，批宽 = n_max+1）这类小批次也走窗口。
    ///
    /// **必须与 process() 的 window_mode 用同一个数**：它还决定建图时用哪套 slot
    /// 张量（每层 `.dec` 视图 vs prefill 共享视图），见 llama-context 的
    /// `moe_decode_phase`。两处判据不一致 → 张量所指的槽段与实际写入的槽段错位。
    int  window_max_rows() const;
    static void map_custom_deferred(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
                                    int ith, int nth, void * userdata);
    // 0 = off (no next-layer predict / prefetch). 1..16 = predicted experts per layer.
    // Safe between llama_decode calls; in-flight prefetch is drained first.
    void set_lookahead_prefetch_depth(int depth);
    void set_lookahead_pred(ggml_tensor * pred_ids);
    /// For a layer without routed experts (a leading dense layer) whose next layer has them:
    /// a CPU op that hands `pred` (target_layer's predicted expert ids) to lookahead prefetch.
    /// Expand the returned tensor into the graph; nullptr when lookahead is off.
    ggml_tensor * build_lookahead_only(ggml_context * ctx, ggml_tensor * pred, int target_layer);
    /// Per-context state for an architecture's trained lookahead predictor (minicpm5moe
    /// current_prev keeps the previous token's ffn_norm here). It lives in this runtime, which
    /// is one per llama_context, so two contexts on one model never share it. Allocated in
    /// `buft` and zero-filled on first use; later calls must ask for the same shape.
    ggml_tensor * predictor_state(int64_t ne0, int64_t ne1, ggml_backend_buffer_type_t buft);
    /// Ubatches this context has run with the predictor state; its parity picks the bank written.
    uint64_t & predictor_step();
    ggml_tensor * build_slot_ids_tensor(
        ggml_context * ctx,
        ggml_tensor * expert_ids,
        int layer);
    void process(int layer, const ggml_tensor * expert_ids, ggml_tensor * slot_ids,
                 const ggml_tensor * lookahead_pred = nullptr);

    /// 累加「图执行期间的专家管理耗时」（`map_custom` 回调内）。
    ///
    /// 为什么单独量它：`process()` 是 CPU 自定义 op，跑在图执行中间，
    /// 每层一次、每 token 40 次，且前后都要 GPU/CPU 同步。它占总时间的比例
    /// 决定了「缓存在端上为什么不下速度」—— 是读得慢，还是这段结构本身贵。
    void add_process_ns(int64_t ns);

    /// 诊断计数快照（累计值，纳秒）。调用方在**一次前向**的前后各取一次做差，
    /// 就得到这次前向的分解，不必依赖 flush 的节流窗口去反推。
    ///
    /// 为什么要把 process() 再拆开：实测 1 行步的「非 I/O 记账」是 3.5 ms，
    /// 3 行步却涨到 19.5 ms（5.5×），而 I/O 只涨 1.72×。要定位这笔异常，
    /// 必须知道它是花在「回读 expert_ids」「读盘」还是「写回 slot_ids」上。
    struct counters {
        int64_t process;    // 整个 process() 回调（= 下列各项 + 其余）
        int64_t readids;    // read_i32_ids：从计算后端回读 expert_ids
        int64_t install;    // install_pending：读盘 + 写进 GPU 可见的槽位
        int64_t writeids;   // write_i32_ids：把 slot_ids 写回计算后端
        int64_t miss;       // 本次前向真正 miss 的 (行, 专家) 次数（各层累计）
        int64_t bytes;      // 本次前向真正读到的字节（各层累计）
        int64_t unique;     // 本次前向各层专家**并集**大小之和（= 实际要做的 matvec 次数）
    };
    counters counters_snapshot() const;

    /// A2 phase B：执行被延后的专家读取（见 process_deferred 实现）。
    void process_deferred(int layer);
    /// 建 A2 的 phase B 节点：输出是 `slot_ids` 的副本，同时依赖 `dep`
    /// （前段专家的 matmul 输出）—— 保证它排在前段算完之后执行。
    ggml_tensor * build_deferred_trigger(ggml_context * ctx, ggml_tensor * slot_ids,
                                         ggml_tensor * dep, int layer);
    /// 末尾保留多少个位置用于「延后读取」。0 = 关闭 A2。
    int  matmul_split() const;

    /// 一轮完整生成结束后拆掉专家缓存的 Metal 包装，丢掉宿主物理页。
    /// 下次 process() / llama_decode 会自动再包。页内容作废，必须清 LRU。
    void park_expert_memory();
    void ensure_unparked();

    /// 在 prefill 之前、空闲期把静态预热表读进各层 decode 窗口（见 LLAMA_MOE_PREWARM）。
    /// decode 段与 prefill 段物理分离，所以这笔读能活过 prefill，阶段切换只重挂映射。
    /// 调用期间不得有 decode 在飞（App：mmproj 卸载后、prefill 前）。
    void prewarm_experts();

private:
    static void map_custom_lookahead_only(ggml_tensor * dst, const ggml_tensor * pred, int ith, int nth, void * userdata);
    static void map_custom(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata);
    static void map_custom_lookahead(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata);

    struct impl;
    std::unique_ptr<impl> pimpl;
};
