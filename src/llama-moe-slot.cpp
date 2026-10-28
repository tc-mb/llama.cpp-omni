#include "llama-moe-slot.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-experts.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace {
// Parallel pread threads per layer wave.
//
// This path is bandwidth-bound (measured: on-device decode time tracks total bytes
// read, not critical-path bytes), so whether the storage can serve more concurrent
// reads decides whether raising this helps at all. It is a tunable rather than a
// constant so it can be swept on device without a rebuild -- set
// LLAMA_MOE_IO_THREADS. Defaults to 4, the value all earlier measurements used.
constexpr int kMoeSsdIoThreadsDefault = 4;

static int moe_ssd_io_threads() {
    static const int n = []() {
        const char * s = std::getenv("LLAMA_MOE_IO_THREADS");
        const int v = s ? std::atoi(s) : 0;
        return (v > 0 && v <= 64) ? v : kMoeSsdIoThreadsDefault;
    }();
    return n;
}

// slot_to_expert 的哨兵值。
//   kSlotFree    -1：空位，可分配
//   kSlotPending -2：已被预读 worker **认领**、内容尚未写完。主路径必须跳过它
//                    （既不能覆盖，也不能当成有效缓存条目）。
constexpr int32_t kSlotFree    = -1;
constexpr int32_t kSlotPending = -2;

// One I/O wave per layer. A layer's unique set never exceeds n_slots (<= 256), so this is
// always a single wave: the wave loop only adds a join + thread respawn per boundary, and
// the disk idles across it. Measured 2026-09-12 (paired A/B, 8/8 reps positive): 13 waves ->
// 1 wave is worth +3.65% prefill (261.9 -> 271.2 t/s), at zero memory cost.
// Kept as a named constant (rather than deleting the loop) so the cap stays explicit.
constexpr int kMoeSsdIoChunk = 256;

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Xcode 能抓住 stderr（和 [MBMtmd] 一样）。llama-cli 的 callback 会吞掉纯 LLAMA_LOG_INFO。
static void moe_log(const char * fmt, ...) {
    char buf[768];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[moe-ssd] %s\n", buf);
    fflush(stderr);
    LLAMA_LOG_INFO("[moe-ssd] %s\n", buf);
}

static bool park_window_enabled() {
    const char * e = std::getenv("LLAMA_MOE_PARK_WINDOW");
    if (!e || !*e) {
        return true;
    }
    return e[0] != '0';
}

using metal_buf_fn = bool (*)(ggml_backend_buffer_t);

static metal_buf_fn lookup_metal_buf_fn(ggml_backend_buffer_t buf, const char * name) {
    if (!buf) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return nullptr;
    }
    return (metal_buf_fn) ggml_backend_reg_get_proc_address(reg, name);
}
} // namespace

// Next-layer router × current hidden (attn_post_norm). Decode only. LLAMA_MOE_LOOKAHEAD_PREFETCH=1.
// Predicted ids are preadv'd into a ping-pong tail of the shared bank after install, overlapping
// the current layer's GPU work. Hits remap for free; misses still install on the critical path.
struct lookahead_expert_prefetch {
    static constexpr int32_t kMaxDepth = 16;

    struct hit {
        int32_t expert;
        int32_t slot;
    };

    bool enabled = false;
    int32_t depth = 8;
    /// 真正**读盘**的专家数；0 = 只算预测、不读盘（把预测用来指导淘汰）。
    /// 由 LLAMA_MOE_PREFETCH_READ 控制，默认 = depth（旧行为）。
    int32_t read_n = 0;
    int32_t n_slots = 0;
    int32_t n_expert = 0;
    int32_t last_moe_layer = -1;

    int32_t slot_floor() const { return n_slots - 2 * depth; }
    int32_t bank_base(int k) const { return slot_floor() + depth * (k & 1); }

    std::thread th;
    std::mutex mtx;
    std::condition_variable cv_go;
    std::condition_variable cv_done;
    bool quit = false;
    bool job_pending = false;
    bool job_running = false;
    int32_t job_layer = -1;
    int32_t job_bank = -1;
    int32_t job_n = 0;
    int32_t job_expert[kMaxDepth] = {};
    int32_t arm_layer = -1;
    int32_t arm_bank = -1;
    int32_t arm_n = 0;
    int32_t arm_expert[kMaxDepth] = {};

    std::function<void(int, int, int)> read_fn;
    // 窗口模式下 slot 由窗口 LRU 决定，不由乒乓区算出来 —— 此时传给 read_fn 的
    // slot 用 -1 作哨兵（表示「你按窗口自己挑」）。见 impl 的那个 lambda。
    bool window_mode = false;
    std::atomic<uint64_t> n_hits{0};
    std::atomic<uint64_t> n_armed{0};
    // 诊断：预读到底有没有在工作。见 arm()/take()。
    std::atomic<uint64_t> n_arm_skipped{0};   // 因上一个 job 未完成而放弃的次数
    std::atomic<uint64_t> n_waits{0};         // take() 里真的阻塞等待的次数
    std::atomic<int64_t>  n_wait_ns{0};       // 阻塞等待的总时长
    std::atomic<uint64_t> n_pred_read{0};     // worker 实际读进去的专家数
    std::atomic<uint64_t> n_pred_published{0};// 预读发布进窗口（真正变成缓存条目）的次数
    std::atomic<uint64_t> n_predict_only{0};  // 只预测不读盘的次数（轻量组合模式）
    std::atomic<uint64_t> n_evict_saved{0};   // 因「预测会用」而被跳过的淘汰次数
    std::atomic<uint64_t> n_evict_forced{0};  // pass0 找不到非预测候选、被迫进入 pass1 的次数
    std::atomic<uint64_t> n_pred_cached{0};   // arm 时因「目标层已有」而被过滤掉的预测数
    std::atomic<uint64_t> n_used_from_pf{0};  // 主路径用到的专家里，落在**预读区**的次数
                                              // （用完不保留，本层结束就被重填冲掉）
    std::atomic<uint64_t> n_evict_total{0};   // 淘汰总次数
    std::atomic<uint64_t> n_pred_bytes{0};    // worker 实际读进去的字节
    bool ready = false;

    void init(int32_t slots, int32_t experts, int32_t last_moe, std::function<void(int, int, int)> fn) {
        n_slots = slots;
        n_expert = experts;
        last_moe_layer = last_moe;
        read_fn = std::move(fn);
        ready = true;

        int want = 0;
        const char * on = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH");
        if (on && on[0] != '\0' && on[0] != '0') {
            want = 8;
            if (const char * d = std::getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")) {
                const int v = atoi(d);
                if (v >= 0 && v <= kMaxDepth) {
                    want = v;
                }
            }
        } else {
            // the mechanism was called FATE before this rename; keep the old spellings loud
            if (std::getenv("LLAMA_MOE_FATE") || std::getenv("LLAMA_MOE_FATE_DEPTH")) {
                moe_log("lookahead-prefetch: off, and LLAMA_MOE_FATE* is no longer read; "
                        "use LLAMA_MOE_LOOKAHEAD_PREFETCH[=_DEPTH]");
            } else {
                moe_log("lookahead-prefetch: off (LLAMA_MOE_LOOKAHEAD_PREFETCH=%s)", on ? on : "unset");
            }
        }
        {
            const char * rs = std::getenv("LLAMA_MOE_PREFETCH_READ");
            read_n = rs ? std::atoi(rs) : -1;   // -1 = 跟随 depth
        }
        set_depth(want);
        if (read_n < 0 || read_n > depth) {
            read_n = depth;                     // 默认与 depth 一致（旧行为）
        }
    }

    void set_depth(int32_t want) {
        if (!ready) {
            return;
        }
        if (want < 0) {
            want = 0;
        }
        if (want > kMaxDepth) {
            want = kMaxDepth;
        }
        if (want > 0 && n_slots < 2 * want + 8) {
            moe_log("lookahead-prefetch: off, bank %d too small for depth %d", (int) n_slots, (int) want);
            want = 0;
        }
        const int32_t cur = enabled ? depth : 0;
        if (want == cur && (want == 0 || th.joinable())) {
            return;
        }

        {
            std::unique_lock<std::mutex> lk(mtx);
            if (job_running) {
                cv_done.wait(lk, [this]() { return !job_running; });
            }
            job_pending = false;
            arm_layer = -1;
            arm_n = 0;
            if (want > 0) {
                depth = want;
                enabled = true;
            } else {
                enabled = false;
            }
        }

        if (enabled && !th.joinable()) {
            quit = false;
            th = std::thread([this]() { worker(); });
        }

        if (enabled) {
            moe_log("lookahead-prefetch: on depth=%d reserved=[%d,%d)", (int) depth, (int) slot_floor(), (int) n_slots);
        } else {
            moe_log("lookahead-prefetch: off (depth=0)");
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            quit = true;
        }
        cv_go.notify_all();
        if (th.joinable()) {
            th.join();
        }
        enabled = false;
    }

    void drain() {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]() { return !job_running && !job_pending; });
        arm_layer = -1;
        arm_n = 0;
    }

    void report() const {
        if (n_armed.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const uint64_t armed = n_armed.load(std::memory_order_relaxed);
        const uint64_t hits = n_hits.load(std::memory_order_relaxed);
        const uint64_t preds = armed * (uint64_t) depth;
        moe_log("lookahead-prefetch: armed=%llu hits=%llu (%.1f%% of predictions)",
            (unsigned long long) armed,
            (unsigned long long) hits,
            preds ? 100.0 * (double) hits / (double) preds : 0.0);
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mtx);
        arm_layer = -1;
        arm_n = 0;
    }

    void take(int32_t layer, std::vector<hit> & out) {
        if (!enabled) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);

        if (window_mode) {
            // 窗口模式：worker 与缓存**共用同一片 slot**，但**不需要等它**。
            //
            // 为什么可以不等（这是「缓存 + 预读结合」的关键）：
            //   1. worker 认领 slot 时标 `kSlotPending`，主路径分配/淘汰都跳过它 ——
            //      既不会覆盖 worker 正在写的位置，也不会把半成品当成有效条目；
            //   2. 主路径要的专家如果 worker 已经**发布**完成，`expert_to_slot` 里就有，
            //      自然算命中；没发布就按未命中自己读（最多重复读一次，代价有界）；
            //   3. worker 读盘时**不持锁**（见 install_prefetch_into_window），
            //      所以不会把主线程挡在 mutex 上。
            //
            // 以前这里无条件等整批 K 个读完，实测每 token 白等 43ms ——
            // 那才是「缓存与预读互斥」的真正原因，不是两者本质冲突。
            arm_layer = -1;
            arm_n = 0;
            return;
        }

        // 乒乓模式：worker 写尾部专用区 [slot_floor, n_slots)，主路径只用
        // [0, slot_floor)（见 process() 里 alloc_hi = slot_floor），两者**不重叠**，
        // 所以不需要等 worker。
        //
        // 以前这里无条件 cv_done.wait()，等的是**整批 K 个专家**（含主路径根本
        // 不需要的那些）。实测代价很大：K=8 时每 token 白等 54.6ms、K=16 时 259ms
        // —— 比整步还长，直接把预读的收益吃光。
        // 现在改成：**只接受已经完成、且正是本层**的预读结果；没好的就跳过，
        // 主路径按未命中正常读（本来也要读）。
        if (job_running) {
            // 结果还没好。arm_layer 仍是上一次的值，下面的一致性检查会拦住它。
            return;
        }
        if (arm_layer != layer || arm_n <= 0) {
            arm_layer = -1;
            arm_n = 0;
            return;
        }
        const int32_t base = bank_base(arm_bank);
        for (int k = 0; k < arm_n; ++k) {
            out.push_back({ arm_expert[k], base + k });
        }
        arm_layer = -1;
        arm_n = 0;
    }

    void arm(int32_t layer_next, const std::vector<int32_t> & pred) {
        if (!enabled || layer_next < 1 || layer_next > last_moe_layer || pred.empty()) {
            return;
        }

        // read_n == 0：只要预测、不读盘。这是「缓存 + 预测」的轻量组合形态。
        // 预测本身由调用方（impl::process）存下来供淘汰参考。
        if (read_n <= 0) {
            n_predict_only.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        std::lock_guard<std::mutex> lk(mtx);
        if (job_pending || job_running) {
            // 上一次预读还没跑完 -> 这次直接放弃。读比算慢时这里会大量命中，
            // 预读的收益就基本归零了。量出来才知道它是不是瓶颈。
            n_arm_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        job_layer = layer_next;
        job_bank = layer_next & 1;
        job_n = 0;
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        for (int32_t e : pred) {
            if (job_n >= read_n) {   // 只读 read_n 个（<= depth）；read_n==0 已在上面返回
                break;
            }
            if (e < 0 || e >= n_expert || seen[(size_t) e]) {
                continue;
            }
            seen[(size_t) e] = 1;
            job_expert[job_n++] = e;
        }
        if (job_n <= 0) {
            return;
        }
        for (int k = job_n; k < kMaxDepth; ++k) {
            job_expert[k] = -1;
        }
        job_pending = true;
        n_armed.fetch_add(1, std::memory_order_relaxed);
        cv_go.notify_one();
    }

    void worker() {
        for (;;) {
            int32_t layer = -1;
            int32_t bank = -1;
            int32_t n = 0;
            int32_t ex[kMaxDepth];
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_go.wait(lk, [this]() { return quit || job_pending; });
                if (quit) {
                    return;
                }
                job_pending = false;
                job_running = true;
                layer = job_layer;
                bank = job_bank;
                n = job_n;
                memcpy(ex, job_expert, sizeof(ex));
            }
            const int32_t base = bank_base(bank);
            const int n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
            std::atomic<int> next{0};
            auto read_one = [&]() {
                for (;;) {
                    const int k = next.fetch_add(1, std::memory_order_relaxed);
                    if (k >= n) {
                        break;
                    }
                    if (ex[k] < 0) {
                        continue;
                    }
                    try {
                        // 窗口模式传 -1：让 impl 按目标层的窗口 LRU 挑 slot。
                        // 预读写进**目标层自己的窗口**，而不是尾部乒乓区 ——
                        // 这样预读成果直接变成那层的缓存命中，不需要再搬一次。
                        read_fn(layer, ex[k], window_mode ? -1 : (base + k));
                        n_pred_read.fetch_add(1, std::memory_order_relaxed);
                        n_pred_published.fetch_add(1, std::memory_order_relaxed);
                    } catch (...) {
                        ex[k] = -1;
                    }
                }
            };
            if (n_th == 1) {
                read_one();
            } else {
                std::vector<std::thread> helpers;
                helpers.reserve((size_t) n_th);
                for (int t = 0; t < n_th; ++t) {
                    helpers.emplace_back(read_one);
                }
                for (auto & h : helpers) {
                    h.join();
                }
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                arm_layer = layer;
                arm_bank = bank;
                arm_n = n;
                memcpy(arm_expert, ex, sizeof(arm_expert));
                job_running = false;
            }
            cv_done.notify_all();
        }
    }
};

struct llama_moe_slot_runtime::impl {
    struct binding {
        ggml_tensor * tensor = nullptr;
        const llama_moe_experts_entry * entry = nullptr;
    };

    // 一次读取要用的「打包视图」：prefill 和 decode 各有自己的一套张量与文件布局。
    // 抽出来是为了让 read_expert_directv 能按阶段选，不用复制读取逻辑。
    struct pack_view {
        const std::vector<binding> * bindings = nullptr;
        const std::vector<size_t>  * order    = nullptr;
        size_t base = 0;
        size_t stride = 0;
        std::string path;
        bool ok = false;

        bool valid() const { return ok && bindings && order; }
    };

    struct layer_state {
        bool enabled = false;
        bool pack_ok = false;
        int32_t window_start = 0;   // 本层 decode 窗口的起始 slot（0 表示未启用窗口）
        /// 本层窗口的**总宽度**（缓存区 c_il + 预读暂存区 K）。各层可不同 ——
        /// 见 llama_model::moe_layer_windows()：热层给大、冷层给小。
        /// window_start 恒为 0（每层有自己的 decode 张量，slot 空间就是 [0, width)）。
        int32_t window_width = 0;
        size_t pack_base = 0;
        size_t pack_bytes = 0;
        size_t pack_stride = 0;
        std::string pack_path;
        std::vector<binding> bindings;
        std::vector<size_t> pack_order; // bindings sorted by file offset, i.e. experts packed order

        // decode 专用的一套（每层独立窗口，ne[2] = c）。prefill 用上面那套。
        bool dec_pack_ok = false;
        size_t dec_pack_base = 0;
        size_t dec_pack_stride = 0;
        std::string dec_pack_path;
        std::vector<binding> dec_bindings;
        std::vector<size_t> dec_pack_order;

        /// 预读成功发布进本窗口的次数（诊断用；与 n_hits 的区别是后者统计的是
        /// 主路径命中，这里统计 worker 的产出）。
        uint64_t n_prefetch_published = 0;
        std::vector<int32_t> slot_to_expert;
        std::vector<int32_t> expert_to_slot;
        std::vector<uint64_t> slot_age;
        uint64_t age = 0;
        std::mutex mutex;
    };
    struct map_userdata {
        llama_moe_slot_runtime * runtime = nullptr;
        int layer = -1;
    };

    struct pending_install {
        int32_t expert = -1;
        int32_t slot = -1;
    };

    struct install_timing {
        int n = 0;
        int n_th = 0;
        bool pack = false;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        int64_t pread_max_ns = 0;
        size_t bytes = 0;
    };

    // 「合并相邻专家读」可行性统计。见 accumulate_runs() 的注释。
    struct run_stats {
        int n          = 0;   // 连续段总数（仅按 expert id 连续划分）
        int experts    = 0;   // 落在段里的专家数（应等于 miss 总数）
        int max        = 0;   // 最长段
        int ok_n       = 0;   // 段内 slot 也连号的段数（可零拷贝合并）
        int ok_ex      = 0;   // 这些段覆盖的专家数
        int scatter_n  = 0;   // slot 不连号、需 scratch 中转的段数
        int scatter_ex = 0;
    };

    struct ubatch_acc {
        int n_tokens = 0;
        int n_used = 0;
        int n_layers = 0;
        int unique_sum = 0;
        int miss_sum = 0;
        int n_th = 0;
        int pack_layers = 0;
        int64_t wall_ns = 0;
        int64_t pread_sum_ns = 0;
        size_t bytes = 0;
        run_stats runs;
    };

    struct io_stats {
        std::atomic<uint64_t> n_miss{0};
        std::atomic<uint64_t> n_bytes{0};
        std::atomic<int64_t>  pread_ns{0};
    };

    /// 每层「下一步预测会用到的专家」列表，由上一层 arm() 写入、本层 process() 消费。
    /// 用途是**零 I/O 成本**地改进淘汰：预测会用的条目不该被换出去。
    std::vector<std::vector<int32_t>> predicted;
    std::mutex pred_mtx;

    // 整个进程的完整 I/O 计量（主路径 + 预读），见 readv_span。
    std::atomic<uint64_t> io_bytes_all{0};
    std::atomic<uint64_t> io_calls_all{0};
    uint64_t last_io_bytes = 0;
    uint64_t last_io_calls = 0;
    // 「图执行期间的专家管理耗时」累计，见 add_process_ns。
    std::atomic<int64_t> process_ns{0};
    // 诊断：把 process() 内部再拆三层，用来定位「记账 3.5 → 19.5 ms」那笔异常。
    // 三者的和是 process_ns 的子集，差值就是其余部分（分配、淘汰、统计等）。
    std::atomic<int64_t> readids_ns{0};
    std::atomic<int64_t> install_ns{0};
    std::atomic<int64_t> writeids_ns{0};
    // 诊断：各层「用到的专家并集」大小之和（= 图里真正要做的 (行,专家) matvec 次数）。
    // 用来区分 ε 是「按并集大小（kernel 次数）」还是「按字节」增长。
    std::atomic<int64_t> unique_sum{0};
    // 诊断：逐层 miss 累计（只统计 decode 步）。用来判断「每层均匀 1.x 个」还是
    // 「少数层很多」—— 前者意味着 40 次串行等待，后者只需解决几层。
    std::vector<uint64_t> miss_per_layer;
    /// miss 在 selected_experts（按分数降序）里的**位置**直方图。
    /// 用来判断能否用「固定尾部预留」代替排序：
    /// 若 miss 集中在末尾几位，就把末尾几位划成第二组，求和顺序不变（可保逐字节一致）。
    /// A2：每层 MoE 的 n_expert_used 里，**末尾**这么多位置允许延后读取。
    ///
    /// 动机：miss 高度集中在分数顺序的末尾（实测 0:0.03 ... 7:0.22）。把这些位置
    /// 的缺失专家延后到「前面那些已驻留专家算完之后」再读，读就与 GPU 计算重叠了。
    /// 固定按位置切（而不是按是否驻留排序切）能保持求和顺序不变 → 输出逐字节一致。
    int32_t matmul_split = 0;

    /// 每层被延后读取的 (expert, slot)，由 phase A 写入、phase B 消费。
    std::vector<std::vector<pending_install>> deferred;

    uint64_t miss_pos[16] = {};
    uint64_t miss_pos_steps = 0;
    uint64_t miss_per_layer_steps = 0;
    int64_t last_process_ns = 0;
    uint64_t last_flush_idx = 0;
    // 预读诊断字段的增量基线，见 flush_ubatch。
    uint64_t last_arm_skipped = 0;
    uint64_t last_wait_count = 0;
    int64_t  last_wait_ns = 0;
    uint64_t last_pred_read = 0;
    uint64_t last_pf_hits = 0;
    uint64_t last_pf_pub = 0;

    explicit impl(const llama_model & model) :
        model(model),
        n_expert((int32_t) model.hparams.n_expert),
        n_slots(model.moe_slot_bank_size()),
        layers(model.layers.size()),
        predicted(model.layers.size()),
        deferred(model.layers.size()),
        miss_per_layer(model.layers.size(), 0),
        userdata(model.layers.size()) {
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto & state = layers[il];
            const auto & layer = model.layers[il];

            bind(state, layer.ffn_gate_up_exps, n_slots);
            bind(state, layer.ffn_gate_exps,    n_slots);
            bind(state, layer.ffn_up_exps,      n_slots);
            bind(state, layer.ffn_down_exps,    n_slots);

            if (state.bindings.empty()) {
                continue;
            }
            if (!layer.ffn_down_exps ||
                (!layer.ffn_gate_up_exps && (!layer.ffn_gate_exps || !layer.ffn_up_exps))) {
                throw std::runtime_error(format("incomplete MoE slot bindings for layer %zu", il));
            }

            state.enabled = true;
            finalize_pack(state);
            state.slot_to_expert.assign(n_slots, kSlotFree);
            state.expert_to_slot.assign(n_expert, kSlotFree);
            state.slot_age.assign(n_slots, 0);
            // window_start 不在这里算 —— decode_window 要到 resolve_decode_window()
            // 才确定，此处还是 0。在那边统一赋值，否则所有层会共用窗口起点 [0, W)，
            // 层间互相覆盖（曾经的实际 bug）。
            userdata[il] = { nullptr, (int) il };
            n_enabled++;
            last_moe_layer = (int) il;
            if (state.pack_ok) {
                n_pack_ok++;
            }
            for (const auto & b : state.bindings) {
                if (b.tensor && b.tensor->view_src) {
                    shared_bank = true;
                    break;
                }
            }
        }
        size_t stride0 = 0;
        if (last_moe_layer >= 0) {
            stride0 = layers[(size_t) last_moe_layer].pack_bytes;
            for (const auto & layer : layers) {
                if (layer.enabled && layer.pack_bytes > 0) {
                    stride0 = layer.pack_bytes;
                    break;
                }
            }
        }
        // Experts loading preadv's the packed expert span straight into its slot, which requires both a
        // packed layout and CPU-addressable slot tensors. Do not guess either, and do not silently fall
        // back to an upload path: private Metal buffers (non-unified memory) expose tensor->data as a
        // virtual address that is not writable from the CPU.
        for (const auto & layer : layers) {
            if (layer.enabled && !layer.pack_ok) {
                throw std::runtime_error(
                    "MoE experts layer is not packed; preadv loading requires gate/up/down contiguous per expert");
            }
            for (const auto & b : layer.bindings) {
                if (!b.tensor->buffer || !ggml_backend_buffer_is_host_addressable(b.tensor->buffer)) {
                    throw std::runtime_error(format(
                        "MoE slot tensor '%s' is not CPU-addressable; preadv loading requires host-visible slots",
                        ggml_get_name(b.tensor)));
                }
            }
        }

        moe_log("init pack=%s pack_layers=%d/%d bank=%d shared=%s experts=%d io_threads=%d stride0=%zu preadv=on park_window=%s",
            n_pack_ok == n_enabled && n_enabled > 0 ? "on" : (n_pack_ok > 0 ? "partial" : "off"),
            n_pack_ok, n_enabled, (int) n_slots, shared_bank ? "on" : "off",
            (int) n_expert, moe_ssd_io_threads(), stride0,
            park_window_enabled() ? "on" : "off");

        lookahead.init((int32_t) n_slots, n_expert, last_moe_layer,
            [this](int layer, int expert, int slot) {
                if (slot < 0) {
                    install_prefetch_into_window(layer, expert);
                } else {
                    read_expert_directv(layers[(size_t) layer], expert, slot);
                }
            });
        resolve_decode_window();
        {
            const char * e = std::getenv("LLAMA_MOE_MATMUL_SPLIT");
            matmul_split = e ? std::atoi(e) : 0;
            if (matmul_split > 0) {
                moe_log("matmul-split: 末尾 %d 个位置的缺失专家延后读取（读与算重叠）",
                        (int) matmul_split);
            }
        }

        // decode 窗口那一套绑定要等 resolve_decode_window() —— 它依赖 shared_bank
        // 和 last_moe_layer，两者都在上面的层循环里才确定。
        if (decode_window > 0) {
            // 模型侧每层张量的宽度是 c_il + K（缓存区 + 预读暂存区），**按层不同**。
            // 见 qwen35moe.cpp 与 install_prefetch_into_window 的分区说明。
            int n_dec = 0;
            std::string fail;
            for (size_t il = 0; il < model.layers.size() && fail.empty(); ++il) {
                auto & state = layers[il];
                if (!state.enabled) {
                    continue;
                }
                const auto & layer = model.layers[il];
                const std::pair<ggml_tensor *, const char *> fam[4] = {
                    { layer.ffn_gate_up_exps_dec, "gate_up" },
                    { layer.ffn_gate_exps_dec,    "gate"    },
                    { layer.ffn_up_exps_dec,      "up"      },
                    { layer.ffn_down_exps_dec,    "down"    },
                };
                for (int k = 0; k < 4 && fail.empty(); ++k) {
                    if (!fam[k].first) {
                        continue;
                    }
                    try {
                        const int32_t expect = layer_width[il] > 0 ? layer_width[il]
                                                                   : decode_window + prefetch_slots;
                        bind(state, fam[k].first, expect, &state.dec_bindings);
                    } catch (const std::exception & e) {
                        ggml_tensor * t = fam[k].first;
                        fail = format("layer %zu family %s: name=%s ne2=%lld nbytes=%zu "
                                      "win=%d :: %s",
                                      il, fam[k].second, ggml_get_name(t),
                                      (long long) t->ne[2], ggml_nbytes(t),
                                      (int) decode_window, e.what());
                    }
                }
                if (fail.empty() && state.dec_bindings.empty()) {
                    fail = format("layer %zu has no decode tensors", il);
                }
                if (fail.empty()) {
                    finalize_pack(state.dec_bindings, state.dec_pack_order, state.dec_pack_ok,
                                  state.dec_pack_base, state.dec_pack_stride, state.dec_pack_path);
                    n_dec++;
                }
            }
            if (!fail.empty()) {
                // 绑不上就退回共享 bank（那是正确路径），但必须**大声**报出来 ——
                // 静默回退会让人以为窗口生效了，测出来的数是假的。
                moe_log("decode-cache: 绑定失败，回退到共享 bank（窗口关闭）。原因: %s",
                        fail.c_str());
                for (auto & st : layers) {
                    st.dec_bindings.clear();
                    st.dec_pack_order.clear();
                    st.dec_pack_ok = false;
                }
                decode_window = 0;
            } else {
                moe_log("decode-cache: %d 层绑定 decode 张量（每层 %d = cache %d + prefetch %d slot，pack %s）",
                        n_dec, (int) (decode_window + prefetch_slots), (int) decode_window,
                        (int) prefetch_slots, layers[0].dec_pack_ok ? "ok" : "off");
            }
        }
        sweep_init();
        prewarm_init();
    }

    /// 预读专用：把一个 expert 读进**目标层自己的窗口**（按窗口 LRU 挑 slot）。
    ///
    /// 与 process() 的安装路径共用同一套 slot 选择规则，所以预读的成果对那层来说
    /// 就是一次普通的缓存命中 —— 不需要乒乓区、也不需要事后搬运。
    ///
    /// 调用时机保证安全：预读在层 L-1 的 process() 里 arm、在层 L 的 process()
    /// 开头 take() 处等它结束，所以 worker 动层 L 的窗口时，层 L 没有在跑。
    /// 清空一层缓存的映射表。
    ///
    /// 保留 `kSlotPending`：那是预读 worker 已认领、正在写的 slot。正常路径上
    /// 清空只发生在 prefill→decode 边界（那时没有 worker 在飞），但保留判断更安全
    /// —— 否则会把 worker 的位置标成空闲，主路径分配过去就撞车了。
    static void clear_layer_cache(layer_state & st) {
        for (auto & e : st.slot_to_expert) {
            if (e != kSlotPending) {
                e = kSlotFree;
            }
        }
        // expert_to_slot 不能整体清：pending 的专家此时还没登记，所以直接全清是安全的。
        std::fill(st.expert_to_slot.begin(), st.expert_to_slot.end(), kSlotFree);
    }

    /// 预读：把一个预测专家读进**目标层自己的窗口**，与缓存共用同一片 slot 池。
    ///
    /// 关键设计：**锁内只做「认领 slot」，读盘在锁外，读完再锁内「发布」**。
    ///
    /// 为什么必须这样：以前这个函数从头到尾持着 `st.mutex` 读盘（每次读 ~1ms），
    /// 而主线程的 `process()` 要拿同一把锁 —— 于是主线程被 worker 的读盘挡住，
    /// 实测每 token 白等 43ms。**这才是「缓存与预读互斥」的真正原因**，
    /// 不是两者本质上冲突。
    ///
    /// 认领期间该 slot 标成 `kSlotPending`：主路径分配/淘汰都会跳过它，
    /// 所以主线程既不会覆盖它、也不会误用未写完的内容。
    /// 认领一个槽位并标 `kSlotPending`（内容尚未写入）。**调用方必须已持 st.mutex**。
    ///
    /// `cache_only` 决定从哪一段挑：
    ///   true  → 只在**缓存区** [window_start, window_start + c_il) 里挑（预热用；
    ///           预热的是「会跨 token 复用」的专家，本就该进缓存区）
    ///   false → 在**预读区**里挑（预读用；统一模式下就是整层窗口）
    ///
    /// 返回 -1 = 该专家已在窗口里，或无处可放。
    int32_t claim_slot_in_window(layer_state & st, int32_t expert, bool cache_only) {
        if (decode_window <= 0 || expert < 0 || expert >= n_expert) {
            return -1;
        }
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            return -1;   // 已经在（缓存区或预读区）—— 白读，跳过
        }
        const int32_t w_il = st.window_width > 0 ? st.window_width : (decode_window + prefetch_slots);
        const int32_t c_il = std::max<int32_t>(1, w_il - prefetch_slots);
        const int32_t lo = st.window_start;
        const int32_t hi = cache_only ? st.window_start + c_il
                                      : st.window_start + w_il;
        const int32_t from = cache_only ? lo
                                        : (prefetch_unified ? lo : st.window_start + c_il);

        int32_t slot = -1;
        for (int32_t cand = from; cand < hi; ++cand) {
            if (st.slot_to_expert[(size_t) cand] == kSlotFree) {
                slot = cand;
                break;
            }
        }
        if (slot < 0) {
            uint64_t oldest = UINT64_MAX;
            for (int32_t cand = from; cand < hi; ++cand) {
                if (st.slot_to_expert[(size_t) cand] == kSlotPending) {
                    continue;
                }
                if (st.slot_age[(size_t) cand] < oldest) {
                    oldest = st.slot_age[(size_t) cand];
                    slot = cand;
                }
            }
        }
        if (slot < 0) {
            return -1;
        }
        const int32_t evicted = st.slot_to_expert[(size_t) slot];
        if (evicted >= 0) {
            st.expert_to_slot[(size_t) evicted] = kSlotFree;
        }
        st.slot_to_expert[(size_t) slot] = kSlotPending;   // 认领，内容尚未写入
        return slot;
    }

    /// 读完后登记为有效缓存条目。**调用方必须已持 st.mutex**。
    /// 返回是否真的登记（专家在读盘期间被主路径抢先装好时不登记，丢弃这份）。
    bool publish_slot_in_window(layer_state & st, int32_t expert, int32_t slot) {
        // 读盘期间主路径可能已经把同一个专家装好了（它自己读的），那就丢弃这份。
        if (st.expert_to_slot[(size_t) expert] >= 0) {
            st.slot_to_expert[(size_t) slot] = kSlotFree;
            return false;
        }
        st.slot_to_expert[(size_t) slot] = expert;
        st.expert_to_slot[(size_t) expert] = slot;
        st.slot_age[(size_t) slot] = ++st.age;
        st.n_prefetch_published++;
        return true;
    }

    void install_prefetch_into_window(int layer, int32_t expert) {
        if (layer < 0 || layer >= (int) layers.size() || decode_window <= 0 || prefetch_slots <= 0) {
            return;
        }
        auto & st = layers[(size_t) layer];
        if (!st.enabled || expert < 0 || expert >= n_expert) {
            return;
        }

        // 只在**预读暂存区** [window_start + c, window_start + c + K) 里分配。
        // 这一块每步重填，不参与跨 token 的 LRU 淘汰 —— 所以预读**不会**把缓存的
        // 条目换出去。这是「缓存 + 预读」并集覆盖的前提：
        //   预测到的专家已在缓存区 -> 白读（跳过，见下面的检查）
        //   不在缓存区 -> 读进暂存区，本层用时不落关键路径
        // 统一模式下预读可写在整层窗口里（见 prefetch_unified 的说明）。
        int32_t slot = -1;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            slot = claim_slot_in_window(st, expert, /*cache_only=*/false);
        }
        if (slot < 0) {
            return;
        }

        // 锁外读盘：这一步不再阻塞主线程。
        read_expert_directv(st, expert, slot, true);

        std::lock_guard<std::mutex> lock(st.mutex);
        publish_slot_in_window(st, expert, slot);
    }

    /// ── 缓存预热（LLAMA_MOE_PREWARM）───────────────────────────────────
    ///
    /// 每轮生成开始时各层窗口都是空的（进窗口模式时整表清空 / park 后又清）。
    /// 预热把一张**静态表**（每层若干专家 id）读进各层 decode 窗口，让第一步不再是
    /// 「全 miss」。
    ///
    /// 关键：decode 窗口与 prefill 段**物理分离**（见 qwen35moe.cpp 的 base 布局），
    /// 所以只要在 prefill 之前把数据读进去，prefill 不会覆盖它，阶段切换时只需
    /// **重新发布映射**（`publish_prewarm_placed`，纯整数操作），不必重读盘。
    /// 这就是 App 侧「mmproj 卸载后、prefill 之前」调用 `prewarm_experts()` 的意义：
    /// 那笔读完全离开首 token 的关键路径。
    ///
    /// 三段式（与 install_prefetch_into_window 同构，但批量做）：
    ///   1. 认领：逐层持锁，给每个专家在**缓存区**找空位 / 淘汰最老的；
    ///   2. 读盘：不持任何层锁，用 io_threads 个线程并行读；
    ///   3. 发布：逐层持锁，并记下落位（层,专家,slot）供阶段切换重挂。
    ///
    /// `cur_layer` 的锁**已被调用方（process）持有**时传该层号；从公开 API 调用
    /// （此时不持任何层锁）传 -1。`std::mutex` 不可重入，再锁会自死锁。
    void prewarm_windows(int cur_layer) {
        if (!prewarm_enabled || decode_window <= 0) {
            return;
        }
        const int64_t t0 = now_ns();
        struct flat { int layer; int32_t expert; int32_t slot; };

        std::vector<std::vector<std::pair<int32_t, int32_t>>> tasks(layers.size());
        size_t n_req = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (il >= prewarm_table.size() || prewarm_table[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            if (!st.enabled) {
                continue;
            }
            n_req += prewarm_table[il].size();
            auto claim_all = [&]() {
                for (int32_t e : prewarm_table[il]) {
                    const int32_t s = claim_slot_in_window(st, e, /*cache_only=*/true);
                    if (s >= 0) {
                        tasks[il].push_back({ e, s });
                    }
                }
            };
            if ((int) il == cur_layer) {
                claim_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                claim_all();
            }
        }

        std::vector<flat> q;
        for (size_t il = 0; il < tasks.size(); ++il) {
            for (const auto & t : tasks[il]) {
                q.push_back({ (int) il, t.first, t.second });
            }
        }
        const int n = (int) q.size();
        const int n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        std::atomic<int> next{0};
        std::atomic<int> n_read{0};
        std::atomic<int64_t> n_bytes{0};
        auto read_one = [&]() {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) {
                    break;
                }
                auto & st = layers[(size_t) q[i].layer];
                try {
                    read_expert_directv(st, q[i].expert, q[i].slot, true);
                    n_read.fetch_add(1, std::memory_order_relaxed);
                    n_bytes.fetch_add((int64_t) st.pack_bytes, std::memory_order_relaxed);
                } catch (...) {
                    q[i].slot = -1;
                }
            }
        };
        if (n > 0) {
            if (n_th == 1) {
                read_one();
            } else {
                std::vector<std::thread> th;
                th.reserve((size_t) n_th);
                for (int t = 0; t < n_th; ++t) {
                    th.emplace_back(read_one);
                }
                for (auto & h : th) {
                    h.join();
                }
            }
        }

        size_t n_pub = 0;
        std::vector<prewarm_place> placed;
        placed.reserve((size_t) n);
        for (size_t il = 0; il < tasks.size(); ++il) {
            if (tasks[il].empty()) {
                continue;
            }
            auto & st = layers[il];
            auto pub_all = [&]() {
                for (const auto & t : tasks[il]) {
                    if (t.second >= 0 && publish_slot_in_window(st, t.first, t.second)) {
                        placed.push_back({ (int32_t) il, t.first, t.second });
                        n_pub++;
                    }
                }
            };
            if ((int) il == cur_layer) {
                pub_all();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                pub_all();
            }
        }
        // 记下落位：阶段切换（prefill → decode）时用它们「重新发布」映射，不必重读盘。
        prewarm_placed = std::move(placed);
        prewarm_ready = !prewarm_placed.empty();

        moe_log("prewarm: 表 %zu 专家 -> 认领 %d / 读入 %d / 发布 %zu，%.1f MiB，wall=%.1f ms "
                "(%d 线程，缓存区)",
                n_req, n, (int) n_read.load(), n_pub,
                (double) n_bytes.load() / (1024.0 * 1024.0), ms_of(now_ns() - t0), n_th);
    }

    /// 阶段切换（prefill → decode）时用预热落位记录**重新发布**映射。
    ///
    /// 数据本身还在（decode 段与 prefill 段物理分离），只是切换时映射表被清了 ——
    /// 所以这里纯整数操作、不读盘。`cur_layer` 的锁已由调用方持有（同 prewarm_windows）。
    void publish_prewarm_placed(int cur_layer) {
        for (const auto & p : prewarm_placed) {
            if (p.layer < 0 || p.layer >= (int) layers.size()) {
                continue;
            }
            auto & st = layers[(size_t) p.layer];
            if (!st.enabled) {
                continue;
            }
            if (p.expert < 0 || p.expert >= n_expert ||
                p.slot < 0 || p.slot >= (int32_t) st.slot_to_expert.size()) {
                continue;
            }
            auto do_pub = [&]() {
                // 刚清过映射（全部 kSlotFree），这里直接登记。
                // 若该 slot/专家已被占用（理论上不会），跳过以免制造冲突。
                if (st.slot_to_expert[(size_t) p.slot] == kSlotFree &&
                    st.expert_to_slot[(size_t) p.expert] < 0) {
                    st.slot_to_expert[(size_t) p.slot] = p.expert;
                    st.expert_to_slot[(size_t) p.expert] = p.slot;
                    st.slot_age[(size_t) p.slot] = ++st.age;
                }
            };
            if ((int) p.layer == cur_layer) {
                do_pub();
            } else {
                std::lock_guard<std::mutex> lk(st.mutex);
                do_pub();
            }
        }
    }

    /// 公开入口（`llama_moe_prewarm_experts`）：把静态表读进各层 decode 窗口并登记映射。
    /// 在 prefill 之前、空闲期调用（App：mmproj 卸载后）。调用期间不得有 decode 在飞。
    void prewarm_experts() {
        if (!prewarm_enabled) {
            return;
        }
        prewarm_windows(-1);   // -1 = 不持任何层锁，全部自取
    }

    /// 决定 decode 缓存的窗口宽度，并把结果打出来。
    ///
    /// 为什么 prefill 用不了窗口：实测 prefill 一层要用 105~215 个专家（`unique/l`），
    /// 而窗口只有 bank/40 个 slot，根本装不下。所以 prefill 必须保持「整块 bank + 逐层清空」，
    /// decode 才切窗口 —— 两者是天然分离的阶段，在 `process()` 里按 `n_tokens == 1` 切换。
    void resolve_decode_window() {
        const char * s = std::getenv("LLAMA_MOE_DECODE_WINDOW");
        if (!s || !*s) {
            moe_log("decode-cache: off（设 LLAMA_MOE_DECODE_WINDOW 开启，留空=按 bank 自动取最大）");
            return;
        }
        if (!shared_bank) {
            moe_log("decode-cache: off（仅共享 bank 模式支持按层窗口）");
            return;
        }
        // 窗口宽度以**模型侧**为唯一来源（`moe_decode_window()`）。
        // 模型加载时按它建了张量布局，运行时若自己另算一遍就会对不上 ——
        // 而且旧的上限（bank / 层数）属于共享张量方案，现在不再成立：
        // 每层有自己的张量，c 只受 RAM 和 ne[2] <= 1024 约束。
        const int32_t want = model.moe_decode_window();
        if (want < 1) {
            moe_log("decode-cache: off（模型侧未开启窗口）");
            return;
        }
        decode_window = want;
        // 预读暂存区大小 = 预读深度（K）。没开预读就是 0，等于退回「只有缓存区」。
        prefetch_slots = lookahead.enabled ? std::max(0, (int32_t) lookahead.depth) : 0;
        {
            // 默认统一 LRU：预读与主路径共用整个窗口 (c+K)。
            // `LLAMA_MOE_PREFETCH_UNIFIED=0` 退回分区模式（只作 A/B 用）。
            const char * u = std::getenv("LLAMA_MOE_PREFETCH_UNIFIED");
            prefetch_unified = !(u && *u && *u == '0');
            moe_log("decode-cache: %s（窗口 %d slot，K=%d）",
                    prefetch_unified ? "预读与缓存共用一个 LRU 窗口" : "预读/缓存分区",
                    (int) (decode_window + prefetch_slots), (int) prefetch_slots);
        }

        // 各层窗口起点**不再是 il*c** —— 现在每层有自己的 decode 张量
        // （base 上的 view），它的 slot 空间就是 [0, 宽度)，所以起点恒为 0。
        // 这一点让运行时更简单：层间天然不冲突。
        //
        // **宽度按层**：模型侧已按 LLAMA_MOE_LAYER_WINDOWS 建好布局（base 容量是
        // 各层宽度之和），这里必须用**同一份**数据，否则张量对不上。
        {
            const std::vector<int32_t> per_layer_c = model.moe_layer_windows();
            layer_width.assign(layers.size(), 0);
            for (size_t il = 0; il < layers.size(); ++il) {
                const int32_t c_il = il < per_layer_c.size() ? per_layer_c[il] : decode_window;
                const int32_t w = c_il + prefetch_slots;
                layers[il].window_start = 0;
                layers[il].window_width = w;
                layer_width[il] = w;
            }
            // 打印一下分布，确认按层生效
            std::string dist;
            for (size_t il = 0; il < layer_width.size() && il < 6; ++il) {
                dist += format("%zu:%d ", il, (int) layer_width[il]);
            }
            moe_log("decode-cache: 按层宽度 %s（前几层 %s）",
                    (getenv("LLAMA_MOE_LAYER_WINDOWS") ? "on" : "off（均匀）"), dist.c_str());
        }

        // ── 窗口覆盖多大的批宽 ──────────────────────────────────────────────
        //
        // 单 token decode 之外，投机验证（DSpark）也会送进批宽 = n_max+1 的小批次。
        // 让它们走窗口，吃的是**跨轮**的专家复用（连续解码同一段文本，路由高度重合）。
        // 上限由**容量**决定：必须保证「该批宽在最坏情况下的专家并集」装得进缓存区。
        //   最坏并集 = min(行数 × top-k, n_expert)——每行最多 top-k 个，全库只有 n_expert 个。
        //
        // 取**各层里最窄的那个**来夹：窗口可逐层不同（LLAMA_MOE_LAYER_WINDOWS），
        // 夹紧必须对最窄的层也成立，否则那一层会在 process() 里抛异常。
        {
            const char * rows_env = std::getenv("LLAMA_MOE_WINDOW_MAX_ROWS");
            const int32_t rows_want = std::max(1, rows_env && *rows_env ? atoi(rows_env) : 1);

            int32_t cache_min = 0;
            for (size_t il = 0; il < layer_width.size() && il < layers.size(); ++il) {
                if (!layers[il].enabled || layer_width[il] <= 0) {
                    continue;
                }
                const int32_t usable = layer_width[il] -
                    (prefetch_unified ? 0 : prefetch_slots);
                if (usable > 0 && (cache_min == 0 || usable < cache_min)) {
                    cache_min = usable;
                }
            }
            if (cache_min <= 0) {
                cache_min = decode_window;
            }

            const int32_t n_used = (int32_t) model.hparams.n_expert_used;
            int32_t rows = rows_want;
            while (rows > 1 && cache_min < std::min(rows * n_used, n_expert)) {
                rows--;
            }
            window_max_rows = rows;

            if (rows != rows_want) {
                moe_log("decode-cache: 缓存区只有 %d 格，装不下 %d 行的最坏并集（%d），"
                        "窗口适用的批宽从 %d 夹到 %d",
                        (int) cache_min, (int) rows_want,
                        (int) std::min(rows_want * n_used, n_expert), (int) rows_want, (int) rows);
            }
            moe_log("decode-cache: 窗口覆盖批宽 1..%d%s", (int) window_max_rows,
                    window_max_rows > 1 ? "（投机验证也走窗口）" : "（只服务单 token decode）");
        }

        // 预读**保留**，并改成写进目标层自己的窗口（见 install_prefetch_into_window）。
        //
        // 为什么不关掉它：预读的字节是在上一层计算期间并发读的，只要
        // `预读字节 / 带宽 <= 单层计算时间` 就完全被隐藏 —— 那部分是白赚的。
        // 真正加在耗时上的是**关键路径**字节（卡住下一层开始的那些）。
        // 缓存的另一个作用正是把预读从「超出预算、部分暴露」压回「完全隐藏」：
        //   无缓存 depth 8: 预读 14.2 MiB/层 → 8.2ms > 6ms 预算  ❌
        //   缓存 24 + K=8 : 预读 6.6 MiB/层 → 3.8ms < 6ms 预算  ✅
        // 所以两者是互补的，不该二选一。
        if (lookahead.enabled && !lookahead.window_mode) {
            lookahead.window_mode = true;
            moe_log("decode-cache: 预读保留，改为写进目标层窗口（depth=%d）",
                    (int) lookahead.depth);
        }

        const int32_t n_moe_layers = model.moe_experts_layer_count();
        const size_t total = (size_t) decode_window * (size_t) n_moe_layers;
        moe_log("decode-cache: on cache=%d + prefetch=%d slots/layer × %d 层 = %zu slots "
                "(%.0f MiB，与 %d-slot 的 prefill 共享 bank 同处一块 base)",
                (int) decode_window, (int) prefetch_slots, (int) n_moe_layers, total,
                (double) total * (double) layers[0].pack_bytes / 1024.0 / 1024.0,
                (int) n_slots);
    }

    ~impl() {
        if (cur.n_layers > 0) {
            flush_ubatch(cur.n_tokens <= 1 ? "decode" : "prefill",
                cur.n_tokens <= 1 ? n_decode_step : n_prefill_ubatch);
        }
        lookahead.report();
        if (miss_pos_steps > 0) {
            std::string s;
            for (int i = 0; i < 16; ++i) {
                if (miss_pos[i] == 0 && i >= 8) break;
                s += format("%d:%.2f ", i, (double) miss_pos[i] / (double) miss_pos_steps);
            }
            moe_log("miss position (score order, avg over %llu steps): %s",
                    (unsigned long long) miss_pos_steps, s.c_str());
        }
        if (miss_per_layer_steps > 0 && decode_window > 0) {
            // 逐层 miss 分布：判断「每层都有几个」还是「少数层很多」。
            std::string s;
            for (size_t il = 0; il < miss_per_layer.size(); ++il) {
                if (miss_per_layer[il] == 0) continue;
                s += format("%zu:%.2f ", il, (double) miss_per_layer[il] / (double) miss_per_layer_steps);
            }
            moe_log("miss/layer (avg over %llu steps): %s",
                    (unsigned long long) miss_per_layer_steps, s.c_str());
        }
        lookahead.stop();
        log_io("final");
        report_merge();
        if (sweep_fp) {
            fclose(sweep_fp);
            sweep_fp = nullptr;
        }
        for (const auto & item : fds) {
#ifdef _WIN32
            _close(item.second);
#else
            close(item.second);
#endif
        }
    }

    void bind(layer_state & state, ggml_tensor * tensor, int32_t expect_slots,
              std::vector<binding> * out = nullptr) {
        if (!tensor) {
            return;
        }
        const auto * entry = model.moe_experts_entry(ggml_get_name(tensor));
        if (!entry) {
            throw std::runtime_error(format("missing experts entry for tensor '%s'", ggml_get_name(tensor)));
        }
        if (tensor->ne[2] != expect_slots ||
            ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) expect_slots) {
            throw std::runtime_error(format("invalid virtual slot tensor '%s'", ggml_get_name(tensor)));
        }
        (out ? *out : state.bindings).push_back({ tensor, entry });
    }

    using pack_out = layer_state;

    static void finalize_pack(std::vector<binding> & bindings, std::vector<size_t> & order,
                              bool & ok, size_t & base, size_t & stride, std::string & path) {
        ok = false;
        if (bindings.empty()) {
            return;
        }
        const std::string & p = bindings[0].entry->path;
        const size_t st = bindings[0].entry->expert_stride;
        if (st == 0) {
            return;
        }
        size_t min_off = (size_t) -1;
        size_t max_end = 0;
        for (const auto & b : bindings) {
            if (b.entry->path != p || b.entry->expert_stride != st) {
                return;
            }
            min_off = std::min(min_off, b.entry->offset);
            max_end = std::max(max_end, b.entry->offset + b.entry->bytes_per_expert);
        }
        if (min_off == (size_t) -1 || max_end <= min_off || (max_end - min_off) > st) {
            return;
        }
        for (const auto & b : bindings) {
            if (b.entry->offset < min_off) {
                return;
            }
            if ((b.entry->offset - min_off) + b.entry->bytes_per_expert > st) {
                return;
            }
        }
        ok = true;
        base = min_off;
        stride = st;
        path = p;

        order.resize(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return bindings[a].entry->offset < bindings[b].entry->offset;
        });
    }

    static void finalize_pack(layer_state & state) {
        finalize_pack(state.bindings, state.pack_order, state.pack_ok,
                      state.pack_base, state.pack_stride, state.pack_path);
        state.pack_bytes = state.pack_stride;
    }

    /// LLAMA_MOE_F_NOCACHE：experts 读是否绕过 page cache。
    ///   未设 / "0" → 全程不绕（默认）
    ///   "auto"     → 按阶段（prefill 绕、decode 不绕）
    ///   "1"        → 全程绕过
    static int nocache_mode() {
        static const int m = [] {
            const char * e = std::getenv("LLAMA_MOE_F_NOCACHE");
            if (!e || !*e || std::strcmp(e, "0") == 0) {
                return 0;
            }
            return std::strcmp(e, "auto") == 0 ? -1 : 1;
        }();
        return m;
    }

    bool experts_nocache_desired() const {
        const int m = nocache_mode();
        return m < 0 ? want_nocache.load(std::memory_order_relaxed) : m != 0;
    }

    /// 把已打开的 experts fd 的 F_NOCACHE 拨到 `on`。只改缓存行为，不动数据。
    void apply_experts_nocache(bool on) {
#if defined(__APPLE__) && defined(F_NOCACHE)
        std::lock_guard<std::mutex> lock(fd_mutex);
        for (const auto & kv : fds) {
            if (fcntl(kv.second, F_NOCACHE, on ? 1 : 0) != 0) {
                moe_log("F_NOCACHE 切换失败: %s", std::strerror(errno));
            }
        }
        // 只在每个方向各打一次，避免长会话刷屏。
        static bool logged[2] = { false, false };
        if (!logged[on ? 0 : 1]) {
            logged[on ? 0 : 1] = true;
            moe_log("experts 读%s page cache（%s 阶段）", on ? "绕过" : "经过",
                    on ? "prefill" : "decode");
        }
#else
        // F_NOCACHE is Apple-specific; other platforms keep normal cached I/O.
        (void) on;
#endif
    }

    int fd_for(const std::string & path) {
        std::lock_guard<std::mutex> lock(fd_mutex);
        const auto found = fds.find(path);
        if (found != fds.end()) {
            return found->second;
        }
#ifdef _WIN32
        const int fd = _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
#if defined(__APPLE__) && defined(F_NOCACHE)
        if (experts_nocache_desired() && fcntl(fd, F_NOCACHE, 1) != 0) {
            moe_log("F_NOCACHE 设置失败: %s", std::strerror(errno));
        }
#endif
#endif
        if (fd < 0) {
            throw std::runtime_error(format("failed to open MoE experts file '%s': %s", path.c_str(), std::strerror(errno)));
        }
        fds.emplace(path, fd);
        return fd;
    }

    // read one contiguous file span into several disjoint buffers, adjusting the iovecs on partial reads
    void readv_span(int fd, size_t offset, struct iovec * iov, int niov) {
        {
            size_t n = 0;
            for (int i = 0; i < niov; ++i) {
                n += iov[i].iov_len;
            }
            // 唯一能给出「完整 I/O」的埋点：主路径和预读 worker 都走这里。
            // 只看 decode 行里的 bytes= 会漏掉预读读的字节（实测那是大头）。
            io_bytes_all.fetch_add(n, std::memory_order_relaxed);
            io_calls_all.fetch_add(1, std::memory_order_relaxed);
        }
        size_t total = 0;
        for (int i = 0; i < niov; ++i) {
            total += iov[i].iov_len;
        }
        size_t done = 0;
        while (done < total) {
            struct iovec cur[8];
            int cn = 0;
            size_t skip = done;
            for (int i = 0; i < niov; ++i) {
                if (skip >= iov[i].iov_len) {
                    skip -= iov[i].iov_len;
                    continue;
                }
                cur[cn].iov_base = (uint8_t *) iov[i].iov_base + skip;
                cur[cn].iov_len  = iov[i].iov_len - skip;
                skip = 0;
                cn++;
            }
#ifdef _WIN32
            std::lock_guard<std::mutex> lock(io_mutex);
            if (_lseeki64(fd, (int64_t) (offset + done), SEEK_SET) < 0) {
                throw std::runtime_error("failed to seek MoE experts");
            }
            const ssize_t result = _read(fd, cur[0].iov_base, (unsigned int) cur[0].iov_len);
#else
            const ssize_t result = preadv(fd, cur, cn, (off_t) (offset + done));
#endif
            if (result < 0 && errno == EINTR) {
                continue;
            }
            if (result <= 0) {
                throw std::runtime_error(format(
                    "failed to read MoE experts at offset %zu", offset + done));
            }
            done += (size_t) result;
        }
    }

    // Reads the packed gate|up|down span (one contiguous run in the experts) with a single preadv,
    // writing each family straight into its own slot. One syscall, one write per byte: no staging
    // buffer and no second copy into the device buffer.
    int64_t read_expert_directv(const layer_state & state, int32_t expert, int32_t slot,
                                bool window_mode = false) {
        // 诊断：LLAMA_MOE_SKIP_READ=1 时走完所有记账但不真读盘。
        //
        // 目的是把「CPU 段 + 每层往返」的成本和「真读盘」的成本分开：
        //   step(真实) - step(空读) = I/O 落在关键路径上的部分
        //   step(空读) - 常驻基线     = 结构开销（40 次 GPU/CPU 往返 + 调度）
        // 输出会是错的（专家内容是垃圾），只用来看时间。
        // 诊断三态，用来把「读」与「写进 GPU 缓冲区」分开：
        //   0（默认）正常：preadv 直接写进 slot 张量（GPU 可见）
        //   1 dummy   ：preadv 读进一块 scratch —— 同样的 I/O，但不碰 GPU 缓冲区
        //   2 skip    ：完全不读
        // 判定：若 0 >> 1 ≈ 2 → 成本在「写 GPU 缓冲区」（缓存行失效）；
        //       若 0 ≈ 1 >> 2 → 单纯是 I/O 本身。
        static const int mode = [] {
            const char * d = std::getenv("LLAMA_MOE_DUMMY_READ");
            if (d && *d && *d != '0') return 1;
            const char * s = std::getenv("LLAMA_MOE_SKIP_READ");
            if (s && *s && *s != '0') return 2;
            return 0;
        }();

        // decode 阶段读进**本层自己的窗口张量**（base 上的 view），prefill 读进共享那段。
        // 两套的张量、文件偏移、步长都不同，所以整组一起切。
        const std::vector<binding> & bindings = window_mode ? state.dec_bindings   : state.bindings;
        const std::vector<size_t>  & order    = window_mode ? state.dec_pack_order : state.pack_order;
        const std::string          & path     = window_mode ? state.dec_pack_path : state.pack_path;
        const size_t base   = window_mode ? state.dec_pack_base   : state.pack_base;
        const size_t stride = window_mode ? state.dec_pack_stride : state.pack_stride;

        struct iovec iov[8];
        const int n = (int) order.size();
        if (n <= 0 || n > 8) {
            throw std::runtime_error("invalid MoE packed family count for preadv");
        }
        size_t nbytes = 0;
        size_t max_exp = 0;
        for (int i = 0; i < n; ++i) {
            max_exp = std::max(max_exp, bindings[order[(size_t) i]].entry->bytes_per_expert);
        }
        if (mode == 1) {
            // dummy：读进一块常驻 scratch（不碰 GPU 缓冲区）。同样是真读盘，
            // 所以 I/O 量与正常模式一致，差别只有「数据落到哪里」。
            static std::vector<uint8_t> scratch;
            if (scratch.size() < max_exp) {
                scratch.resize(max_exp);
            }
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = scratch.data();
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const binding & b = bindings[order[(size_t) i]];
                iov[i].iov_base = (uint8_t *) b.tensor->data + (size_t) slot * b.entry->bytes_per_expert;
                iov[i].iov_len  = b.entry->bytes_per_expert;
                nbytes += b.entry->bytes_per_expert;
            }
        }
        const int fd = fd_for(path);
        const size_t offset = base + (size_t) expert * stride;

        if (mode == 2) {
            // 记账照做（让上层以为读成功了），但不落盘 —— 用于隔离 I/O 与结构开销。
            stats.n_miss.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        // LLAMA_MOE_READ_DELAY_US：给每次读加固定延迟。
        // 用来判断「读时间是否 1:1 落在关键路径上」—— 若总时间按 delay × 读次数
        // 等量增长，说明读在关键路径上是线性的；若增长更多，说明有放大效应。
        static const int delay_us = [] {
            const char * d = std::getenv("LLAMA_MOE_READ_DELAY_US");
            return d ? std::atoi(d) : 0;
        }();
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }

        const int64_t t0 = now_ns();
        readv_span(fd, offset, iov, n);
        const int64_t dt = now_ns() - t0;
        stats.pread_ns.fetch_add(dt, std::memory_order_relaxed);
        stats.n_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        stats.n_miss.fetch_add(1, std::memory_order_relaxed);
        return dt;
    }

    static const uint8_t * host_data(const ggml_tensor * tensor) {
        if (!tensor || !tensor->data || (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer))) {
            return nullptr;
        }
        return (const uint8_t *) tensor->data;
    }

    static uint8_t * host_data(ggml_tensor * tensor) {
        return const_cast<uint8_t *>(host_data((const ggml_tensor *) tensor));
    }

    void read_i32_ids(const ggml_tensor * tensor, std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        ids.resize((size_t) (n_ids * n_tokens));
        if (const uint8_t * src = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(ids.data() + token * n_ids, src + (size_t) token * tensor->nb[1], row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_get(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    void write_i32_ids(ggml_tensor * tensor, const std::vector<int32_t> & ids) const {
        const int64_t n_ids = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t row_bytes = (size_t) n_ids * sizeof(int32_t);
        if (ids.size() != (size_t) (n_ids * n_tokens)) {
            throw std::runtime_error("MoE slot id count does not match tensor shape");
        }
        if (uint8_t * dst = host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                memcpy(dst + (size_t) token * tensor->nb[1], ids.data() + token * n_ids, row_bytes);
            }
            return;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_set(
                tensor,
                ids.data() + token * n_ids,
                (size_t) token * tensor->nb[1],
                row_bytes);
        }
    }

    // 统计 pending 的「连续段」结构，用来判断「合并相邻专家的读」是否可行。
    //
    // 背景：experts 是 expert-major，expert k 的字节在文件里位于
    // `base + k * expert_stride`，所以**专家 id 连号 == 文件地址连号**。
    // 而 pending 是按 expert id 升序构建的，因此天然带着可用于合并的连续段。
    //
    // 两类可合并：
    //   1. slot 也连号 → 可以直接用一组 iovec 覆盖整段，**零拷贝**、syscall 数减到 1/段长
    //   2. slot 不连号 → 得先读进 scratch 再散写到各 slot，多一次内存拷贝
    //
    // 共享 bank 的 prefill 里 LRU 每层清空、slot 从 0 顺序分配，而 pending 又是按
    // expert 升序压入的，所以 1 类应该占绝大多数。这个统计就是来验证这一点的。
    static run_stats accumulate_runs(const std::vector<pending_install> & pending) {
        run_stats st;
        const size_t n = pending.size();
        if (n == 0) {
            return st;
        }
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;   // 只按 expert 连号扩展，slot 单独判
            while (j < n && pending[j].expert == pending[j - 1].expert + 1) {
                ++j;
            }
            const int run_len = (int) (j - i);

            bool slots_consecutive = true;
            for (size_t k = i + 1; k < j; ++k) {
                if (pending[k].slot != pending[k - 1].slot + 1) {
                    slots_consecutive = false;
                    break;
                }
            }

            st.n++;
            st.experts += run_len;
            st.max = std::max(st.max, run_len);
            if (slots_consecutive) {
                st.ok_n++;
                st.ok_ex += run_len;
            } else {
                st.scatter_n++;
                st.scatter_ex += run_len;
            }
            i = j;
        }
        return st;
    }

    static void add_runs(run_stats & dst, const run_stats & src) {
        dst.n          += src.n;
        dst.experts    += src.experts;
        dst.max         = std::max(dst.max, src.max);
        dst.ok_n       += src.ok_n;
        dst.ok_ex      += src.ok_ex;
        dst.scatter_n  += src.scatter_n;
        dst.scatter_ex += src.scatter_ex;
    }

    install_timing install_pending(layer_state & state, const std::vector<pending_install> & pending,
                                   bool window_mode) {
        install_timing total;
        const int n = (int) pending.size();
        if (n <= 0) {
            return total;
        }
        for (int offset = 0; offset < n; offset += kMoeSsdIoChunk) {
            const int n_wave = std::min(kMoeSsdIoChunk, n - offset);
            std::vector<pending_install> wave(
                pending.begin() + offset, pending.begin() + offset + n_wave);
            const install_timing t = install_packed_wave(state, wave, window_mode);
            total.n += t.n;
            total.n_th = std::max(total.n_th, t.n_th);
            total.pack = true;
            total.wall_ns += t.wall_ns;
            total.pread_sum_ns += t.pread_sum_ns;
            total.pread_max_ns = std::max(total.pread_max_ns, t.pread_max_ns);
            total.bytes += t.bytes;
        }
        return total;
    }

    install_timing install_packed_wave(layer_state & state, const std::vector<pending_install> & pending,
                                       bool window_mode) {
        install_timing t;
        const int n = (int) pending.size();
        if (n <= 0) {
            return t;
        }

        t.n = n;
        t.pack = true;
        t.bytes = (size_t) n * state.pack_bytes;
        std::vector<std::string> errs((size_t) n);
        std::vector<int64_t> dts((size_t) n, 0);
        auto read_one = [&](int i) {
            try {
                dts[(size_t) i] = read_expert_directv(
                    state, pending[(size_t) i].expert, pending[(size_t) i].slot, window_mode);
            } catch (const std::exception & e) {
                errs[(size_t) i] = e.what();
            }
        };

        t.n_th = std::max(1, std::min(moe_ssd_io_threads(), n));
        const int64_t wall0 = now_ns();
        // 注意：单线程不能只调 read_one(0) —— 那只会装第一个专家，
        // 其余 N-1 个被静默跳过，slot 里留着上一层/上一个 token 的旧权重，
        // 模型输出会错但不报错。这条路径只在 io_threads==1 时进入，
        // 默认 4 所以长期潜伏；把它做成可调后才会踩到。
        // 单线程还有个副作用：pending 是按 expert id 升序构建的，而 experts 是
        // expert-major（offset = base + expert * stride），所以单线程等于**严格
        // 顺序读文件**，可能接近顺序读速率（实测比 4 线程的随机交织快）。
        if (t.n_th == 1) {
            for (int i = 0; i < n; ++i) {
                read_one(i);
            }
        } else {
            std::atomic<int> next{0};
            std::vector<std::thread> threads;
            threads.reserve((size_t) t.n_th);
            for (int th = 0; th < t.n_th; ++th) {
                threads.emplace_back([&]() {
                    for (;;) {
                        const int i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= n) {
                            break;
                        }
                        read_one(i);
                    }
                });
            }
            for (auto & th : threads) {
                th.join();
            }
        }
        t.wall_ns = now_ns() - wall0;
        for (int i = 0; i < n; ++i) {
            if (!errs[(size_t) i].empty()) {
                throw std::runtime_error(errs[(size_t) i]);
            }
            t.pread_sum_ns += dts[(size_t) i];
            t.pread_max_ns = std::max(t.pread_max_ns, dts[(size_t) i]);
        }
        return t;
    }

    static double ms_of(int64_t ns) {
        return ns / 1e6;
    }

    static double mib_of(size_t bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    static double par_of(int64_t sum_ns, int64_t wall_ns) {
        if (wall_ns <= 0) {
            return 0.0;
        }
        return (double) sum_ns / (double) wall_ns;
    }

    bool should_log_ubatch(int n_tokens, uint64_t idx) const {
        // LLAMA_MOE_LOG_EVERY_STEP=1：每步都打，供性能归因用。
        // 稀疏打印时 counter 的累计窗口和 span 容易搞错（实测踩过），
        // 逐步打印后 span 恒为 1，消费方不用做任何归一。
        static const bool every = [] {
            const char * s = std::getenv("LLAMA_MOE_LOG_EVERY_STEP");
            return s && *s && *s != '0';
        }();
        if (every) {
            return true;
        }
        if (idx <= 8) {
            return true;
        }
        if (n_tokens <= 1) {
            return idx <= 4 || (idx % 32) == 0;
        }
        return (idx % 8) == 0;
    }

    void flush_ubatch(const char * kind, uint64_t idx) {
        if (cur.n_layers <= 0) {
            return;
        }
        // 本步的**完整** I/O：主路径 + 预读。用全局计数器的增量，避免口径混淆。
        const uint64_t all_now = io_bytes_all.load(std::memory_order_relaxed);
        const uint64_t calls_now = io_calls_all.load(std::memory_order_relaxed);
        const int64_t proc_now = process_ns.load(std::memory_order_relaxed);
        const uint64_t arm_skip_now = lookahead.n_arm_skipped.load(std::memory_order_relaxed);
        const uint64_t wait_now     = lookahead.n_waits.load(std::memory_order_relaxed);
        const int64_t  wait_ns_now  = lookahead.n_wait_ns.load(std::memory_order_relaxed);
        const uint64_t pred_read_now = lookahead.n_pred_read.load(std::memory_order_relaxed);
        const uint64_t pf_hits_now = lookahead.n_hits.load(std::memory_order_relaxed);
        const uint64_t pf_pub_now  = lookahead.n_pred_published.load(std::memory_order_relaxed);
        const uint64_t all_delta = all_now - last_io_bytes;
        const uint64_t calls_delta = calls_now - last_io_calls;
        const int64_t proc_delta = proc_now - last_process_ns;
        const uint64_t arm_skip_delta = arm_skip_now - last_arm_skipped;
        const uint64_t wait_delta = wait_now - last_wait_count;
        const int64_t wait_ns_delta = wait_ns_now - last_wait_ns;
        const uint64_t pred_read_delta = pred_read_now - last_pred_read;
        const uint64_t pf_hits_delta = pf_hits_now - last_pf_hits;
        const uint64_t pf_pub_delta  = pf_pub_now - last_pf_pub;
        last_io_bytes = all_now;
        last_io_calls = calls_now;
        last_process_ns = proc_now;
        last_arm_skipped = arm_skip_now;
        last_wait_count = wait_now;
        last_wait_ns = wait_ns_now;
        last_pred_read = pred_read_now;
        last_pf_hits = pf_hits_now;
        last_pf_pub  = pf_pub_now;

        // 这些计数器只在 flush 时前进，而 flush 受 should_log_ubatch 节流。
        // 所以必须把「本次 flush 覆盖了多少步」一起打出来，否则消费方会把
        // 多步累计当成单步值（实测虚报过 24 倍）。`span=` 就是干这个的。
        const uint64_t span = idx > last_flush_idx ? idx - last_flush_idx : 1;
        last_flush_idx = idx;
        const double wall_s = cur.wall_ns / 1e9;
        const double mibs = wall_s > 0.0 ? (cur.bytes / (1024.0 * 1024.0)) / wall_s : 0.0;
        const double uniq = cur.n_layers > 0 ? (double) cur.unique_sum / (double) cur.n_layers : 0.0;
        const double avg_run = cur.runs.n > 0 ? (double) cur.runs.experts / (double) cur.runs.n : 0.0;
        moe_log("%s #%llu tok=%d used=%d layers=%d unique/l=%.0f miss=%d hit=%d pack=%d/%d nth=%d "
                "pread_wall=%.0fms pread_sum=%.0fms bytes=%.0fMiB par=%.2f io=%.0fMiB/s "
                "readall=%.0fMiB calls=%llu realio=%.0fMiB/s span=%llu pmgmt=%.1fms "
                "pfskip=%llu pfwait=%llu/%.1fms pfread=%llu pfpub=%llu pfhit=%llu pfpredict=%llu pevsave=%llu pevforced=%llu pfcached=%llu usedpf=%llu "
                "runs=%d avg_run=%.1f max_run=%d zerocopy=%.0f%%",
            kind,
            (unsigned long long) idx,
            cur.n_tokens,
            cur.n_used,
            cur.n_layers,
            uniq,
            cur.miss_sum,
            cur.unique_sum - cur.miss_sum,
            cur.pack_layers,
            cur.n_layers,
            cur.n_th,
            ms_of(cur.wall_ns),
            ms_of(cur.pread_sum_ns),
            mib_of(cur.bytes),
            par_of(cur.pread_sum_ns, cur.wall_ns),
            mibs,
            mib_of(all_delta),
            (unsigned long long) calls_delta,
            // 真·带宽：完整字节 / 本步墙钟（含 compute，所以这是下限）。
            wall_s > 0.0 ? (double) all_delta / (1024.0 * 1024.0) / wall_s : 0.0,
            (unsigned long long) span,
            ms_of(proc_delta),
            (unsigned long long) arm_skip_delta,
            (unsigned long long) wait_delta,
            ms_of(wait_ns_delta),
            (unsigned long long) pred_read_delta,
            (unsigned long long) pf_pub_delta,
            (unsigned long long) pf_hits_delta,
            (unsigned long long) lookahead.n_predict_only.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_saved.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_evict_forced.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_pred_cached.load(std::memory_order_relaxed),
            (unsigned long long) lookahead.n_used_from_pf.load(std::memory_order_relaxed),
            cur.runs.n,
            avg_run,
            cur.runs.max,
            cur.runs.experts > 0 ? 100.0 * (double) cur.runs.ok_ex / (double) cur.runs.experts : 0.0);
        cur = ubatch_acc{};
    }

    /// 整个进程的「合并读」可行性汇总。用来回答一个问题：
    /// **把相邻专家的读合并成大 preadv，到底能减少多少次 syscall、传输会变成多大。**
    void report_merge() {
        if (g_runs.experts <= 0) {
            return;
        }
        const double avg_run = (double) g_runs.experts / (double) g_runs.n;
        const double zc_pct  = 100.0 * (double) g_runs.ok_ex / (double) g_runs.experts;
        // 假设全合并：syscall 数从「每专家一次」降到「每段一次」
        const double syscall_ratio = (double) g_runs.n / (double) g_runs.experts;
        moe_log("merge-scan: experts=%d runs=%d avg_run=%.2f max_run=%d "
                "zerocopy=%.1f%% scatter_experts=%d "
                "syscalls=%.1f%% of current (avg read = %.2f experts)",
            g_runs.experts, g_runs.n, avg_run, g_runs.max,
            zc_pct, g_runs.scatter_ex,
            100.0 * syscall_ratio, avg_run);
    }

    void log_io(const char * tag) {        const uint64_t miss = stats.n_miss.load(std::memory_order_relaxed);
        if (miss == 0) {
            return;
        }
        const int64_t pread_ns = stats.pread_ns.load(std::memory_order_relaxed);
        const uint64_t bytes = stats.n_bytes.load(std::memory_order_relaxed);
        moe_log("total[%s]: miss=%llu bytes=%.1f MiB pread=%.0fms",
            tag,
            (unsigned long long) miss,
            mib_of((size_t) bytes),
            pread_ns / 1e6);
    }

    const llama_model & model;
    int32_t n_expert;
    int32_t n_slots;
    int n_enabled = 0;
    int n_pack_ok = 0;
    bool shared_bank = false;
    // 按层 decode 缓存：每层分到 [layer*W, (layer+1)*W) 一段私有 slot 窗口，
    // 窗口内 LRU 跨 token 保留。prefill 不用窗口（一层要用 100+ 专家，装不下 W），
    // 仍然是「整块 bank + 每层清空」。
    //
    // W = 0 表示关闭（退回原行为：所有层共享全部 slot、每层清空）。
    int32_t decode_window = 0;
    /// 窗口适用的最大批宽（行数）。1 = 只服务单 token decode（旧行为）；更大的值让
    /// 投机验证（DSpark，批宽 = n_max+1）这类**小批次**也走窗口，靠跨轮专家复用省读盘。
    ///
    /// 由 LLAMA_MOE_WINDOW_MAX_ROWS 给出想要值，再由 resolve_decode_window() 按
    /// 缓存区容量夹紧 —— 夹紧是必须的：装不下时 process() 是抛异常（运行期崩），
    /// 不是降级。上限另外受「批宽 < 32」约束：再大就走 mul_mat_id 路径，寻址不到
    /// 每层的 `.dec` 视图。
    int32_t window_max_rows = 1;
    /// 每层**预读暂存区**的 slot 数（= 预读深度 K）。
    ///
    /// 为什么单列一块：预读的条目是「下一步可能用到」，与「跨 token 复用」是两种
    /// 不同的价值。共用一个 LRU 时，预读会在窗口满时**把跨 token 的缓存条目淘汰掉**
    /// —— 等于预读破坏缓存。分区之后两者互不干扰，覆盖变成**并集**：
    ///   缓存区 [window_start, window_start + c)          跨 token LRU
    ///   预读区 [window_start + c, window_start + c + K)  每步重填，不参与 LRU 淘汰
    int32_t prefetch_slots = 0;
    /// 每层窗口的总宽度（c_il + K），下标 = layer index。见 layer_state::window_width。
    std::vector<int32_t> layer_width;
    /// 统一 LRU：预读与主路径共用**整个窗口**（不分区）。**默认开启**。
    ///
    /// 为什么分区反而慢（实测，每 24 步）：
    ///   分区   miss 53 | 预读写入 3022 | 用到的专家来自预读区 4568 | pmgmt 174ms
    ///   统一   miss 29 | 预读写入 1932 | 用到的专家来自预读区  888 | pmgmt 101ms
    /// 分区下预读区只保留 ~3 步（每步被重填），而 78% 被用到的专家都落在那里 ——
    /// 于是「热专家」用完就被冲掉、下次重读。统一 LRU 让它们自然进入跨 token LRU。
    ///
    /// 分区当初是为了防「预读把缓存成批挤出去」，但那个前提已被 arm 时的过滤消掉：
    /// 预读每层每步只新增约 2.4 个（而不是 8 个），不会造成成批淘汰。
    bool prefetch_unified = true;
    bool was_window = false;
    /// 仅在 LLAMA_MOE_F_NOCACHE=auto（按阶段）时被翻转：prefill 绕、decode 不绕。
    /// 必须 atomic：主线程在 process() 里写它时**不持** fd_mutex，而 fd_for() 会被
    /// 预读 worker 的子线程调用并读它 → 无同步的并发读写。relaxed 够用：
    /// 只需不撕裂，不依赖可见顺序。
    std::atomic<bool> want_nocache{false};
    int last_moe_layer = -1;
    uint64_t n_prefill_ubatch = 0;
    uint64_t n_decode_step = 0;
    ubatch_acc cur;
    run_stats g_runs;   // 整进程累计；process() 只在 ith==0 上跑，单线程，无需原子
    std::vector<layer_state> layers;
    std::vector<map_userdata> userdata;
    std::unordered_map<std::string, int> fds;
    std::mutex fd_mutex;
    std::mutex io_mutex;
    io_stats stats;
    lookahead_expert_prefetch lookahead;
    ggml_tensor * lookahead_pending = nullptr;

    std::atomic<bool> parked{false};
    std::mutex park_mutex;

    // ── 命中率扫描（诊断，默认关闭）──────────────────────────────────────
    // LLAMA_MOE_SWEEP=<path> 打开，把每个 decode 步、每一层的原始记录写成一行：
    //
    //   L<layer> E:<本层真实用的专家> P:<上一层给本层预测的专家（按分数降序）>
    //
    // 为什么 dump 原始数据而不是在 C++ 里算表：命中率只取决于路由本身，与硬件无关，
    // 所以可以在 Mac 上跑一次、把文件拿来随便分析。dump 之后 cap / K 想怎么扫都行，
    // 不用为每个组合重编译。分析脚本见项目文档里的 analyze_hits.py。
    FILE * sweep_fp = nullptr;
    std::vector<std::vector<int32_t>> sweep_pred;   // sweep_pred[L] = 给层 L 的预测（来自层 L-1）

    // ── 缓存预热表（LLAMA_MOE_PREWARM=<path>）────────────────────────────
    // 文件：每行一层，行号 = 模型层号（与 sweep dump 的 `L<layer>` 同一编号），
    // 行内是逗号/空白分隔的专家 id（`#` 起注释，空行/注释行不占层号）。
    // 第一行 = 层 0，第二行 = 层 1……
    std::vector<std::vector<int32_t>> prewarm_table;
    bool prewarm_enabled = false;

    /// 预热读入后的落位记录（层、专家、窗口内 slot 下标）。
    /// 阶段切换时用它「重新发布」映射 —— 数据没被 prefill 碰（物理分离），只有映射被清。
    struct prewarm_place { int32_t layer; int32_t expert; int32_t slot; };
    std::vector<prewarm_place> prewarm_placed;
    /// 是否已读入且尚未被阶段切换消费。consumed 时置回 false（一次性），
    /// park（每轮结束）也会清掉 —— 下一轮必须重新 prewarm_experts()。
    bool prewarm_ready = false;

    void prewarm_init() {
        const char * path = std::getenv("LLAMA_MOE_PREWARM");
        if (!path || !*path) {
            return;
        }
        FILE * fp = std::fopen(path, "r");
        if (!fp) {
            moe_log("prewarm: 打不开表文件 %s", path);
            return;
        }
        prewarm_table.assign(layers.size(), {});
        char buf[16384];
        int li = 0;
        int total = 0;
        while (li < (int) layers.size() && std::fgets(buf, sizeof(buf), fp)) {
            const char * p = buf;
            while (*p == ' ' || *p == '\t') {
                ++p;
            }
            if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') {
                continue;   // 注释 / 空行不占层号
            }
            std::vector<int32_t> ids;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == ',') {
                    ++p;
                }
                if (!*p || *p == '\n' || *p == '\r' || *p == '#') {
                    break;
                }
                char * end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) {
                    ++p;
                    continue;
                }
                if (v >= 0 && v < n_expert) {
                    ids.push_back((int32_t) v);
                }
                p = end;
            }
            prewarm_table[(size_t) li] = std::move(ids);
            total += (int) prewarm_table[(size_t) li].size();
            li++;
        }
        std::fclose(fp);
        prewarm_enabled = total > 0 && decode_window > 0;
        moe_log("prewarm: 表 %s -> %d 层 / %d 个专家, decode_window=%d, enabled=%s",
                path, li, total, (int) decode_window, prewarm_enabled ? "on" : "off");
        if (decode_window <= 0) {
            moe_log("prewarm: 窗口未开（LLAMA_MOE_DECODE_WINDOW=0），预热不会生效");
        }
    }

    void sweep_init() {
        const char * path = std::getenv("LLAMA_MOE_SWEEP");
        if (!path || !*path) {
            return;
        }
        sweep_fp = fopen(path, "w");
        if (!sweep_fp) {
            moe_log("sweep: 无法打开 %s", path);
            return;
        }
        sweep_pred.assign(layers.size(), {});
        fprintf(sweep_fp, "# layer real_experts | predicted_experts(desc)\n");
        moe_log("sweep: 记录到 %s（每 decode 步每层一行）", path);
    }

    void sweep_write(int layer, const std::vector<int32_t> & used, const std::vector<int32_t> & pred) {
        if (!sweep_fp) {
            return;
        }
        fprintf(sweep_fp, "L%d E:", layer);
        for (size_t i = 0; i < used.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", used[i]);
        }
        fprintf(sweep_fp, " P:");
        for (size_t i = 0; i < pred.size(); ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", pred[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    /// prefill 时记录**最后一个 token** 用的专家（`P<layer> E:...`）。
    ///
    /// 用途：验证「prefill 能否给 decode 留一份初始缓存」。
    /// prefill 每层要用 100+ 个专家，装不进按层窗口（c=24），所以 prefill 必须用
    /// 整块 bank、逐层清空 —— 结束时 bank 里只剩最后一层的数据。但**最后一个 token
    /// 用的 8 个专家是 decode 第一步最可能的邻居**，可以单独留下来当种子。
    void sweep_write_prefill(int layer, const int32_t * ids, int n_ids, int n_tokens) {
        if (!sweep_fp || n_tokens <= 0) {
            return;
        }
        const int32_t * last = ids + (size_t) (n_tokens - 1) * n_ids;
        fprintf(sweep_fp, "P%d E:", layer);
        for (int i = 0; i < n_ids; ++i) {
            fprintf(sweep_fp, "%s%d", i ? "," : "", last[i]);
        }
        fprintf(sweep_fp, "\n");
    }

    std::vector<ggml_backend_buffer_t> expert_buffers() const {
        std::set<ggml_backend_buffer_t> uniq;
        auto add = [&](const std::vector<binding> & bs) {
            for (const auto & b : bs) {
                ggml_tensor * t = b.tensor;
                if (!t) {
                    continue;
                }
                ggml_backend_buffer_t buf = t->buffer;
                if (!buf && t->view_src) {
                    buf = t->view_src->buffer;
                }
                if (buf) {
                    uniq.insert(buf);
                }
            }
        };
        for (const auto & st : layers) {
            add(st.bindings);
            add(st.dec_bindings);
        }
        return { uniq.begin(), uniq.end() };
    }

    void park_expert_memory() {
        if (!park_window_enabled()) {
            return;
        }
        lookahead.drain();
        std::lock_guard<std::mutex> lk(park_mutex);
        if (parked.load(std::memory_order_relaxed)) {
            return;
        }
        // 一轮结束：窗口数据随本轮 decode churn 过，落位记录作废。
        prewarm_ready = false;
        prewarm_placed.clear();
        int n_ok = 0;
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_park");
            if (fn && fn(buf)) {
                n_ok++;
            }
        }
        for (auto & st : layers) {
            if (st.enabled) {
                clear_layer_cache(st);
            }
        }
        was_window = false;
        parked.store(true, std::memory_order_release);
        moe_log("park expert memory: %d metal buffer(s)", n_ok);
    }

    void ensure_unparked() {
        if (!parked.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lk(park_mutex);
        if (!parked.load(std::memory_order_relaxed)) {
            return;
        }
        for (ggml_backend_buffer_t buf : expert_buffers()) {
            metal_buf_fn fn = lookup_metal_buf_fn(buf, "ggml_backend_metal_buffer_unpark");
            if (fn && !fn(buf)) {
                throw std::runtime_error("failed to unpark MoE expert Metal buffer");
            }
        }
        parked.store(false, std::memory_order_release);
        moe_log("unparked expert memory");
    }
};

void llama_moe_slot_runtime::map_custom(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    // 这里在**图执行期间**被调用（CPU 自定义 op），所以这段耗时直接落在关键路径上。
    // 每层一次、每 token 40 次，量出来才能知道它是不是瓶颈。
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, src, dst);
    data->runtime->add_process_ns(now_ns() - t0);
}

void llama_moe_slot_runtime::map_custom_lookahead(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    const int64_t t0 = now_ns();
    data->runtime->process(data->layer, a, dst, b);
    data->runtime->add_process_ns(now_ns() - t0);
}

llama_moe_slot_runtime::llama_moe_slot_runtime(const llama_model & model) :
    pimpl(std::make_unique<impl>(model)) {
    for (auto & item : pimpl->userdata) {
        item.runtime = this;
    }
}

llama_moe_slot_runtime::~llama_moe_slot_runtime() = default;

bool llama_moe_slot_runtime::uses_layer(int layer) const {
    return layer >= 0 && layer < (int) pimpl->layers.size() && pimpl->layers[layer].enabled;
}

int llama_moe_slot_runtime::matmul_split() const {
    return pimpl->matmul_split;
}

/// A2 的 phase B 节点：把 slot_ids 拷一份作为输出（这样本节点自然依赖 phase A），
/// 同时依赖前段专家的 matmul 输出 `b` —— 于是调度器会把它排在前段算完之后、
/// 后段 matmul 之前执行。
void llama_moe_slot_runtime::map_custom_deferred(
        ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
        int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(b);
    if (ith != 0) {
        return;
    }
    auto * data = static_cast<llama_moe_slot_runtime::impl::map_userdata *>(userdata);
    memcpy(dst->data, a->data, ggml_nbytes(a));
    data->runtime->process_deferred(data->layer);
}

ggml_tensor * llama_moe_slot_runtime::build_deferred_trigger(
        ggml_context * ctx, ggml_tensor * slot_ids, ggml_tensor * dep, int layer) {
    return ggml_map_custom2(ctx, slot_ids, dep, map_custom_deferred, 1, &pimpl->userdata[(size_t) layer]);
}

void llama_moe_slot_runtime::add_process_ns(int64_t ns) {
    pimpl->process_ns.fetch_add(ns, std::memory_order_relaxed);
}

bool llama_moe_slot_runtime::lookahead_prefetch_enabled() const {
    return pimpl->lookahead.enabled;
}

int llama_moe_slot_runtime::lookahead_prefetch_depth() const {
    return pimpl->lookahead.enabled ? (int) pimpl->lookahead.depth : 0;
}

int llama_moe_slot_runtime::decode_window() const {
    return (int) pimpl->decode_window;
}

int llama_moe_slot_runtime::window_max_rows() const {
    return (int) pimpl->window_max_rows;
}

void llama_moe_slot_runtime::park_expert_memory() {
    pimpl->park_expert_memory();
}

void llama_moe_slot_runtime::ensure_unparked() {
    pimpl->ensure_unparked();
}

void llama_moe_slot_runtime::prewarm_experts() {
    pimpl->prewarm_experts();
}

llama_moe_slot_runtime::counters llama_moe_slot_runtime::counters_snapshot() const {
    counters c;
    c.process  = pimpl->process_ns.load(std::memory_order_relaxed);
    c.readids  = pimpl->readids_ns.load(std::memory_order_relaxed);
    c.install  = pimpl->install_ns.load(std::memory_order_relaxed);
    c.writeids = pimpl->writeids_ns.load(std::memory_order_relaxed);
    c.miss     = (int64_t) pimpl->stats.n_miss.load(std::memory_order_relaxed);
    c.bytes    = (int64_t) pimpl->stats.n_bytes.load(std::memory_order_relaxed);
    c.unique   = pimpl->unique_sum.load(std::memory_order_relaxed);
    return c;
}

void llama_moe_slot_runtime::set_lookahead_prefetch_depth(int depth) {
    pimpl->lookahead.set_depth(depth);
}

void llama_moe_slot_runtime::set_lookahead_pred(ggml_tensor * pred_ids) {
    pimpl->lookahead_pending = pred_ids;
}

ggml_tensor * llama_moe_slot_runtime::build_slot_ids_tensor(
        ggml_context * ctx,
        ggml_tensor * expert_ids,
        int layer) {
    if (!uses_layer(layer)) {
        return expert_ids;
    }
    ggml_tensor * pred = pimpl->lookahead_pending;
    pimpl->lookahead_pending = nullptr;
    if (pimpl->lookahead.enabled && pred) {
        return ggml_map_custom2(ctx, expert_ids, pred, map_custom_lookahead, 1, &pimpl->userdata[(size_t) layer]);
    }
    return ggml_map_custom1(ctx, expert_ids, map_custom, 1, &pimpl->userdata[layer]);
}

void llama_moe_slot_runtime::process(
        int layer,
        const ggml_tensor * expert_ids,
        ggml_tensor * slot_ids,
        const ggml_tensor * lookahead_pred) {
    pimpl->ensure_unparked();
    if (!uses_layer(layer) ||
        expert_ids->type != GGML_TYPE_I32 ||
        slot_ids->type != GGML_TYPE_I32 ||
        expert_ids->ne[0] != slot_ids->ne[0] ||
        expert_ids->ne[1] != slot_ids->ne[1]) {
        throw std::runtime_error(format(
            "invalid MoE slot map tensors at layer %d type=%s/%s shape=%lldx%lld vs %lldx%lld",
            layer,
            ggml_type_name(expert_ids ? expert_ids->type : GGML_TYPE_COUNT),
            ggml_type_name(slot_ids ? slot_ids->type : GGML_TYPE_COUNT),
            expert_ids ? (long long) expert_ids->ne[0] : -1,
            expert_ids ? (long long) expert_ids->ne[1] : -1,
            slot_ids ? (long long) slot_ids->ne[0] : -1,
            slot_ids ? (long long) slot_ids->ne[1] : -1));
    }

    auto & state = pimpl->layers[layer];

    const int32_t n_rows = (int32_t) expert_ids->ne[1];

    const bool decode = n_rows == 1;
    // 窗口也覆盖投机验证这类**小批次**（批宽 = n_max+1）：它们与单 token decode 走
    // 同一条 mul_mv_id 路径、同一套每层 `.dec` 张量，跨轮复用同一批专家。
    // 大批次（prefill）不走 —— 它拿的是 prefill 共享视图，只能用 [0, prefill_cap) 那段。
    //
    // 上限由 window_max_rows 按缓存区容量夹紧（见 resolve_decode_window），
    // 「并集装不下」的风险在那里就被消掉，不会落到这里的运行期。
    const bool window_mode = pimpl->decode_window > 0 && n_rows <= pimpl->window_max_rows;
    const bool lookahead_step = pimpl->lookahead.enabled && decode;

    // auto 模式下按阶段切 experts 的缓存策略：prefill 绕过、decode 不绕过。
    //
    // 判据用批宽而不是 window_mode —— 后者还要求 decode 窗口开着，而窗口关掉时
    // decode 阶段照样存在，用它会连 decode 一起绕过缓存。本判据与 llama-context 的
    // moe_decode_phase 同源。默认（非 auto）下这段不生效。
    if (llama_moe_slot_runtime::impl::nocache_mode() < 0) {
        const bool want = n_rows > pimpl->window_max_rows;
        if (pimpl->want_nocache.load(std::memory_order_relaxed) != want) {
            pimpl->want_nocache.store(want, std::memory_order_relaxed);
            pimpl->apply_experts_nocache(want);
        }
    }

    // **只在窗口模式下**需要在加锁之前等预读收工。
    //
    // 窗口模式里预读 worker 会锁**目标层**的 mutex 去写它的窗口；若本函数已持
    // state.mutex 再等 worker，就成了锁序反转 → 死锁（实际踩到过）。
    // 非窗口模式 worker 只写尾部乒乓区、不碰任何层锁，所以照旧在锁后 take 取结果。
    if (lookahead_step && window_mode) {
        std::vector<lookahead_expert_prefetch::hit> ignored;   // 窗口模式不返回 hit
        pimpl->lookahead.take(layer, ignored);
    }

    std::lock_guard<std::mutex> lock(state.mutex);

    // 阶段切换：prefill 用的是整块 bank / prefill 共享视图，与每层窗口在物理上可能
    // 落在同一块内存，所以**进入窗口阶段时**必须把每层的映射整表清空。
    //
    // 只清窗口内那 W 个 slot 是不够的：prefill 留下的 `expert_to_slot` 里还有大量
    // 指向**窗口之外**的陈旧映射，那些专家会被 process() 误判成「已在内存」而跳过安装，
    // 于是 matmul 读到的是别的专家权重 —— 输出错但不报错。（这是实际踩到的 bug。）
    if (window_mode && !pimpl->was_window) {
        for (auto & st : pimpl->layers) {
            if (!st.enabled) {
                continue;
            }
            // 当前层（layer 0）的 mutex 已经被上面那个 lock_guard 持有了，
            // 再锁一次会**自死锁**（std::mutex 不可重入）。它由下面那段直接清。
            if (&st == &state) {
                continue;
            }
            std::lock_guard<std::mutex> lk(st.mutex);
            llama_moe_slot_runtime::impl::clear_layer_cache(st);
        }
        // 当前层自己在锁内清（不用再锁）
        llama_moe_slot_runtime::impl::clear_layer_cache(state);

        // 缓存预热：窗口刚开、缓存刚清空（每轮生成都会走到这里一次）。
        //
        // 若这轮已经在 prefill 之前调过 `prewarm_experts()`（App 的正常路径），
        // 那批专家**已经读进窗口**且 prefill 碰不到它（decode 段与 prefill 段物理分离），
        // 这里只需**重新发布映射**（纯整数操作、零 I/O）；否则退回「现在读」的老路径。
        // 此刻 layer 0 的锁已被本函数持有，两个函数对当前层都不再加锁。
        if (pimpl->prewarm_ready) {
            const size_t n_before = pimpl->prewarm_placed.size();
            pimpl->publish_prewarm_placed(layer);
            pimpl->prewarm_ready = false;   // 一次性：本轮已消费
            moe_log("prewarm: 重挂 %zu 个已预热专家（无 I/O，数据活过 prefill）", n_before);
        } else {
            pimpl->prewarm_windows(layer);
        }
    }
    // 阶段判据必须是 `window_mode` 而不是 `decode`：投机验证（批宽 = n_max+1）与
    // 单 token decode 属于**同一个阶段**，它们之间不能清缓存 —— 否则窗口每步都被
    // 清空，跨轮复用无从谈起（实测那样会让窗口命中率归零）。
    pimpl->was_window = window_mode;

    if (pimpl->shared_bank && !window_mode) {
        llama_moe_slot_runtime::impl::clear_layer_cache(state);
    }
    std::vector<int32_t> experts;
    {
        const int64_t t0 = now_ns();
        pimpl->read_i32_ids(expert_ids, experts);
        pimpl->readids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    const size_t count = experts.size();
    std::vector<int32_t> slots(count);
    std::vector<uint8_t> requested(pimpl->n_expert, 0);

    // 诊断：这个 token 里哪些**位置**会 miss（该位置还不在缓存里）。
    // 注意此时槽位分配还没做，所以只用 expert_to_slot 判断「已驻留」。
    if (window_mode && expert_ids->ne[1] == 1) {
        pimpl->miss_pos_steps++;
        for (size_t i = 0; i < count && i < 16; ++i) {
            if (experts[i] >= 0 && experts[i] < pimpl->n_expert &&
                state.expert_to_slot[experts[i]] < 0) {
                pimpl->miss_pos[i]++;
            }
        }
    }

    for (size_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= pimpl->n_expert) {
            throw std::runtime_error("MoE expert id is out of range");
        }
        requested[experts[i]] = 1;
    }

    if (pimpl->lookahead.enabled && !decode) {
        pimpl->lookahead.reset();
    }
    // 非窗口模式：在这里取预读结果（命中已在乒乓区，需搬到目标 slot）。
    // 窗口模式的结果由 worker 直接写进目标窗口了，这里无事可做。
    if (lookahead_step && !window_mode) {
        std::vector<lookahead_expert_prefetch::hit> hits;
        pimpl->lookahead.take(layer, hits);
        for (const auto & h : hits) {
            if (h.expert < 0 || h.expert >= pimpl->n_expert) {
                continue;
            }
            if (!requested[(size_t) h.expert] || state.expert_to_slot[h.expert] >= 0) {
                continue;
            }
            state.slot_to_expert[h.slot] = h.expert;
            state.expert_to_slot[h.expert] = h.slot;
            state.slot_age[h.slot] = ++state.age;
            pimpl->lookahead.n_hits.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 分配范围。窗口模式下只在**本层窗口**内找空位 / 淘汰 —— 这是整个机制的要点：
    // 每层有独立的 slot 段，所以层间的 LRU 互不干扰，缓存能跨 token 存活。
    // 非窗口模式仍是整块 bank（预读开启时让出尾部乒乓区）。
    int32_t alloc_lo = 0;
    int32_t alloc_hi = pimpl->n_slots;
    if (window_mode) {
        alloc_lo = state.window_start;
        // 分区模式：主路径只从缓存区分配，暂存区留给预读（互不挤占）。
        // 统一模式：主路径也可用整个窗口 —— 用到的专家因此天然进入跨 token LRU，
        // 不会被「预读区每步重填」冲掉。
        // 边界用**本层宽度**（各层不同，见 layer_state::window_width）。
        const int32_t w_il = state.window_width > 0
            ? state.window_width : (pimpl->decode_window + pimpl->prefetch_slots);
        alloc_hi = state.window_start + w_il -
                   (pimpl->prefetch_unified ? 0 : pimpl->prefetch_slots);
    } else if (lookahead_step) {
        alloc_hi = pimpl->lookahead.slot_floor();
    }

    // 「缓存 + 预测」的结合点（零 I/O 成本）：
    // 上一层给出的预测里，有些专家本层这次没用到、但**下一步**会用。淘汰时必须
    // 避开它们 —— 否则「刚被预测会用」的条目恰好是最老的，正好被换出去。
    //
    // 这不是预读（不读任何字节），只是把已有的预测信号用来改进淘汰决策。
    std::vector<uint8_t> keep;
    if (window_mode && pimpl->lookahead.enabled) {
        std::vector<int32_t> pred;
        {
            std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
            auto & slot = pimpl->predicted[(size_t) layer];
            if (!slot.empty()) {
                pred.swap(slot);
            }
        }
        if (!pred.empty()) {
            keep.assign((size_t) pimpl->n_expert, 0);
            for (int32_t e : pred) {
                if (e >= 0 && e < pimpl->n_expert) {
                    keep[(size_t) e] = 1;
                }
            }
        }
    }

    std::vector<llama_moe_slot_runtime::impl::pending_install> pending;
    pending.reserve((size_t) std::min((int32_t) pimpl->n_slots, pimpl->n_expert));

    // A2：按**名次**决定「现在读」还是「延后读」。末尾 matmul_split 个**名次**的
    // 缺失专家延后到 phase B —— 实测 miss 高度集中在尾部（名次 7 的 miss 是名次 0
    // 的 7 倍），那里最可能真的有东西要读。
    //
    // 按**固定名次**切而不是按「是否驻留」排序切，是为了保持求和顺序不变 ——
    // 输出因此可以逐字节校验。
    //
    // 判据必须用**名次**（`pos % n_used`）而不是线性下标：展平顺序是
    // `pos = 名次 + n_used * 行号`（ne[0] 变化最快，见 read_i32_ids），所以
    // 「末尾 K 个线性下标」多行时只覆盖最后 K/n_used 行的**全部**名次；而图侧的切分
    // （build_moe_ffn 的 slice_ids）切的是**所有行**的名次 [split_at, n_used)。
    // 两边对不上，phase A 就会读到 phase B 还没写入的槽位（静默读错专家权重）。
    // 单 token 时 pos == 名次，两种写法等价。
    auto & defer_list = pimpl->deferred[(size_t) layer];
    defer_list.clear();
    const int32_t n_used_row = (int32_t) expert_ids->ne[0];
    const int32_t split_at   = n_used_row - pimpl->matmul_split;
    const bool defer_by_rank = window_mode && pimpl->matmul_split > 0 && split_at > 0;

    for (size_t pos = 0; pos < count; ++pos) {
        const int32_t expert = experts[pos];
        if (state.expert_to_slot[expert] >= 0) {
            slots[pos] = state.expert_to_slot[expert];   // 已驻留，无需读取
            continue;
        }

        int32_t slot = -1;
        for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
            if (state.slot_to_expert[candidate] == kSlotFree) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) {
            // 淘汰时跳过 kSlotPending：那是预读 worker 正在写的 slot，
            // 动它会让 worker 把内容写到一个已经属于别人的位置。
            //
            // 两轮挑选：第一轮只考虑「预测下一步不会用」的条目（keep[]==0），
            // 第二轮才放宽。这样预测会用的条目能多留一步 —— 零 I/O 成本的收益。
            for (int pass = 0; pass < 2 && slot < 0; ++pass) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t candidate = alloc_lo; candidate < alloc_hi; ++candidate) {
                    const int32_t resident = state.slot_to_expert[candidate];
                    if (resident == kSlotPending) {
                        continue;
                    }
                    if (pass == 0 && !keep.empty() && keep[(size_t) resident]) {
                        continue;   // 预测下一步要用，先不动它
                    }
                    if (!requested[resident] && state.slot_age[candidate] < oldest) {
                        oldest = state.slot_age[candidate];
                        slot = candidate;
                    }
                }
                if (pass == 0) {
                    if (slot >= 0) {
                        pimpl->lookahead.n_evict_saved.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // pass0 一个都没找到 —— 说明候选**全在预测集里**，
                        // 也就是「保护预测条目」这个策略真的起了约束作用。
                        pimpl->lookahead.n_evict_forced.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        if (slot < 0) {
            throw std::runtime_error(format(
                "layer %d requires more than %d unique experts in one graph; reduce ubatch or increase --moe-slot-bank",
                layer, alloc_hi - alloc_lo));
        }

        const int32_t evicted = state.slot_to_expert[slot];
        if (evicted >= 0) {
            state.expert_to_slot[evicted] = kSlotFree;
        }
        state.slot_age[slot] = ++state.age;
        slots[pos] = slot;
        const bool defer_this = defer_by_rank &&
            (int32_t) (pos % (size_t) n_used_row) >= split_at;
        if (defer_this) {
            // 延后：槽位标 kSlotPending 占住（别处不会动它），phase B 读完再发布。
            state.slot_to_expert[slot] = kSlotPending;
            defer_list.push_back({ expert, slot });
        } else {
            state.slot_to_expert[slot] = expert;
            state.expert_to_slot[expert] = slot;
            pending.push_back({ expert, slot });
        }
    }

    int n_unique = 0;
    for (int32_t expert = 0; expert < pimpl->n_expert; ++expert) {
        if (requested[expert]) {
            n_unique++;
        }
    }
    pimpl->unique_sum.fetch_add(n_unique, std::memory_order_relaxed);

    const llama_moe_slot_runtime::impl::install_timing timing =
        pimpl->install_pending(state, pending, window_mode);
    pimpl->install_ns.fetch_add(timing.wall_ns, std::memory_order_relaxed);

    if (!pending.empty()) {
        // 仅统计，不影响读取路径。按层取增量，避免 cur 累计后重复计数。
        const auto st = llama_moe_slot_runtime::impl::accumulate_runs(pending);
        llama_moe_slot_runtime::impl::add_runs(pimpl->cur.runs, st);
        llama_moe_slot_runtime::impl::add_runs(pimpl->g_runs, st);
    }

    if (lookahead_step && lookahead_pred) {
        std::vector<int32_t> pred;
        pimpl->read_i32_ids(lookahead_pred, pred);
        // 「缓存 + 预测」的结合点（零 I/O）：把预测留给**目标层**，
        // 它下一步淘汰时可以避开这些条目。不是预读 —— 不读任何字节。
        {
            std::lock_guard<std::mutex> lk(pimpl->pred_mtx);
            pimpl->predicted[(size_t) layer + 1] = pred;
        }
        // arm 前先去掉**目标层缓存里已有**的预测专家。
        //
        // 它们本来就不需要读（worker 也会查表跳过），但会让 arm 的批次看起来更大、
        // worker 跑得更久 —— 而 arm 在 worker 忙时是**直接丢弃**的（pfskip）。
        // 实测每 token 约 9 次 arm 因此被丢（40 次里的 22%），等于白扔预测。
        // 先过滤就只把「真需要读的」交给 worker，批次更小、完成更快、丢得更少。
        std::vector<int32_t> need_read;
        need_read.reserve(pred.size());
        {
            auto & tgt = pimpl->layers[(size_t) layer + 1];
            std::lock_guard<std::mutex> lk(tgt.mutex);
            for (int32_t e : pred) {
                if (e >= 0 && e < pimpl->n_expert && tgt.expert_to_slot[(size_t) e] < 0) {
                    need_read.push_back(e);
                }
            }
        }
        pimpl->lookahead.n_pred_cached.fetch_add(pred.size() - need_read.size(),
                                                 std::memory_order_relaxed);
        if (!need_read.empty()) {
            pimpl->lookahead.arm(layer + 1, need_read);
        }
        // 记下「给层 layer+1 的预测」，供 layer+1 的扫描记录用。
        // 注意预测本身是用层 layer 的 hidden 算的（见 qwen35moe.cpp），所以
        // 层 0 永远拿不到预测 —— 预读覆盖率的结构性上界是 39/40。
        if (pimpl->sweep_fp && layer + 1 < (int) pimpl->sweep_pred.size()) {
            pimpl->sweep_pred[layer + 1] = pred;
        }
    }

    // 命中率扫描：decode 记录整批的 E（真实用到）；prefill 只记**最后一个 token** 的 E
    // （用于判断 prefill 能否给 decode 留初始缓存）。
    if (pimpl->sweep_fp) {
        if (decode) {
            std::vector<int32_t> used;
            used.reserve(32);
            for (int32_t e = 0; e < pimpl->n_expert; ++e) {
                if (requested[e]) {
                    used.push_back(e);
                }
            }
            pimpl->sweep_write(layer, used, pimpl->sweep_pred[layer]);
        } else {
            pimpl->sweep_write_prefill(layer, experts.data(),
                                       (int) expert_ids->ne[0], (int) expert_ids->ne[1]);
        }
    }

    const int n_tokens = (int) expert_ids->ne[1];
    const int n_used = (int) expert_ids->ne[0];
    const int n_miss = (int) pending.size();
    const int n_hit = n_unique - n_miss;

    if (layer == 0) {
        if (pimpl->cur.n_layers > 0) {
            pimpl->flush_ubatch(pimpl->cur.n_tokens <= 1 ? "decode" : "prefill",
                pimpl->cur.n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch);
        }
        if (n_tokens <= 1) {
            pimpl->n_decode_step++;
            pimpl->miss_per_layer_steps++;   // 每 decode 步一次（不是每层）
        } else {
            pimpl->n_prefill_ubatch++;
        }
        pimpl->cur.n_tokens = n_tokens;
        pimpl->cur.n_used = n_used;
    }

    pimpl->cur.n_layers++;
    pimpl->cur.unique_sum += n_unique;
    pimpl->cur.miss_sum += n_miss;
    pimpl->cur.n_th = std::max(pimpl->cur.n_th, timing.n_th);
    pimpl->cur.pack_layers += state.pack_ok ? 1 : 0;
    pimpl->cur.wall_ns += timing.wall_ns;
    pimpl->cur.pread_sum_ns += timing.pread_sum_ns;
    pimpl->cur.bytes += timing.bytes;
    if (window_mode && n_tokens <= 1 && layer < (int) pimpl->miss_per_layer.size()) {
        pimpl->miss_per_layer[(size_t) layer] += (uint64_t) n_miss;
    }

    const uint64_t ubatch_idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
    const bool sample_layer = (layer == 0 || layer == 1 ||
        layer == pimpl->last_moe_layer / 2 || layer == pimpl->last_moe_layer);
    if (ubatch_idx <= 2 && sample_layer && (n_miss > 0 || layer == 0)) {
        moe_log("layer l=%d tok=%d used=%d unique=%d miss=%d hit=%d pack=%d nth=%d "
                "pread_wall=%.1fms pread_sum=%.1fms pread_max=%.1fms bytes=%.1fMiB par=%.2f",
            layer, n_tokens, n_used, n_unique, n_miss, n_hit,
            timing.pack ? 1 : 0, timing.n_th,
            llama_moe_slot_runtime::impl::ms_of(timing.wall_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_sum_ns),
            llama_moe_slot_runtime::impl::ms_of(timing.pread_max_ns),
            llama_moe_slot_runtime::impl::mib_of(timing.bytes),
            llama_moe_slot_runtime::impl::par_of(timing.pread_sum_ns, timing.wall_ns));
    }

    if (layer == pimpl->last_moe_layer) {
        const char * kind = n_tokens <= 1 ? "decode" : "prefill";
        const uint64_t idx = n_tokens <= 1 ? pimpl->n_decode_step : pimpl->n_prefill_ubatch;
        if (pimpl->should_log_ubatch(n_tokens, idx)) {
            pimpl->flush_ubatch(kind, idx);
        } else {
            pimpl->cur = llama_moe_slot_runtime::impl::ubatch_acc{};
        }
        if (idx == 8 || (idx > 8 && (idx % 8) == 0)) {
            pimpl->log_io(kind);
        }
    }

    // slots[] 已在分配循环里按位置填好（被延后的也有有效槽位地址）。这里只补
    // LRU 时间戳：位置越靠后越新，与原来一致。
    for (size_t i = 0; i < count; ++i) {
        if (slots[i] < 0) {
            throw std::runtime_error("MoE expert was not installed");
        }
        state.slot_age[(size_t) slots[i]] = ++state.age;
        // 诊断：这个专家来自预读区吗？预读区每步重填，所以它**用完就没了** ——
        // 下次再需要就得重读。理想情况应该把它升格到缓存区。
        if (window_mode && pimpl->prefetch_slots > 0 &&
            slots[i] >= state.window_start + (state.window_width > 0
                        ? state.window_width - pimpl->prefetch_slots : pimpl->decode_window)) {
            pimpl->lookahead.n_used_from_pf.fetch_add(1, std::memory_order_relaxed);
        }
    }
    {
        const int64_t t0 = now_ns();
        pimpl->write_i32_ids(slot_ids, slots);
        pimpl->writeids_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
}

/// A2 的 phase B：把 phase A 里被延后的缺失专家读进来，并发布到窗口。
///
/// 调用时机：**在「前段专家（位置 < n - matmul_split）已经算完」之后**。
/// 那时 GPU 正在算那些已驻留的专家，这里的读就与它重叠了 —— 这是整个 A2 的收益来源。
///
/// 锁：读盘不持锁（槽位在 phase A 已标 kSlotPending，别处不会动它）；只有发布
/// （写 expert_to_slot / slot_to_expert）才持层锁，和预读 worker 的写法一致。
void llama_moe_slot_runtime::process_deferred(int layer) {
    if (layer < 0 || layer >= (int) pimpl->deferred.size()) {
        return;
    }
    auto & list = pimpl->deferred[(size_t) layer];
    if (list.empty()) {
        return;
    }
    auto & st = pimpl->layers[(size_t) layer];
    if (!st.enabled) {
        list.clear();
        return;
    }

    for (const auto & d : list) {
        pimpl->read_expert_directv(st, d.expert, d.slot, true);
    }
    {
        std::lock_guard<std::mutex> lk(st.mutex);
        for (const auto & d : list) {
            st.slot_to_expert[(size_t) d.slot] = d.expert;
            st.expert_to_slot[(size_t) d.expert] = d.slot;
        }
        st.n_prefetch_published += list.size();
    }
    list.clear();
}
