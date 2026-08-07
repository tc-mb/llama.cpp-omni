#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <type_traits>

struct omni_context;

struct OmniPerfTokenStats {
    long long calls = 0;
    long long tokens = 0;
    double duration_ms = 0.0;
};

struct OmniPerfState {
    std::atomic<int> current_chunk_index{-1};
    std::mutex token_stats_mtx;
    OmniPerfTokenStats llm_prefill;
    OmniPerfTokenStats llm_decode;
    OmniPerfTokenStats tts_prefill;
    OmniPerfTokenStats tts_decode;
};

bool omni_tts_debug_dump_enabled();

void omni_gpu_perf_sampler_start();
void omni_gpu_perf_sampler_stop();

void omni_perf_print_token_stats(struct omni_context * ctx_omni);
void omni_perf_record_tokens(struct omni_context * ctx_omni, const char * stage, long long tokens, double duration_ms);
std::string omni_perf_speed_detail(long long tokens, double duration_ms);

void omni_perf_mark(struct omni_context * ctx_omni,
                    const char * stage,
                    const char * event,
                    int chunk_index = -1,
                    double duration_ms = -1.0,
                    const char * detail = nullptr);

class OmniPerfScope {
public:
    OmniPerfScope(struct omni_context * ctx, const char * stage, int chunk_index, const std::string & detail);
    ~OmniPerfScope();

    void set_detail(const std::string & detail);
    void set_tokens(long long tokens);

private:
    struct omni_context * ctx;
    const char * stage;
    int chunk_index;
    std::string detail;
    long long tokens = 0;
    bool has_tokens = false;
    std::chrono::steady_clock::time_point start;
};

// 插桩 detail 构造器：把 "k=v,k=v,..." 的拼接收敛成链式调用，
// 业务侧只负责传参，格式化逻辑集中在 omni-perf.cpp。
//   OmniPerfDetail().i("frame_id", idx).s("mode", "pending")          // "frame_id=3,mode=pending"
//   OmniPerfDetail(base).f("vpm_ms", vpm).i("chunks", n)             // 在已有串后追加
// i() 接收任意整型（含 bool，输出 0/1），f() 接收浮点，s() 接收字符串。
class OmniPerfDetail {
public:
    OmniPerfDetail() = default;
    explicit OmniPerfDetail(std::string base);

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    OmniPerfDetail & i(const char * key, T value) { return add_i(key, (long long) value); }

    OmniPerfDetail & f(const char * key, double value);
    OmniPerfDetail & s(const char * key, const std::string & value);

    std::string str() const { return buf; }
    operator std::string() const { return buf; }

private:
    OmniPerfDetail & add_i(const char * key, long long value);
    void append(const char * key, const std::string & value);
    std::string buf;
};

// ========================= 插桩专用宏（PERF PROBE）=========================
// 仅用于性能采集。所有用法都带 `// 插桩` 注释，出 PR 时按前缀 OMNI_PERF_PROBE_
// 全局搜索即可整体删除，不影响业务逻辑。
//
// 两套接口：
//   1) 自计时（计时变量只服务于插桩时用）：
//        OMNI_PERF_PROBE_BEGIN(ctx, stage, idx, start_detail);              // 插桩
//        ... 被测代码 ...
//        OMNI_PERF_PROBE_END(ctx, stage, end_detail);                       // 插桩（无 token）
//        OMNI_PERF_PROBE_END_TOKENS(ctx, stage, tokens, end_detail);        // 插桩（带 token，自动追加速度）
//      BEGIN 在当前作用域声明 _omni_probe_{idx,npast0,t0}，END/END_TOKENS 中
//      可在 end_detail 里引用 _omni_probe_ms（耗时）与 _omni_probe_npast0。
//
//   2) 显式时长（计时/打印已由业务代码持有，插桩只复用其结果）：
//        OMNI_PERF_PROBE_START(ctx, stage, idx, detail);                    // 插桩
//        OMNI_PERF_PROBE_FINISH(ctx, stage, idx, dur_ms, detail);           // 插桩（无 token）
//        OMNI_PERF_PROBE_FINISH_TOKENS(ctx, stage, idx, dur_ms, tok, detail);// 插桩（带 token，自动追加速度）
//
// detail 可为 const char* 或 std::string 表达式（速度后缀由 *_TOKENS 自动追加）。

// ---- 自计时 ----
#define OMNI_PERF_PROBE_BEGIN(ctx, stage_name, idx, start_detail)                                     \
    [[maybe_unused]] const int  _omni_probe_idx    = (idx);                                           \
    [[maybe_unused]] const int  _omni_probe_npast0 = (ctx)->n_past;                                   \
    [[maybe_unused]] const auto _omni_probe_t0     = std::chrono::high_resolution_clock::now();       \
    omni_perf_mark((ctx), (stage_name), "start", _omni_probe_idx, -1.0,                               \
                   std::string(start_detail).c_str())

#define OMNI_PERF_PROBE_END(ctx, stage_name, end_detail)                                              \
    do {                                                                                              \
        [[maybe_unused]] const double _omni_probe_ms = std::chrono::duration<double, std::milli>(     \
            std::chrono::high_resolution_clock::now() - _omni_probe_t0).count();                      \
        omni_perf_mark((ctx), (stage_name), "end", _omni_probe_idx, _omni_probe_ms,                   \
                       std::string(end_detail).c_str());                                              \
    } while (0)

#define OMNI_PERF_PROBE_END_TOKENS(ctx, stage_name, tok, end_detail)                                  \
    do {                                                                                              \
        const double    _omni_probe_ms     = std::chrono::duration<double, std::milli>(               \
            std::chrono::high_resolution_clock::now() - _omni_probe_t0).count();                      \
        const long long _omni_probe_tokens = (tok);                                                   \
        omni_perf_record_tokens((ctx), (stage_name), _omni_probe_tokens, _omni_probe_ms);             \
        omni_perf_mark((ctx), (stage_name), "end", _omni_probe_idx, _omni_probe_ms,                   \
            (std::string(end_detail) +                                                                \
             omni_perf_speed_detail(_omni_probe_tokens, _omni_probe_ms)).c_str());                    \
    } while (0)

// ---- 显式时长 ----
#define OMNI_PERF_PROBE_START(ctx, stage_name, idx, detail)                                           \
    omni_perf_mark((ctx), (stage_name), "start", (idx), -1.0, std::string(detail).c_str())

#define OMNI_PERF_PROBE_FINISH(ctx, stage_name, idx, dur_ms, detail)                                  \
    omni_perf_mark((ctx), (stage_name), "end", (idx), (dur_ms), std::string(detail).c_str())

#define OMNI_PERF_PROBE_FINISH_TOKENS(ctx, stage_name, idx, dur_ms, tok, detail)                      \
    do {                                                                                              \
        const long long _omni_probe_tokens = (tok);                                                   \
        const double    _omni_probe_dur    = (dur_ms);                                                 \
        omni_perf_record_tokens((ctx), (stage_name), _omni_probe_tokens, _omni_probe_dur);            \
        omni_perf_mark((ctx), (stage_name), "end", (idx), _omni_probe_dur,                            \
            (std::string(detail) +                                                                   \
             omni_perf_speed_detail(_omni_probe_tokens, _omni_probe_dur)).c_str());                   \
    } while (0)
// ======================= 插桩专用宏 END =======================
