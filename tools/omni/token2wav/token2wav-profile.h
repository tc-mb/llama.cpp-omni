#pragma once

// Token2Wav profiling helpers (header-only).
//
// 设计目标：
//   - 尽量零成本：关闭时只有一次原子读 + 一次时间戳，不做任何 I/O。
//   - 无侵入：通过环境变量控制，不改公共接口。
//   - 便于 baseline：统一给出 init / token2mel / vocoder / total / callback 的
//     p50/p95/p99/mean，并区分 "首 chunk" 和 "稳态 chunk"。
//
// 环境变量：
//   OMNI_T2W_PROFILE        : 0=关闭（默认）
//                             1=收集统计 + 结束时 print_summary()
//                             2=每次 push_tokens_window 都打印一行 [timing]
//   OMNI_T2W_PRINT_GRAPH    : 1=第一次调用时 ggml_graph_print() 一次性转储图结构
//
// 使用：
//   profile::ScopeTimer t("token2mel");           // 超出作用域自动记录
//   profile::Profiler::instance().print_summary(stderr);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace omni {
namespace flow {
namespace profile {

class Profiler;
inline Profiler & profiler_instance_();
inline void       print_summary(std::FILE * f);

// 首次调用 enabled() 时通过 atexit 注册一次性 summary dump，这样 llama-omni-cli、
// token2wav-example、test-duplex 等任何链接 libomni 的 binary 都会在退出前自动打印。
//
// 关键点：必须先"触摸"一次 Profiler::instance() 让 Meyers singleton 完成构造，
// 然后再调 std::atexit(print_summary)。这样 atexit 的 LIFO 顺序会保证
// print_summary() 先于 Profiler 的隐式析构器执行，避免访问已销毁的 vector。
inline void ensure_autoprint_() {
    static bool once = [] {
        (void) profiler_instance_();
        std::atexit([] { print_summary(stderr); });
        return true;
    }();
    (void) once;
}

inline int level() {
    static int v = [] {
        const char * s = std::getenv("OMNI_T2W_PROFILE");
        if (!s || !*s) {
            return 0;
        }
        const int n = std::atoi(s);
        return n < 0 ? 0 : n;
    }();
    if (v >= 1) {
        ensure_autoprint_();
    }
    return v;
}

inline bool enabled() {
    return level() >= 1;
}

inline bool verbose() {
    return level() >= 2;
}

inline bool print_graph_enabled() {
    static int v = [] {
        const char * s = std::getenv("OMNI_T2W_PRINT_GRAPH");
        if (!s || !*s) {
            return 0;
        }
        return std::atoi(s) != 0 ? 1 : 0;
    }();
    return v != 0;
}

struct Stage {
    std::string         name;
    std::vector<double> samples_ms;
};

class Profiler {
  public:
    static Profiler & instance() { return profiler_instance_(); }

    void record_init(double ms) {
        std::lock_guard<std::mutex> lk(mu_);
        init_ms_ = ms;
    }

    void record(const char * stage, double ms, bool is_first_chunk = false) {
        std::lock_guard<std::mutex> lk(mu_);
        Stage & s = get_or_create(stage);
        s.samples_ms.push_back(ms);
        if (is_first_chunk && std::strcmp(stage, "total") == 0) {
            first_chunk_ms_ = ms;
        }
    }

    void add_audio_samples(int64_t n_samples, int32_t sample_rate) {
        std::lock_guard<std::mutex> lk(mu_);
        total_audio_samples_ += n_samples;
        if (sample_rate > 0) {
            sample_rate_ = sample_rate;
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        stages_.clear();
        init_ms_             = 0.0;
        first_chunk_ms_      = 0.0;
        total_audio_samples_ = 0;
    }

    // 注意：percentile 使用线性插值；样本量小（<10）时 p95/p99 意义不大但仍能给出上界
    void print_summary(std::FILE * f = stderr) {
        std::lock_guard<std::mutex> lk(mu_);
        if (stages_.empty() && init_ms_ == 0.0) {
            return;
        }

        std::fprintf(f, "\n[profile] ===================== token2wav profile summary =====================\n");
        if (init_ms_ > 0.0) {
            std::fprintf(f, "[profile] init                                          = %9.3f ms\n", init_ms_);
        }
        if (first_chunk_ms_ > 0.0) {
            std::fprintf(f, "[profile] first-chunk total (warmup, includes JIT etc.) = %9.3f ms\n", first_chunk_ms_);
        }

        std::fprintf(f,
                     "[profile] %-12s %8s %10s %10s %10s %10s %10s %10s\n",
                     "stage", "n", "min", "p50", "p95", "p99", "max", "mean");
        for (Stage & s : stages_) {
            const auto st = compute_stats_(s.samples_ms);
            std::fprintf(f,
                         "[profile] %-12s %8zu %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                         s.name.c_str(), s.samples_ms.size(), st.min_v, st.p50, st.p95, st.p99, st.max_v, st.mean);
        }

        if (total_audio_samples_ > 0 && sample_rate_ > 0) {
            const Stage * total_stage = find_stage_("total");
            if (total_stage && !total_stage->samples_ms.empty()) {
                double sum_ms = 0.0;
                for (double v : total_stage->samples_ms) {
                    sum_ms += v;
                }
                const double audio_ms = 1000.0 * (double) total_audio_samples_ / (double) sample_rate_;
                const double rtf      = (audio_ms > 0.0) ? (sum_ms / audio_ms) : 0.0;
                std::fprintf(f,
                             "[profile] audio=%.3fs compute=%.3fs RTF=%.4f (<1 is faster than realtime)\n",
                             audio_ms / 1000.0, sum_ms / 1000.0, rtf);
            }
        }
        std::fprintf(f, "[profile] ==========================================================================\n\n");
    }

  private:
    struct Stats {
        double min_v = 0.0;
        double max_v = 0.0;
        double mean  = 0.0;
        double p50   = 0.0;
        double p95   = 0.0;
        double p99   = 0.0;
    };

    static double percentile_(std::vector<double> & v, double p) {
        if (v.empty()) {
            return 0.0;
        }
        if (v.size() == 1) {
            return v.front();
        }
        const double h     = p * (double) (v.size() - 1);
        const size_t lo    = (size_t) std::floor(h);
        const size_t hi    = std::min(lo + 1, v.size() - 1);
        const double frac  = h - (double) lo;
        return v[lo] + frac * (v[hi] - v[lo]);
    }

    static Stats compute_stats_(const std::vector<double> & in) {
        Stats st{};
        if (in.empty()) {
            return st;
        }
        std::vector<double> sorted = in;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        for (double v : sorted) {
            sum += v;
        }
        st.min_v = sorted.front();
        st.max_v = sorted.back();
        st.mean  = sum / (double) sorted.size();
        st.p50   = percentile_(sorted, 0.50);
        st.p95   = percentile_(sorted, 0.95);
        st.p99   = percentile_(sorted, 0.99);
        return st;
    }

    Stage & get_or_create(const char * name) {
        for (Stage & s : stages_) {
            if (s.name == name) {
                return s;
            }
        }
        stages_.push_back(Stage{ name, {} });
        return stages_.back();
    }

    const Stage * find_stage_(const char * name) const {
        for (const Stage & s : stages_) {
            if (s.name == name) {
                return &s;
            }
        }
        return nullptr;
    }

    std::mutex         mu_;
    std::vector<Stage> stages_;
    double             init_ms_             = 0.0;
    double             first_chunk_ms_      = 0.0;
    int64_t            total_audio_samples_ = 0;
    int32_t            sample_rate_         = 0;
};

// RAII: 超出作用域时记录一次耗时样本。关闭 profiling 时不会进入 Profiler 内部加锁路径。
class ScopeTimer {
  public:
    explicit ScopeTimer(const char * stage_name, bool is_first_chunk = false) :
        stage_(stage_name),
        t0_(std::chrono::steady_clock::now()),
        first_(is_first_chunk) {}

    ~ScopeTimer() {
        const auto   t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        ms_             = ms;
        if (record_ && enabled()) {
            Profiler::instance().record(stage_, ms, first_);
        }
    }

    // 禁止复制，允许"吞掉"记录（用于业务代码自行决定是否记录）
    ScopeTimer(const ScopeTimer &)             = delete;
    ScopeTimer & operator=(const ScopeTimer &) = delete;

    void cancel() { record_ = false; }

    // 读取已过去的毫秒数（不停止计时）
    double elapsed_ms() const {
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

    // 析构后可取到最终耗时
    double ms() const { return ms_; }

  private:
    const char *                               stage_;
    std::chrono::steady_clock::time_point      t0_;
    bool                                       first_  = false;
    bool                                       record_ = true;
    double                                     ms_     = 0.0;
};

// 便捷函数
inline void record_init_ms(double ms) {
    if (enabled()) {
        Profiler::instance().record_init(ms);
    }
}

inline void record_ms(const char * stage, double ms, bool is_first_chunk = false) {
    if (enabled()) {
        Profiler::instance().record(stage, ms, is_first_chunk);
    }
}

inline void record_audio_samples(int64_t n_samples, int32_t sample_rate) {
    if (enabled()) {
        Profiler::instance().add_audio_samples(n_samples, sample_rate);
    }
}

inline Profiler & profiler_instance_() {
    static Profiler p;
    return p;
}

inline void print_summary(std::FILE * f) {
    profiler_instance_().print_summary(f);
}

}  // namespace profile
}  // namespace flow
}  // namespace omni
