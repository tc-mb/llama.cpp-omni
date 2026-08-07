#include "omni-perf.h"

#include "omni.h"

#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

void print_with_timestamp(const char * format, ...);

namespace {

std::atomic<uint64_t> g_omni_perf_seq{0};
std::mutex g_omni_gpu_sampler_mutex;
std::mutex g_omni_gpu_file_mutex;

bool omni_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return !normalized.empty() && normalized != "0" && normalized != "false" &&
           normalized != "off" && normalized != "no";
}

std::string omni_perf_wall_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif

    std::ostringstream ss;
    ss << std::put_time(&buf, "%H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void omni_perf_write_line(const char * kind, const std::string & line) {
    const bool is_gpu_line = kind != nullptr && std::strcmp(kind, "DUPLEX_GPU") == 0;
    const char * gpu_file = is_gpu_line ? std::getenv("OMNI_GPU_PROF_FILE") : nullptr;
    if (!is_gpu_line || gpu_file == nullptr || gpu_file[0] == '\0') {
        print_with_timestamp("%s", line.c_str());
        return;
    }

    std::lock_guard<std::mutex> lock(g_omni_gpu_file_mutex);
    std::ofstream out(gpu_file, std::ios::app);
    if (out.good()) {
        out << omni_perf_wall_timestamp() << " " << line;
        if (line.empty() || line.back() != '\n') {
            out << "\n";
        }
    }
}

bool omni_gpu_prof_enabled() {
    return omni_env_enabled("OMNI_GPU_PROF");
}

double omni_perf_now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

double omni_perf_rss_mb() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long long total_pages = 0;
    long long resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) {
        return -1.0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return -1.0;
    }
    return (double) resident_pages * (double) page_size / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

bool omni_perf_gpu_memory_mb(double & used_mb, double & total_mb) {
    used_mb = -1.0;
    total_mb = -1.0;
#ifdef GGML_USE_CUDA
    const int device_count = ggml_backend_cuda_get_device_count();
    if (device_count <= 0) {
        return false;
    }
    size_t free_bytes_sum = 0;
    size_t total_bytes_sum = 0;
    for (int i = 0; i < device_count; ++i) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_cuda_get_device_memory(i, &free_bytes, &total_bytes);
        free_bytes_sum += free_bytes;
        total_bytes_sum += total_bytes;
    }
    if (total_bytes_sum == 0) {
        return false;
    }
    used_mb = (double) (total_bytes_sum - free_bytes_sum) / (1024.0 * 1024.0);
    total_mb = (double) total_bytes_sum / (1024.0 * 1024.0);
    return true;
#else
    return false;
#endif
}

#if !defined(_WIN32)
using nvmlReturn_t = int;
using nvmlDevice_t = void *;

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

static constexpr nvmlReturn_t OMNI_NVML_SUCCESS = 0;
static constexpr unsigned int OMNI_NVML_TEMPERATURE_GPU = 0;
static constexpr unsigned int OMNI_NVML_CLOCK_GRAPHICS = 0;
static constexpr unsigned int OMNI_NVML_CLOCK_MEM = 2;

class OmniNvmlLoader {
public:
    ~OmniNvmlLoader() {
        unload();
    }

    bool load(std::string & reason) {
        if (loaded) {
            return true;
        }
        handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) {
            handle = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
        }
        if (handle == nullptr) {
            reason = "nvml_not_found";
            return false;
        }
        if (!load_symbol(nvmlInit_v2, "nvmlInit_v2") ||
            !load_symbol(nvmlShutdown, "nvmlShutdown") ||
            !load_symbol(nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2") ||
            !load_symbol(nvmlDeviceGetHandleByIndex_v2, "nvmlDeviceGetHandleByIndex_v2") ||
            !load_symbol(nvmlDeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates") ||
            !load_symbol(nvmlDeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo") ||
            !load_symbol(nvmlDeviceGetPowerUsage, "nvmlDeviceGetPowerUsage") ||
            !load_symbol(nvmlDeviceGetTemperature, "nvmlDeviceGetTemperature") ||
            !load_symbol(nvmlDeviceGetClockInfo, "nvmlDeviceGetClockInfo")) {
            reason = "nvml_symbol_missing";
            unload();
            return false;
        }
        loaded = true;
        return true;
    }

    void unload() {
        if (handle != nullptr) {
            dlclose(handle);
        }
        handle = nullptr;
        loaded = false;
    }

    using nvmlInit_v2_fn = nvmlReturn_t (*)();
    using nvmlShutdown_fn = nvmlReturn_t (*)();
    using nvmlDeviceGetCount_v2_fn = nvmlReturn_t (*)(unsigned int *);
    using nvmlDeviceGetHandleByIndex_v2_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
    using nvmlDeviceGetUtilizationRates_fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t *);
    using nvmlDeviceGetMemoryInfo_fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *);
    using nvmlDeviceGetPowerUsage_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    using nvmlDeviceGetTemperature_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *);
    using nvmlDeviceGetClockInfo_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *);

    nvmlInit_v2_fn nvmlInit_v2 = nullptr;
    nvmlShutdown_fn nvmlShutdown = nullptr;
    nvmlDeviceGetCount_v2_fn nvmlDeviceGetCount_v2 = nullptr;
    nvmlDeviceGetHandleByIndex_v2_fn nvmlDeviceGetHandleByIndex_v2 = nullptr;
    nvmlDeviceGetUtilizationRates_fn nvmlDeviceGetUtilizationRates = nullptr;
    nvmlDeviceGetMemoryInfo_fn nvmlDeviceGetMemoryInfo = nullptr;
    nvmlDeviceGetPowerUsage_fn nvmlDeviceGetPowerUsage = nullptr;
    nvmlDeviceGetTemperature_fn nvmlDeviceGetTemperature = nullptr;
    nvmlDeviceGetClockInfo_fn nvmlDeviceGetClockInfo = nullptr;

private:
    template <typename T>
    bool load_symbol(T & fn, const char * name) {
        fn = reinterpret_cast<T>(dlsym(handle, name));
        return fn != nullptr;
    }

    void * handle = nullptr;
    bool loaded = false;
};
#endif

class OmniGpuPerfSampler {
public:
    bool start() {
        if (running.load()) {
            return true;
        }
        if (!omni_gpu_prof_enabled()) {
            return false;
        }

        interval_ms = omni_gpu_prof_interval_ms();

#if defined(_WIN32)
        omni_perf_write_line("DUPLEX_GPU", "[DUPLEX_GPU] event=unavailable reason=\"nvml_unsupported_platform\"");
        return false;
#else
        std::string reason;
        if (nvml.load(reason) &&
            nvml.nvmlInit_v2() == OMNI_NVML_SUCCESS &&
            nvml.nvmlDeviceGetCount_v2(&device_count) == OMNI_NVML_SUCCESS &&
            device_count > 0) {
            using_nvml = true;
            sample_devices = resolve_sample_devices(device_count);
            if (sample_devices.empty()) {
                omni_perf_write_line("DUPLEX_GPU", "[DUPLEX_GPU] event=unavailable reason=\"no_selected_devices\"");
                nvml.nvmlShutdown();
                nvml.unload();
                using_nvml = false;
                return false;
            }
        } else {
            if (device_count > 0) {
                nvml.nvmlShutdown();
            }
            nvml.unload();
            using_nvml = false;
            device_count = 1;
            if (!jetson_gpu_sysfs_available()) {
                const std::string fallback_reason = reason.empty() ? "nvml_unavailable_and_jetson_sysfs_missing" : reason;
                omni_perf_write_line("DUPLEX_GPU", "[DUPLEX_GPU] event=unavailable reason=\"" + fallback_reason + "\"");
                return false;
            }
            sample_devices = {0};
        }
        {
            std::ostringstream line;
            line << "[DUPLEX_GPU] event=devices";
            const char * source = using_nvml ? (jetson_gpu_sysfs_available() ? "nvml+jetson_sysfs" : "nvml") : "jetson_sysfs";
            line << " source=\"" << source << "\"";
            line << " selected=\"";
            for (size_t i = 0; i < sample_devices.size(); ++i) {
                if (i > 0) {
                    line << ",";
                }
                line << sample_devices[i];
            }
            line << "\"";
            omni_perf_write_line("DUPLEX_GPU", line.str());
        }

        running = true;
        worker = std::thread(&OmniGpuPerfSampler::run, this);
        return true;
#endif
    }

    void stop() {
        if (!running.exchange(false)) {
            return;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
#if !defined(_WIN32)
        if (using_nvml && device_count > 0) {
            nvml.nvmlShutdown();
        }
        nvml.unload();
        device_count = 0;
        using_nvml = false;
        sample_devices.clear();
#endif
    }

    ~OmniGpuPerfSampler() {
        stop();
    }

private:
    static int omni_gpu_prof_interval_ms() {
        const char * value = std::getenv("OMNI_GPU_PROF_INTERVAL_MS");
        if (value == nullptr || value[0] == '\0') {
            return 20;
        }
        char * end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value || parsed <= 0) {
            return 20;
        }
        return (int) std::max<long>(1, std::min<long>(parsed, 1000));
    }

#if !defined(_WIN32)
    static std::string fmt_na_or_int(bool ok, unsigned int value) {
        return ok ? std::to_string(value) : "NA";
    }

    static std::string fmt_na_or_mb(bool ok, unsigned long long bytes) {
        if (!ok) {
            return "NA";
        }
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2)
           << (double) bytes / (1024.0 * 1024.0);
        return ss.str();
    }

    static std::string fmt_na_or_power(bool ok, unsigned int milliwatts) {
        if (!ok) {
            return "NA";
        }
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (double) milliwatts / 1000.0;
        return ss.str();
    }

    static std::string fmt_na_or_double(bool ok, double value, int precision = 2) {
        if (!ok) {
            return "NA";
        }
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    static bool read_double_file(const char * path, double & value) {
        std::ifstream in(path);
        if (!in.good()) {
            return false;
        }
        double parsed = 0.0;
        if (!(in >> parsed)) {
            return false;
        }
        value = parsed;
        return true;
    }

    static bool read_jetson_gpu_load_pct(double & value) {
        double raw = 0.0;
        if (!read_double_file("/sys/devices/platform/17000000.gpu/load", raw)) {
            return false;
        }
        // Jetson exposes GR3D load as either percent or permille depending on BSP.
        value = raw > 100.0 ? raw / 10.0 : raw;
        value = std::max(0.0, std::min(100.0, value));
        return true;
    }

    static bool read_jetson_gpu_clock_mhz(double & value) {
        double hz = 0.0;
        if (!read_double_file("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq", hz)) {
            return false;
        }
        value = hz / 1000000.0;
        return true;
    }

    static bool jetson_gpu_sysfs_available() {
        double ignored = 0.0;
        return read_jetson_gpu_load_pct(ignored);
    }

    static void append_device_if_valid(std::vector<unsigned int> & out,
                                       long parsed,
                                       unsigned int device_count) {
        if (parsed < 0 || parsed >= (long) device_count) {
            return;
        }
        unsigned int device = (unsigned int) parsed;
        if (std::find(out.begin(), out.end(), device) == out.end()) {
            out.push_back(device);
        }
    }

    static std::vector<unsigned int> parse_device_list(const char * value,
                                                       unsigned int device_count) {
        std::vector<unsigned int> devices;
        if (value == nullptr || value[0] == '\0') {
            return devices;
        }
        const std::string text(value);
        size_t pos = 0;
        while (pos < text.size()) {
            size_t comma = text.find(',', pos);
            std::string item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            size_t first = item.find_first_not_of(" \t");
            size_t last = item.find_last_not_of(" \t");
            if (first != std::string::npos) {
                item = item.substr(first, last - first + 1);
                char * end = nullptr;
                long parsed = std::strtol(item.c_str(), &end, 10);
                if (end != item.c_str()) {
                    append_device_if_valid(devices, parsed, device_count);
                }
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        return devices;
    }

    static std::vector<unsigned int> resolve_sample_devices(unsigned int device_count) {
        // Explicit override: NVML physical indices, e.g. OMNI_GPU_PROF_DEVICES=0 or 0,2.
        std::vector<unsigned int> devices = parse_device_list(std::getenv("OMNI_GPU_PROF_DEVICES"), device_count);
        if (!devices.empty()) {
            return devices;
        }

        // In common single-GPU runs CUDA_VISIBLE_DEVICES limits the process to one card.
        // NVML still sees physical indices, so sample the first visible physical id.
        devices = parse_device_list(std::getenv("CUDA_VISIBLE_DEVICES"), device_count);
        if (!devices.empty()) {
            devices.resize(1);
            return devices;
        }

        // Avoid sampling every GPU on shared servers by default.
        return {0};
    }

    void run() {
        auto next = std::chrono::steady_clock::now();
        while (running.load()) {
            sample_once();
            next += std::chrono::milliseconds(interval_ms);
            std::unique_lock<std::mutex> lock(wait_mutex);
            cv.wait_until(lock, next, [&] { return !running.load(); });
            if (next < std::chrono::steady_clock::now() - std::chrono::milliseconds(interval_ms)) {
                next = std::chrono::steady_clock::now();
            }
        }
    }

    void sample_once() {
        for (unsigned int device : sample_devices) {
            if (device >= device_count) {
                continue;
            }
            nvmlDevice_t handle = nullptr;
            if (using_nvml && nvml.nvmlDeviceGetHandleByIndex_v2(device, &handle) != OMNI_NVML_SUCCESS) {
                continue;
            }

            nvmlUtilization_t util{};
            nvmlMemory_t mem{};
            unsigned int power_mw = 0;
            unsigned int temp_c = 0;
            unsigned int graphics_clock = 0;
            unsigned int mem_clock = 0;

            const bool has_util = using_nvml && nvml.nvmlDeviceGetUtilizationRates(handle, &util) == OMNI_NVML_SUCCESS;
            const bool has_mem = using_nvml && nvml.nvmlDeviceGetMemoryInfo(handle, &mem) == OMNI_NVML_SUCCESS;
            const bool has_power = using_nvml && nvml.nvmlDeviceGetPowerUsage(handle, &power_mw) == OMNI_NVML_SUCCESS;
            const bool has_temp = using_nvml && nvml.nvmlDeviceGetTemperature(handle, OMNI_NVML_TEMPERATURE_GPU, &temp_c) == OMNI_NVML_SUCCESS;
            const bool has_graphics_clock = using_nvml && nvml.nvmlDeviceGetClockInfo(handle, OMNI_NVML_CLOCK_GRAPHICS, &graphics_clock) == OMNI_NVML_SUCCESS;
            const bool has_mem_clock = using_nvml && nvml.nvmlDeviceGetClockInfo(handle, OMNI_NVML_CLOCK_MEM, &mem_clock) == OMNI_NVML_SUCCESS;

            double jetson_load_pct = 0.0;
            const bool has_jetson_load = !has_util && read_jetson_gpu_load_pct(jetson_load_pct);
            double jetson_clock_mhz = 0.0;
            const bool has_jetson_clock = !has_graphics_clock && read_jetson_gpu_clock_mhz(jetson_clock_mhz);
            double cuda_used_mb = -1.0;
            double cuda_total_mb = -1.0;
            const bool has_cuda_mem = !has_mem && omni_perf_gpu_memory_mb(cuda_used_mb, cuda_total_mb);

            const uint64_t sample_id = next_sample_id++;
            std::ostringstream line;
            line << "[DUPLEX_GPU] sample_id=" << sample_id
                 << " t_ms=" << std::fixed << std::setprecision(3) << omni_perf_now_ms()
                 << " device=" << device
                 << " sm_util_pct=" << (has_util ? fmt_na_or_int(true, util.gpu) : fmt_na_or_double(has_jetson_load, jetson_load_pct, 1))
                 << " mem_util_pct=" << fmt_na_or_int(has_util, util.memory)
                 << " gpu_used_mb=" << (has_mem ? fmt_na_or_mb(true, mem.used) : fmt_na_or_double(has_cuda_mem, cuda_used_mb))
                 << " gpu_total_mb=" << (has_mem ? fmt_na_or_mb(true, mem.total) : fmt_na_or_double(has_cuda_mem, cuda_total_mb))
                 << " power_w=" << fmt_na_or_power(has_power, power_mw)
                 << " temp_c=" << fmt_na_or_int(has_temp, temp_c)
                 << " graphics_clock_mhz=" << (has_graphics_clock ? fmt_na_or_int(true, graphics_clock) : fmt_na_or_double(has_jetson_clock, jetson_clock_mhz, 1))
                 << " mem_clock_mhz=" << fmt_na_or_int(has_mem_clock, mem_clock);
            omni_perf_write_line("DUPLEX_GPU", line.str());
        }
    }

    OmniNvmlLoader nvml;
    unsigned int device_count = 0;
    std::vector<unsigned int> sample_devices;
    bool using_nvml = false;
#endif

    std::atomic<bool> running{false};
    std::thread worker;
    std::mutex wait_mutex;
    std::condition_variable cv;
    int interval_ms = 20;
    uint64_t next_sample_id = 0;
};

std::unique_ptr<OmniGpuPerfSampler> g_omni_gpu_sampler;

double omni_perf_tokens_per_s(long long tokens, double duration_ms) {
    if (tokens <= 0 || duration_ms <= 0.0) {
        return 0.0;
    }
    return (double) tokens * 1000.0 / duration_ms;
}

OmniPerfTokenStats * omni_perf_stage_stats(struct omni_context * ctx_omni, const char * stage) {
    if (ctx_omni == nullptr || stage == nullptr) {
        return nullptr;
    }
    if (std::strcmp(stage, "duplex.llm.prefill") == 0) {
        return &ctx_omni->perf.llm_prefill;
    }
    if (std::strcmp(stage, "duplex.llm.decode") == 0) {
        return &ctx_omni->perf.llm_decode;
    }
    if (std::strcmp(stage, "tts.prefill") == 0) {
        return &ctx_omni->perf.tts_prefill;
    }
    if (std::strcmp(stage, "tts.decode") == 0) {
        return &ctx_omni->perf.tts_decode;
    }
    return nullptr;
}

} // namespace

std::string omni_perf_speed_detail(long long tokens, double duration_ms) {
    std::ostringstream oss;
    oss << ",tokens_per_s=" << std::fixed << std::setprecision(2)
        << omni_perf_tokens_per_s(tokens, duration_ms);
    if (tokens > 0 && duration_ms > 0.0) {
        oss << ",ms_per_token=" << std::setprecision(4)
            << (duration_ms / (double) tokens);
    } else {
        oss << ",ms_per_token=0.0000";
    }
    return oss.str();
}

bool omni_tts_debug_dump_enabled() {
    return omni_env_enabled("OMNI_TTS_DEBUG_DUMP");
}

void omni_gpu_perf_sampler_start() {
    if (!omni_gpu_prof_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_omni_gpu_sampler_mutex);
    if (!g_omni_gpu_sampler) {
        g_omni_gpu_sampler = std::make_unique<OmniGpuPerfSampler>();
        if (!g_omni_gpu_sampler->start()) {
            g_omni_gpu_sampler.reset();
        }
    }
}

void omni_gpu_perf_sampler_stop() {
    std::lock_guard<std::mutex> lock(g_omni_gpu_sampler_mutex);
    if (g_omni_gpu_sampler) {
        g_omni_gpu_sampler->stop();
        g_omni_gpu_sampler.reset();
    }
}

void omni_perf_record_tokens(struct omni_context * ctx_omni, const char * stage, long long tokens, double duration_ms) {
    if (tokens <= 0 || duration_ms < 0.0) {
        return;
    }
    OmniPerfTokenStats * stats = omni_perf_stage_stats(ctx_omni, stage);
    if (stats == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->perf.token_stats_mtx);
    stats->calls += 1;
    stats->tokens += tokens;
    stats->duration_ms += duration_ms;
}

void omni_perf_print_token_stats(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->perf.token_stats_mtx);
    const auto print_one = [](const char * stage, const OmniPerfTokenStats & stats) {
        if (stats.calls <= 0) {
            return;
        }
        const double tokens_per_s = omni_perf_tokens_per_s(stats.tokens, stats.duration_ms);
        const double ms_per_token = stats.tokens > 0 ? stats.duration_ms / (double) stats.tokens : 0.0;
        print_with_timestamp("[DUPLEX_PERF_SUMMARY] stage=%s calls=%lld tokens=%lld total_ms=%.3f avg_tokens_per_s=%.2f avg_ms_per_token=%.4f\n",
                             stage, stats.calls, stats.tokens, stats.duration_ms, tokens_per_s, ms_per_token);
    };
    print_one("duplex.llm.prefill", ctx_omni->perf.llm_prefill);
    print_one("duplex.llm.decode", ctx_omni->perf.llm_decode);
    print_one("tts.prefill", ctx_omni->perf.tts_prefill);
    print_one("tts.decode", ctx_omni->perf.tts_decode);
}

void omni_perf_mark(struct omni_context * ctx_omni,
                    const char * stage,
                    const char * event,
                    int chunk_index,
                    double duration_ms,
                    const char * detail) {
    double gpu_used_mb = -1.0;
    double gpu_total_mb = -1.0;
    const bool has_gpu = omni_perf_gpu_memory_mb(gpu_used_mb, gpu_total_mb);
    const double rss_mb = omni_perf_rss_mb();
    const int n_past = ctx_omni ? ctx_omni->n_past : -1;
    const char * safe_stage = stage ? stage : "unknown";
    const char * safe_event = event ? event : "mark";
    const char * safe_detail = detail ? detail : "";
    const uint64_t seq = g_omni_perf_seq++;

    std::ostringstream line;
    if (has_gpu) {
        line << "[DUPLEX_PERF] seq=" << seq
             << " stage=" << safe_stage
             << " event=" << safe_event
             << " chunk=" << chunk_index
             << " t_ms=" << std::fixed << std::setprecision(3) << omni_perf_now_ms()
             << " dur_ms=" << duration_ms
             << " rss_mb=" << std::setprecision(2) << rss_mb
             << " gpu_used_mb=" << gpu_used_mb
             << " gpu_total_mb=" << gpu_total_mb
             << " n_past=" << n_past
             << " detail=\"" << safe_detail << "\"";
    } else {
        line << "[DUPLEX_PERF] seq=" << seq
             << " stage=" << safe_stage
             << " event=" << safe_event
             << " chunk=" << chunk_index
             << " t_ms=" << std::fixed << std::setprecision(3) << omni_perf_now_ms()
             << " dur_ms=" << duration_ms
             << " rss_mb=" << std::setprecision(2) << rss_mb
             << " gpu_used_mb=NA gpu_total_mb=NA"
             << " n_past=" << n_past
             << " detail=\"" << safe_detail << "\"";
    }
    omni_perf_write_line("DUPLEX_PERF", line.str());
}

OmniPerfDetail::OmniPerfDetail(std::string base) : buf(std::move(base)) {}

void OmniPerfDetail::append(const char * key, const std::string & value) {
    if (!buf.empty()) {
        buf += ',';
    }
    buf += key;
    buf += '=';
    buf += value;
}

OmniPerfDetail & OmniPerfDetail::add_i(const char * key, long long value) {
    append(key, std::to_string(value));
    return *this;
}

OmniPerfDetail & OmniPerfDetail::f(const char * key, double value) {
    append(key, std::to_string(value));
    return *this;
}

OmniPerfDetail & OmniPerfDetail::s(const char * key, const std::string & value) {
    append(key, value);
    return *this;
}

OmniPerfScope::OmniPerfScope(struct omni_context * ctx_, const char * stage_, int chunk_index_, const std::string & detail_)
    : ctx(ctx_), stage(stage_), chunk_index(chunk_index_), detail(detail_),
      start(std::chrono::steady_clock::now()) {
    omni_perf_mark(ctx, stage, "start", chunk_index, -1.0, detail.c_str());
}

OmniPerfScope::~OmniPerfScope() {
    const auto end = std::chrono::steady_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::string final_detail = detail;
    if (has_tokens) {
        final_detail += omni_perf_speed_detail(tokens, duration_ms);
        omni_perf_record_tokens(ctx, stage, tokens, duration_ms);
    }
    omni_perf_mark(ctx, stage, "end", chunk_index, duration_ms, final_detail.c_str());
}

void OmniPerfScope::set_detail(const std::string & detail_) {
    detail = detail_;
}

void OmniPerfScope::set_tokens(long long tokens_) {
    tokens = tokens_;
    has_tokens = true;
}
