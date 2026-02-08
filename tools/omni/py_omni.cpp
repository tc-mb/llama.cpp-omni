/**
 * pybind11 binding for libomni — MiniCPM-o 推理引擎 Python 直连接口
 *
 * 核心类:
 *   OmniEngine — 封装 omni_context 生命周期，提供 Python 友好的 API
 *
 * 设计:
 *   - 单进程调用，无 HTTP/IPC 开销
 *   - text/wav 通过回调直推 Python，无文件 I/O 中转
 *   - GIL 在 C++ 推理期间释放，回调时自动重新获取
 *   - 向后兼容：不影响 server.cpp 的 HTTP 模式
 */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "omni.h"
#include "common.h"
#include "llama.h"

#include <string>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <functional>

namespace py = pybind11;

/**
 * OmniEngine — libomni 的 Python 接口封装
 *
 * 生命周期: init() → [stream_prefill() → stream_decode()]* → free()
 * 线程模型: LLM/TTS/T2W 线程由 libomni 内部管理，Python 只需调用顶层 API
 */
class OmniEngine {
public:
    OmniEngine() = default;
    ~OmniEngine() {
        free();
    }

    /**
     * 初始化推理引擎（加载模型、创建上下文）
     *
     * Args:
     *   llm_model_path: LLM 主模型路径 (.gguf)
     *   model_dir: 模型根目录，包含 vision/, audio/, tts/ 子目录
     *   media_type: 1=语音, 2=视频(omni)
     *   use_tts: 是否启用 TTS
     *   duplex_mode: 是否启用双工模式
     *   n_gpu_layers: GPU offload 层数 (-1=全部)
     *   n_ctx: 上下文长度
     *   n_threads: CPU 线程数
     *   tts_gpu_layers: TTS 模型 GPU offload 层数
     *   coreml_path: CoreML vision 模型路径 (.mlmodelc)，空字符串表示不使用
     *   output_dir: 输出目录路径
     *   voice_audio: 参考音频路径（用于音色克隆），空字符串表示不使用
     *   language: 语言设置 "zh" 或 "en"
     *
     * Raises:
     *   RuntimeError: 模型加载失败
     */
    void init(
        const std::string & llm_model_path,
        const std::string & model_dir,
        int media_type,
        bool use_tts,
        bool duplex_mode,
        int n_gpu_layers,
        int n_ctx,
        int n_threads,
        int tts_gpu_layers,
        const std::string & coreml_path,
        const std::string & output_dir,
        const std::string & voice_audio,
        const std::string & language
    ) {
        if (ctx_ != nullptr) {
            throw std::runtime_error("OmniEngine already initialized, call free() first");
        }

        // 构造 common_params
        params_ = common_params{};
        params_.model.path = llm_model_path;
        params_.n_gpu_layers = n_gpu_layers;
        params_.n_ctx = n_ctx;
        params_.cpuparams.n_threads = n_threads;
        params_.n_predict = 2048;
        params_.n_batch = 2048;

        // 模型路径
        std::string dir = model_dir;
        if (!dir.empty() && dir.back() != '/') dir += '/';
        params_.vpm_model = dir + "vision/MiniCPM-o-4_5-vision-F16.gguf";
        params_.apm_model = dir + "audio/MiniCPM-o-4_5-audio-F16.gguf";
        params_.tts_model = dir + "tts/MiniCPM-o-4_5-tts-F16.gguf";

        // CoreML
        if (!coreml_path.empty()) {
            params_.vision_coreml_model_path = coreml_path;
        }

        // 🔧 [CRITICAL FIX] tts_bin_dir 必须指向 tts/ 目录，因为 omni_init 从中加载:
        //   1. Projector: {tts_bin_dir}/MiniCPM-o-4_5-projector-F16.gguf
        //   2. Token2Wav: 先尝试 {tts_bin_dir}/encoder.gguf，找不到自动 fallback 到
        //      tools/omni/models/token2wav/
        // 之前错误地指向 token2wav/，导致 Projector 加载失败，TTS 语音质量异常
        std::string tts_bin_dir = dir + "tts";
        std::string token2wav_device = "gpu:0";

        // 释放 GIL 执行模型加载（耗时操作）
        {
            py::gil_scoped_release release;

            llama_backend_init();
            llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

            ctx_ = omni_init(&params_, media_type, use_tts, tts_bin_dir,
                            tts_gpu_layers, token2wav_device, duplex_mode,
                            nullptr, nullptr, output_dir);
        }

        if (ctx_ == nullptr) {
            throw std::runtime_error("omni_init failed — check model paths and GPU memory");
        }

        ctx_->async = true;
        ctx_->duplex_mode = duplex_mode;
        ctx_->language = language;

        // CoreML warmup
        if (!coreml_path.empty()) {
            py::gil_scoped_release release;
            omni_warmup_ane(ctx_);
        }

        // Voice cloning (index=0 prefill)
        // 🔧 [修复] 设置 ref_audio_path，供 stream_prefill simplex 分支使用
        // 之前只传给 stream_prefill 作为 aud_fname，但 simplex 路径读的是 ref_audio_path
        if (!voice_audio.empty()) {
            ctx_->ref_audio_path = voice_audio;
            py::gil_scoped_release release;
            if (!stream_prefill(ctx_, voice_audio, "", 0)) {
                throw std::runtime_error("stream_prefill(voice_audio) failed during init");
            }
        }

        initialized_ = true;
    }

    /**
     * Prefill 音频/图像到 KV cache
     *
     * Args:
     *   audio_path: 音频文件路径 (.wav, 16kHz mono)
     *   image_path: 图像文件路径 (.png/.jpg)，空字符串表示无图像
     *   index: 帧索引 (0=系统 prompt 初始化, >=1=用户输入)
     *   max_slice_nums: 高清模式 slice 数量 (-1=使用全局设置)
     *
     * Raises:
     *   RuntimeError: engine 未初始化或 prefill 失败
     */
    void prefill(
        const std::string & audio_path,
        const std::string & image_path,
        int index,
        int max_slice_nums
    ) {
        check_initialized("prefill");
        bool ok;
        {
            py::gil_scoped_release release;
            ok = stream_prefill(ctx_, audio_path, image_path, index, max_slice_nums);
        }
        if (!ok) {
            throw std::runtime_error("stream_prefill failed");
        }
    }

    /**
     * 从内存 buffer 直接 prefill（零文件 I/O）
     *
     * 仅支持 index >= 1 的 async 路径。index=0（系统 prompt 初始化）在 init() 中完成。
     *
     * 内存安全:
     *   - Python bytes 在函数调用期间由 pybind11 持有引用
     *   - C++ 内部拷贝一次到 audition_audio_u8/vision_image_u8，函数返回后不持有引用
     *   - omni_embeds 由 LLM 线程队列管理，处理后 delete
     *
     * Args:
     *   audio_wav_bytes: WAV 文件内容 (bytes)，None/空表示无音频
     *   image_bytes: PNG/JPEG 图像内容 (bytes)，None/空表示无图像
     *   index: 帧索引 (必须 >= 1)
     *   max_slice_nums: 高清模式 slice 数量 (-1=使用全局设置)
     *
     * Raises:
     *   RuntimeError: engine 未初始化或 prefill 失败
     */
    void prefill_from_memory(
        py::bytes audio_wav_bytes,
        py::bytes image_bytes,
        int index,
        int max_slice_nums
    ) {
        check_initialized("prefill_from_memory");

        // 提取 bytes 指针和长度（零拷贝读取，pybind11 持有引用）
        const unsigned char * audio_ptr = nullptr;
        size_t audio_len = 0;
        const unsigned char * image_ptr = nullptr;
        size_t image_len = 0;

        std::string audio_str = static_cast<std::string>(audio_wav_bytes);
        std::string image_str = static_cast<std::string>(image_bytes);

        if (!audio_str.empty()) {
            audio_ptr = reinterpret_cast<const unsigned char*>(audio_str.data());
            audio_len = audio_str.size();
        }
        if (!image_str.empty()) {
            image_ptr = reinterpret_cast<const unsigned char*>(image_str.data());
            image_len = image_str.size();
        }

        bool ok;
        {
            py::gil_scoped_release release;
            ok = stream_prefill_from_memory(
                ctx_, audio_ptr, audio_len, image_ptr, image_len, index, max_slice_nums);
        }
        if (!ok) {
            throw std::runtime_error("stream_prefill_from_memory failed");
        }
    }

    /**
     * 启动 decode 循环，通过回调流式输出文本和音频
     *
     * 此函数阻塞直到 LLM 生成完毕（listen/turn_eos/eos）。
     * TTS/T2W 线程可能在 decode 返回后仍在异步产出音频。
     *
     * Args:
     *   on_text: 文本回调 (str,) — 普通文本片段、"__IS_LISTEN__"、"__END_OF_TURN__"
     *   on_audio: 音频回调 (bytes, int, int) — (PCM int16 LE bytes, wav_index, n_input_tokens)
     *   on_tts_chunk: TTS chunk 完成回调 (str, int, int) — (text, n_speech_tokens, chunk_idx)
     *                 可选，传 None 则不触发
     *   debug_dir: 调试输出目录
     *   round_idx: 轮次索引 (-1=使用内部计数)
     *
     * Raises:
     *   RuntimeError: engine 未初始化或 decode 失败
     */
    void decode(
        py::function on_text,
        py::function on_audio,
        py::object on_tts_chunk,
        const std::string & debug_dir,
        int round_idx
    ) {
        check_initialized("decode");

        // 设置 C++ 回调 → Python 回调（需要 GIL）
        ctx_->text_callback = [on_text](const std::string & text) {
            py::gil_scoped_acquire acquire;
            on_text(text);
        };

        ctx_->wav_callback = [on_audio](const int16_t * pcm, size_t num_samples, int wav_idx, int n_input_tokens) {
            py::gil_scoped_acquire acquire;
            // 将 PCM 数据包装为 bytes 对象（拷贝，安全跨线程）
            py::bytes audio_bytes(reinterpret_cast<const char*>(pcm), num_samples * sizeof(int16_t));
            on_audio(audio_bytes, wav_idx, n_input_tokens);
        };

        // tts_chunk_callback: 可选，传 None 则不设置
        if (!on_tts_chunk.is_none()) {
            py::function tts_cb = on_tts_chunk.cast<py::function>();
            ctx_->tts_chunk_callback = [tts_cb](const std::string & text, int n_speech_tokens, int chunk_idx) {
                py::gil_scoped_acquire acquire;
                tts_cb(text, n_speech_tokens, chunk_idx);
            };
        } else {
            ctx_->tts_chunk_callback = nullptr;
        }

        bool ok;
        {
            py::gil_scoped_release release;
            ok = stream_decode(ctx_, debug_dir, round_idx);
        }

        // 注意：不在这里清除回调！
        // stream_decode 返回时 LLM 已完成，但 TTS/T2W 线程可能仍在异步产出音频。
        // 回调在下次 decode() 调用时被新回调覆盖，或在 free() 时随 ctx_ 销毁。
        // Python 侧的 SSE generator 会等待 T2W 完成后再关闭连接。

        if (!ok) {
            throw std::runtime_error("stream_decode failed");
        }
    }

    /**
     * 清除 text/wav 回调（SSE generator 结束后调用）
     */
    void clear_callbacks() {
        if (ctx_ != nullptr) {
            ctx_->text_callback = nullptr;
            ctx_->wav_callback = nullptr;
            ctx_->tts_chunk_callback = nullptr;
        }
    }

    /**
     * 中断当前生成（双工模式下用户打断）
     */
    void stop() {
        if (ctx_ != nullptr) {
            py::gil_scoped_release release;
            stop_speek(ctx_);
        }
    }

    /**
     * 清理 KV cache
     */
    void clear_kv_cache() {
        if (ctx_ != nullptr) {
            py::gil_scoped_release release;
            clean_kvcache(ctx_);
        }
    }

    /**
     * 设置 break_event 打断标志（双工模式打断）
     */
    void break_generation() {
        if (ctx_ != nullptr) {
            ctx_->break_event.store(true);
        }
    }

    /**
     * 释放所有资源
     */
    void free() {
        if (ctx_ != nullptr) {
            py::gil_scoped_release release;
            omni_stop_threads(ctx_);
            omni_free(ctx_);
            ctx_ = nullptr;
        }
        initialized_ = false;
    }

    /**
     * 获取当前 n_past（KV cache 使用量）
     */
    int get_n_past() const {
        return ctx_ ? ctx_->n_past : 0;
    }

    /**
     * 获取当前 n_keep（系统 prompt 保护长度）
     */
    int get_n_keep() const {
        return ctx_ ? ctx_->n_keep : 0;
    }

    /**
     * 是否已初始化
     */
    bool is_initialized() const {
        return initialized_;
    }

    /**
     * 是否以 listen 结束（双工模式）
     */
    bool ended_with_listen() const {
        return ctx_ ? ctx_->ended_with_listen.load() : false;
    }

private:
    omni_context * ctx_ = nullptr;
    common_params params_;
    bool initialized_ = false;

    void check_initialized(const char * func_name) const {
        if (!initialized_ || ctx_ == nullptr) {
            throw std::runtime_error(
                std::string(func_name) + ": OmniEngine not initialized, call init() first"
            );
        }
    }
};


PYBIND11_MODULE(omni_engine, m) {
    m.doc() = "MiniCPM-o 推理引擎 — pybind11 直连接口（零 IPC、零文件 I/O）";

    py::class_<OmniEngine>(m, "OmniEngine",
        "libomni 推理引擎封装\n\n"
        "生命周期: init() → [prefill() → decode()]* → free()\n"
        "线程模型: LLM/TTS/T2W 线程由 C++ 内部管理")
        .def(py::init<>())
        .def("init", &OmniEngine::init,
            py::arg("llm_model_path"),
            py::arg("model_dir"),
            py::arg("media_type") = 2,
            py::arg("use_tts") = true,
            py::arg("duplex_mode") = true,
            py::arg("n_gpu_layers") = 99,
            py::arg("n_ctx") = 4096,
            py::arg("n_threads") = 4,
            py::arg("tts_gpu_layers") = 99,
            py::arg("coreml_path") = "",
            py::arg("output_dir") = "./tools/omni/output",
            py::arg("voice_audio") = "",
            py::arg("language") = "zh",
            "初始化推理引擎（加载模型）")
        .def("prefill", &OmniEngine::prefill,
            py::arg("audio_path"),
            py::arg("image_path") = "",
            py::arg("index") = 1,
            py::arg("max_slice_nums") = -1,
            "Prefill 音频/图像到 KV cache（文件路径版）")
        .def("prefill_from_memory", &OmniEngine::prefill_from_memory,
            py::arg("audio_wav_bytes") = py::bytes(""),
            py::arg("image_bytes") = py::bytes(""),
            py::arg("index") = 1,
            py::arg("max_slice_nums") = -1,
            "Prefill 音频/图像到 KV cache（内存版，零文件 I/O）\n\n"
            "Args:\n"
            "  audio_wav_bytes: WAV 文件内容 (bytes)\n"
            "  image_bytes: PNG/JPEG 图像内容 (bytes)")
        .def("decode", &OmniEngine::decode,
            py::arg("on_text"),
            py::arg("on_audio"),
            py::arg("on_tts_chunk") = py::none(),
            py::arg("debug_dir") = "./tools/omni/output",
            py::arg("round_idx") = -1,
            "启动 decode，通过回调流式输出文本和音频\n\n"
            "Args:\n"
            "  on_text: 文本回调 (str,)\n"
            "  on_audio: 音频回调 (bytes, int)\n"
            "  on_tts_chunk: TTS chunk 回调 (str, int, int) — (text, n_speech_tokens, chunk_idx)，可选")
        .def("stop", &OmniEngine::stop,
            "中断当前生成")
        .def("clear_kv_cache", &OmniEngine::clear_kv_cache,
            "清理 KV cache")
        .def("break_generation", &OmniEngine::break_generation,
            "设置 break_event 打断标志")
        .def("clear_callbacks", &OmniEngine::clear_callbacks,
            "清除 text/wav 回调（SSE generator 结束后调用）")
        .def("free", &OmniEngine::free,
            "释放所有资源")
        .def_property_readonly("n_past", &OmniEngine::get_n_past,
            "当前 KV cache 使用量")
        .def_property_readonly("n_keep", &OmniEngine::get_n_keep,
            "系统 prompt 保护长度")
        .def_property_readonly("is_initialized", &OmniEngine::is_initialized,
            "是否已初始化")
        .def_property_readonly("ended_with_listen", &OmniEngine::ended_with_listen,
            "上次 decode 是否以 listen 结束");
}
