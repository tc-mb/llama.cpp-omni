"""MiniCPM-o macOS App — pybind11 直连版

核心改进:
  - 单进程: Python 直接调用 C++ libomni，无 HTTP 转发
  - 零文件 I/O: T2W 音频通过回调直推，无 WAV 文件轮询

架构:
  Client ──HTTP/SSE──→ Python FastAPI (:9060) ──pybind11──→ libomni (in-process)

目录结构:
  llama.cpp-omni/
  ├── build/bin/omni_engine.so    ← pybind11 模块
  └── tools/omni/
      ├── app/                    ← 本文件所在（SCRIPT_DIR）
      │   ├── server.py
      │   ├── run.sh
      │   └── tests/
      ├── models/                 ← GGUF 模型 + CoreML（gitignore）
      │   ├── MiniCPM-o-4_5-Q4_K_M.gguf
      │   ├── audio/ vision/ tts/ token2wav/
      │   └── coreml/
      └── assets/                 ← 参考音频等

启动:
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/server.py [--duplex]
"""

import sys
import os
import argparse
import glob
import json
import base64
import asyncio
import tempfile
import time
import logging
import struct
import threading
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 注意: soundfile/librosa/PIL 不再需要 —— prefill 走内存路径，不写文件

# ==================== 路径设置 ====================
# 目录层级: llama.cpp-omni / tools / omni / app / server.py
#                                          ↑ OMNI_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # tools/omni/app/
OMNI_DIR = os.path.dirname(SCRIPT_DIR)                        # tools/omni/
LLAMACPP_ROOT = os.path.dirname(os.path.dirname(OMNI_DIR))    # llama.cpp-omni/

# 将 omni_engine.so 所在目录加入 Python 路径
_BUILD_BIN_DIR = os.path.join(LLAMACPP_ROOT, "build", "bin")
if _BUILD_BIN_DIR not in sys.path:
    sys.path.insert(0, _BUILD_BIN_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("omni_server")


# ==================== 路径自动探测 ====================

def find_dir(parent: str, keyword: str) -> str:
    """在 parent 下查找包含 keyword 的子目录"""
    for name in sorted(os.listdir(parent)):
        full = os.path.join(parent, name)
        if os.path.isdir(full) and keyword in name.lower():
            return full
    return ""


def auto_detect_llm_model(model_dir: str) -> str:
    """自动从模型目录检测 LLM GGUF 文件

    优先级：Q4_K_M > Q8_0 > F16 > 其他 .gguf
    """
    if not model_dir or not os.path.isdir(model_dir):
        return ""
    priority_patterns = ["*Q4_K_M*.gguf", "*Q8_0*.gguf", "*F16*.gguf", "*.gguf"]
    for pattern in priority_patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        # 排除子目录中的模型文件（vision/audio/tts）
        top_level = [m for m in matches if os.path.dirname(m) == model_dir]
        if top_level:
            return top_level[0]
    return ""


def auto_detect_paths() -> Dict[str, str]:
    """自动探测模型目录、参考音频、CoreML 等

    基于新目录结构:
      llama.cpp-omni/tools/omni/models/  ← 所有模型
      llama.cpp-omni/tools/omni/assets/  ← 参考音频
      llama.cpp-omni/tools/omni/models/coreml/  ← CoreML 编译产物
    """
    llamacpp_root = LLAMACPP_ROOT

    # 模型目录: tools/omni/models/
    model_dir = os.path.join(OMNI_DIR, "models")
    if not os.path.isdir(model_dir):
        # 兼容旧结构: 在上层目录搜索
        old_root = os.path.dirname(LLAMACPP_ROOT)
        model_dir = find_dir(old_root, "gguf") or find_dir(old_root, "minicpm-o")

    # 参考音频
    ref_audio = ""
    ref_audio_candidates = [
        os.path.join(OMNI_DIR, "assets/default_ref_audio/default_ref_audio.wav"),
        os.path.join(OMNI_DIR, "assets/default_ref_audio.wav"),
    ]
    for candidate in ref_audio_candidates:
        if os.path.exists(candidate):
            ref_audio = candidate
            break

    # CoreML 编译产物 (*.mlmodelc)
    vision_coreml = ""
    coreml_dir = os.path.join(model_dir, "coreml") if model_dir else ""
    if coreml_dir and os.path.isdir(coreml_dir):
        coreml_candidates = glob.glob(os.path.join(coreml_dir, "*.mlmodelc"))
        if coreml_candidates:
            vision_coreml = coreml_candidates[0]

    return {
        "llamacpp_root": llamacpp_root,
        "model_dir": model_dir,
        "ref_audio": ref_audio,
        "vision_coreml": vision_coreml,
    }


# ==================== 全局状态 ====================

class ServerState:
    """服务器全局状态"""

    def __init__(self) -> None:
        self.engine: Any = None  # OmniEngine instance
        self.initialized: bool = False
        self.prefill_counter: int = 0
        self.temp_dir: str = ""
        self.lock: threading.Lock = threading.Lock()
        # 配置（由 main 填充）
        self.llm_model: str = ""
        self.model_dir: str = ""
        self.llamacpp_root: str = ""
        self.ref_audio: str = ""
        self.vision_coreml: str = ""
        self.output_dir: str = ""
        self.duplex_mode: bool = True
        self.n_gpu_layers: int = 99
        self.n_ctx: int = 4096
        self.n_threads: int = 4
        self.tts_gpu_layers: int = 99
        self.language: str = "zh"


STATE = ServerState()


# ==================== FastAPI App ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭时的资源管理"""
    # 创建临时文件目录
    STATE.temp_dir = os.path.join(SCRIPT_DIR, "temp_streaming_prefill_v2")
    os.makedirs(STATE.temp_dir, exist_ok=True)
    os.makedirs(STATE.output_dir, exist_ok=True)

    logger.info("omni_server started (pybind11 direct mode)")
    yield

    # 清理
    if STATE.engine is not None:
        logger.info("释放 OmniEngine...")
        STATE.engine.free()
        STATE.engine = None
    logger.info("omni_server shutdown")


app = FastAPI(title="MiniCPM-o Server (pybind11)", lifespan=lifespan)


# ==================== 工具函数 ====================

def _pcm_float32_to_wav_bytes(raw_pcm: bytes, sample_rate: int = 16000) -> bytes:
    """将 raw PCM float32 数据在内存中构造为 WAV 字节

    Args:
        raw_pcm: float32 PCM 数据 (little-endian)
        sample_rate: 采样率

    Returns:
        完整的 WAV 文件字节（RIFF header + PCM int16 data）
    """
    samples = np.frombuffer(raw_pcm, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (samples * 32767).astype(np.int16)
    data_bytes = pcm_int16.tobytes()

    # 构造 WAV header（44 bytes）
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(data_bytes)
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", data_size,
    )
    return header + data_bytes


# ==================== API 端点 ====================

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "backend": "pybind11_direct", "initialized": STATE.initialized}


@app.post("/omni/init_sys_prompt")
async def init_sys_prompt(request: Request):
    """初始化系统 prompt / 会话

    Body:
        media_type: "audio" | "omni" | 1 | 2
        duplex_mode: bool
        language: "zh" | "en"
    """
    body = await request.json()

    # 解析 media_type
    mt = body.get("media_type", body.get("msg_type", 2))
    if isinstance(mt, str):
        mt = 1 if mt == "audio" else 2
    media_type: int = mt

    duplex_mode: bool = body.get("duplex_mode", STATE.duplex_mode)
    language: str = body.get("language", STATE.language)

    logger.info(f"init_sys_prompt: media_type={media_type}, duplex={duplex_mode}, lang={language}")

    try:
        # 如果已初始化且参数相同，跳过完整重新加载
        # 避免 Metal GPU 资源释放不及时导致的 OOM crash
        if STATE.initialized and STATE.engine is not None:
            logger.info("已初始化，跳过重新加载（避免 Metal OOM）")
            return JSONResponse({
                "success": True,
                "media_type": media_type,
                "duplex_mode": duplex_mode,
                "language": language,
                "backend": "pybind11_direct",
                "note": "already_initialized",
            })

        # 如果有旧 engine（但未完成初始化），先释放
        if STATE.engine is not None:
            logger.info("释放旧 engine...")
            STATE.engine.free()
            STATE.engine = None
            STATE.initialized = False
            # 等待 Metal GPU 资源完全释放
            import gc
            gc.collect()
            import time as _time
            _time.sleep(2.0)
            logger.info("Metal 资源释放等待完成")

        import omni_engine
        engine = omni_engine.OmniEngine()

        # 在事件循环的线程池中执行（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: engine.init(
            llm_model_path=STATE.llm_model,
            model_dir=STATE.model_dir,
            media_type=media_type,
            use_tts=True,
            duplex_mode=duplex_mode,
            n_gpu_layers=STATE.n_gpu_layers,
            n_ctx=STATE.n_ctx,
            n_threads=STATE.n_threads,
            tts_gpu_layers=STATE.tts_gpu_layers,
            coreml_path=STATE.vision_coreml,
            output_dir=STATE.output_dir,
            voice_audio=STATE.ref_audio,
            language=language,
        ))

        STATE.engine = engine
        STATE.initialized = True
        STATE.prefill_counter = 0

        logger.info("OmniEngine 初始化成功")

        return JSONResponse({
            "success": True,
            "media_type": media_type,
            "duplex_mode": duplex_mode,
            "language": language,
            "backend": "pybind11_direct",
        })
    except Exception as e:
        logger.error(f"init_sys_prompt 失败: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: Request):
    """Prefill 音频/图像（内存直传，零文件 I/O）

    Body:
        audio: base64 编码的 WAV 音频
        image: base64 编码的 PNG/JPEG 图像 (可选)

    内存管理:
        - base64 解码后的 bytes 由 Python GC 管理
        - C++ 内部拷贝一次到 audition/vision 结构体，不持有 Python bytes 引用
        - 请求结束后 Python bytes 自动释放
    """
    if not STATE.initialized or STATE.engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)

    body = await request.json()
    audio_b64: str = body.get("audio", "")
    image_b64: str = body.get("image", "")

    # 递增 prefill 计数器
    with STATE.lock:
        idx = STATE.prefill_counter
        STATE.prefill_counter += 1

    try:
        # base64 → bytes（内存中，不写文件）
        audio_bytes: bytes = b""
        image_bytes: bytes = b""

        if audio_b64:
            raw_audio = base64.b64decode(audio_b64)
            # 如果已经是 WAV 格式，直接用
            if raw_audio[:4] == b"RIFF" and raw_audio[8:12] == b"WAVE":
                audio_bytes = raw_audio
            else:
                # raw PCM float32 → WAV bytes（在内存中构造 WAV header）
                audio_bytes = _pcm_float32_to_wav_bytes(raw_audio)

        if image_b64:
            image_bytes = base64.b64decode(image_b64)

        # 执行 prefill（在线程池中，避免阻塞事件循环）
        prefill_index = idx + 1  # index=0 是 voice cloning（在 init 中完成）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: STATE.engine.prefill_from_memory(
                audio_bytes, image_bytes, prefill_index, -1
            ),
        )

        return JSONResponse({"success": True, "index": prefill_index})

    except Exception as e:
        logger.error(f"streaming_prefill 失败: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/omni/streaming_generate")
async def streaming_generate(request: Request):
    """流式生成 — SSE 推送文本和音频

    返回 SSE 事件流:
        data: {"chunk_data": {"wav": "<base64>", "sample_rate": 24000, "text": "..."}}
        data: {"is_listen": true}
        data: [DONE]
    """
    if not STATE.initialized or STATE.engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)

    # 创建 asyncio queue 用于桥接 C++ 回调 → SSE 流
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_text(text: str) -> None:
        """C++ text_callback（从 C++ 线程调用，GIL 已获取）"""
        loop.call_soon_threadsafe(queue.put_nowait, ("text", text))

    def on_audio(pcm_bytes: bytes, wav_idx: int) -> None:
        """C++ wav_callback（从 C++ T2W 线程调用，GIL 已获取）"""
        loop.call_soon_threadsafe(queue.put_nowait, ("audio", pcm_bytes, wav_idx))

    # 在后台线程中运行 stream_decode（阻塞调用）
    decode_done = asyncio.Event()

    def run_decode() -> None:
        try:
            STATE.engine.decode(
                on_text=on_text,
                on_audio=on_audio,
                debug_dir=STATE.output_dir,
                round_idx=-1,
            )
        except Exception as e:
            logger.error(f"decode 异常: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(decode_done.set)
            # 发送结束信号
            loop.call_soon_threadsafe(queue.put_nowait, ("done",))

    decode_thread = threading.Thread(target=run_decode, name="decode_worker", daemon=True)
    decode_thread.start()

    async def event_generator():
        """SSE 事件生成器

        文本-音频对应策略:
          C++ 回调产出 text 和 audio 两种事件。在单工模式下 LLM 先完整生成
          所有文本（多次 text 回调），然后 TTS 逐 chunk 产出音频。

          每个 audio chunk ≈ 1s，中文语速 ≈ 3 字/秒。将累积的文本按
          ~3 字/chunk 的粒度逐步随 audio 发出，实现文本与音频的均匀对应。

          在双工模式下 text/audio 自然交替，效果一致。
        """
        # 中文 TTS 语速约 3 字/秒，每个 audio chunk 约 1 秒
        CHARS_PER_AUDIO_CHUNK = 3
        text_buffer: str = ""  # 待分配给 audio chunk 的文本

        def _drain_text() -> str:
            """从 text_buffer 中取出一小段文本（约 CHARS_PER_AUDIO_CHUNK 字）"""
            nonlocal text_buffer
            if not text_buffer:
                return ""
            cut = min(CHARS_PER_AUDIO_CHUNK, len(text_buffer))
            chunk_text = text_buffer[:cut]
            text_buffer = text_buffer[cut:]
            return chunk_text

        while True:
            # 等待新事件或 decode 完成
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                # 没有新事件，检查是否还有音频在队列中
                if decode_done.is_set() and queue.empty():
                    break
                continue

            if item[0] == "done":
                # 发送缓冲中剩余的文本（如果有）
                if text_buffer:
                    event = json.dumps({
                        "chunk_data": {"text": text_buffer, "wav": "", "sample_rate": 24000}
                    }, ensure_ascii=False)
                    yield f"data: {event}\n\n"
                    text_buffer = ""
                break

            elif item[0] == "text":
                text = item[1]
                if text == "__IS_LISTEN__":
                    # 先发缓冲中剩余文本
                    if text_buffer:
                        event = json.dumps({
                            "chunk_data": {"text": text_buffer, "wav": "", "sample_rate": 24000}
                        }, ensure_ascii=False)
                        yield f"data: {event}\n\n"
                        text_buffer = ""
                    yield f"data: {json.dumps({'is_listen': True})}\n\n"
                elif text == "__END_OF_TURN__":
                    pass  # 由 "done" 信号处理
                else:
                    text_buffer += text

            elif item[0] == "audio":
                pcm_bytes = item[1]
                wav_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                # 从 text 缓冲中按字符粒度取出一段，与此 audio chunk 对应
                chunk_text = _drain_text()
                event = json.dumps({
                    "chunk_data": {
                        "wav": wav_b64,
                        "sample_rate": 24000,
                        "text": chunk_text,
                    }
                }, ensure_ascii=False)
                yield f"data: {event}\n\n"

            elif item[0] == "error":
                yield f"data: {json.dumps({'error': item[1]})}\n\n"
                break

        # 等待剩余音频（decode 结束后 T2W 线程可能还在产出）
        # 给 T2W 线程最多 3 秒收尾
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                if item[0] == "audio":
                    pcm_bytes = item[1]
                    wav_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                    chunk_text = _drain_text()
                    event = json.dumps({
                        "chunk_data": {"wav": wav_b64, "sample_rate": 24000, "text": chunk_text}
                    }, ensure_ascii=False)
                    yield f"data: {event}\n\n"
                elif item[0] == "done":
                    break
            except asyncio.TimeoutError:
                if queue.empty():
                    break

        yield "data: [DONE]\n\n"

        # SSE 完全结束后清除 C++ 回调引用
        if STATE.engine is not None:
            STATE.engine.clear_callbacks()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/omni/stop")
async def stop_generation():
    """停止当前生成"""
    if STATE.engine is not None:
        STATE.engine.break_generation()
    return JSONResponse({"success": True})


@app.post("/omni/break")
async def break_generation():
    """打断当前生成（等同于 /omni/stop）"""
    if STATE.engine is not None:
        STATE.engine.break_generation()
    return JSONResponse({"success": True})


# ==================== 静态文件服务 ====================

STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
if os.path.isdir(STATIC_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================== 入口 ====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o Server V2 (pybind11 直连，零 IPC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/omni/app/server.py                  # 自动探测路径，默认双工
  python tools/omni/app/server.py --simplex        # 单工模式
  python tools/omni/app/server.py --port 9060      # 指定端口
""",
    )
    parser.add_argument("--port", type=int, default=9060, help="服务端口 (默认: 9060)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("--duplex", action="store_true", help="默认双工模式")
    parser.add_argument("--simplex", action="store_true", help="默认单工模式")
    parser.add_argument("--llamacpp-root", type=str, default="")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--vision-coreml", type=str, default="")
    args = parser.parse_args()

    # 路径探测
    detected = auto_detect_paths()
    llamacpp_root = args.llamacpp_root or detected["llamacpp_root"]
    model_dir = args.model_dir or detected["model_dir"]
    ref_audio = detected["ref_audio"]
    vision_coreml = args.vision_coreml or detected.get("vision_coreml", "")

    if not llamacpp_root or not os.path.isdir(llamacpp_root):
        print(f"错误: llama.cpp-omni 目录未找到: {llamacpp_root}")
        sys.exit(1)
    if not model_dir or not os.path.isdir(model_dir):
        print(f"错误: GGUF 模型目录未找到: {model_dir}")
        sys.exit(1)

    llm_model = auto_detect_llm_model(model_dir)
    if not llm_model:
        print(f"错误: 在 {model_dir} 中未找到 LLM GGUF 模型")
        sys.exit(1)

    duplex_mode = args.duplex and not args.simplex
    if not args.duplex and not args.simplex:
        duplex_mode = True  # 默认双工

    output_dir = os.path.join(OMNI_DIR, f"output_{args.port}")
    os.makedirs(output_dir, exist_ok=True)

    # 填充全局状态
    STATE.llm_model = llm_model
    STATE.model_dir = model_dir
    STATE.llamacpp_root = llamacpp_root
    STATE.ref_audio = ref_audio
    STATE.vision_coreml = vision_coreml
    STATE.output_dir = output_dir
    STATE.duplex_mode = duplex_mode
    STATE.n_gpu_layers = args.n_gpu_layers
    STATE.n_ctx = args.ctx_size
    STATE.language = "zh"

    mode_name = "双工 (Duplex)" if duplex_mode else "单工 (Simplex)"
    print()
    print("=" * 60)
    print("  MiniCPM-o Server (pybind11 直连)")
    print("=" * 60)
    print(f"  Web UI:        http://localhost:{args.port}")
    print(f"  API:           http://localhost:{args.port}/health")
    print(f"  默认模式:      {mode_name}")
    print(f"  后端:          pybind11 in-process (零 IPC)")
    print()
    print(f"  LLAMACPP_ROOT: {llamacpp_root}")
    print(f"  MODEL_DIR:     {model_dir}")
    print(f"  LLM_MODEL:     {llm_model}")
    print(f"  REF_AUDIO:     {ref_audio}")
    print(f"  OUTPUT_DIR:    {output_dir}")
    if vision_coreml:
        print(f"  VISION_COREML: {vision_coreml}")
        print(f"  并行模式:      Vision(ANE) + LLM(GPU) 零竞争")
    else:
        print(f"  VISION_COREML: 未启用")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
