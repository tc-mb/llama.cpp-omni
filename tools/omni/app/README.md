# MiniCPM-o macOS 简易 Demo

基于 pybind11 直连的本地语音对话 demo，零 Docker、零 IPC，单进程运行。

## 与其他部署方式的区别

本仓库提供三种运行方式，各有适用场景：

### 架构对比

```
方式 A: llama-omni-cli（C++ 命令行）
  终端 stdin/stdout ──→ libomni (C++)
  音频输入: 文件路径
  音频输出: WAV 文件写入 tools/omni/output/

方式 B: Docker WebRTC Demo（旧方案）
  Browser ──WebRTC──→ Frontend Container (:3000)
       ──LiveKit──→ Backend Container (:8021)
            ──HTTP──→ Python HTTP API (:9060)
                 ──HTTP──→ llama-server (C++)
  音频: WebRTC 实时流

方式 C: pybind11 直连 Demo（本方案）✅
  Browser ──HTTP/SSE──→ Python FastAPI (:9060) ──pybind11──→ libomni (C++)
  音频: SSE base64 推送，Web Audio 播放
```

### 详细对比

| 维度 | A. CLI | B. Docker WebRTC | C. pybind11 直连（本方案） |
|---|---|---|---|
| **调用方式** | 终端交互，文件 I/O | 4 个 Docker 容器 + LiveKit | 单进程，Python 直调 C++ |
| **IPC 开销** | 无（单进程） | HTTP 转发（Python→C++ server） | 无（in-process 函数调用） |
| **音频传输** | 读写 WAV 文件 | WebRTC 实时流 | 内存回调 → SSE base64 |
| **部署复杂度** | 最简单，编译即用 | 最复杂，需 Docker + LiveKit + 4 个服务 | 中等，编译 .so + pip install |
| **前端** | 无 | React WebRTC 前端 | 单页 HTML（433 行） |
| **双工支持** | 支持 | 完整双工 + 视频流 | 支持（实验性） |
| **适用场景** | 快速测试、调试 | 生产级 demo、多人演示 | 本地开发、快速验证、二次开发 |
| **额外依赖** | 无 | Docker Desktop, LiveKit | pybind11, FastAPI, uvicorn |
| **首响延迟** | ~3.2s | ~3.2s + HTTP 转发开销 | ~3.2s（零额外开销） |

### 为什么用 pybind11 直连？

1. **零开销**：C++ 推理引擎和 Python 服务跑在同一个进程，函数调用级延迟，无序列化/网络开销
2. **简单部署**：不需要 Docker、LiveKit、多个容器协调，`pip install` + 一行命令即可启动
3. **易于调试**：单进程，Python debugger 可以直接断点到 C++ 回调边界
4. **灵活扩展**：Python 侧可以自由添加业务逻辑（鉴权、日志、数据处理），C++ 侧只负责推理

### 本方案的局限

- 前端是简易 demo（单页 HTML），不支持视频流输入
- SSE 而非 WebRTC，音频延迟略高于 WebRTC 方案（但对单工模式影响极小）
- 单 worker，不支持多用户并发（推理引擎是有状态的单实例）

## 前置条件

### 硬件
- Apple Silicon Mac（M1/M2/M3/M4），推荐 16GB+ 统一内存
- M4 系列性能最佳，首响 ~3.2s，稳态 RTF ~1.06

### 模型文件

将 GGUF 模型放在 `tools/omni/models/` 下：

```
tools/omni/models/
├── MiniCPM-o-4_5-Q4_K_M.gguf       # LLM 主模型 (~5GB)
├── audio/
│   └── MiniCPM-o-4_5-audio-F16.gguf
├── vision/
│   └── MiniCPM-o-4_5-vision-F16.gguf
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf
│   └── MiniCPM-o-4_5-projector-F16.gguf
├── token2wav/
│   ├── encoder.gguf
│   ├── flow_matching.gguf
│   ├── flow_extra.gguf
│   ├── hifigan2.gguf
│   └── prompt_cache.gguf
└── coreml/                          # 可选，Vision ANE 加速
    └── *.mlmodelc
```

### 参考音频（语音克隆）

默认使用 `tools/omni/assets/default_ref_audio/default_ref_audio.wav`（6s, 16kHz mono）。
如需自定义音色，替换此文件即可。

## 编译

```bash
cd llama.cpp-omni

# 创建 venv 并安装依赖
python3 -m venv .venv/base
.venv/base/bin/pip install pybind11 numpy fastapi uvicorn

# 编译 pybind11 模块
cmake -B build -DGGML_METAL=ON -DBUILD_PYBIND=ON \
  -DPython3_EXECUTABLE=$(pwd)/.venv/base/bin/python \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j 8 --target omni_engine
```

编译产物：`build/bin/omni_engine.cpython-*.so`

## 启动

### 方式一：一键启动（推荐）

```bash
cd llama.cpp-omni
bash tools/omni/app/run.sh --simplex    # 单工模式（推荐，RTF ~1.06）
bash tools/omni/app/run.sh --duplex     # 双工模式（实验性，RTF 3-4x）
```

脚本会自动检测 venv、安装依赖、启动服务并打开浏览器。

### 方式二：手动启动

```bash
cd llama.cpp-omni
PYTHONPATH=. .venv/base/bin/python tools/omni/app/server.py --simplex
```

### 启动参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--simplex` | — | 单工模式（LLM 先生成完文本，再 TTS） |
| `--duplex` | 默认 | 双工模式（LLM 边生成边 TTS） |
| `--port` | 9060 | 服务端口 |
| `--host` | 0.0.0.0 | 绑定地址 |
| `--model-dir` | 自动探测 | 模型目录路径 |
| `--ctx-size` | 4096 | 上下文长度 |
| `--n-gpu-layers` | 99 | GPU offload 层数 |
| `--vision-coreml` | 自动探测 | CoreML vision 模型路径 |

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOCODER_THREADS` | 4 | Vocoder CPU 线程数（实测 4 线程最优） |

## 使用

启动后访问 http://localhost:9060 ：

1. **初始化**：页面加载后点击「初始化」，等待模型加载完成（~30s）
2. **录音**：点击「录音」按钮开始说话，再次点击结束
3. **回复**：系统自动进行语音识别 → LLM 推理 → TTS 语音合成，文本和音频流式输出
4. **重置**：点击「重置」清除对话历史（不重载模型）

### 前端界面功能

- 流式文本显示（LLM 每 ~10 token 推送一次）
- 流式音频播放（Web Audio API，T2W 每 ~1s 推送一段 PCM）
- 实时统计：首响延迟、音频时长、RTF
- 支持多轮对话

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查，返回 `initialized` 状态 |
| `/omni/init_sys_prompt` | POST | 初始化引擎（加载模型，~30s） |
| `/omni/streaming_prefill` | POST | Prefill 音频/图像（base64 编码） |
| `/omni/streaming_generate` | POST | SSE 流式生成 |
| `/omni/reset` | POST | 重置对话（清 KV cache，不重载模型） |
| `/omni/stop` | POST | 中断当前生成 |

### SSE 事件类型（`/omni/streaming_generate`）

| 事件 | 数据格式 | 说明 |
|---|---|---|
| `text` | `{"text": "..."}` | LLM 文本片段 |
| `tts_chunk` | `{"text": "...", "n_speech_tokens": N, "chunk_idx": I}` | TTS 完成一个 text chunk |
| `audio` | `{"audio": "<base64>", "wav_idx": I}` | PCM int16 LE @ 24kHz |

### 调用示例

```python
import requests, json

# 1. 初始化
requests.post("http://localhost:9060/omni/init_sys_prompt", json={
    "media_type": 1,       # 1=语音, 2=视频(omni)
    "duplex_mode": False,  # 单工
    "language": "zh"
})

# 2. Prefill 音频（base64 编码的 WAV）
import base64
with open("test.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()
requests.post("http://localhost:9060/omni/streaming_prefill", json={
    "audio": audio_b64
})

# 3. SSE 流式获取结果
resp = requests.post("http://localhost:9060/omni/streaming_generate", stream=True)
for line in resp.iter_lines(decode_unicode=True):
    if line.startswith("data: "):
        data = json.loads(line[6:])
        if "text" in data:
            print(data["text"], end="", flush=True)
```

## 性能参考（Apple Silicon，单工模式）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 首响 | ~3.2s | LLM 934ms + TTS 1180ms + T2W 932ms |
| LLM | ~68ms/token | Metal GPU |
| TTS | ~39ms/speech_token | Metal GPU |
| Token2Mel | ~490ms/window | Metal GPU |
| Vocoder | ~430ms/window | CPU (4 threads) |
| 稳态 RTF | ~1.06 | 三级流水线并行 |

## 目录结构

```
tools/omni/app/
├── README.md               ← 本文件
├── server.py               ← FastAPI 服务端（HTTP/SSE）
├── run.sh                  ← 一键启动脚本
├── static/
│   └── index.html          ← 前端 demo（单页应用）
└── tests/
    ├── test_backend_api.py         ← API 集成测试
    ├── test_simplex_rtf.py         ← 单工 RTF 测试
    ├── test_duplex_stream.py       ← 双工流式测试
    ├── test_duplex_timeline.py     ← 双工时间线分析
    ├── test_omni_realdata.py       ← 真实数据端到端测试
    ├── collect_audio.py            ← 音频采集工具
    └── generate_timeline_html.py   ← 时间线可视化
```
