# llama.cpp-omni

**llama.cpp-omni** is a high-performance Omni multimodal inference engine built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

- 🚀 **First Full-Duplex Omni Streaming Engine** — The first open-source C++ inference framework supporting full-duplex, omni-modal streaming video calls
- ⚡ **Lightweight & Efficient** — Inherits llama.cpp's high-performance characteristics with GGUF quantization support and low memory footprint
- 🔌 **Fully Ecosystem Compatible** — Compatible with llama.cpp interfaces and ecosystem for seamless integration with existing toolchains
- 🌐 **Cross-Platform Deployment** — Supports Windows, Linux, and macOS, enabling efficient Omni model inference on consumer-grade hardware
- 🎙️ **End-to-End Voice Interaction** — Supports the complete pipeline of streaming audio input, LLM inference, and TTS speech synthesis

---

## MiniCPM-o

[**MiniCPM-o 4.5**](https://github.com/OpenBMB/MiniCPM-o) is a 9B-parameter on-device omni-modal large language model jointly developed by ModelBest and Tsinghua University, featuring powerful vision, speech, and full-duplex streaming capabilities.

---

## Omni Architecture & Runtime Mechanism

### Model Architecture

Built on the MiniCPM-o 4.5 end-to-end omni-modal architecture, where modality encoders/decoders are densely connected to the LLM through hidden states. This design enables better information flow and control while fully leveraging the rich multimodal knowledge acquired during training.

llama.cpp-omni splits the original PyTorch model into multiple independent GGUF modules, each with specific responsibilities:

- **VPM**: Vision encoder based on SigLip2 architecture, responsible for encoding images into visual embeddings. Includes a Resampler module that compresses visual features into a fixed number of query tokens before projecting them into the LLM's hidden space.
- **APM**: Audio encoder based on Whisper architecture, responsible for encoding 16kHz audio into audio embeddings. Features AvgPool and Projector layers to project into the LLM's hidden space.
- **LLM**: Main language model based on Qwen3-8B, which receives visual and audio embeddings as input and generates text token sequences. Supports multiple quantization formats (F16/Q8_0/Q4_K_M).
- **TTS**: Text-to-speech model based on LLaMA architecture, which projects LLM hidden states through Projector Semantic and autoregressively generates audio token sequences.
- **Token2Wav**: Flow Matching-based vocoder that converts audio tokens into 24kHz waveform audio.

### Full-Duplex Streaming Mechanism

llama.cpp-omni implements a full-duplex streaming mechanism where input streams (video + audio) and output streams (speech + text) operate without blocking each other:

- **Streaming Encoders**: Transforms offline modality encoders into online streaming versions for real-time input processing. Audio is sliced into 1-second chunks for APM, while images are fed frame-by-frame to VPM.
- **Time-Division Multiplexing (TDM)**: Within the LLM backbone, TDM divides parallel omni-modal streams into sequential information groups within periodic time slices, achieving millisecond-level input/output stream synchronization.
- **Interleaved Speech Generation**: The TTS module models text and speech tokens in an interleaved manner, supporting full-duplex speech generation where output can synchronize with new input in real-time while ensuring stability for long speech generation (>1 minute).

### Proactive Interaction Mechanism

In duplex mode, the LLM continuously monitors incoming video and audio streams, deciding whether to speak proactively at 1Hz frequency. This high-frequency decision-making capability, combined with full-duplex features, enables proactive interactions such as spontaneous reminders and comments.

### Runtime Pipeline

The core runtime pipeline of llama.cpp-omni consists of three stages:

1. **Initialization (omni_init)**: Loads all GGUF models, initializes LLM/TTS/Token2Wav contexts, and configures simplex/duplex mode along with reference audio (for voice cloning).

2. **Streaming Prefill (stream_prefill)**: 
   - When `index=0`: Initializes System Prompt, including text system prompt and audio system prompt (reference audio embedding)
   - When `index>0`: Processes user input — audio is encoded via APM, images via VPM, and embeddings are fed into LLM prefill
   - Supports high-resolution mode (max_slice_nums=2) and high-FPS mode (main image + stacked images)

3. **Streaming Decode (stream_decode)**:
   - LLM autoregressively generates text tokens, entering speech generation upon `<|speak|>` and switching to listening state upon `<|listen|>`
   - TTS projects LLM hidden states to generate audio tokens
   - Token2Wav synthesizes WAV audio in real-time using a sliding window approach (28 tokens input, 25 tokens stride)
   - All three modules execute in parallel via asynchronous queues, enabling streaming output

---

## Performance Benchmarks

### Inference Latency (RTX 4090, F16)

| Stage | Latency | Notes |
|-------|---------|-------|
| **Time to First Token (TTFT)** | **< 550ms** | First audio output |
| Prefill (vision + audio) | ~65ms | Audio-only ~21ms |
| Decode-LLM | ~38ms/token | 3 tokens ~115ms |
| TTS Generation | ~8.5ms/token | 25 tokens ~215ms |
| Token2Wav | RTF ~0.15x | 25 tokens → 1s audio ~150ms |

### Inference Latency (Apple M4 Max, Metal)

| Stage | Latency | Notes |
|-------|---------|-------|
| **Time to First Token (TTFT)** | **< 650ms** | First audio output |
| Prefill (audio) | ~30ms | Audio-only |
| Decode-LLM | ~12ms/token | Metal accelerated |
| TTS Generation | ~10ms/token | Metal accelerated |
| Token2Wav (Token2Mel) | ~235ms/chunk | Metal accelerated |
| Token2Wav (Vocoder) | ~220ms/chunk | CPU (HiFiGAN) |
| **Token2Wav Total** | RTF ~0.47x | 28 tokens → 1s audio ~450ms |

### Memory Usage (NVIDIA GPU)

| Configuration | LLM Quantization | Model Size | VRAM Estimate |
|---------------|------------------|------------|---------------|
| Full Omni | F16 | ~18 GB | ~20 GB |
| Full Omni | Q8_0 | ~11 GB | ~13 GB |
| Full Omni | Q4_K_M | ~8 GB | ~9 GB |
| Vision Only | Q8_0 | ~9 GB | ~10 GB |
| Audio Only | Q8_0 | ~10 GB | ~12 GB |

### Memory Usage (Apple Silicon)

| Configuration | LLM Quantization | Model Size | Unified Memory |
|---------------|------------------|------------|----------------|
| Full Omni | F16 | ~15 GB | ~19 GB |
| Full Omni | Q8_0 | ~8.1 GB | ~12 GB |
| Full Omni | Q4_K_M | ~4.7 GB | ~8.5 GB |

> **Note**: Apple Silicon uses unified memory architecture. Recommended: 16GB Mac for Q4_K_M/Q8_0, 32GB+ Mac for F16.

---

## Quick Start

### Prerequisites

**Model Files**: Download MiniCPM-o 4.5 GGUF models with the following directory structure:

```
MiniCPM-o-4_5-gguf/
├── MiniCPM-o-4_5-Q4_K_M.gguf         # LLM (or F16/Q8_0)
├── audio/
│   └── MiniCPM-o-4_5-audio-F16.gguf
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf
│   └── MiniCPM-o-4_5-projector-F16.gguf
├── token2wav-gguf/
│   ├── encoder.gguf                  # ~144MB
│   ├── flow_matching.gguf            # ~437MB
│   ├── flow_extra.gguf               # ~13MB
│   ├── hifigan2.gguf                 # ~79MB
│   └── prompt_cache.gguf             # ~67MB
└── vision/
    └── MiniCPM-o-4_5-vision-F16.gguf
```

### Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --target llama-omni-server --target llama-omni-cli -j
```

> CMake will auto-detect and enable Metal (macOS) or CUDA (Linux with NVIDIA GPU).

### Usage

```bash
# Basic usage (auto-detect all model paths from LLM path)
./build/bin/llama-omni-cli \
    -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf

# With custom reference audio (voice cloning)
./build/bin/llama-omni-cli \
    -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    --ref-audio /path/to/your_voice.wav

# Disable TTS (text-only output)
./build/bin/llama-omni-cli \
    -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
    --no-tts
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-m <path>` | **Required**. Path to LLM GGUF model |
| `--vision <path>` | Override vision model path |
| `--audio <path>` | Override audio model path |
| `--tts <path>` | Override TTS model path |
| `--projector <path>` | Override projector model path |
| `--ref-audio <path>` | Reference audio for voice cloning |
| `-c, --ctx-size <n>` | Context size (default: 4096) |
| `-ngl <n>` | Number of GPU layers (default: 99) |
| `--no-tts` | Disable TTS output |
| `--vision-batch-encode` | Encode same-size image slices in one batched pass (off by default; see below) |
| `--test <prefix> <n>` | Run test with audio files |
| `--bench-vision <img>` | Benchmark serial vs batched vision encoding on an image, then exit |

### Vision Batch Encoding (optional optimization)

For high-resolution / high-refresh inputs, an image is split into one overview plus
many equally-sized slices, and each slice is encoded by the ViT. By default these
slices are encoded **one at a time** (serial). `--vision-batch-encode` instead packs
all same-size slices into a **single batched** ViT pass, which is significantly faster
when there are many slices.

- **When to enable**: large images / high-res / high-refresh modes, i.e. cases that
  produce many slices. On a 4821×2259 image this gives roughly **1.5–2.3× faster**
  vision encoding (more slices → larger speedup).
- **Why it's off by default**: batched cuBLAS GEMM uses a different accumulation order
  than per-slice GEMM, so the embeddings are *numerically very close but not bit-exact*
  (avg diff ~1e-2). It also uses somewhat more VRAM. It is therefore opt-in rather than
  a universal default.

```bash
# Enable the optimization
./build/bin/llama-omni-cli \
    -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    --omni --vision-batch-encode

# Benchmark serial vs batched (prints a per-slice-count comparison table)
./build/bin/llama-omni-cli \
    -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    --bench-vision /path/to/large_image.png
```

Programmatically the same switch is exposed via `common_params.vpm_batch_encode`
(applied in `omni_init`) and `vision_set_batch_encode(ctx_vision, true)`.

### Output

Generated audio files are saved to `tools/omni/output/`:

```
tools/omni/output/
├── round_000/
│   └── tts_wav/
│       ├── wav_0.wav
│       ├── wav_1.wav
│       └── ...
└── round_001/
    └── tts_wav/
        └── wav_1000.wav
```

---

## 🌐 Recommended Companion Demo — MiniCPM-o-Demo (Comni)

For an out-of-the-box, end-to-end omni video-call experience built on top of `llama-omni-server`, we recommend the **Comni** branch of the official demo:

🔗 **[OpenBMB/MiniCPM-o-Demo @ Comni](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/Comni)**

It bundles a Python gateway + worker (which spawns and orchestrates `llama-omni-server`) and a desktop + mobile React frontend, supporting **macOS (Metal)**, **Linux (CUDA)**, and **Windows (CUDA)**. Use it when you want a turnkey video-call demo without writing your own HTTP integration.

> 💡 **Don't want to compile?** Pre-built one-click installers (**Comni for Windows / macOS**) are available on the [llama.cpp-omni Releases page](https://github.com/tc-mb/llama.cpp-omni/releases).

### TL;DR — Five Commands From Scratch

If you already have the GGUF weights from [Prerequisites](#prerequisites):

```bash
# 1. Build the C++ engine
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni && git checkout feat/web-demo \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-omni-server --target llama-omni-cli -j
cd ..

# 2. Set up the demo (Python venv + mobile frontend)
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo && git checkout Comni
bash install.sh
( cd frontend/mobile && bun install && bun run --bun build:static )   # or `npm`

# 3. Configure (use absolute paths)
cp config.example.json config.json
# Edit config.json:
#   "backend": "cpp"
#   "cpp_backend.llamacpp_root" = absolute path to ../llama.cpp-omni
#   "cpp_backend.model_dir"     = absolute path to MiniCPM-o-4_5-gguf

# 4. Launch
CUDA_VISIBLE_DEVICES=0 bash start_all.sh

# 5. Open in browser
#    https://localhost:8040/         (desktop)
#    https://localhost:8040/mobile/  (mobile React)
```

The detailed walkthrough below is the same content the demo repo's
[`README.md`](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/Comni/README.md) /
[`README_zh.md`](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/Comni/README_zh.md) covers — kept here so you don't have to bounce between repos.

### Architecture

```
gateway.py        :8040 (HTTPS)        ─┐
                                        │  HTTP / WS  (internal)
worker.py         :22440 + i  GPU i    ─┘
    │  spawns + HTTP-calls
    ▼
llama-omni-server      :19080 + i  GPU i
    /v1/stream/omni_init
    /v1/stream/update_session_config
    /v1/stream/prefill
    /v1/stream/decode    (SSE)
    /v1/stream/break
```

### Step-by-Step

**1. Build `llama-omni-server` from this repo**

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni
git checkout feat/web-demo
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-server --target llama-omni-cli -j
```

CMake auto-detects CUDA (Linux + NVIDIA) and Metal (macOS). After the build, `build/bin/llama-omni-server` is the binary `worker.py` will spawn — you do **not** need to start `llama-omni-server` yourself.

**2. Install Python dependencies**

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo
git checkout Comni
bash install.sh                   # creates .venv/base/ + installs deps
# PYTHON=python3.11 bash install.sh   # to use a different interpreter
```

`install.sh` creates `.venv/base/` (Python 3.10), upgrades `pip`, installs `torch==2.8.0` + `torchaudio==2.8.0`, and finally installs `requirements.txt`. The C++ backend doesn't use PyTorch at runtime, but the worker is still a Python process so the venv is needed.

**3. Configure `config.json`**

Copy the template and set `backend` to `cpp`:

```bash
cp config.example.json config.json
```

```json
{
    "backend": "cpp",
    "cpp_backend": {
        "llamacpp_root":   "/abs/path/to/llama.cpp-omni",
        "model_dir":       "/abs/path/to/MiniCPM-o-4_5-gguf",
        "llm_model":       "MiniCPM-o-4_5-Q4_K_M.gguf",
        "cpp_server_port": 19080,
        "ctx_size":        8192,
        "n_gpu_layers":    99
    },
    "audio":   { "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
                 "playback_delay_ms": 200 },
    "service": {
        "gateway_port":     8040,
        "worker_base_port": 22440,
        "num_workers":      1,
        "max_queue_size":   1000,
        "request_timeout":  300.0,
        "data_dir":         "data"
    },
    "duplex":  { "pause_timeout": 60.0 }
}
```

| Field | Purpose |
|-------|---------|
| `cpp_backend.llamacpp_root` | Absolute path to your `llama.cpp-omni` checkout. `worker.py` runs `${llamacpp_root}/build/bin/llama-omni-server` and uses `${llamacpp_root}/tools/omni/output_<port>/` as the TTS WAV output dir |
| `cpp_backend.model_dir` | Absolute path to the GGUF directory (LLM + `audio/` + `tts/` + `vision/` + `token2wav-gguf/`) |
| `cpp_backend.llm_model` | LLM filename inside `model_dir`. Pick the quantization you downloaded (`Q4_K_M` / `Q8_0` / `F16`) |
| `cpp_backend.cpp_server_port` | HTTP port `worker.py` will start `llama-omni-server` on. Worker `i` uses `cpp_server_port + i` |
| `cpp_backend.ctx_size` / `n_gpu_layers` | Forwarded to `llama-omni-server` as `--ctx-size` / `--n-gpu-layers` |

**4. Build the mobile frontend (one-time)**

The `/mobile/` route is served from `static/mobile/`, which is **gitignored** — it's the build output of the React + Vite project under `frontend/mobile/`:

```bash
cd frontend/mobile
bun install                    # or `npm install` (Node ≥ 20.19)
bun run --bun build:static     # publishes to ../../static/mobile/
cd ../..
```

See the demo repo's [`frontend/mobile/README.md`](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/Comni/frontend/mobile/README.md) for dev proxy / npm-only / hot-reload details.

**5. Start the stack**

```bash
CUDA_VISIBLE_DEVICES=0 bash start_all.sh
```

First boot loads all GGUF modules (VPM, APM, LLM, TTS, Token2Wav) and takes 10–60 s. The worker's `/health` returns `worker_status: "idle"` once `omni_init` finishes.

Then open:

- `https://localhost:8040/`            — desktop entry (Home / Omni / Audio-Duplex / Turnbased / Half-Duplex)
- `https://localhost:8040/mobile/`     — mobile React frontend
- `https://localhost:8040/mobile-omni/` — mobile-adapted Omni page (DOM bridge over the desktop `omni-app.js`)

> ⚠️ Camera / microphone require HTTPS. The self-signed certs under `certs/` work locally — accept the browser warning. Falling back to `bash start_all.sh --http` will only allow text input (browsers block `MediaDevices` on insecure origins).

### Stop

```bash
pkill -f "gateway.py|worker.py|llama-omni-server"
```

`worker.py` automatically restarts `llama-omni-server` after each session (`full_reinit`) to keep KV cache state clean across runs.

### Multi-GPU

Set `service.num_workers > 1` in `config.json` and pass the visible devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh
```

Each worker is bound to its own GPU (via `CUDA_VISIBLE_DEVICES`) and spawns its own `llama-omni-server` on `cpp_server_port + worker_index`.

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Worker log: `llama-omni-server not found` | `cpp_backend.llamacpp_root` is wrong, or `cmake --build … --target llama-omni-server` was not run |
| Worker `/health` stays at `worker_status: "loading"` for a long time | `omni_init` is still loading GGUF modules. Check `tmp/worker_<i>.log` for lines tagged `[CPP]` |
| WAV files appear under `${llamacpp_root}/tools/omni/output_<port>/round_XXX/` but the browser plays nothing | The gateway is HTTP — many browsers block `Audio` / `MediaDevices` on insecure origins. Use the default HTTPS mode |
| `kv_cache_length` keeps shrinking mid-conversation | C++ side sliding-window pruning is kicking in. The desktop and mobile UIs expose a "Stop on KV pruning" toggle (default on) that ends the session cleanly when this happens |

For more details and Chinese documentation, see the demo repo: [`README.md`](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/Comni/README.md) / [`README_zh.md`](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/Comni/README_zh.md).





## HTTP API & Integration Guide
> 📝 This section is based on community integration experience.

This section documents the HTTP API call sequence for integrating llama-omni-server into your own application (e.g. a Tauri/Electron desktop app). The official CLI is a black box — if you want programmatic control, you need to call these endpoints directly.

> This guide is based on real-world integration experience. Several critical details are **not documented elsewhere**.

---

### 1. Start llama-omni-server

```bash
./llama-omni-server \
  --host 0.0.0.0 \
  --port 9060 \
  --model /path/to/MiniCPM-o-4_5-Q4_K_M.gguf \
  -ngl 99 \
  --ctx-size 8192 \
  --repeat-penalty 1.05 \
  --temp 0.7
```

Poll `GET /health` until it returns 200 before proceeding. It typically takes 10–60 seconds.

```bash
# Wait for ready
curl http://localhost:9060/health
```

---

### 2. Initialize — `POST /v1/stream/omni_init`

Call this **once per application lifecycle**. It loads all model modules, sets up voice cloning, and internally executes the `index=0` prefill (system prompt initialization).

```json
POST /v1/stream/omni_init

{
  "media_type": 2,
  "use_tts": true,
  "duplex_mode": true,
  "model_dir": "/path/to/MiniCPM-o-4_5-gguf",
  "tts_bin_dir": "/path/to/MiniCPM-o-4_5-gguf/tts",
  "tts_gpu_layers": 100,
  "token2wav_device": "gpu:0",
  "output_dir": "/path/to/output",
  "voice_audio": "/path/to/reference_voice.wav"
}
```

| Field | Description |
|-------|-------------|
| `media_type` | `2` = vision + audio (full omni) |
| `duplex_mode` | `true` enables full-duplex streaming |
| `voice_audio` | Reference WAV for voice cloning. Omit to use default voice |
| `output_dir` | Directory where TTS WAV files will be written |

**Expected response:**
```json
{ "success": true, ... }
```

> ⚠️ `omni_init` internally completes `index=0` prefill. **Do not** send a separate `cnt=0` prefill after this call. Start your prefill counter at `1`.

---

### 3. Prefill Loop — `POST /v1/stream/prefill`

After `omni_init`, enter a continuous loop. Each iteration sends 1 second of audio + 1 screenshot frame. The counter `cnt` increments by 1 each call and **never resets** within a session.

```json
POST /v1/stream/prefill

{
  "audio_path_prefix": "/path/to/audio_chunk.wav",
  "img_path_prefix": "/path/to/screenshot.png",
  "cnt": 1
}
```

| Field | Description |
|-------|-------------|
| `cnt` | Starts at `1`, increments every call. `0` is reserved for `omni_init` |
| `audio_path_prefix` | 1-second audio chunk (16kHz WAV). Send a silence chunk if mic is muted |
| `img_path_prefix` | Current screen frame. Can reuse last frame if no update |

> ⚠️ **Always send an audio chunk**, even when muted. Submitting a silence segment keeps the duplex loop rhythm intact. Skipping will cause timing drift.

**Recommended loop cadence:** 1000ms per iteration.

---

### 4. Decode — `POST /v1/stream/decode`

Call decode **after each prefill**. It triggers the LLM to generate a response and returns an SSE stream.

```json
POST /v1/stream/decode

{
  "debug_dir": "/path/to/output",
  "stream": true
}
```

**SSE stream response format:**

```
data: {"content": "Hello", "is_listen": false, "stop": false}
data: {"content": "!", "is_listen": false, "stop": false}
data: {"is_listen": true, "stop": false}
data: [DONE]
```

| Field | Description |
|-------|-------------|
| `content` | Text token chunk. Empty string is possible, filter before display |
| `is_listen` | `true` = model has switched to listening state (stop playing audio) |
| `stop` | `true` = generation fully complete |

> ⚠️ The text field is **`content`**, not `text`. This is inconsistent with standard OpenAI-compatible SSE format.

---

### 5. Audio Output

TTS WAV files are written incrementally to `output_dir/round_XXX/tts_wav/`. Watch this directory for new files and play them in order.

Use a filesystem watcher (e.g. `notify` in Rust) to detect new WAV files as they appear during decode.

```
output_dir/
├── round_000/
│   └── tts_wav/
│       ├── wav_0.wav
│       ├── wav_1.wav
│       └── ...
└── round_001/
    └── tts_wav/
        └── wav_1000.wav
```

> ⚠️ Mute your microphone input while playing back TTS audio to prevent echo feedback into the prefill loop.

---

### Full Call Sequence Summary

```
start llama-omni-server
    ↓
GET /health  (poll until 200)
    ↓
POST /v1/stream/omni_init  (cnt=0 handled internally, start your counter at 1)
    ↓
loop every ~1000ms:
    POST /v1/stream/prefill  { cnt: N, audio, image }
    POST /v1/stream/decode   → consume SSE → play WAV files from output_dir
    N++
```
