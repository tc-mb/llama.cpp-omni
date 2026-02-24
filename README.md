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
cmake --build build --target llama-omni-cli -j
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
| `--test <prefix> <n>` | Run test with audio files |

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

## 🌐 WebRTC Demo — Real-Time Video Interaction

Full-duplex real-time video interaction demo based on WebRTC. Supports **macOS (Metal)**, **Linux (CUDA)**, and **Windows (CUDA)**.

### Fastest Way: oneclick.sh (No Docker Needed)

```bash
# One command — auto-downloads everything, compiles, and starts all services
PYTHON_CMD=/path/to/python bash oneclick.sh start
```

Open **https://localhost:8088** after startup.

### Alternative: Docker Deployment

```bash
# Build llama-server
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j

# Download and load Docker images
# 📦 Download: https://drive.google.com/file/d/191h2OJYir9aAL4KIE-mFF_XJ1jT6gnxj/view?usp=sharing

# One-click deployment (simplex)
./deploy_all.sh \
    --cpp-dir /path/to/llama.cpp-omni \
    --model-dir /path/to/MiniCPM-o-4_5-gguf

# Duplex mode
./deploy_all.sh \
    --cpp-dir /path/to/llama.cpp-omni \
    --model-dir /path/to/MiniCPM-o-4_5-gguf \
    --duplex
```

Open **http://localhost:3000** after startup.

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 (Docker) / 8088 (oneclick) | Web UI |
| Backend | 8025 (Docker) / 8021 (oneclick) | Backend API |
| LiveKit | 7880 | Real-time communication |
| Inference | 9060 | C++ HTTP API |

📖 **Full Documentation**: [MiniCPM-o-cookbook WebRTC Demo](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/demo/web_demo/WebRTC_Demo/README.md)
