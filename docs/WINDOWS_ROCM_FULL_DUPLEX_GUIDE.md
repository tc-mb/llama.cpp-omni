# Running MiniCPM-o 4.5 Full-Duplex Voice on Windows with AMD ROCm / HIP

This guide details how to build and run **MiniCPM-o 4.5** with real-time **bidirectional Full-Duplex audio conversation** on Windows natively using **AMD Radeon GPUs** (RX 6000 series `gfx1030`, RX 7000 series `gfx1100`, etc.) via **ROCm 7.x / HIP**.

---

## 📑 Architecture Overview

The system consists of three coordinated layers:

1. **`llama-omni-server` (C++ Backend):**
   - High-performance LLM streaming inference (GGUF Q4_K_M).
   - Native hardware-accelerated **Audition APM** audio encoder.
   - **TTS Flow Model** + **Token2Wav Vocoder** for real-time speech synthesis.
   - Parallel pipeline (`encoder_thread`, `llm_thread`, `tts_thread`, `t2w_thread`).
2. **`worker.py` & `gateway.py` (MiniCPM-o-Demo):**
   - Session management, KV Cache sliding window preservation.
   - WebSocket streaming router (`/ws/duplex`).
   - Reference audio & voice cloning manager (`/api/default_ref_audio`, `/api/presets`).
3. **Web Frontend:**
   - Real-time Audio Duplex Interface (`https://localhost:8006/audio-duplex/audio_duplex.html`).
   - Live microphone waveform, dynamic latency control, voice presets.

---

## 🛠️ Prerequisites

1. **Operating System:** Windows 10 / 11 (64-bit).
2. **GPU:** AMD Radeon RX 6000 / 7000 / 8000 series (e.g. RX 6800 XT `gfx1030`, RX 7900 XTX `gfx1100`) with >= 12-16 GB VRAM.
3. **Software Toolchain:**
   - **Visual Studio 2022** (Desktop development with C++ & Windows 10/11 SDK).
   - **CMake** (>= 3.21) and **Ninja** (`pip install ninja`).
   - **Python** (3.10 - 3.12).
   - **ROCm / HIP SDK for Windows**: Official AMD ROCm TheRock tarball for Windows gfx103X (RDNA2): [therock-dist-windows-gfx103X-all-7.14.0.tar.gz](https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-windows-gfx103X-all-7.14.0.tar.gz) (or via [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)).

---

## 🔧 Essential C++ Fixes in `llama.cpp-omni`

When building `llama.cpp-omni` on Windows for real-time duplex streaming, several fixes are required:

1. **Windows Sockets & SSL (`tools/server/server-omni.cpp`):**
   - Initialize Winsock with `WSAStartup(MAKEWORD(2, 2), &wsaData)` in `main()`.
   - Use plain `httplib::Server` when SSL certificate paths are empty to avoid aborting.

2. **Duplex Mutex Deadlock Elimination (`tools/server/server-omni.cpp`):**
   - Removed broad `state.octx_mutex` locks around `stream_decode()` and `stream_prefill()`.
   - Thread safety is internally managed by `dup->llm_mtx`, `dup->encoder_mtx`, and `text_mtx`. Removing the outer transport lock allows concurrent audio ingestion and text/speech decoding.

3. **Duplex Prefill Routing & Nullptr Guard (`tools/omni/omni.cpp`):**
   - Route all duplex prefill requests (including `index = 0`) to `duplex_prefill(...)` when `system_prompt_initialized == true`.
   - Guard `ctx_omni->llm_thread_info` access in simplex fallback to prevent null pointer dereference.

4. **ROCm Device Library Paths (`ggml/src/ggml-hip/CMakeLists.txt`):**
   - Pass `--rocm-path` and `--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode` to `ggml-hip` compiler options.

---

## 🎯 AMD GPU Architecture Targets

Set `-DAMDGPU_TARGETS` based on your hardware:

| Architecture | Example GPUs / APUs | CMake Target Flag |
| :--- | :--- | :--- |
| **RDNA2 (dGPU)** | RX 6800 XT, RX 6800, RX 6700 XT, RX 6600 | `-DAMDGPU_TARGETS="gfx1030"` |
| **RDNA3 (dGPU)** | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600 | `-DAMDGPU_TARGETS="gfx1100"` |
| **RDNA3 (iGPU)** | Radeon 780M, 760M, 740M | `-DAMDGPU_TARGETS="gfx1103"` |
| **Strix Point / Halo** | Ryzen AI 300 series / Strix Halo | `-DAMDGPU_TARGETS="gfx1150"` (or `gfx1151`) |
| **RDNA4 (dGPU)** | RX 9070 XT, RX 9070, RX 9060 XT | `-DAMDGPU_TARGETS="gfx1200;gfx1201"` |

---

## ⚡ Building `llama-omni-server` with ROCm / HIP

Open **PowerShell** and execute:

```powershell
# 1. Set ROCm paths (adjust to your extracted TheRock ROCm SDK directory, e.g. C:\opt\rocm or $env:USERPROFILE\.cache\...)
$env:ROCM_PATH = "C:\path\to\rocm-sdk"
$env:HIP_PATH  = "$env:ROCM_PATH"
$env:PATH      = "$env:ROCM_PATH\bin;$env:ROCM_PATH\lib\llvm\bin;" + $env:PATH

cd C:\path\to\llama.cpp-omni

# 2. Configure with CMake & Ninja
cmake -B build_hip -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS="gfx1030" `
  -DCMAKE_C_COMPILER="$env:ROCM_PATH/lib/llvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:ROCM_PATH/lib/llvm/bin/clang++.exe" `
  -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe"

# 3. Compile the server executable
ninja -C build_hip llama-omni-server -j 8
```

This outputs `build_hip/bin/llama-omni-server.exe`, `build_hip/bin/ggml-hip.dll`, and `build_hip/bin/omni.dll`.

---

## 📦 Setting Up `MiniCPM-o-Demo`

### 1. Install Python Dependencies

```powershell
pip install fastapi uvicorn websockets httpx numpy soundfile librosa pyyaml ninja
```

### 2. Configure `config.json`

Edit `MiniCPM-o-Demo/config.json`:

```json
{
  "backend": "cpp",
  "llamacpp_root": "C:/path/to/llama.cpp-omni",
  "model_dir": "C:/path/to/models/MiniCPM-o-4_5-gguf",
  "llm_model": "MiniCPM-o-4_5-Q4_K_M.gguf",
  "ctx_size": 4096,
  "n_gpu_layers": 99,
  "gateway_port": 8006,
  "playback_delay_ms": 400
}
```

---

## 🚀 Running the Services

### Terminal 1: Start Worker (GPU Backend)
```powershell
cd C:\path\to\MiniCPM-o-Demo
python worker.py --port 22400
```
*Look for: `vision using ROCm0 backend` and `flowGGUFModelLoader: backend=ROCm0`.*

### Terminal 2: Start Gateway
```powershell
cd C:\path\to\MiniCPM-o-Demo
python gateway.py
```

---

## 🎙️ Using Full-Duplex Audio in Browser

1. Open **`https://localhost:8006/audio_duplex`** (or `https://localhost:8006/audio-duplex/audio_duplex.html`).
2. Accept the self-signed HTTPS certificate (required for browser microphone capture).
3. Choose a voice preset:
   - **English Call:** Uses `ref_en_dlc_1.wav` for natural English conversational tone.
   - **中文通话:** Standard Chinese assistant voice.
   - **Advanced ▴:** Enter a custom system prompt or upload your own `.wav` sample for instant voice cloning.
4. Click **Start** and begin speaking!

---

## 🧪 Hardware Benchmark & Real-World Test Background

### Test Rig Specifications
- **GPU:** AMD Radeon RX 6800 XT (16 GB GDDR6 VRAM, `gfx1030`, RDNA2)
- **OS:** Windows 11 Pro 64-bit (Native, no WSL)
- **Toolchain:** Visual Studio 2022 + AMD ROCm 7.14.0 TheRock SDK (`Clang 23`) + Ninja
- **Models Tested:**
  * LLM: `MiniCPM-o-4_5-Q4_K_M.gguf` (99 layers offloaded to VRAM)
  * Audio APM: `MiniCPM-o-4_5-audio-F16.gguf`
  * Vision VPM: `MiniCPM-o-4_5-vision-F16.gguf`
  * TTS Model: `tts/` (LLaMA architecture acoustic model)
  * Vocoder: `token2wav-gguf/`

### Key WebUI (`MiniCPM-o-Demo`) Integration Details
1. **Dynamic DLL Loading:** `cpp_backend.py` prepends `$ROCM_PATH/bin` and `$LLAMACPP_ROOT/build_hip/bin` to `os.environ["PATH"]`, enabling automatic loading of `amdhip64.dll`, `hipblas.dll`, and `rocblas.dll` without manual DLL copying.
2. **Workload Partitioning:**
   - **GPU (`ROCm0`):** LLM backbone (99 layers), SigLip2 vision encoder, and TTS Flow acoustic model run 100% on GPU VRAM.
   - **CPU:** HiFiGAN Token2Wav vocoder runs across 8 CPU threads for maximum stability and assertion safety.
3. **Playback Buffer Tuning:**
   - Setting `playback_delay_ms: 400` in `config.json` provides adequate buffer margin (`Ahead > 200ms`), achieving continuous, zero-dropout speech synthesis.

### Verified Startup & Performance Logs
```text
vision_ctx: vision using ROCm0 backend
alloc_compute_meta:      ROCm0 compute buffer size =   100.30 MiB
init tts....init t2w....flowGGUFModelLoader: init_backend device=gpu:0, backend=ROCm0
voc_hg2_model: CPU backend using 8 threads
omni_init success: {'success': True}
[prof] llm decode n_past=39->41 tokens=2 ms=25.2 listen=1 (~40 tokens/sec)
```
Inference latency (*Time to First Sound*) is instantaneous in real-time duplex streaming.

---

## 💡 Troubleshooting & Best Practices

- **Avoid Audio Feedback:** Use headphones during full-duplex conversations so your microphone does not capture the speaker output, preventing accidental interruptions.
- **Buffer Continuity:** If you encounter audio stuttering or dropouts on high-latency connections, increase `playback_delay_ms` to `350`–`400` ms in `config.json` or the web UI slider.
- **Unlimited Conversation Length:** Leave `Stop on KV pruning (sliding window)` unchecked in the Web UI to let the sliding window automatically evict old context tokens and continue the call indefinitely.

