# VoxCPM / VoxCPM2 for llama.cpp-omni

C++/ggml inference implementation of VoxCPM-0.5B, VoxCPM-1.5, and VoxCPM2.

## Supported Versions

The three model versions share the same high-level inference pipeline:

```text
BaseLM -> ResidualLM -> FSQ -> LocEnc/LocDiT CFM -> AudioVAE
```

They differ in model dimensions, acoustic topology, LocDiT prefix layout, and AudioVAE sample rate:

| Model | Config architecture | Version metadata | BaseLM | ResidualLM | LocEnc / LocDiT | patch | FSQ dim | AudioVAE output |
|-------|---------------------|------------------|--------|------------|-----------------|-------|---------|-----------------|
| VoxCPM-0.5B | `voxcpm` | `voxcpm.model_version = 0.5` | 24 layers, h=1024 | 6 layers, h=1024, RoPE | 4 / 4 layers, h=1024 | 2 | 256 | 16 kHz, dec_dim=1536 |
| VoxCPM-1.5 | `voxcpm` | `voxcpm.model_version = 1.5` | 24 layers, h=1024 | 8 layers, h=1024, RoPE | 8 / 8 layers, h=1024 | 4 | 256 | 44.1 kHz, dec_dim=2048 |
| VoxCPM2 | `voxcpm2` | `voxcpm.model_version = 2.0` | 28 layers, h=2048 | 8 layers, h=2048, no RoPE | 12 / 12 layers, h=1024 | 4 | 512 | 48 kHz, dec_dim=2048 |

Additional implementation differences:

| Area | VoxCPM-0.5B / VoxCPM-1.5 | VoxCPM2 |
|------|---------------------------|---------|
| Model checkpoint | 0.5B may use `pytorch_model.bin`; 1.5 uses `model.safetensors` | `model.safetensors` |
| Acoustic projection fusion | No `fusion_concat_proj`; residual fusion and DiT condition use elementwise add | Has `fusion_concat_proj`; residual fusion uses concat + linear and DiT condition uses concat |
| LocDiT generated prefix | One token: `(mu + time)` | Separate token(s): `mu_token(s), time` |
| ResidualLM `no_rope` | `false` | `true` |
| AudioVAE conditioning | No sample-rate scale/bias conditioning | Uses `scale_bias` with sample-rate bins |
| Default output file names | `VoxCPM-0.5B-*.gguf` or `VoxCPM-1.5-*.gguf` | `VoxCPM2-*.gguf` |

Current GGUF files use the `voxcpm.*` metadata namespace. Old acoustic GGUF files exported with `voxcpm2.*` metadata should be regenerated with the current converter.

## Model Conversion

Convert official PyTorch weights to GGUF format. The converter accepts a model file, a model directory containing `model.safetensors` / `pytorch_model.bin`, or a HuggingFace-style directory with `*.safetensors` files:

```bash
cd tools/omni/voxcpm2
pip install torch safetensors numpy gguf

python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors_or_pytorch_model.bin \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output /path/to/output \
    [--tokenizer-dir /path/to/tokenizer_dir] \
    [--dtype f16|f32]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Model file (`model.safetensors`, `pytorch_model.bin`) or directory |
| `--vae` | (required) | Path to `audiovae.pth` |
| `--config` | (required) | Path to `config.json` |
| `--output` | `./` | Output directory for GGUF files |
| `--tokenizer-dir` | same as `--config` dir | Directory containing `tokenizer.json` / `tokenizer_config.json` |
| `--dtype` | `f16` | Output weight dtype: `f16` or `f32` (use `f32` for quantization input) |

The converter auto-detects model version from `config.json`'s `architecture` field and names output files accordingly.

By default, this produces F16 GGUF files:

| Model | BaseLM output | Acoustic output |
|-------|---------------|-----------------|
| VoxCPM-0.5B | `VoxCPM-0.5B-BaseLM-F16.gguf` | `VoxCPM-0.5B-Acoustic-F16.gguf` |
| VoxCPM-1.5 | `VoxCPM-1.5-BaseLM-F16.gguf` | `VoxCPM-1.5-Acoustic-F16.gguf` |
| VoxCPM2 | `VoxCPM2-BaseLM-F16.gguf` | `VoxCPM2-Acoustic-F16.gguf` |

To produce F32 GGUF files (needed for quantization):

```bash
python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output /path/to/output \
    --dtype f32
```

## BaseLM Quantization (Q8_0)

The BaseLM GGUF can be quantized to Q8_0 to reduce model size by ~2x with minimal quality loss. Quantization requires F32 input:

```bash
# 1. Convert to F32
python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output ./gguf \
    --dtype f32

# 2. Build llama-quantize if not already built
cmake --build build -j$(nproc) --target llama-quantize

# 3. Quantize BaseLM to Q8_0 (~1.5GB vs ~3.0GB F16)
./build/bin/llama-quantize \
    ./gguf/VoxCPM2-BaseLM-F32.gguf \
    ./gguf/VoxCPM2-BaseLM-Q8_0.gguf \
    Q8_0

# 4. Run inference with quantized BaseLM
./build/bin/voxcpm2-cli \
    -t "Hello, welcome to VoxCPM2." \
    -o output.wav \
    ./gguf/VoxCPM2-BaseLM-Q8_0.gguf \
    ./gguf/VoxCPM2-Acoustic-F32.gguf
```

## Build

```bash
cd /path/to/llama.cpp-omni

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target voxcpm2-cli
```

Binary: `build/bin/voxcpm2-cli`

## Usage

```bash
./build/bin/voxcpm2-cli [options] <BaseLM.gguf> <Acoustic.gguf>
```

### Basic TTS

```bash
./build/bin/voxcpm2-cli \
    -t "Hello, welcome to VoxCPM2." \
    -o output.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### Voice Design

Describe the desired voice in parentheses at the start of the text:

```bash
./build/bin/voxcpm2-cli \
    -t "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!" \
    -o voice_design.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

Also works for Chinese and other languages:

```bash
./build/bin/voxcpm2-cli \
    -t "(A deep, calm male voice)This model supports thirty languages natively." \
    -o multilingual.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### Voice Cloning

Provide a reference audio file (WAV, mono, any sample rate):

```bash
./build/bin/voxcpm2-cli \
    -t "This is a cloned voice generated by VoxCPM2." \
    -r speaker.wav \
    -o clone.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### Streaming

```bash
./build/bin/voxcpm2-cli \
    -t "Streaming test with VoxCPM2." \
    --stream \
    -o streaming.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --text` | (required) | Input text to synthesize |
| `-o, --output` | `output.wav` | Output WAV file path |
| `-r, --reference` | — | Reference WAV for voice cloning |
| `--stream` | — | Streaming output mode |
| `--steps` | 200 | Max decode loop steps. For non-streaming text-only TTS, the default is treated as auto and reduced from text length |
| `--timesteps` | 10 | CFM inference timesteps |
| `--cfg` | 2.0 | CFG guidance scale |
| `--temperature` | 1.0 | Noise temperature |
| `--seed` | 42 | Random seed |
| `--cpu` | — | Use CPU backend (default: GPU) |
| `--n-gpu-layers` | -1 | Number of layers to offload to GPU (default: all) |

### Decode Step Limit

`--steps` is an upper bound for acoustic decode patches, not always the exact number of generated patches.

For non-streaming text-only TTS, any `--steps` value of 200 or higher is treated as automatic mode with that value as the upper cap:

```text
effective_max_steps = min(steps, max(15, text_token_count * 3 + 15))
```

With the CLI default, this is `min(200, max(15, text_token_count * 3 + 15))`. This avoids overly long output when the stop predictor does not fire early. Values below 200 are used directly as the max step count. Voice cloning and streaming paths currently use `--steps` directly and do not apply this text-length auto reduction.

One decode step corresponds to one latent patch. The approximate audio duration per step depends on model version:

| Model | patch_size | Approx audio per step |
|-------|------------|-----------------------|
| VoxCPM-0.5B | 2 | 80 ms |
| VoxCPM-1.5 | 4 | 160 ms |
| VoxCPM2 | 4 | 160 ms |

## Performance

RTX 4090, F16 weights:

| Text Length | Effective Steps | Audio Duration | Elapsed | RTF |
|-------------|-----------------|---------------|---------|-----|
| Short | 8 | 1.28s | 0.49s | 0.38 |
| Short | 30 | 4.80s | 1.24s | 0.26 |
| Long | ~100+ | ~16s+ | — | ~0.25–0.35 |

## Online Serving (OpenAI-compatible API)

The `llama-tts-server` provides a standalone OpenAI-compatible TTS endpoint at `/v1/audio/speech`, similar to vLLM-Omni. It runs independently without requiring an LLM model.

### Build

```bash
cmake --build build -j$(nproc) --target llama-tts-server
```

### Start Server

```bash
./build/bin/llama-tts-server \
    --voxcpm2-base-lm /path/to/VoxCPM2-BaseLM-F16.gguf \
    --voxcpm2-acoustic /path/to/VoxCPM2-Acoustic-F16.gguf \
    --port 8080
```

### Dynamic Loading (without restart)

```bash
curl -X POST http://localhost:8080/v1/voxcpm2/init \
  -H "Content-Type: application/json" \
  -d '{
    "base_lm": "/path/to/VoxCPM2-BaseLM-F16.gguf",
    "acoustic": "/path/to/VoxCPM2-Acoustic-F16.gguf",
    "n_gpu_layers": -1
  }'
```

### Basic TTS

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "Hello, welcome to VoxCPM2!",
    "voice": "default"
  }' \
  --output out.wav
```

### Voice Design

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    "voice": "default"
  }' \
  --output voice_design.wav
```

### Voice Cloning

Provide a base64-encoded reference WAV:

```bash
REF_B64=$(base64 -w0 speaker.wav)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"voxcpm2\",
    \"input\": \"This is a cloned voice.\",
    \"voice\": \"default\",
    \"reference_audio\": \"$REF_B64\"
  }" \
  --output clone.wav
```

### Streaming

```bash
curl http://localhost:8080/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "Streaming test with VoxCPM2.",
    "voice": "default"
  }' \
  --output streaming.wav
```

### API Reference

The `model` parameter accepts `"voxcpm"` (also accepts `"voxcpm2"` for backward compatibility). A single server instance loads one VoxCPM model at a time — the loaded version (0.5B, 1.5, or 2) depends on which GGUF files are provided at model loading or via `/v1/voxcpm2/init`.

#### `POST /v1/audio/speech`

OpenAI-compatible TTS endpoint.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model identifier. Use `"voxcpm"` (supports VoxCPM-0.5B, VoxCPM-1.5, and VoxCPM2) |
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice identifier (reserved) |
| `response_format` | string | `"wav"` | Output format: `wav` or `pcm` |
| `reference_audio` | string | — | Base64-encoded WAV for voice cloning |
| `seed` | int | `42` | Random seed |
| `cfg_value` | float | `2.0` | CFG guidance scale |
| `inference_timesteps` | int | `10` | CFM inference timesteps |
| `max_steps` | int | `200` | Max decode steps |
| `temperature` | float | `1.0` | Noise temperature |

#### `POST /v1/audio/speech/stream`

Streaming variant. Returns chunked WAV audio as it's generated.

#### `POST /v1/voxcpm2/init`

Load or reload VoxCPM2 runtime dynamically.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_lm` | string | (required) | BaseLM GGUF path |
| `acoustic` | string | (required) | Acoustic GGUF path |
| `n_gpu_layers` | int | `-1` | GPU layers (-1 = all) |

#### `GET /v1/audio/speech/models`

List loaded VoxCPM2 model info.
