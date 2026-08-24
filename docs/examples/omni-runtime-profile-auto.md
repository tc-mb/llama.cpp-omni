# Omni runtime profile example

`--profile auto` requires a static JSON file. By default the server reads
`<model-dir>/omni-runtime-profile.json`; use `--profile-config PATH` to place
the file elsewhere. Relative model paths below are resolved from
`--model-dir`. The server exits if the file, a required field, a model file, or
a requested device is unavailable.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./build/bin/llama-omni-server \
    --profile auto \
    --model-dir /models/MiniCPM-o-4_5-gguf \
    --host 0.0.0.0 \
    --port 9060
```

Example `omni-runtime-profile.json`:

```json
{
  "schema_version": 1,
  "profile": "auto",
  "llm": {
    "model": "MiniCPM-o-4_5-Q4_K_M.gguf",
    "quantization": "Q4_K_M",
    "device": "CUDA0",
    "n_gpu_layers": -2
  },
  "vision": {
    "model": "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "device": "CUDA1"
  },
  "audio": {
    "model": "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "device": "CUDA1"
  },
  "tts": {
    "model": "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "device": "CUDA0",
    "gpu_layers": -1
  },
  "projector": {
    "model": "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "device": "CUDA1"
  },
  "token2wav": {
    "model_dir": "token2wav-gguf",
    "device": "CUDA0",
    "threads": 8
  },
  "runtime": {
    "n_ctx": 8192,
    "duplex": true,
    "async": true,
    "vpm_batch_encode": true
  }
}
```

The example uses fixed GGML backend device names: `CUDA0` for the first visible
CUDA device and `CUDA1` for the second. Use `cpu` for a CPU module. The names
must match the devices visible to the current process; a missing device causes
profile validation to fail. `primary` and `secondary` remain accepted as
backward-compatible aliases. `--print-effective-config` validates the file and
prints the resulting paths and placements without loading model weights.

To inspect the resolved configuration without loading model weights or opening
the HTTP port, add `--print-effective-config` to the same command.
