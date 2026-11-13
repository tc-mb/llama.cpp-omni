# Qwen3-ForcedAligner

Forced alignment: given audio and a list of alignment units, return the start and end time of
every unit. Useful for subtitle timing, for driving animation off synthesized speech, and as a
sanity check on a TTS run (a transcript that does not match the audio produces spans that go
backwards or collapse to zero length).

The runtime lives in `qwen3-aligner.{h,cpp}` as a static library, `llama-aligner-cli` is a smoke
tool, and `llama-tts-server` exposes it over HTTP as `POST /v1/audio/align`.

## Model files

Two GGUFs, converted from [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf):

| File | Contents |
|---|---|
| `Qwen3-Aligner-LM-F16.gguf` | the qwen3 backbone (0.6B) and its vocabulary, standard `qwen3` architecture |
| `Qwen3-Aligner-Audio-F16.gguf` | the audio tower and multimodal projector under the existing `qwen3a` projector type, plus the timestamp head |

`convert_hf_to_gguf.py` does not know the `Qwen3ASRForTokenClassification` architecture, so the
conversion is done out of tree for now. What it produces beyond the usual clip/qwen3 tensors:

- `score.weight`, F32, `[n_embd, n_buckets]` (1024 × 5000), stored in the audio GGUF as an extra
  tensor. The clip loader tolerates unknown tensors; the aligner reads it by name through the gguf
  API and applies it on the CPU.
- `aligner.timestamp_segment_ms` (80), `aligner.timestamp_token_id` (151705) and
  `aligner.n_buckets` (5000) in the audio GGUF metadata. All three have defaults, so a file
  without them still loads.
- The original weights carry no audio positional embedding; the converter generates a 1500 ×
  d_model sinusoidal matrix (Whisper's formula), of which the `qwen3a` graph uses the first 13 rows.

## How it works

The input is `<|audio_start|>` + the audio embeddings + `<|audio_end|>`, then each unit followed by
two `<timestamp>` slots — one for its start, one for its end. One forward pass with
`embeddings = true` and `pooling_type = NONE` gives a hidden state per position; the timestamp
positions go through the 5000-way head and the argmax bucket index times `timestamp_segment_ms`
is the time in milliseconds.

Raw predictions can be locally out of order, so they are repaired exactly as the reference
implementation does: the longest non-decreasing subsequence marks the sane values, runs of up to
two outliers take the closer neighbour, longer runs are interpolated. The output is therefore
always non-decreasing across the flattened `[start0, end0, start1, end1, …]` sequence.

## Usage

```bash
llama-aligner-cli \
    --aligner-lm    Qwen3-Aligner-LM-F16.gguf \
    --aligner-audio Qwen3-Aligner-Audio-F16.gguf \
    -f speech.wav --units "hello,world" --json
```

Units are taken as given; nothing is tokenized or segmented here. Pass `--units-file` with one
unit per line when the units are not ASCII, since Windows consoles rewrite arguments through the
active code page.

Serving it:

```bash
llama-tts-server \
    --voxcpm2-base-lm  VoxCPM2-BaseLM-F16.gguf \
    --voxcpm2-acoustic VoxCPM2-Acoustic-F16.gguf \
    --aligner-lm       Qwen3-Aligner-LM-F16.gguf \
    --aligner-audio    Qwen3-Aligner-Audio-F16.gguf
```

```bash
curl http://127.0.0.1:8080/v1/audio/align \
  -H 'Content-Type: application/json' \
  -d '{"audio": "<base64 wav>", "units": ["hel", "lo"]}'
```

```json
{
  "sample_rate": 16000,
  "duration": 1.28,
  "units": [
    { "text": "hel", "start": 0.08, "end": 0.56 },
    { "text": "lo",  "start": 0.56, "end": 1.20 }
  ]
}
```

`/health` reports `"aligner": true` once it is loaded. Both `--aligner-lm` and `--aligner-audio`
must be given; if loading fails the endpoint reports that it is unavailable and TTS keeps working.
`--aligner-ctx-size` defaults to 4096, about five minutes of audio.
