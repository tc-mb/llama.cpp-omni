# Duplex profiling

This directory contains a small profiling tool for checking whether a machine can run MiniCPM-o duplex mode in real time.

The profiler reuses the normal omni duplex API and writes two reports:

- `perf_report.json`: structured timing data
- `perf_report.md`: human-readable summary and pass/fail result

## What It Measures

The report focuses on these real-time metrics:

| Metric | Meaning | Pass criterion |
|---|---|---|
| LLM decision latency | Time from pushing one input frame to receiving the LISTEN/SPEAK decision | P95 below the input frame interval |
| First audio latency (e2e) | Time from the first SPEAK frame push in a turn to the first generated wav chunk | P95 below the input frame interval |
| TTS RTF | Wall time from LLM decision done (`t_done`) to the last wav of that turn, divided by generated audio duration | Average TTS RTF below 1.0 |
| e2e RTF (info only) | Wall time from SPEAK first-frame push to the last wav, divided by audio duration | Not used for pass/fail |

`RTF` means real-time factor:

```text
TTS RTF = (t_last_wav - t_llm_done) / generated_audio_duration
e2e RTF = (t_last_wav - t_speak_push) / generated_audio_duration
```

For example, `TTS RTF = 0.7` means producing 1 second of audio takes about 0.7 seconds of TTS/pipeline wall time after the LLM decision, which is faster than real-time playback.

SPEAK turns and audio turns are matched by timestamps (latest unused SPEAK whose `t_push <=` first wav time), not by array index. A SPEAK turn with no audio does not shift later matches.

The report also includes a wav chunk duration section. This is informational only: input frames and output wav chunks are not expected to map one-to-one.

## Usage

Build and run the default duplex profiling case:

```bash
tools/omni/perf/run_perf.sh --build \
  -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
```

Run with a custom test set:

```bash
tools/omni/perf/run_perf.sh \
  -m <llm.gguf> \
  --test <input-prefix> <frame-count>
```

Analyze an existing JSON report:

```bash
python3 tools/omni/perf/analyze_perf.py tools/omni/output/perf_report.json \
  --interval-ms 1000 \
  --md tools/omni/output/perf_report.md
```

## Options

- `--stream-interval <ms>` controls how often input frames are pushed. The default is `1000`, which simulates one frame per second.
- `--interval-ms <ms>` controls the real-time threshold used by `analyze_perf.py`. If omitted, the analyzer reads it from the JSON metadata.
- `--no-tts` skips audio generation. The report can still show LLM latency, but the final duplex verdict is `incomplete` (exit code 3), not pass.
- `--vision-backend <metal|coreml>` selects the vision backend when supported.

## Interpreting Results

The machine is considered suitable for duplex mode only when TTS was actually exercised and all required checks pass:

```text
[PASS] LLM decision latency
[PASS] First audio latency (e2e)
[PASS] TTS RTF
```

If LLM decision latency fails, frame processing is slower than the input rate. If first audio latency fails, users may notice delayed responses. If TTS RTF fails, audio generation is slower than playback and may underrun.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Pass: machine looks suitable for duplex |
| 2 | Fail: at least one real-time check failed |
| 3 | Incomplete: `--no-tts`, no SPEAK/audio coverage, or missing audio timeline; do not treat as duplex-ready |
