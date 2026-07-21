# Duplex profiling

This directory contains a small profiling tool for checking whether a machine can run MiniCPM-o duplex mode in real time.

The profiler reuses the normal omni duplex API and writes two reports:

- `perf_report.json`: structured timing data
- `perf_report.md`: human-readable summary and pass/fail result

## What It Measures

The report focuses on three real-time metrics:

| Metric | Meaning | Pass criterion |
|---|---|---|
| LLM decision latency | Time from pushing one input frame to receiving the LISTEN/SPEAK decision | P95 below the input frame interval |
| First audio latency | Time from the first SPEAK frame in a turn to the first generated wav chunk | P95 below the input frame interval |
| Audio RTF | Wall time spent to generate one second of audio | Average RTF below 1.0 |

`RTF` means real-time factor:

```text
RTF = audio generation wall time / generated audio duration
```

For example, `RTF = 0.7` means generating 1 second of audio takes about 0.7 seconds, which is faster than real-time playback.

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
- `--no-tts` skips audio generation and reports only frame-level results.
- `--vision-backend <metal|coreml>` selects the vision backend when supported.

## Interpreting Results

The machine is considered suitable for duplex mode when all required checks pass:

```text
[PASS] LLM decision latency
[PASS] First audio latency
[PASS] Audio RTF
```

If LLM decision latency fails, frame processing is slower than the input rate. If first audio latency fails, users may notice delayed responses. If audio RTF fails, audio generation is slower than playback and may underrun.

The profiler exits with `0` when the report passes and `2` when the machine does not meet the real-time criteria.
