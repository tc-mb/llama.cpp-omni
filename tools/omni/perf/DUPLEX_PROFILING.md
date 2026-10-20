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
| First audio latency (e2e) | Time from the first SPEAK frame push to the first generated wav chunk, plus configured startup buffering | P95 below the input frame interval |
| TTS RTF | Wall time from LLM decision done (`t_done`) to the last wav of that turn, divided by generated audio duration | Every matched turn below 1.0 |
| e2e RTF (info only) | Wall time from SPEAK first-frame push to the last wav, divided by audio duration | Not used for pass/fail |
| Playback continuity estimate | Simulated 1x playback of each turn using chunk completion times | No gap above the configured tolerance |
| Coverage | Failed frames and unmatched SPEAK/audio turns | No failed frames; complete matching and final markers |

`RTF` means real-time factor:

```text
TTS RTF = (t_last_wav - t_llm_done) / generated_audio_duration
e2e RTF = (t_last_wav - t_speak_push) / generated_audio_duration
```

For example, `TTS RTF = 0.7` means producing 1 second of audio takes about 0.7 seconds of TTS/pipeline wall time after the LLM decision, which is faster than real-time playback.

SPEAK turns and audio turns are matched by timestamps (latest unused SPEAK whose `t_push <=` first wav time), not by array index. A SPEAK turn with no audio does not shift later matches.

Zero-duration markers are retained for turn-closure validation but excluded from
first/last playable-chunk timing and playback simulation.

The report also includes a wav chunk duration section. This is informational only: input frames and output wav chunks are not expected to map one-to-one.

### Playback continuity

Total RTF does not capture a stall between chunks. For example, four one-second
chunks ready at `[200, 1000, 1800, 2600]` ms play continuously from 200 ms. The same
chunks ready at `[200, 2300, 2400, 2600]` ms have identical first/last completion
times and RTF, but playback stalls for 1100 ms after the first chunk.

For each turn, initialize `playback_end = first_chunk_ready + startup_buffer`.
For every playable chunk, accumulate `max(0, ready - playback_end)` as starvation,
then update `playback_end = max(playback_end, ready) + duration`. This models
immediate resumption after a stall. Zero-duration terminal markers do not count
as playable chunks. The report includes total starvation, largest gap, number
of gaps, and the minimum startup delay that would eliminate starvation in the
recorded trace. That minimum is a retrospective diagnostic, not an online
buffer-selection policy.

This is a **generation-side estimate**, using one server clock with instantaneous
delivery. It does not measure network jitter, client receipt, actual playback,
barge-in latency, or user-perceived quality. A pass applies only to this trace
under the stated buffering assumptions. Timestamp matching is also a heuristic;
it does not prove response ownership. Causal output identities and client
playback telemetry are separate future work.

Invalid/nonfinite timings, unclosed audio turns, partial IDs, interleaved audio
turns, missing SPEAK/audio coverage, and empty reports cannot pass. Explicitly
canceled turns are not currently represented by the report schema; do not omit
their data and interpret the remainder as a complete successful run.

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
- `--startup-buffer-ms <ms>` adds a non-negative startup delay to simulated playback (default `0`). This delay also increases reported first-audio latency, so extra buffering cannot conceal its responsiveness cost.
- `--max-gap-ms <ms>` sets the largest allowed simulated starvation gap (default `0`). Both this tolerance and startup delay are printed in the report.
- `--no-tts` skips audio generation. The report can still show LLM latency, but the final duplex verdict is `incomplete` (exit code 3), not pass.
- `--vision-backend <metal|coreml>` selects the vision backend when supported.

## Interpreting Results

The recorded run passes generation-side duplex checks only when TTS was actually exercised and all required checks pass:

```text
[PASS] LLM decision latency
[PASS] First audio latency (e2e)
[PASS] TTS RTF
[PASS] No failed frames
[PASS] Playback continuity estimate
```

If LLM decision latency fails, frame processing is slower than the input rate. If first audio latency fails, users may notice delayed responses. If TTS RTF fails, audio generation is slower than playback and may underrun.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Pass: the recorded generation-side timeline meets the configured checks |
| 2 | Fail: at least one real-time check failed |
| 3 | Incomplete: `--no-tts`, no SPEAK/audio coverage, or missing audio timeline; do not treat as duplex-ready |

Run the model-free regression suite (also run by `omni-perf-tests.yml`):

```bash
python3 -m unittest discover -s tools/omni/perf -p 'test_*.py' -v
```
