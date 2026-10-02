#!/usr/bin/env python3
# Duplex profiling usage:
#   OMNI_GPU_PROF=1 \
#   OMNI_GPU_PROF_INTERVAL_MS=20 \
#   OMNI_GPU_PROF_DEVICES=0 \
#   OMNI_GPU_PROF_FILE=duplex.gpu.log \
#   ./bin/llama-omni-test-duplex \
#     -m <model.gguf> \
#     --omni \
#     --test <duplex_test_case_prefix> <case_count> \
#     --stream-interval 1000 \
#     --ref-audio <ref_audio.wav> \
#     -ngl 99 \
#     -c 4096 \
#     -o <output_dir> \
#     2>&1 | tee duplex.log
#
# GPU sampling:
#   - OMNI_GPU_PROF_DEVICES uses NVML physical GPU indexes, e.g. 0 or 0,2.
#   - Without OMNI_GPU_PROF_DEVICES, the first CUDA_VISIBLE_DEVICES entry is sampled.
#   - Without either variable, only device 0 is sampled.
#
# Analyze:
#   python3 tools/omni/test/analyze_duplex_perf.py \
#     --log <duplex.log> \
#     --gpu-log <duplex.gpu.log> \
#     --out-dir <report_dir>
"""Parse duplex omni perf logs and generate SVG/CSV/Markdown assets.

This script uses only the Python standard library so it can run on the benchmark
host without creating a conda environment.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PERF_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[DUPLEX_PERF\] (?:seq=(?P<seq>\d+) )?stage=(?P<stage>\S+) event=(?P<event>\S+) "
    r"chunk=(?P<chunk>-?\d+) t_ms=(?P<t_ms>[-\d.]+) dur_ms=(?P<dur_ms>[-\d.]+) "
    r"rss_mb=(?P<rss_mb>[-\d.]+|NA) gpu_used_mb=(?P<gpu_used_mb>[-\d.]+|NA) "
    r"gpu_total_mb=(?P<gpu_total_mb>[-\d.]+|NA) n_past=(?P<n_past>-?\d+) "
    r'detail="(?P<detail>[^"]*)"'
)
GPU_SAMPLE_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[DUPLEX_GPU\] sample_id=(?P<sample_id>\d+) t_ms=(?P<t_ms>[-\d.]+) "
    r"device=(?P<device>\d+) sm_util_pct=(?P<sm_util_pct>[-\d.]+|NA) "
    r"mem_util_pct=(?P<mem_util_pct>[-\d.]+|NA) gpu_used_mb=(?P<gpu_used_mb>[-\d.]+|NA) "
    r"gpu_total_mb=(?P<gpu_total_mb>[-\d.]+|NA) power_w=(?P<power_w>[-\d.]+|NA) "
    r"temp_c=(?P<temp_c>[-\d.]+|NA) graphics_clock_mhz=(?P<graphics_clock_mhz>[-\d.]+|NA) "
    r"mem_clock_mhz=(?P<mem_clock_mhz>[-\d.]+|NA)"
)
GPU_STATUS_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r'\[DUPLEX_GPU\] event=(?P<event>\S+) reason="(?P<reason>[^"]*)"'
)
CHUNK_RE = re.compile(r"prefill:\s*([0-9.]+) s \| decode:\s*([0-9.]+) s \| total:\s*([0-9.]+) s \| n_past:\s*(\d+)")
OPT4_CHUNK_RE = re.compile(
    r"--- Chunk (?P<seq>\d+)/(?P<count>\d+) --- "
    r"prefill (?P<prefill_ms>[0-9.]+)ms \| decode (?P<decode_ms>[0-9.]+)ms \| "
    r"e2e (?P<e2e_ms>[0-9.]+)ms \| n_past (?P<n_past>\d+) \| "
    r"<\|(?P<decision>speak|listen)\|>(?: \"(?P<text>[^\"]*)\")?"
)
DECISION_RE = re.compile(r"决策:\s*<\|(speak|listen)\|>(?:\s*→\s*\"([^\"]*)\")?")

STAGE_ROWS = [
    ("duplex.encode", "#4c78a8"),
    ("duplex.llm.prefill", "#b279a2"),
    ("duplex.llm.decode", "#e45756"),
    ("tts.condition", "#8cd17d"),
    ("tts.prefill", "#54a24b"),
    ("tts.decode", "#2ca02c"),
    ("t2w.infer", "#3182bd"),
    ("t2w.write", "#9e9ac8"),
]

REQUESTED_STAGES = {stage for stage, _ in STAGE_ROWS}
TOKEN_SPEED_STAGES = ["duplex.llm.prefill", "duplex.llm.decode", "tts.prefill", "tts.decode"]

FRAME_STAGE_ALIASES = {
    "encode": ["duplex.encode"],
    "llm_prefill": ["duplex.llm.prefill"],
    "llm_decode": ["duplex.llm.decode"],
    "tts": ["tts.condition", "tts.prefill", "tts.decode"],
    "t2w_infer": ["t2w.infer"],
    "t2w_write": ["t2w.write"],
}

FRAME_CORE_STAGES = {
    stage
    for names in FRAME_STAGE_ALIASES.values()
    for stage in names
}


@dataclass
class TokenStageAccumulator:
    n: int = 0
    tokens: int = 0
    total_ms: float = 0.0
    event_tokens_per_s: list[float] = field(default_factory=list)
    event_ms_per_token: list[float] = field(default_factory=list)


@dataclass
class GpuStageAccumulator:
    n_intervals: int = 0
    n_samples: int = 0
    total_ms: float = 0.0
    estimated_samples: int = 0
    sm: list[float] = field(default_factory=list)
    mem: list[float] = field(default_factory=list)
    power: list[float] = field(default_factory=list)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def avg(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def pctl(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def parse_metric(value: str) -> float:
    return -1.0 if value == "NA" else float(value)


def parse_detail(detail: str):
    result = {}
    for part in detail.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def detail_int(detail, key, default=0):
    try:
        return int(detail.get(key, default))
    except (TypeError, ValueError):
        return default


def token_count_for_event(event):
    detail = parse_detail(event["detail"])
    if "tokens" in detail:
        return detail_int(detail, "tokens", -1)
    if event["stage"] == "tts.decode":
        return detail_int(detail, "audio_tokens_generated", -1)
    if event["stage"] == "duplex.llm.prefill":
        return detail_int(detail, "n_past_delta", -1)
    if event["stage"] == "duplex.llm.decode":
        return detail_int(detail, "tokens", -1)
    return -1


def parse_log(path: Path, gpu_log_path=None):
    text = path.read_text(errors="replace")
    gpu_text = ""
    if gpu_log_path and gpu_log_path != path:
        gpu_text = gpu_log_path.read_text(errors="replace")
    combined_gpu_text = text + "\n" + gpu_text

    events = []
    for match in PERF_RE.finditer(text):
        group = match.groupdict()
        events.append({
            "wall": group["wall"],
            "seq": int(group["seq"]) if group.get("seq") is not None else -1,
            "stage": group["stage"],
            "event": group["event"],
            "chunk": int(group["chunk"]),
            "t_ms": float(group["t_ms"]),
            "dur_ms": float(group["dur_ms"]),
            "rss_mb": parse_metric(group["rss_mb"]),
            "gpu_used_mb": parse_metric(group["gpu_used_mb"]),
            "gpu_total_mb": parse_metric(group["gpu_total_mb"]),
            "n_past": int(group["n_past"]),
            "detail": group["detail"],
        })

    gpu_samples = []
    for match in GPU_SAMPLE_RE.finditer(combined_gpu_text):
        group = match.groupdict()
        gpu_samples.append({
            "wall": group["wall"],
            "sample_id": int(group["sample_id"]),
            "t_ms": float(group["t_ms"]),
            "device": int(group["device"]),
            "sm_util_pct": parse_metric(group["sm_util_pct"]),
            "mem_util_pct": parse_metric(group["mem_util_pct"]),
            "gpu_used_mb": parse_metric(group["gpu_used_mb"]),
            "gpu_total_mb": parse_metric(group["gpu_total_mb"]),
            "power_w": parse_metric(group["power_w"]),
            "temp_c": parse_metric(group["temp_c"]),
            "graphics_clock_mhz": parse_metric(group["graphics_clock_mhz"]),
            "mem_clock_mhz": parse_metric(group["mem_clock_mhz"]),
        })
    gpu_samples.sort(key=lambda row: (row["device"], row["t_ms"], row["sample_id"]))

    gpu_statuses = []
    for match in GPU_STATUS_RE.finditer(combined_gpu_text):
        group = match.groupdict()
        gpu_statuses.append({
            "wall": group["wall"],
            "event": group["event"],
            "reason": group["reason"],
        })

    chunks = []
    for match in OPT4_CHUNK_RE.finditer(text):
        group = match.groupdict()
        chunks.append({
            "chunk": int(group["seq"]),
            "prefill_ms": float(group["prefill_ms"]),
            "decode_ms": float(group["decode_ms"]),
            "total_ms": float(group["e2e_ms"]),
            "n_past": int(group["n_past"]),
            "decision": group["decision"],
            "text": group["text"] or "",
        })
    if not chunks:
        decisions = list(DECISION_RE.finditer(text))
        for idx, match in enumerate(CHUNK_RE.finditer(text)):
            decision = decisions[idx].group(1) if idx < len(decisions) else ""
            decision_text = decisions[idx].group(2) if idx < len(decisions) and decisions[idx].group(2) else ""
            chunks.append({
                "chunk": idx,
                "prefill_ms": float(match.group(1)) * 1000.0,
                "decode_ms": float(match.group(2)) * 1000.0,
                "total_ms": float(match.group(3)) * 1000.0,
                "n_past": int(match.group(4)),
                "decision": decision,
                "text": decision_text,
            })
    return text, events, chunks, gpu_samples, gpu_statuses


def stage_stats(events):
    grouped = defaultdict(list)
    for event in events:
        if event["stage"] not in REQUESTED_STAGES:
            continue
        if event["event"] == "end" and event["dur_ms"] >= 0:
            grouped[event["stage"]].append(event["dur_ms"])
    stats = {}
    for stage, values in grouped.items():
        stats[stage] = {
            "n": len(values),
            "total": sum(values),
            "avg": avg(values),
            "p50": statistics.median(values),
            "p90": pctl(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    return stats


def stage_token_stats(events, speak_frame_ids=None):
    filter_speak_frames = speak_frame_ids is not None
    speak_frame_ids = set(speak_frame_ids or [])
    grouped = defaultdict(TokenStageAccumulator)
    for event in events:
        if event["stage"] not in TOKEN_SPEED_STAGES or event["event"] != "end" or event["dur_ms"] < 0:
            continue
        if filter_speak_frames and event["chunk"] not in speak_frame_ids:
            continue
        tokens = token_count_for_event(event)
        if tokens < 0:
            continue
        row = grouped[event["stage"]]
        row.n += 1
        row.tokens += tokens
        row.total_ms += event["dur_ms"]
        if tokens > 0 and event["dur_ms"] > 0:
            row.event_tokens_per_s.append(tokens * 1000.0 / event["dur_ms"])
            row.event_ms_per_token.append(event["dur_ms"] / tokens)

    stats = {}
    for stage, row in grouped.items():
        tokens = row.tokens
        total_ms = row.total_ms
        stats[stage] = {
            "n": row.n,
            "tokens": tokens,
            "total_ms": total_ms,
            "tokens_per_s": tokens * 1000.0 / total_ms if tokens > 0 and total_ms > 0 else 0.0,
            "ms_per_token": total_ms / tokens if tokens > 0 else 0.0,
            "avg_event_tokens_per_s": avg(row.event_tokens_per_s),
            "p50_event_tokens_per_s": statistics.median(row.event_tokens_per_s) if row.event_tokens_per_s else 0.0,
            "avg_event_ms_per_token": avg(row.event_ms_per_token),
        }
    return stats


def tts_token_rows(events):
    rows = []
    for event in sorted(events, key=lambda item: (item["t_ms"], item.get("seq", -1))):
        if event["stage"] != "tts.decode" or event["event"] != "end":
            continue
        detail = parse_detail(event["detail"])
        def to_int(key, default=0):
            try:
                return int(detail.get(key, default))
            except ValueError:
                return default
        llm_tokens = to_int("llm_tokens")
        filtered_llm_tokens = to_int("filtered_llm_tokens", llm_tokens)
        condition_tokens = to_int("condition_tokens")
        compute_tokens = to_int("compute_tokens", to_int("tokens", -1))
        audio_tokens = to_int("audio_tokens_generated", -1)
        if compute_tokens < 0:
            compute_tokens = audio_tokens
        rows.append({
            "chunk": event["chunk"],
            "tts_chunk": to_int("tts_chunk", event["chunk"]),
            "dur_ms": event["dur_ms"],
            "llm_tokens": llm_tokens,
            "filtered_llm_tokens": filtered_llm_tokens,
            "condition_tokens": condition_tokens,
            "compute_tokens": compute_tokens,
            "audio_tokens_generated": audio_tokens,
            "is_end_of_turn": to_int("is_end_of_turn"),
            "flush_only": to_int("flush_only"),
            "compute_tokens_per_s": compute_tokens * 1000.0 / event["dur_ms"] if compute_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "audio_tokens_per_ms": audio_tokens / event["dur_ms"] if audio_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "audio_tokens_per_s": audio_tokens * 1000.0 / event["dur_ms"] if audio_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "ms_per_audio_token": event["dur_ms"] / audio_tokens if audio_tokens > 0 else 0.0,
            "has_audio_token_detail": int(audio_tokens >= 0),
        })
    return rows


def build_stage_intervals(events):
    starts = defaultdict(list)
    intervals = []
    for event in sorted(events, key=lambda item: (item["t_ms"], item.get("seq", -1))):
        key = (event["stage"], event["chunk"])
        if event["event"] == "start":
            starts[key].append(event)
        elif event["event"] == "end" and event["dur_ms"] >= 0:
            start = starts[key].pop(0) if starts[key] else None
            begin = start["t_ms"] if start else event["t_ms"] - event["dur_ms"]
            finish = event["t_ms"]
            if finish >= begin:
                intervals.append({
                    "stage": event["stage"],
                    "chunk": event["chunk"],
                    "begin_ms": begin,
                    "end_ms": finish,
                    "duration_ms": finish - begin,
                })
    return intervals


def intervals_for_aliases(by_stage, aliases):
    for stage in aliases:
        if by_stage.get(stage):
            return by_stage[stage]
    rows = []
    for stage in aliases:
        rows.extend(by_stage.get(stage, []))
    return rows


def sum_duration(rows):
    return sum(row["duration_ms"] for row in rows)


def max_end(rows):
    return max((row["end_ms"] for row in rows), default=0.0)


def infer_frame_decisions(events, chunks):
    decisions = {chunk["chunk"]: chunk.get("decision", "") for chunk in chunks if chunk.get("chunk", -1) > 0}
    for event in events:
        chunk = event["chunk"]
        if chunk <= 0 or decisions.get(chunk):
            continue
        detail = parse_detail(event["detail"])
        if event["stage"] == "duplex.llm.decode" and event["event"] == "end":
            if detail.get("is_speak") == "1":
                decisions[chunk] = "speak"
            elif detail.get("is_speak") == "0":
                decisions[chunk] = "listen"
    return decisions


def build_frame_summaries(events, chunks, intervals, sla_ms=1000.0):
    chunk_info = {chunk["chunk"]: chunk for chunk in chunks if chunk.get("chunk", -1) > 0}
    decisions = infer_frame_decisions(events, chunks)
    frame_ids = {interval["chunk"] for interval in intervals if interval["chunk"] > 0 and interval["stage"] in FRAME_CORE_STAGES}
    frame_ids.update(chunk_info)
    frame_ids.update(decisions)

    rows = []
    for frame_id in sorted(frame_ids):
        frame_intervals = [
            interval for interval in intervals
            if interval["chunk"] == frame_id and interval["stage"] in FRAME_CORE_STAGES
        ]
        if not frame_intervals and frame_id not in chunk_info:
            continue
        by_stage = defaultdict(list)
        for interval in frame_intervals:
            by_stage[interval["stage"]].append(interval)

        encode_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["encode"])
        llm_prefill_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["llm_prefill"])
        llm_decode_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["llm_decode"])
        tts_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["tts"])
        t2w_infer_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["t2w_infer"])
        t2w_write_rows = intervals_for_aliases(by_stage, FRAME_STAGE_ALIASES["t2w_write"])

        decision = decisions.get(frame_id, "")
        if not decision:
            if t2w_write_rows or tts_rows:
                decision = "speak"
            else:
                decision = "unknown"

        begin_ms = min((row["begin_ms"] for row in frame_intervals), default=0.0)
        decode_end_ms = max_end(llm_decode_rows)
        t2w_write_end_ms = max_end(t2w_write_rows)
        if decision == "speak":
            end_ms = t2w_write_end_ms
            complete = bool(begin_ms and t2w_write_end_ms)
        elif decision == "listen":
            end_ms = decode_end_ms
            complete = bool(begin_ms and decode_end_ms)
        else:
            end_ms = max((row["end_ms"] for row in frame_intervals), default=0.0)
            complete = bool(begin_ms and end_ms)

        e2e_ms = end_ms - begin_ms if complete and end_ms >= begin_ms else 0.0
        missing = []
        if not encode_rows:
            missing.append("encode")
        if not llm_prefill_rows:
            missing.append("llm_prefill")
        if not llm_decode_rows:
            missing.append("llm_decode")
        if decision == "speak":
            if not tts_rows:
                missing.append("tts")
            if not t2w_infer_rows:
                missing.append("t2w_infer")
            if not t2w_write_rows:
                missing.append("t2w_write")

        chunk_meta = chunk_info.get(frame_id, {})
        rows.append({
            "frame_id": frame_id,
            "decision": decision,
            "complete": complete and not (decision == "speak" and missing),
            "begin_ms": begin_ms,
            "end_ms": end_ms,
            "e2e_ms": e2e_ms,
            "encode_ms": sum_duration(encode_rows),
            "llm_prefill_ms": sum_duration(llm_prefill_rows),
            "llm_decode_ms": sum_duration(llm_decode_rows),
            "tts_ms": sum_duration(tts_rows),
            "t2w_infer_ms": sum_duration(t2w_infer_rows),
            "t2w_write_ms": sum_duration(t2w_write_rows),
            "n_past": chunk_meta.get("n_past", 0),
            "text": chunk_meta.get("text", ""),
            "over_1s": bool(decision == "speak" and complete and e2e_ms > sla_ms),
            "missing": ",".join(missing),
        })
    assign_utterance_ids(rows)
    return rows


def assign_utterance_ids(frame_rows):
    utterance_id = 0
    in_speak = False
    for row in frame_rows:
        if row["decision"] == "speak":
            if not in_speak:
                utterance_id += 1
                in_speak = True
            row["utterance_id"] = utterance_id
        else:
            row["utterance_id"] = 0
            in_speak = False


def choose_focus_frames(frame_rows):
    groups = defaultdict(list)
    for row in frame_rows:
        if row.get("utterance_id", 0) > 0:
            groups[row["utterance_id"]].append(row)
    if not groups:
        return frame_rows[: min(len(frame_rows), 6)]

    # Prefer the longest complete SPEAK segment; it best shows a full duplex turn.
    best_utt, speak_rows = max(
        groups.items(),
        key=lambda item: (
            len([row for row in item[1] if row["complete"]]),
            sum(row["e2e_ms"] for row in item[1]),
        ),
    )
    first_idx = next(i for i, row in enumerate(frame_rows) if row.get("utterance_id") == best_utt)
    last_idx = max(i for i, row in enumerate(frame_rows) if row.get("utterance_id") == best_utt)
    start_idx = max(0, first_idx - 2)
    end_idx = min(len(frame_rows) - 1, last_idx + 2)
    return frame_rows[start_idx:end_idx + 1]


def nearest_sample(samples, times, target):
    if not samples:
        return None
    idx = bisect.bisect_left(times, target)
    candidates = []
    if idx < len(samples):
        candidates.append(samples[idx])
    if idx > 0:
        candidates.append(samples[idx - 1])
    return min(candidates, key=lambda sample: abs(sample["t_ms"] - target)) if candidates else None


def stage_gpu_stats(intervals, gpu_samples):
    by_device = defaultdict(list)
    for sample in gpu_samples:
        by_device[sample["device"]].append(sample)
    for samples in by_device.values():
        samples.sort(key=lambda sample: sample["t_ms"])

    grouped = defaultdict(GpuStageAccumulator)

    for interval in intervals:
        if interval["stage"] not in REQUESTED_STAGES:
            continue
        for device, samples in by_device.items():
            times = [sample["t_ms"] for sample in samples]
            lo = bisect.bisect_left(times, interval["begin_ms"])
            hi = bisect.bisect_right(times, interval["end_ms"])
            selected = samples[lo:hi]
            estimated = False
            if not selected:
                sample = nearest_sample(samples, times, (interval["begin_ms"] + interval["end_ms"]) / 2.0)
                selected = [sample] if sample else []
                estimated = bool(sample)

            row = grouped[(interval["stage"], device)]
            row.n_intervals += 1
            row.total_ms += interval["duration_ms"]
            if estimated:
                row.estimated_samples += 1
            row.n_samples += len(selected)
            row.sm.extend(sample["sm_util_pct"] for sample in selected if sample["sm_util_pct"] >= 0)
            row.mem.extend(sample["mem_util_pct"] for sample in selected if sample["mem_util_pct"] >= 0)
            row.power.extend(sample["power_w"] for sample in selected if sample["power_w"] >= 0)

    stats = {}
    for (stage, device), row in grouped.items():
        avg_sm = avg(row.sm)
        stats[(stage, device)] = {
            "stage": stage,
            "device": device,
            "n_intervals": row.n_intervals,
            "n_samples": row.n_samples,
            "total_ms": row.total_ms,
            "avg_sm_util_pct": avg_sm,
            "p50_sm_util_pct": pctl(row.sm, 0.50),
            "p90_sm_util_pct": pctl(row.sm, 0.90),
            "max_sm_util_pct": max(row.sm) if row.sm else 0.0,
            "avg_mem_util_pct": avg(row.mem),
            "max_mem_util_pct": max(row.mem) if row.mem else 0.0,
            "avg_power_w": avg(row.power),
            "max_power_w": max(row.power) if row.power else 0.0,
            "estimated_samples": row.estimated_samples,
            "stage_gpu_busy_ms": row.total_ms * avg_sm / 100.0,
        }
    return stats


def svg_base(width, height, body):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#172033}",
        ".title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#5b6475}",
        ".label{font-size:13px}.small{font-size:11px;fill:#5b6475}.tiny{font-size:10px;fill:#5b6475}",
        ".lane{fill:#f6f8fb;stroke:#d6dde8}.box{rx:12;ry:12;stroke:#29415f;stroke-width:1.2}",
        "</style>",
        *body,
        "</svg>",
        "",
    ])


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def cleanup_report_dir(out_dir: Path):
    stale_files = [
        "omni-duplex-frame-summary.csv",
        "omni-duplex-gpu-samples.csv",
        "omni-duplex-sla.md",
        "omni-duplex-stage-gpu-stats.csv",
        "omni-duplex-stage-stats.csv",
        "omni-duplex-stage-timing-table.md",
        "omni-duplex-stage-token-speed.csv",
        "omni-duplex-tts-token-stats.csv",
        "figures/omni-duplex-overlap-timeline.svg",
        "figures/omni-duplex-stage-latency.svg",
    ]
    for name in stale_files:
        path = out_dir / name
        if path.exists():
            path.unlink()


def pipeline_svg(path: Path):
    body = [
        '<rect width="1180" height="520" fill="#ffffff"/>',
        '<text x="40" y="42" class="title">Omni Duplex 实测流水线</text>',
        '<text x="40" y="66" class="subtitle">只展示用户关心的数据处理阶段；计时从线程拿到队列数据后开始，不包含等待队列时间</text>',
    ]
    lanes = [("Encode", 120), ("LLM", 235), ("TTS", 350), ("T2W", 465)]
    for lane, y in lanes:
        body += [f'<rect x="30" y="{y - 42}" width="1120" height="76" class="lane" rx="14"/>', f'<text x="48" y="{y - 14}" class="label" font-weight="700">{esc(lane)}</text>']
    boxes = [
        (225, 96, 190, 48, "duplex.encode", "frame -> embeddings", "#dcecff"),
        (225, 211, 190, 48, "duplex.llm.prefill", "embeddings -> KV", "#e6dcff"),
        (505, 211, 190, 48, "duplex.llm.decode", "KV -> text/hidden", "#ffd6a5"),
        (420, 326, 150, 48, "tts.condition", "hidden -> condition", "#d8f3d0"),
        (590, 326, 150, 48, "tts.prefill", "condition -> KV", "#d0f4de"),
        (760, 326, 150, 48, "tts.decode", "KV -> audio tokens", "#c2e8c2"),
        (505, 441, 190, 48, "t2w.infer", "audio tokens -> wav", "#cde7ff"),
        (775, 441, 190, 48, "t2w.write", "write wav", "#e6e1f2"),
    ]
    for x, y, w, h, title, sub, color in boxes:
        body += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" class="box"/>', f'<text x="{x + 12}" y="{y + 20}" class="label" font-weight="700">{esc(title)}</text>', f'<text x="{x + 12}" y="{y + 38}" class="small">{esc(sub)}</text>']
    body.append('<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#29415f"/></marker></defs>')
    for x1, y1, x2, y2 in [
        (415, 120, 225, 235),
        (415, 235, 505, 235),
        (695, 235, 420, 350),
        (570, 350, 590, 350),
        (740, 350, 760, 350),
        (910, 350, 505, 465),
        (695, 465, 775, 465),
    ]:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#29415f" stroke-width="1.8" marker-end="url(#arrow)"/>')
    body.append(f'<text x="60" y="505" class="small">• {esc("LISTEN 帧只要求 encode + LLM；SPEAK 帧 e2e 到最后一次 t2w.write.end")}</text>')
    write(path, svg_base(1180, 520, body))


def frame_pipeline_svg(path: Path, frame_rows, intervals):
    if not frame_rows:
        pipeline_svg(path)
        return

    lane_names = ["Encode", "LLM", "TTS", "T2W"]
    stage_lane = {
        "duplex.encode": "Encode",
        "duplex.llm.prefill": "LLM",
        "duplex.llm.decode": "LLM",
        "tts.condition": "TTS",
        "tts.prefill": "TTS",
        "tts.decode": "TTS",
        "t2w.infer": "T2W",
        "t2w.write": "T2W",
    }
    colors = {stage: color for stage, color in STAGE_ROWS}
    colors.update({
        "duplex.encode": "#4c78a8",
        "duplex.llm.prefill": "#b279a2",
        "duplex.llm.decode": "#e45756",
    })

    focus_rows = choose_focus_frames(frame_rows)
    focus_frame_ids = {row["frame_id"] for row in focus_rows}
    focus_utterances = sorted({row.get("utterance_id", 0) for row in focus_rows if row.get("utterance_id", 0) > 0})
    focus_utterance = focus_utterances[0] if focus_utterances else 0

    drawable = [
        interval for interval in intervals
        if interval["chunk"] in focus_frame_ids and interval["stage"] in stage_lane
    ]
    if not drawable:
        pipeline_svg(path)
        return

    start_ms = min(row["begin_ms"] for row in focus_rows if row["begin_ms"] > 0)
    end_ms = max(max(row["end_ms"], row["begin_ms"] + 1000.0) for row in focus_rows if row["begin_ms"] > 0)
    span = max(1.0, end_ms - start_ms)
    width = 1320
    left, chart_w = 170, 1080
    top, lane_h = 140, 82
    height = top + len(lane_names) * lane_h + 118

    def x_of(t_ms):
        return left + (t_ms - start_ms) / span * chart_w

    lane_y = {lane: top + idx * lane_h for idx, lane in enumerate(lane_names)}
    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="40" y="42" class="title">Duplex Pipeline: One SPEAK Turn</text>',
        '<text x="40" y="66" class="subtitle">竖线标记 1s 输入间隔；同色细线连接同一 frame 的各阶段产物</text>',
    ]

    for row in focus_rows:
        if row["begin_ms"] <= 0:
            continue
        x = x_of(row["begin_ms"])
        body += [
            f'<line x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="{top + len(lane_names) * lane_h - 26}" stroke="#cfd8e6" stroke-width="1.5"/>',
            f'<text x="{x + 4:.1f}" y="{top - 24}" class="tiny">f{row["frame_id"]}</text>',
        ]
    # Draw regular 1s cadence labels so the input rhythm is obvious.
    cadence = 0.0
    while cadence <= span + 1.0:
        x = left + cadence / span * chart_w
        body.append(f'<text x="{x - 8:.1f}" y="{height - 28}" class="tiny">{cadence / 1000.0:.0f}s</text>')
        cadence += 1000.0

    for lane in lane_names:
        y = lane_y[lane]
        body += [
            f'<rect x="40" y="{y - 18}" width="{width - 80}" height="54" fill="#f6f8fb" stroke="#d6dde8" rx="10"/>',
            f'<text x="56" y="{y + 5}" class="label" font-weight="700">{esc(lane)}</text>',
        ]

    for row in focus_rows:
        if row["begin_ms"] <= 0 or row["end_ms"] <= 0:
            continue
        x = x_of(row["begin_ms"])
        w = max(2.0, x_of(row["end_ms"]) - x)
        stroke = "#2ca25f" if row["decision"] == "speak" else "#8b95a5"
        body.append(
            f'<rect x="{x:.1f}" y="{top - 24}" width="{w:.1f}" height="{len(lane_names) * lane_h - 32}" '
            f'fill="none" stroke="{stroke}" stroke-width="1" stroke-dasharray="4 4" rx="8"/>'
        )
        label = f'f{row["frame_id"]} {row["decision"]}'
        if row.get("utterance_id", 0) > 0:
            label += f' u{row["utterance_id"]}'
        body.append(f'<text x="{x + 4:.1f}" y="{top - 10}" class="tiny">{esc(label)}</text>')

    stage_offsets = {
        "duplex.encode": -8,
        "duplex.llm.prefill": -8,
        "duplex.llm.decode": 10,
        "tts.condition": -8,
        "tts.prefill": 0,
        "tts.decode": 10,
        "t2w.infer": -8,
        "t2w.write": 10,
    }
    stage_order = {
        "duplex.encode": 0,
        "duplex.llm.prefill": 1,
        "duplex.llm.decode": 2,
        "tts.condition": 3,
        "tts.prefill": 4,
        "tts.decode": 5,
        "t2w.infer": 6,
        "t2w.write": 7,
    }
    row_by_frame = {row["frame_id"]: row for row in focus_rows}

    def bar_geometry(interval):
        lane = stage_lane[interval["stage"]]
        x = x_of(interval["begin_ms"])
        raw_w = (interval["end_ms"] - interval["begin_ms"]) / span * chart_w
        w = max(2.0, raw_w)
        y = lane_y[lane] - 2 + stage_offsets.get(interval["stage"], 0)
        return {
            "x": x,
            "y": y,
            "w": w,
            "raw_w": raw_w,
            "cx": x + w / 2.0,
            "cy": y + 7.0,
        }

    bar_rows = [
        (interval, bar_geometry(interval))
        for interval in sorted(drawable, key=lambda item: (item["begin_ms"], item["chunk"], item["stage"]))
    ]

    by_frame = defaultdict(list)
    for interval, geom in bar_rows:
        by_frame[interval["chunk"]].append((interval, geom))

    # Draw same-frame data-flow connectors before the bars so colored stage bars
    # stay readable while overlapping SPEAK windows can still be traced.
    for frame_id in sorted(by_frame):
        frame_row = row_by_frame.get(frame_id, {})
        stroke = "#2ca25f" if frame_row.get("decision") == "speak" else "#6b7280"
        points = [
            (geom["cx"], geom["cy"])
            for interval, geom in sorted(
                by_frame[frame_id],
                key=lambda item: (stage_order.get(item[0]["stage"], 99), item[0]["begin_ms"], item[0]["end_ms"]),
            )
        ]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            body.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{stroke}" stroke-width="1.3" opacity="0.55"/>'
            )

    for interval, geom in bar_rows:
        x = geom["x"]
        y = geom["y"]
        w = geom["w"]
        raw_w = geom["raw_w"]
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="14" '
            f'fill="{colors.get(interval["stage"], "#8b95a5")}" rx="3"/>'
        )
        if raw_w < 4.0:
            # Keep tiny stages honest in time scale: the bar remains short, while
            # a small tick makes the stage visible without implying a longer duration.
            body.append(
                f'<line x1="{x:.1f}" y1="{y - 2:.1f}" x2="{x:.1f}" y2="{y + 16:.1f}" '
                f'stroke="{colors.get(interval["stage"], "#8b95a5")}" stroke-width="2"/>'
            )
        if w > 42:
            label = interval["stage"].replace("duplex.", "")
            body.append(f'<text x="{x + 4:.1f}" y="{y + 11:.1f}" class="tiny">{esc(label)} {interval["duration_ms"]:.0f}ms</text>')

    legend_x = 40
    legend_y = height - 88
    legend_items = [
        ("SPEAK frame", "#2ca25f"),
        ("LISTEN frame", "#8b95a5"),
        ("duplex.encode", colors["duplex.encode"]),
        ("duplex.llm.prefill", colors["duplex.llm.prefill"]),
        ("duplex.llm.decode", colors["duplex.llm.decode"]),
        ("tts.condition", colors["tts.condition"]),
        ("tts.prefill", colors["tts.prefill"]),
        ("tts.decode", colors["tts.decode"]),
        ("t2w.infer", colors["t2w.infer"]),
        ("t2w.write", colors["t2w.write"]),
    ]
    body.append(f'<text x="{legend_x}" y="{legend_y}" class="label" font-weight="700">Legend</text>')
    for idx, (label, color) in enumerate(legend_items):
        col = idx % 4
        row = idx // 4
        x = legend_x + col * 295
        y = legend_y + 16 + row * 22
        body += [
            f'<rect x="{x}" y="{y}" width="22" height="12" fill="{color}" rx="3"/>',
            f'<text x="{x + 30}" y="{y + 11}" class="small">{esc(label)}</text>',
        ]

    write(path, svg_base(width, height, body))


def stage_latency_svg(path: Path, stats):
    width = 1080
    left, top, bar_w, row_h = 270, 88, 650, 42
    chart_bottom = top + len(STAGE_ROWS) * row_h + 10
    height = max(470, chart_bottom + 50)
    max_value = max([stats.get(stage, {}).get("avg", 0) for stage, _ in STAGE_ROWS] + [260])
    body = [f'<rect width="1080" height="{height}" fill="#ffffff"/>', '<text x="40" y="42" class="title">阶段耗时均值</text>', '<text x="40" y="66" class="subtitle">单位 ms；横条为平均值，灰线为 p50 到 max</text>']
    for tick in range(0, 301, 50):
        x = left + tick / max_value * bar_w
        body += [f'<line x1="{x:.1f}" y1="78" x2="{x:.1f}" y2="{chart_bottom}" stroke="#edf1f7"/>', f'<text x="{x - 8:.1f}" y="{chart_bottom + 19}" class="tiny">{tick}</text>']
    for idx, (stage, color) in enumerate(STAGE_ROWS):
        y = top + idx * row_h
        row = stats.get(stage, {})
        avgv, p50, maxv, n = row.get("avg", 0), row.get("p50", 0), row.get("max", 0), int(row.get("n", 0))
        w = avgv / max_value * bar_w
        p50x, maxx = left + p50 / max_value * bar_w, left + maxv / max_value * bar_w
        body += [f'<text x="40" y="{y + 18}" class="label">{esc(stage)}</text>', f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" fill="{color}" rx="5"/>', f'<line x1="{p50x:.1f}" y1="{y + 31}" x2="{maxx:.1f}" y2="{y + 31}" stroke="#7f8897" stroke-width="2"/>', f'<circle cx="{p50x:.1f}" cy="{y + 31}" r="3" fill="#7f8897"/>', f'<circle cx="{maxx:.1f}" cy="{y + 31}" r="3" fill="#7f8897"/>', f'<text x="{left + w + 8:.1f}" y="{y + 16}" class="small">{avgv:.1f} ms</text>', f'<text x="940" y="{y + 18}" class="tiny">n={n} p50={p50:.1f} max={maxv:.1f}</text>']
    write(path, svg_base(width, height, body))


def overlap_svg(path: Path, events, start_ms=850.0, end_ms=1400.0):
    width, height = 1180, 530
    left, top, chart_w = 210, 86, 900
    lanes = [
        ("encode", ["duplex.encode"]),
        ("llm", ["duplex.llm.prefill", "duplex.llm.decode"]),
        ("tts", ["tts.condition", "tts.prefill", "tts.decode"]),
        ("t2w", ["t2w.infer", "t2w.write"]),
    ]
    colors = {
        "duplex.encode":"#4c78a8", "duplex.llm.prefill":"#b279a2", "duplex.llm.decode":"#e45756",
        "tts.condition":"#8cd17d", "tts.prefill":"#54a24b", "tts.decode":"#2ca02c",
        "t2w.infer":"#3182bd", "t2w.write":"#9e9ac8",
    }
    stage_lane = {stage: lane for lane, stages in lanes for stage in stages}
    lane_y = {}
    body = ['<rect width="1180" height="530" fill="#ffffff"/>', '<text x="40" y="42" class="title">异步重叠时间线：约 0.85s 到 1.40s</text>', '<text x="40" y="66" class="subtitle">只显示 encode、LLM、TTS、T2W 数据处理区间；不包含队列等待</text>']
    for tick in range(int(start_ms), int(end_ms) + 1, 50):
        x = left + (tick - start_ms) / (end_ms - start_ms) * chart_w
        body += [f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="455" stroke="#edf1f7"/>', f'<text x="{x - 14:.1f}" y="478" class="tiny">{tick}</text>']
    for idx, (lane, _) in enumerate(lanes):
        y = top + idx * 92
        lane_y[lane] = y
        body += [f'<rect x="40" y="{y - 20}" width="1070" height="62" fill="#f6f8fb" stroke="#d6dde8" rx="10"/>', f'<text x="58" y="{y + 5}" class="label" font-weight="700">{esc(lane)}</text>']
    starts = defaultdict(list)
    bars = []
    for event in sorted(events, key=lambda item: item["t_ms"]):
        key = (event["stage"], event["chunk"])
        if event["event"] == "start":
            starts[key].append(event)
        elif event["event"] == "end" and event["stage"] in stage_lane:
            start = starts[key].pop(0) if starts[key] else None
            begin = start["t_ms"] if start else event["t_ms"] - max(event["dur_ms"], 0)
            finish = event["t_ms"]
            if finish >= start_ms and begin <= end_ms:
                bars.append((begin, finish, event["stage"], event["chunk"]))
    offsets = {stage: (i % 3) * 16 for i, stage in enumerate(colors)}
    for begin, finish, stage, chunk in bars:
        lane = stage_lane[stage]
        x = left + (max(begin, start_ms) - start_ms) / (end_ms - start_ms) * chart_w
        w = max(2, (min(finish, end_ms) - max(begin, start_ms)) / (end_ms - start_ms) * chart_w)
        y = lane_y[lane] - 4 + offsets.get(stage, 0)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="13" fill="{colors[stage]}" rx="3"/>')
        if w > 62:
            body.append(f'<text x="{x + 4:.1f}" y="{y + 10:.1f}" class="tiny">c{chunk} {esc(stage.split(".")[-1])}</text>')
    write(path, svg_base(width, height, body))


def gpu_utilization_svg(path: Path, gpu_samples, intervals):
    width = 1180
    left, chart_w = 80, 1010
    device_ids = sorted({sample["device"] for sample in gpu_samples})
    if not gpu_samples or not device_ids:
        body = ['<rect width="1180" height="180" fill="#ffffff"/>', '<text x="40" y="42" class="title">GPU 利用率时间线</text>', '<text x="40" y="72" class="subtitle">未解析到 [DUPLEX_GPU] sample</text>']
        write(path, svg_base(width, 180, body))
        return
    if gpu_valid_metric_count(gpu_samples) == 0:
        body = [
            '<rect width="1180" height="210" fill="#ffffff"/>',
            '<text x="40" y="42" class="title">GPU 利用率时间线</text>',
            '<text x="40" y="72" class="subtitle">解析到了 [DUPLEX_GPU] sample，但所有 GPU 指标均为 NA</text>',
            '<text x="40" y="104" class="small">Jetson/Orin 上 NVML 常不支持 util/memory/power 字段；请使用带 jetson_sysfs fallback 的新采样器重新运行。</text>',
        ]
        write(path, svg_base(width, 210, body))
        return

    sample_min = min(sample["t_ms"] for sample in gpu_samples)
    sample_max = max(sample["t_ms"] for sample in gpu_samples)
    interval_min = min([interval["begin_ms"] for interval in intervals] + [sample_min])
    interval_max = max([interval["end_ms"] for interval in intervals] + [sample_max])
    start_ms = min(sample_min, interval_min)
    end_ms = max(sample_max, interval_max)
    span = max(1.0, end_ms - start_ms)
    device_h = 115
    stage_top = 96 + len(device_ids) * device_h
    height = stage_top + 270
    body = ['<rect width="1180" height="%d" fill="#ffffff"/>' % height, '<text x="40" y="42" class="title">GPU 利用率时间线</text>', '<text x="40" y="66" class="subtitle">蓝线=SM util，橙线=memory controller util；下方为主要阶段 interval</text>']

    def x_of(t_ms):
        return left + (t_ms - start_ms) / span * chart_w

    for tick in range(int(start_ms // 100 * 100), int(end_ms) + 101, 100):
        x = x_of(tick)
        body += [f'<line x1="{x:.1f}" y1="84" x2="{x:.1f}" y2="{height - 40}" stroke="#edf1f7"/>', f'<text x="{x - 18:.1f}" y="{height - 18}" class="tiny">{tick}</text>']

    for idx, device in enumerate(device_ids):
        y0 = 90 + idx * device_h
        body += [f'<rect x="40" y="{y0 - 10}" width="1070" height="92" fill="#f6f8fb" stroke="#d6dde8" rx="10"/>', f'<text x="50" y="{y0 + 10}" class="label" font-weight="700">device {device}</text>']
        for pct in (0, 50, 100):
            y = y0 + 70 - pct / 100.0 * 62
            body += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e8edf5"/>', f'<text x="46" y="{y + 4:.1f}" class="tiny">{pct}%</text>']
        samples = [sample for sample in gpu_samples if sample["device"] == device]
        for key, color in (("sm_util_pct", "#2f6fed"), ("mem_util_pct", "#f58518")):
            points = []
            for sample in samples:
                value = sample[key]
                if value < 0:
                    continue
                points.append(f'{x_of(sample["t_ms"]):.1f},{y0 + 70 - min(100.0, value) / 100.0 * 62:.1f}')
            if len(points) >= 2:
                body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"/>')

    stage_colors = {stage: color for stage, color in STAGE_ROWS}
    focus_stages = ["duplex.encode", "duplex.llm.prefill", "duplex.llm.decode", "tts.condition", "tts.prefill", "tts.decode", "t2w.infer", "t2w.write"]
    lane_y = {stage: stage_top + 22 + idx * 26 for idx, stage in enumerate(focus_stages)}
    body += [f'<text x="40" y="{stage_top}" class="label" font-weight="700">stage intervals</text>']
    for stage in focus_stages:
        y = lane_y[stage]
        body += [f'<text x="40" y="{y + 10}" class="tiny">{esc(stage)}</text>', f'<line x1="{left}" y1="{y + 5}" x2="{left + chart_w}" y2="{y + 5}" stroke="#edf1f7"/>']
    for interval in intervals:
        stage = interval["stage"]
        if stage not in lane_y:
            continue
        x = x_of(interval["begin_ms"])
        w = max(1.5, (interval["end_ms"] - interval["begin_ms"]) / span * chart_w)
        y = lane_y[stage]
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="11" fill="{stage_colors.get(stage, "#8b95a5")}" opacity="0.58" rx="3"/>')

    body += ['<rect x="875" y="90" width="220" height="52" fill="#ffffff" stroke="#d6dde8" rx="10"/>', '<line x1="895" y1="110" x2="930" y2="110" stroke="#2f6fed" stroke-width="2"/><text x="940" y="114" class="small">SM util</text>', '<line x1="895" y1="132" x2="930" y2="132" stroke="#f58518" stroke-width="2"/><text x="940" y="136" class="small">mem util</text>']
    write(path, svg_base(width, height, body))


def write_csvs(out_dir: Path, stats, gpu_samples, gpu_stats, tts_tokens, token_stats):
    with (out_dir / "omni-duplex-stage-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "total_ms", "avg_ms", "p50_ms", "p90_ms", "min_ms", "max_ms"])
        for stage in sorted(stats):
            row = stats[stage]
            writer.writerow([stage, row["n"], f'{row["total"]:.3f}', f'{row["avg"]:.3f}', f'{row["p50"]:.3f}', f'{row["p90"]:.3f}', f'{row["min"]:.3f}', f'{row["max"]:.3f}'])
    with (out_dir / "omni-duplex-gpu-samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "t_ms", "device", "sm_util_pct", "mem_util_pct", "gpu_used_mb", "gpu_total_mb", "power_w", "temp_c", "graphics_clock_mhz", "mem_clock_mhz"])
        for sample in gpu_samples:
            writer.writerow([sample["sample_id"], f'{sample["t_ms"]:.3f}', sample["device"], sample["sm_util_pct"], sample["mem_util_pct"], sample["gpu_used_mb"], sample["gpu_total_mb"], sample["power_w"], sample["temp_c"], sample["graphics_clock_mhz"], sample["mem_clock_mhz"]])
    with (out_dir / "omni-duplex-stage-gpu-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "device", "n_intervals", "n_samples", "total_ms", "avg_sm_util_pct", "p50_sm_util_pct", "p90_sm_util_pct", "max_sm_util_pct", "avg_mem_util_pct", "max_mem_util_pct", "avg_power_w", "max_power_w", "estimated_samples", "stage_gpu_busy_ms"])
        for key in sorted(gpu_stats):
            row = gpu_stats[key]
            writer.writerow([row["stage"], row["device"], row["n_intervals"], row["n_samples"], f'{row["total_ms"]:.3f}', f'{row["avg_sm_util_pct"]:.3f}', f'{row["p50_sm_util_pct"]:.3f}', f'{row["p90_sm_util_pct"]:.3f}', f'{row["max_sm_util_pct"]:.3f}', f'{row["avg_mem_util_pct"]:.3f}', f'{row["max_mem_util_pct"]:.3f}', f'{row["avg_power_w"]:.3f}', f'{row["max_power_w"]:.3f}', row["estimated_samples"], f'{row["stage_gpu_busy_ms"]:.3f}'])
    with (out_dir / "omni-duplex-stage-token-speed.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "tokens", "total_ms", "tokens_per_s", "ms_per_token", "avg_event_tokens_per_s", "p50_event_tokens_per_s", "avg_event_ms_per_token"])
        for stage in TOKEN_SPEED_STAGES:
            row = token_stats.get(stage, {})
            writer.writerow([
                stage,
                int(row.get("n", 0)),
                int(row.get("tokens", 0)),
                f'{row.get("total_ms", 0.0):.3f}',
                f'{row.get("tokens_per_s", 0.0):.3f}',
                f'{row.get("ms_per_token", 0.0):.6f}',
                f'{row.get("avg_event_tokens_per_s", 0.0):.3f}',
                f'{row.get("p50_event_tokens_per_s", 0.0):.3f}',
                f'{row.get("avg_event_ms_per_token", 0.0):.6f}',
            ])
    with (out_dir / "omni-duplex-tts-token-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "tts_chunk", "dur_ms", "llm_tokens", "filtered_llm_tokens", "condition_tokens", "compute_tokens", "audio_tokens_generated", "is_end_of_turn", "flush_only", "compute_tokens_per_s", "audio_tokens_per_ms", "audio_tokens_per_s", "ms_per_audio_token", "has_audio_token_detail"])
        for row in tts_tokens:
            writer.writerow([row["chunk"], row["tts_chunk"], f'{row["dur_ms"]:.3f}', row["llm_tokens"], row["filtered_llm_tokens"], row["condition_tokens"], row["compute_tokens"], row["audio_tokens_generated"], row["is_end_of_turn"], row["flush_only"], f'{row["compute_tokens_per_s"]:.3f}', f'{row["audio_tokens_per_ms"]:.6f}', f'{row["audio_tokens_per_s"]:.3f}', f'{row["ms_per_audio_token"]:.3f}', row["has_audio_token_detail"]])


def write_frame_summary_csv(out_dir: Path, frame_rows):
    with (out_dir / "omni-duplex-frame-summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "utterance_id",
            "decision",
            "complete",
            "audio_e2e_ms",
            "encode_ms",
            "llm_prefill_ms",
            "llm_decode_ms",
            "tts_ms",
            "t2w_infer_ms",
            "t2w_write_ms",
            "n_past",
            "over_1s",
            "missing",
            "text",
        ])
        for row in frame_rows:
            writer.writerow([
                row["frame_id"],
                row.get("utterance_id", 0),
                row["decision"],
                int(row["complete"]),
                f'{row["e2e_ms"]:.3f}',
                f'{row["encode_ms"]:.3f}',
                f'{row["llm_prefill_ms"]:.3f}',
                f'{row["llm_decode_ms"]:.3f}',
                f'{row["tts_ms"]:.3f}',
                f'{row["t2w_infer_ms"]:.3f}',
                f'{row["t2w_write_ms"]:.3f}',
                row["n_past"],
                int(row["over_1s"]),
                row["missing"],
                row["text"],
            ])


def write_sla_report(path: Path, frame_rows, sla_ms=1000.0):
    speak_rows = [row for row in frame_rows if row["decision"] == "speak"]
    complete_speak = [row for row in speak_rows if row["complete"] and row["e2e_ms"] > 0]
    listen_rows = [row for row in frame_rows if row["decision"] == "listen"]
    incomplete_speak = [row for row in speak_rows if not row["complete"]]
    values = [row["e2e_ms"] for row in complete_speak]
    passed = bool(values) and not incomplete_speak and max(values) <= sla_ms
    result = "PASS" if passed else "FAIL"

    lines = [
        "# Omni Duplex SPEAK SLA",
        "",
        f"- SLA：SPEAK 帧 audio e2e `<= {sla_ms:.0f} ms`，起点为该帧底层 encode 开始，终点为该帧最后一次 `t2w.write.end`。",
        f"- 结论：`{result}`。",
        f"- Frames：总计 `{len(frame_rows)}`，SPEAK `{len(speak_rows)}`，LISTEN `{len(listen_rows)}`，完整 SPEAK `{len(complete_speak)}`，不完整 SPEAK `{len(incomplete_speak)}`。",
    ]
    if values:
        lines.extend([
            f"- SPEAK e2e：avg `{avg(values):.1f} ms`，p50 `{statistics.median(values):.1f} ms`，p95 `{pctl(values, 0.95):.1f} ms`，max `{max(values):.1f} ms`。",
            "",
            "| frame | e2e ms | encode | llm prefill | llm decode | tts | t2w infer | t2w write | status |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in complete_speak:
            status = "over_1s" if row["over_1s"] else "ok"
            lines.append(
                f"| {row['frame_id']} | {row['e2e_ms']:.1f} | {row['encode_ms']:.1f} | "
                f"{row['llm_prefill_ms']:.1f} | {row['llm_decode_ms']:.1f} | {row['tts_ms']:.1f} | "
                f"{row['t2w_infer_ms']:.1f} | {row['t2w_write_ms']:.1f} | `{status}` |"
            )
    else:
        lines.append("- 未找到完整 SPEAK 帧，无法给出有效 e2e 统计。")

    if incomplete_speak:
        lines.extend([
            "",
            "## 不完整 SPEAK 帧",
            "",
            "| frame | missing | text |",
            "| ---: | --- | --- |",
        ])
        for row in incomplete_speak:
            lines.append(
                f"| {row['frame_id']} | `{row['missing'] or 'unknown'}` | {esc(row['text'][:80])} |"
            )

    lines.append("")
    write(path, "\n".join(lines))


def stage_kind(stage: str) -> str:
    if stage.endswith(".write"):
        return "io"
    return "compute"


def ordered_stage_names(stats):
    preferred = [stage for stage, _ in STAGE_ROWS]
    return preferred + [stage for stage in sorted(stats) if stage not in preferred]


def write_stage_timing_table(path: Path, stats):
    lines = [
        "# Duplex 阶段用时统计表",
        "",
        "单位均为 ms。`total` 是该 stage 所有 end 事件的耗时总和；异步/父子阶段不要直接相加。",
        "",
        "| type | stage | n | total | avg | p50 | p90 | min | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in ordered_stage_names(stats):
        row = stats.get(stage, {})
        if not row:
            continue
        lines.append(
            f"| `{stage_kind(stage)}` | `{stage}` | {int(row['n'])} | "
            f"{row['total']:.1f} | {row['avg']:.1f} | {row['p50']:.1f} | "
            f"{row['p90']:.1f} | {row['min']:.1f} | {row['max']:.1f} |"
        )
    lines.append("")
    write(path, "\n".join(lines))


def gpu_stage_summary(gpu_stats, stage):
    rows = [row for (row_stage, _), row in gpu_stats.items() if row_stage == stage]
    if not rows:
        return None
    total_samples = sum(row["n_samples"] for row in rows)
    avg_sm = avg(row["avg_sm_util_pct"] for row in rows)
    max_sm = max(row["max_sm_util_pct"] for row in rows)
    avg_power = avg(row["avg_power_w"] for row in rows)
    estimated = sum(row["estimated_samples"] for row in rows)
    devices = ",".join(str(row["device"]) for row in sorted(rows, key=lambda item: item["device"]))
    return {
        "devices": devices,
        "total_samples": total_samples,
        "avg_sm": avg_sm,
        "max_sm": max_sm,
        "avg_power": avg_power,
        "estimated": estimated,
    }


def active_gpu_ids(gpu_samples, max_devices=1):
    by_device = defaultdict(list)
    for sample in gpu_samples:
        by_device[sample["device"]].append(sample)
    scores = []
    for device, samples in by_device.items():
        sm_values = [sample["sm_util_pct"] for sample in samples if sample["sm_util_pct"] >= 0]
        mem_values = [sample["gpu_used_mb"] for sample in samples if sample["gpu_used_mb"] >= 0]
        max_sm = max(sm_values) if sm_values else 0.0
        avg_sm = avg(sm_values)
        mem_delta = (max(mem_values) - min(mem_values)) if mem_values else 0.0
        score = max_sm * 1000.0 + avg_sm * 100.0 + mem_delta
        if score > 0:
            scores.append((score, device))
    scores.sort(reverse=True)
    return [device for _, device in scores[:max_devices]]


def gpu_device_names(device_ids):
    device_ids = sorted(set(device_ids))
    if not device_ids:
        return {}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}

    names = {}
    for line in proc.stdout.splitlines():
        if "," not in line:
            continue
        idx_text, name = line.split(",", 1)
        try:
            idx = int(idx_text.strip())
        except ValueError:
            continue
        if idx in device_ids:
            names[idx] = name.strip()
    return names


def gpu_device_summary(gpu_samples):
    device_ids = sorted({sample["device"] for sample in gpu_samples})
    if not device_ids:
        return ""
    names = gpu_device_names(device_ids)
    parts = []
    for device in device_ids:
        name = names.get(device)
        parts.append(f"device {device} ({name})" if name else f"device {device}")
    return ", ".join(parts)


def filter_gpu_samples(gpu_samples, device_ids):
    if not device_ids:
        return gpu_samples
    keep = set(device_ids)
    return [sample for sample in gpu_samples if sample["device"] in keep]


def gpu_valid_metric_count(gpu_samples):
    keys = ("sm_util_pct", "mem_util_pct", "gpu_used_mb", "power_w", "graphics_clock_mhz")
    return sum(1 for sample in gpu_samples if any(sample.get(key, -1.0) >= 0 for key in keys))


def stage_display_name(stage):
    return {
        "duplex.encode": "Encode",
        "duplex.llm.prefill": "LLM Prefill",
        "duplex.llm.decode": "LLM Decode",
        "tts.condition": "TTS Condition",
        "tts.prefill": "TTS Prefill",
        "tts.decode": "TTS Decode",
        "t2w.infer": "T2W",
        "t2w.write": "Wav Write",
    }.get(stage, stage)


FRAME_STAGE_METRICS = [
    ("encode_ms", "Encode"),
    ("llm_prefill_ms", "LLM Prefill"),
    ("llm_decode_ms", "LLM Decode"),
    ("tts_ms", "TTS"),
    ("t2w_infer_ms", "T2W"),
    ("t2w_write_ms", "Wav Write"),
]


def stage_bottleneck_note(supports_duplex, complete_speak_frames, stats):
    if supports_duplex:
        return ""

    if complete_speak_frames:
        avg_e2e = avg(row["e2e_ms"] for row in complete_speak_frames)
        stage_avgs = []
        for key, label in FRAME_STAGE_METRICS:
            values = [row.get(key, 0.0) for row in complete_speak_frames]
            stage_avgs.append((avg(values), key, label))

        bottleneck_avg, _, bottleneck_label = max(stage_avgs, key=lambda item: item[0])
        share = bottleneck_avg / avg_e2e * 100.0 if avg_e2e > 0 else 0.0

        worst_frame = max(complete_speak_frames, key=lambda row: row["e2e_ms"])
        worst_stage_key, worst_stage_label = max(
            FRAME_STAGE_METRICS,
            key=lambda item: worst_frame.get(item[0], 0.0),
        )
        worst_stage_ms = worst_frame.get(worst_stage_key, 0.0)
        return (
            f"- 瓶颈阶段：按完整 SPEAK 帧平均耗时，`{bottleneck_label}` 最长，"
            f"avg `{bottleneck_avg:.1f} ms`，约占 SPEAK e2e 平均 `{share:.1f}%`；"
            f"最慢 frame `f{worst_frame['frame_id']}` 的最长阶段是 `{worst_stage_label}` "
            f"`{worst_stage_ms:.1f} ms`。"
        )

    stage_avgs = [
        (row.get("avg", 0.0), stage_display_name(stage))
        for stage, row in stats.items()
        if row
    ]
    if not stage_avgs:
        return "- 瓶颈阶段：缺少完整 SPEAK frame 和阶段统计，无法定位耗时最长阶段。"

    bottleneck_avg, bottleneck_label = max(stage_avgs, key=lambda item: item[0])
    return (
        f"- 瓶颈阶段：未找到完整 SPEAK frame，退化使用全局阶段平均耗时；"
        f"`{bottleneck_label}` 最长，avg `{bottleneck_avg:.1f} ms`。"
    )


def write_report(path: Path, log_path: Path, text: str, events, chunks, stats, gpu_samples, gpu_stats, gpu_statuses, token_stats, frame_rows):
    decisions = Counter(c["decision"] for c in chunks)
    def speed(stage):
        return token_stats.get(stage, {}).get("tokens_per_s", 0.0)
    complete_speak_frames = [
        row for row in frame_rows
        if row["decision"] == "speak" and row["complete"] and row["e2e_ms"] > 0
    ]
    speak_rows = [row for row in frame_rows if row["decision"] == "speak"]
    listen_rows = [row for row in frame_rows if row["decision"] == "listen"]
    incomplete_speak = [row for row in speak_rows if not row["complete"]]
    speak_e2e = [row["e2e_ms"] for row in complete_speak_frames]
    sla_ms = 1000.0
    supports_duplex = bool(speak_e2e) and not incomplete_speak and max(speak_e2e) <= sla_ms
    conclusion = "支持双工" if supports_duplex else "暂不满足双工"
    reason = (
        f"完整 SPEAK 帧最慢 {max(speak_e2e):.1f} ms，低于 1s 输入间隔"
        if supports_duplex and speak_e2e
        else "存在不完整 SPEAK 帧或最慢 SPEAK e2e 超过 1s"
    )
    bottleneck_note = stage_bottleneck_note(supports_duplex, complete_speak_frames, stats)

    utterances = sorted({row.get("utterance_id", 0) for row in speak_rows if row.get("utterance_id", 0) > 0})

    lines = [
        "# Duplex 1s 流式测试结论",
        "",
        f"**结论：{conclusion}。** {reason}。",
        "",
        f"- 输入节奏：`1s/frame`；日志：`{log_path.name}`。",
        f"- Frames：`{len(frame_rows)}`；SPEAK `{len(speak_rows)}`，LISTEN `{len(listen_rows)}`，SPEAK utterance `{len(utterances)}` 段。",
        f"- 完整 SPEAK e2e：avg `{avg(speak_e2e):.1f} ms`，p95 `{pctl(speak_e2e, 0.95):.1f} ms`，max `{max(speak_e2e) if speak_e2e else 0.0:.1f} ms`。",
        f"- 判定口径：从 `duplex.encode.start` 到该 SPEAK frame 最后一次 `t2w.write.end`；不统计队列等待。",
        *([bottleneck_note] if bottleneck_note else []),
        "",
        "## 流水线",
        "",
        "![Omni Duplex 实测流水线](figures/omni-duplex-pipeline.svg)",
        "",
        "图中只展示一个连续 SPEAK turn，并保留前后 LISTEN frame；竖线标出 1s 输入节奏，同色细线连接同一 frame 的各阶段产物。",
        "",
        "## 阶段概览",
        "",
        "| 阶段 | 平均耗时 | p90 | SPEAK tokens | SPEAK 速度 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage, _ in STAGE_ROWS:
        row = stats.get(stage, {})
        if not row:
            continue
        speed_text = ""
        token_text = ""
        if stage in TOKEN_SPEED_STAGES:
            unit = "tok/s" if not stage.startswith("tts.") else "TTS tok/s"
            speed_text = f"{speed(stage):.1f} {unit}"
            token_text = str(int(token_stats.get(stage, {}).get("tokens", 0)))
        lines.append(
            f"| `{stage_display_name(stage)}` | {row.get('avg', 0.0):.1f} ms | "
            f"{row.get('p90', 0.0):.1f} ms | {token_text or '-'} | {speed_text or '-'} |"
        )

    lines += [
        "",
        "## GPU 利用率",
        "",
    ]
    if gpu_samples and gpu_valid_metric_count(gpu_samples) == 0:
        gpu_summary = gpu_device_summary(gpu_samples)
        if gpu_summary:
            lines.append(f"- 使用 GPU：`{gpu_summary}`。")
        lines.append(
            f"- GPU sample：`{len(gpu_samples)}` 条，但所有利用率/显存/功耗字段均为 `NA`；"
            "当前采样源未拿到有效指标，Jetson 上通常需要 `jetson_sysfs` fallback 或外部 `tegrastats`。"
        )
        lines.extend([
            "",
            "![GPU 利用率时间线](figures/omni-duplex-gpu-utilization.svg)",
        ])
    elif gpu_samples:
        devices = ", ".join(str(d) for d in sorted({s["device"] for s in gpu_samples}))
        gpu_summary = gpu_device_summary(gpu_samples)
        if gpu_summary:
            lines.append(f"- 使用 GPU：`{gpu_summary}`。")
        lines.append(f"- GPU sample：`{len(gpu_samples)}` 条；device：`{devices}`。")
        for stage in ["tts.prefill", "tts.decode", "t2w.infer", "duplex.llm.decode"]:
            row = gpu_stage_summary(gpu_stats, stage)
            if row:
                lines.append(f"- `{stage_display_name(stage)}`: avg SM `{row['avg_sm']:.1f}%`, max SM `{row['max_sm']:.1f}%`, avg power `{row['avg_power']:.1f} W`。")
        lines.extend([
            "",
            "![GPU 利用率时间线](figures/omni-duplex-gpu-utilization.svg)",
        ])
    elif gpu_statuses:
        reasons = ", ".join(f"{status['event']}:{status['reason']}" for status in gpu_statuses)
        lines.append(f"- 未解析到 GPU sample；状态：`{reasons}`。")
    else:
        lines.append("- 未解析到 `[DUPLEX_GPU]` sample。运行时设置 `OMNI_GPU_PROF=1` 可启用 NVML 采样。")
    lines.append("")
    write(path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("duplex_perf.log"))
    parser.add_argument("--gpu-log", type=Path, default=None, help="Optional separate OMNI_GPU_PROF_FILE log")
    parser.add_argument("--out-dir", type=Path, default=Path("duplex_perf_report"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_report_dir(args.out_dir)

    text, events, chunks, gpu_samples, gpu_statuses = parse_log(args.log, args.gpu_log)
    stats = stage_stats(events)
    intervals = build_stage_intervals(events)
    frame_rows = build_frame_summaries(events, chunks, intervals)
    speak_frame_ids = {row["frame_id"] for row in frame_rows if row["decision"] == "speak"}
    token_stats = stage_token_stats(events, speak_frame_ids)
    active_devices = active_gpu_ids(gpu_samples, max_devices=1)
    focused_gpu_samples = filter_gpu_samples(gpu_samples, active_devices)
    gpu_stats = stage_gpu_stats(intervals, focused_gpu_samples)
    figures = args.out_dir / "figures"
    frame_pipeline_svg(figures / "omni-duplex-pipeline.svg", frame_rows, intervals)
    gpu_utilization_svg(figures / "omni-duplex-gpu-utilization.svg", focused_gpu_samples, intervals)
    write_report(args.out_dir / "omni-duplex-perf-report.md", args.log, text, events, chunks, stats, focused_gpu_samples, gpu_stats, gpu_statuses, token_stats, frame_rows)
    print(f"parsed_events={len(events)} raw_markers={text.count('[DUPLEX_PERF]')} chunks={len(chunks)} frames={len(frame_rows)} gpu_samples={len(focused_gpu_samples)} active_gpus={','.join(str(d) for d in active_devices)}")
    print(f"wrote={args.out_dir / 'omni-duplex-perf-report.md'}")
    print(f"pipeline={figures / 'omni-duplex-pipeline.svg'}")
    print(f"gpu={figures / 'omni-duplex-gpu-utilization.svg'}")


if __name__ == "__main__":
    main()
