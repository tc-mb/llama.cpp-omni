#!/usr/bin/env python3
"""Aggregate repeated RTS judge reports without changing the protected evaluator."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def report_for(meta_path: Path, judge_root: Path) -> Path:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sessions = meta.get("sessions") or [meta.get("session_dir")]
    if len(sessions) != 1 or not sessions[0]:
        raise ValueError(f"expected one session in {meta_path}")
    return judge_root / sessions[0] / "eval_e2e_report.json"


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} RUN_ROOT [OUTPUT_JSON]", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else root / "summary.json"
    repo = Path(__file__).resolve().parents[2]
    judge_root = repo / "evaluation" / "judge-final"
    rows: list[dict[str, object]] = []

    for meta_path in sorted(root.glob("*/rts_runs/*/run_meta.json")):
        label = meta_path.parents[2].name
        if not label.startswith("measured_"):
            continue
        report_path = report_for(meta_path, judge_root)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rtf = report["rtf"]
        core = rtf["core"]
        rows.append({
            "label": label,
            "report": str(report_path),
            "all_pooled_rtf": rtf["rtf_aggregate"],
            "all_frame_mean_rtf": rtf["rtf"]["mean"],
            "all_frame_p95_rtf": rtf["rtf"]["p95"],
            "all_audio_ms": rtf["audio_total_ms"],
            "all_compute_ms": rtf["compute_total_ms"],
            "core_pooled_rtf": core["rtf_aggregate"],
            "core_frame_mean_rtf": core["rtf"]["mean"],
            "core_frame_p95_rtf": core["rtf"]["p95"],
            "speak_to_wav_mean_ms": report["e2e_speak_recv_to_wav_poll_ms"]["mean_ms"],
            "speak_to_wav_median_ms": report["e2e_speak_recv_to_wav_poll_ms"]["median_ms"],
            "speak_wall_mean_ms": report["speak_chunk_wall_ms"]["mean_ms"],
            "decode_to_wav_mean_ms": report["e2e_decode_end_to_wav_poll_ms"]["mean_ms"],
            "stage_rtf": rtf["stage_rtf"],
            "core_stage_rtf": core["stage_rtf"],
        })

    if not rows:
        raise SystemExit(f"no measured reports under {root}")

    total_audio = sum(float(row["all_audio_ms"]) for row in rows)
    total_compute = sum(float(row["all_compute_ms"]) for row in rows)
    numeric = [
        "all_pooled_rtf",
        "all_frame_mean_rtf",
        "all_frame_p95_rtf",
        "core_pooled_rtf",
        "core_frame_mean_rtf",
        "core_frame_p95_rtf",
        "speak_to_wav_mean_ms",
        "speak_to_wav_median_ms",
        "speak_wall_mean_ms",
        "decode_to_wav_mean_ms",
    ]
    aggregate: dict[str, object] = {
        key: stats([float(row[key]) for row in rows]) for key in numeric
    }
    aggregate["all_runs_pooled_rtf"] = total_compute / total_audio
    aggregate["stage_rtf"] = {
        stage: stats([float(row["stage_rtf"][stage]) for row in rows])
        for stage in ("encode", "llm_prefill", "llm_decode", "tts", "token2wav")
    }
    result = {"root": str(root), "runs": rows, "aggregate": aggregate}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
