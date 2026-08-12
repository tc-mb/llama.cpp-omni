#!/usr/bin/env python3
"""Summarize validation logs without discarding their raw evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULT_DIR", file=sys.stderr)
        return 2

    result_dir = Path(sys.argv[1])
    logs = result_dir / "logs"
    if not logs.is_dir():
        print(f"missing logs directory: {logs}", file=sys.stderr)
        return 2

    summary: dict[str, object] = {"result_dir": str(result_dir), "logs": {}}
    apm_rows: list[dict[str, object]] = []
    timing_rows: list[str] = []
    apm_pattern = re.compile(
        r"\[apm-ab\] summary mode=(\w+) prepared=(\d+) live=(\d+) hit=(\d+) miss=(\d+)"
    )

    for path in sorted(logs.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        summary["logs"][path.name] = {
            "bytes": path.stat().st_size,
            "ctest_passed": "100% tests passed" in text,
            "unittest_ok": bool(re.search(r"^OK$", text, re.MULTILINE)),
        }
        for match in apm_pattern.finditer(text):
            apm_rows.append(
                dict(zip(("mode", "prepared", "live", "hit", "miss"), match.groups()))
            )
        timing_rows.extend(
            line for line in text.splitlines()
            if line.startswith("[timing]") or line.startswith("[timing-total]")
        )

    summary["apm"] = apm_rows
    summary["timings"] = timing_rows
    output = result_dir / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
