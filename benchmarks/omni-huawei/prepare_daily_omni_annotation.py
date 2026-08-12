#!/usr/bin/env python3
"""Convert the official Daily-Omni annotation JSON to evaluator JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("Question", "Choice", "Answer", "video_id", "Type")
OPTIONAL_FIELDS = (
    "content_parent_category",
    "content_fine_category",
    "video_category",
    "video_duration",
)


def convert_record(record: dict[str, Any], index: int, dataset_root: Path) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record {index}: missing fields: {', '.join(missing)}")

    video_id = record["video_id"]
    choices = record["Choice"]
    answer = record["Answer"]
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"record {index}: video_id must be a non-empty string")
    if not isinstance(choices, list) or not choices or not all(isinstance(x, str) for x in choices):
        raise ValueError(f"record {index}: Choice must be a non-empty string list")
    if answer not in "ABCDEFGHIJKL"[: len(choices)]:
        raise ValueError(f"record {index}: Answer {answer!r} is outside the choice range")

    video_rel = Path("Videos") / video_id / f"{video_id}_video.mp4"
    audio_rel = Path("Videos") / video_id / f"{video_id}_audio.wav"
    for media_path in (video_rel, audio_rel):
        if not (dataset_root / media_path).is_file():
            raise FileNotFoundError(f"record {index}: media file not found: {dataset_root / media_path}")

    converted = {
        "video_id": video_id,
        "VideoPath": video_rel.as_posix(),
        "WavPath": audio_rel.as_posix(),
        "question": record["Question"],
        "choices": choices,
        "gt_answer": answer,
        "qa_type": record["Type"],
    }
    converted.update({field: record[field] for field in OPTIONAL_FIELDS if field in record})
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official Daily-Omni qa.json")
    parser.add_argument("dataset_root", type=Path, help="Daily-Omni dataset root")
    parser.add_argument("output", type=Path, help="Destination evaluator JSONL")
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("source annotation must be a JSON list")

    converted = [convert_record(record, index, args.dataset_root) for index, record in enumerate(records)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in converted:
            json.dump(record, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")

    unique_videos = len({record["video_id"] for record in converted})
    print(f"wrote {len(converted)} records ({unique_videos} unique videos) to {args.output}")


if __name__ == "__main__":
    main()
