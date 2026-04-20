#!/usr/bin/env bash
# 合并 tts_wav 目录下的 wav_0.wav, wav_1.wav, ... 为一个完整音频。
# 按数字顺序排序（避免 wav_10 排到 wav_2 前面），用 ffmpeg concat 无损拼接。
#
# 用法:
#   ./merge_wav.sh [wav_dir] [output_file]
# 示例:
#   ./tools/omni/merge_wav.sh
#   ./tools/omni/merge_wav.sh tools/omni/output/round_000/tts_wav merged.wav

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAV_DIR="${1:-$SCRIPT_DIR/output/round_000/tts_wav}"
OUT="${2:-${WAV_DIR%/}/merged.wav}"

if [[ ! -d "$WAV_DIR" ]]; then
    echo "Error: 目录不存在: $WAV_DIR" >&2
    exit 1
fi

count=$(find "$WAV_DIR" -maxdepth 1 -name 'wav_*.wav' | wc -l)
if [[ "$count" -eq 0 ]]; then
    echo "Error: $WAV_DIR 下没有 wav_*.wav 文件" >&2
    exit 1
fi

LIST_FILE="$(mktemp --suffix=.txt)"
trap 'rm -f "$LIST_FILE"' EXIT

find "$WAV_DIR" -maxdepth 1 -name 'wav_*.wav' -printf '%f\n' \
    | awk -F'wav_|\\.wav' '{print $2"\t"$0}' \
    | sort -n \
    | cut -f2 \
    | while read -r fname; do
        printf "file '%s'\n" "$(readlink -f "$WAV_DIR/$fname")" >> "$LIST_FILE"
    done

echo "Merging $count wav files -> $OUT"
ffmpeg -y -hide_banner -loglevel warning -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUT"
echo "Done: $OUT ($(du -h "$OUT" | cut -f1), $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT")s)"
