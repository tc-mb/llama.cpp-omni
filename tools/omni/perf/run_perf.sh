#!/usr/bin/env bash
#
# 双工可行性 profiling 一键脚本
#   1. (可选) 编译 llama-omni-perf-duplex target
#   2. 跑 perf-duplex，按 1s 节奏推帧，产出 perf_report.json
#   3. 跑 analyze_perf.py，输出可读报告 + 可行性判定
#
# 用法:
#   tools/omni/perf/run_perf.sh -m <llm.gguf> [--omni] \
#       [--test <prefix> <n>] [--stream-interval 1000] [--build] [-- <额外 perf-duplex 参数>]
#
# 退出码: 0=可支撑双工, 2=不满足, 其它=运行/编译失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tools/omni/output}"
JSON_PATH="${OUTPUT_DIR}/perf_report.json"
MD_PATH="${OUTPUT_DIR}/perf_report.md"
DO_BUILD=0
INTERVAL=1000

PERF_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) DO_BUILD=1; shift ;;
        --stream-interval) INTERVAL="$2"; PERF_ARGS+=("--stream-interval" "$2"); shift 2 ;;
        -o) OUTPUT_DIR="$2"; JSON_PATH="$2/perf_report.json"; MD_PATH="$2/perf_report.md"; PERF_ARGS+=("-o" "$2"); shift 2 ;;
        --out-json) JSON_PATH="$2"; PERF_ARGS+=("--out-json" "$2"); shift 2 ;;
        --) shift; while [[ $# -gt 0 ]]; do PERF_ARGS+=("$1"); shift; done ;;
        *) PERF_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

if [[ "${DO_BUILD}" == "1" ]]; then
    echo "[run_perf] 编译 llama-omni-perf-duplex ..."
    cmake --build "${BUILD_DIR}" --target llama-omni-perf-duplex -j
fi

PERF_BIN="${BUILD_DIR}/bin/llama-omni-perf-duplex"
if [[ ! -x "${PERF_BIN}" ]]; then
    PERF_BIN="${BUILD_DIR}/tools/omni/llama-omni-perf-duplex"
fi
if [[ ! -x "${PERF_BIN}" ]]; then
    echo "[run_perf] 未找到 llama-omni-perf-duplex，请先用 --build 编译。" >&2
    exit 3
fi

echo "[run_perf] 运行 profiler: ${PERF_BIN}"
echo "[run_perf] 进帧间隔=${INTERVAL}ms  输出=${JSON_PATH}"
# perf-duplex 默认 --stream-interval 1000；这里把 OUTPUT_DIR/JSON 透传
"${PERF_BIN}" --out-json "${JSON_PATH}" -o "${OUTPUT_DIR}" "${PERF_ARGS[@]}"

echo ""
echo "[run_perf] 生成可行性报告 ..."
set +e
python3 "${SCRIPT_DIR}/analyze_perf.py" "${JSON_PATH}" --interval-ms "${INTERVAL}" --md "${MD_PATH}"
RC=$?
set -e
echo "[run_perf] markdown 报告: ${MD_PATH}"
exit ${RC}
