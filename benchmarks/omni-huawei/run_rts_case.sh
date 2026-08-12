#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 LABEL GPU WARMUPS RUNS [NAME=VALUE ...]" >&2
    exit 2
fi

label=$1
gpu=$2
warmups=$3
runs=$4
shift 4

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
judge_root="${repo}/evaluation/judge-final"
result_base=${RESULT_BASE:-/workspace/MiniCPM--competition/result/llama_huawei_migration}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root="${result_base}/${label}_${stamp}"
server_bin=${OMNI_SERVER_BIN:-${repo}/build/bin/llama-omni-server}
ctx_size=${CTX_SIZE:-40960}

source "${repo}/.local-eval/config.env"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GGML_CANN_WEIGHT_NZ=off
export GGML_CANN_ACL_GRAPH=off
export OMNI_T2W_DEVICE=${OMNI_T2W_DEVICE:-gpu:0}
export OMNI_T2M_DEVICE=${OMNI_T2M_DEVICE:-gpu:0}
export OMNI_VOC_DEVICE=${OMNI_VOC_DEVICE:-gpu:0}
export OMNI_SAMPLER_SEED=${OMNI_SAMPLER_SEED:-42}
export OMNI_SERVER_BIN="${server_bin}"
export PYTHONPATH="${repo}/.local-eval/python:${PYTHONPATH:-}"

mkdir -p "${result_root}"
{
    echo "label=${label}"
    echo "utc_started=$(date -u +%FT%TZ)"
    echo "git_head=$(git -C "${repo}" rev-parse HEAD)"
    echo "server_bin=${server_bin}"
    echo "server_sha256=$(sha256sum "${server_bin}" | awk '{print $1}')"
    echo "gpu=${gpu}"
    echo "ctx_size=${ctx_size}"
    echo "warmups=${warmups}"
    echo "runs=${runs}"
    printf 'env=%q\n' "$@"
} > "${result_root}/manifest.txt"

run_phase() {
    local phase=$1
    local count=$2
    shift 2
    local index out_dir meta report
    for index in $(seq -w 1 "${count}"); do
        out_dir="${result_root}/${phase}_${index}"
        mkdir -p "${out_dir}"
        echo "[$(date -u +%FT%TZ)] START ${label} ${phase} ${index}"
        env "$@" "${repo}/.venv-eval/bin/python" "${judge_root}/run_judge_direct.py" \
            --model "${RTS_MODEL_LLM:-${MODEL_LLM}}" \
            --llamacpp-root "${repo}" \
            --video "${repo}/evaluation/judge-final/assets/video/omni_duplex1.mp4" \
            --max-duration 120 \
            --runs-dir "${out_dir}/rts_runs" \
            --gpu "${gpu}" \
            --ctx-size "${ctx_size}" \
            > "${out_dir}/console.log" 2>&1
        meta=$(find "${out_dir}/rts_runs" -name run_meta.json -print -quit)
        report=$("${repo}/.venv-eval/bin/python" - "${meta}" "${judge_root}" <<'PY'
import json, pathlib, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
print(pathlib.Path(sys.argv[2]) / meta["sessions"][0] / "eval_e2e_report.json")
PY
)
        "${repo}/.venv-eval/bin/python" - "${report}" "${phase}" "${index}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
r = d["rtf"]
print(
    f"DONE {sys.argv[2]} {sys.argv[3]} "
    f"all_pooled={r['rtf_aggregate']:.4f} all_mean={r['rtf']['mean']:.4f} "
    f"core={r['core']['rtf_aggregate']:.4f} "
    f"speak_ms={d['e2e_speak_recv_to_wav_poll_ms']['mean_ms']:.1f}",
    flush=True,
)
PY
    done
}

if (( warmups > 0 )); then
    run_phase warmup "${warmups}" "$@"
fi
run_phase measured "${runs}" "$@"
"${repo}/.venv-eval/bin/python" \
    "${repo}/benchmarks/omni-huawei/summarize_rts.py" "${result_root}"
echo "result_root=${result_root}"
