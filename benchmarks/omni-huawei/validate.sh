#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-static}
build_dir=${BUILD_DIR:-"${TMPDIR:-/tmp}/llama-omni-validation-build-${USER:-user}"}
result_root=${RESULT_DIR:-"${repo_root}/benchmarks/omni-huawei/results"}
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
result_dir="${result_root}/${run_id}"

case "${mode}" in
    static|cpu|full) ;;
    *) echo "usage: $0 [static|cpu|full]" >&2; exit 2 ;;
esac

mkdir -p "${result_dir}/logs"
exec > >(tee "${result_dir}/logs/validate.log") 2>&1

finish() {
    status=$?
    trap - EXIT
    set +e
    echo "exit_status=${status}" >> "${result_dir}/manifest.txt"
    python3 "${repo_root}/benchmarks/omni-huawei/summarize.py" "${result_dir}"
    echo "validation result: ${result_dir} (exit=${status})"
    exit "${status}"
}
trap finish EXIT

declare -Ar protected_hashes=(
    [tools/omni/omni-eval-cli.cpp]=f1d1a0c8169cd9ca356251854411d6c439ceeff234ad79b0bb25a312d5cbd457
    [tools/omni/omni-eval-daily-cli.cpp]=60c211b7e7889cea1a116b0a5155cc3fe748fd7d62898dab99936cd759fd03c6
    [tools/omni/omni-tts-eval.cpp]=e0dfbd068638f5d3f4df49dee2ea54f18fb9a6db7c548da319ae315c54178fb3
    [tools/omni/CMakeLists.txt]=8e67311fa4ff47ada77475f3b6e0edfb9fb49f65685bc4ec0feabcd71bfd7ca1
)
expected_evaluation_hash=f7c02a99eae0cda6df4b1806814fa049495add6e17c8b557201dcfdd64f30e43

cd "${repo_root}"
{
    echo "run_id=${run_id}"
    echo "mode=${mode}"
    echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=$(git rev-parse HEAD)"
    echo "dirty=$(test -n "$(git status --short)" && echo true || echo false)"
    echo "uname=$(uname -srvmo)"
    echo "cmake=$(cmake --version | head -n 1)"
    echo "compiler=$(${CXX:-c++} --version | head -n 1)"
    echo "GGML_CANN_WEIGHT_NZ=${GGML_CANN_WEIGHT_NZ:-off}"
    echo "GGML_CANN_ACL_GRAPH=${GGML_CANN_ACL_GRAPH:-off}"
    echo "CTX_SIZE=${CTX_SIZE:-40960}"
} > "${result_dir}/manifest.txt"

for path in "${!protected_hashes[@]}"; do
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${protected_hashes[${path}]}" ]]; then
        echo "protected hash mismatch: ${path}: ${actual}" >&2
        exit 1
    fi
done

evaluation_hash=$(git ls-files -z evaluation | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
if [[ "${evaluation_hash}" != "${expected_evaluation_hash}" ]]; then
    echo "evaluation tracked tree hash mismatch: ${evaluation_hash}" >&2
    exit 1
fi
echo "protected hashes: ok"

git diff --check
python3 -m py_compile \
    tools/omni/pyt2w/test_service.py \
    tools/omni/pyt2w/test_cpp_oneshot_wiring.py \
    tools/omni/pyt2w/test_runtime_optimization_wiring.py \
    benchmarks/omni-huawei/summarize.py \
    benchmarks/omni-huawei/summarize_rts.py
python3 tools/omni/pyt2w/test_cpp_oneshot_wiring.py -v \
    2>&1 | tee "${result_dir}/logs/python-wiring-tests.log"
python3 tools/omni/pyt2w/test_runtime_optimization_wiring.py -v \
    2>&1 | tee "${result_dir}/logs/python-runtime-optimization-tests.log"

cmake -S . -B "${build_dir}" \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DGGML_CANN=OFF \
    2>&1 | tee "${result_dir}/logs/cmake-configure.log"

if [[ "${mode}" == "cpu" || "${mode}" == "full" ]]; then
    targets=(
        test-token2wav-graph-policy
        test-token2wav-conv-state-policy
        test-token2wav-adaln-silu-policy
        test-token2wav-adaln-cache-policy
        test-token2wav-est-att-writeback-policy
        test-tts-head-code-executor
        test-duplex-terminal-tracker
        test-cann-graph-bypass
        test-cann-set-rows-f32-f16-policy
        test-cann-kv-pair-update-policy
        test-cann-modulate-fusion-policy
        test-cann-attn-time-pack-policy
    )
    cmake --build "${build_dir}" --target "${targets[@]}" -j "${JOBS:-4}" \
        2>&1 | tee "${result_dir}/logs/build-targeted.log"
    ctest --test-dir "${build_dir}" --output-on-failure \
        -R '^(test-token2wav-|test-tts-head-code-executor$|test-duplex-terminal-tracker$|test-cann-(graph-bypass|set-rows-f32-f16-policy|kv-pair-update-policy|modulate-fusion-policy|attn-time-pack-policy)$)' \
        2>&1 | tee "${result_dir}/logs/ctest-targeted.log"
    python3 tools/omni/pyt2w/test_service.py -v \
        2>&1 | tee "${result_dir}/logs/python-service-tests.log"
fi

if [[ "${mode}" == "full" ]]; then
    (
        cd evaluation
        EVAL_CONFIG=${EVAL_CONFIG:-"${repo_root}/.local-eval/config.env"} \
            CTX_SIZE=${CTX_SIZE:-40960} \
            GGML_CANN_WEIGHT_NZ=${GGML_CANN_WEIGHT_NZ:-off} \
            GGML_CANN_ACL_GRAPH=${GGML_CANN_ACL_GRAPH:-off} \
            ./run_all.sh --no-build --smoke 2
    ) 2>&1 | tee "${result_dir}/logs/smoke.log"
fi
