#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${1:-${PROJECT_ROOT}/Results/reproduce_$(date +%Y%m%d_%H%M%S)}"
BIN_DIR="${RESULTS_DIR}/bin"
TIMING_CSV="${RESULTS_DIR}/benchmark.csv"
IFS=' ' read -r -a DTYPE_LIST <<< "${SCAN_DTYPES:-fp32 bf16}"

mkdir -p "${RESULTS_DIR}" "${BIN_DIR}"

"${SCRIPT_DIR}/collect_env.sh" "${RESULTS_DIR}/env.txt"

g++ -O2 -std=c++17 -o "${BIN_DIR}/generate_inputs" "${PROJECT_ROOT}/SyntheticData/generate_inputs.cpp"
mkdir -p "${PROJECT_ROOT}/SyntheticData/inputs"
"${BIN_DIR}/generate_inputs" "${PROJECT_ROOT}/SyntheticData/inputs" | tee "${RESULTS_DIR}/generate_inputs.log"

g++ -O2 -std=c++17 -o "${BIN_DIR}/run_reference" "${PROJECT_ROOT}/SequentialBaseline/run_reference.cpp"
mkdir -p "${PROJECT_ROOT}/SequentialBaseline/SequentialData"
"${BIN_DIR}/run_reference" "${PROJECT_ROOT}/SyntheticData/inputs" "${PROJECT_ROOT}/SequentialBaseline/SequentialData" | tee "${RESULTS_DIR}/run_reference.log"

GPU_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
fi
if [[ -z "${GPU_CC}" ]]; then
    GPU_CC="80"
fi

nvcc -O3 -std=c++17 -arch="sm_${GPU_CC}" -o "${BIN_DIR}/benchmark" "${PROJECT_ROOT}/Kernels/benchmark.cu"

rm -f "${TIMING_CSV}"
merge_csv_into_timing() {
    local source_csv="$1"
    local target_csv="$2"
    python3 - "$source_csv" "$target_csv" <<'PY'
import csv
import os
import sys

source_csv, target_csv = sys.argv[1], sys.argv[2]
with open(source_csv, newline="") as src:
    reader = csv.DictReader(src)
    rows = list(reader)
    fieldnames = reader.fieldnames
if not rows:
    raise SystemExit(0)
need_header = not os.path.exists(target_csv)
with open(target_csv, "a", newline="") as dst:
    writer = csv.DictWriter(dst, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY
}

for DTYPE in "${DTYPE_LIST[@]}"; do
    DTYPE_OUT_DIR="${RESULTS_DIR}/${DTYPE}_sweep"
    mkdir -p "${DTYPE_OUT_DIR}"
    "${BIN_DIR}/benchmark" \
        "${PROJECT_ROOT}/SyntheticData/inputs" \
        "${PROJECT_ROOT}/SequentialBaseline/SequentialData" \
        "${DTYPE_OUT_DIR}" \
        --dtype "${DTYPE}" \
        --warmup 10 \
        --repeat 50 | tee "${RESULTS_DIR}/benchmark_${DTYPE}.stdout.txt"
    merge_csv_into_timing "${DTYPE_OUT_DIR}/benchmark.csv" "${TIMING_CSV}"
done

rm -f "${RESULTS_DIR}/selective_scan_stdout.txt"
for DTYPE in "${DTYPE_LIST[@]}"; do
    python3 "${PROJECT_ROOT}/Benchmarking/benchmark_selective_scan.py" \
        --input_dir "${PROJECT_ROOT}/SyntheticData/inputs" \
        --ref_dir "${PROJECT_ROOT}/SequentialBaseline/SequentialData" \
        --output_csv "${TIMING_CSV}" \
        --dtype "${DTYPE}" \
        --warmup 10 \
        --repeat 50 | tee -a "${RESULTS_DIR}/selective_scan_stdout.txt" || true
done

if [[ "${RUN_REAL_MAMBA:-0}" == "1" ]]; then
    REAL_SURROGATE_DIR="${RESULTS_DIR}/real_surrogate_inputs"
    REAL_REF_DIR="${RESULTS_DIR}/real_surrogate_refs"
    REAL_FULL_DIR="${RESULTS_DIR}/real_scan_tensors"
    REAL_BENCH_DIR="${RESULTS_DIR}/real_surrogate_benchmark"

    python3 "${PROJECT_ROOT}/Benchmarking/dump_mamba_activations.py" \
        --dtype bf16 \
        --full_out_dir "${REAL_FULL_DIR}" \
        --surrogate_out_dir "${REAL_SURROGATE_DIR}" | tee "${RESULTS_DIR}/dump_mamba_activations.txt"

    mkdir -p "${REAL_REF_DIR}"
    "${BIN_DIR}/run_reference" "${REAL_SURROGATE_DIR}" "${REAL_REF_DIR}" | tee "${RESULTS_DIR}/run_reference_real_inputs.log"

    mkdir -p "${REAL_BENCH_DIR}"
    "${BIN_DIR}/benchmark" \
        "${REAL_SURROGATE_DIR}" \
        "${REAL_REF_DIR}" \
        "${REAL_BENCH_DIR}" \
        --dtype bf16 \
        --warmup 10 \
        --repeat 50 | tee "${RESULTS_DIR}/benchmark_real_surrogate.stdout.txt"

    python3 "${PROJECT_ROOT}/Benchmarking/benchmark_mamba_layer.py" \
        --dtype bf16 \
        --output_csv "${RESULTS_DIR}/mamba_layer_benchmark.csv" | tee "${RESULTS_DIR}/benchmark_mamba_layer.stdout.txt"
fi

printf 'Reproduction outputs saved in: %s\n' "${RESULTS_DIR}"
