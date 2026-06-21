#!/bin/bash
#SBATCH --job-name=profile_ssm_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=logs/profile_metrics_%j.out
#SBATCH --error=logs/profile_metrics_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.3.0
    module load cuda/12.6.3
fi

D_LIST=(1 16 64 256 512)
L_LIST=(1024 2048 4096 8192 16384 32768 65536 131072)
PROFILE_KERNELS=(warp_shuffle blelloch hillis_steele cub)
IFS=' ' read -r -a DTYPE_LIST <<< "${SCAN_DTYPES:-fp32 bf16}"

INPUT_DIR="${PROJECT_ROOT}/SyntheticData/inputs"
REF_DIR="${PROJECT_ROOT}/SequentialBaseline/SequentialData"
RUN_TAG="${SLURM_JOB_ID:-colab_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${PROJECT_ROOT}/Results/profile_run_${RUN_TAG}"
RAW_UTIL_DIR="${OUT_ROOT}/raw_ncu_util"
RAW_BYTES_DIR="${OUT_ROOT}/raw_ncu_bytes"
RAW_FLOPS_DIR="${OUT_ROOT}/raw_ncu_flops"
PROBE_DIR="${OUT_ROOT}/raw_ncu_probe"
BIN_DIR="${OUT_ROOT}/bin"
TIMING_CSV="${OUT_ROOT}/timing.csv"
ENV_TXT="${OUT_ROOT}/env.txt"

mkdir -p "${OUT_ROOT}" "${RAW_UTIL_DIR}" "${RAW_BYTES_DIR}" "${RAW_FLOPS_DIR}" "${PROBE_DIR}" "${BIN_DIR}"
rm -f "${TIMING_CSV}"

if [[ -x "${PROJECT_ROOT}/scripts/collect_env.sh" ]]; then
    "${PROJECT_ROOT}/scripts/collect_env.sh" "${ENV_TXT}"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
if [[ -z "${GPU_CC}" ]]; then
    GPU_CC="80"
fi
NVCC_ARCH_FLAG="-arch=sm_${GPU_CC}"

infer_peak_bw_gbs() {
    local name="$1"
    case "${name}" in
        *"A100-SXM4-80GB"*) printf '2039' ;;
        *"A100 80GB"*) printf '2039' ;;
        *"A100-SXM4-40GB"*) printf '1555' ;;
        *"A100"*) printf '1555' ;;
        *"H100 SXM"*|*"H100-SXM"*) printf '3350' ;;
        *"H100 PCIe"*) printf '2000' ;;
        *"RTX 5070 Ti"*) printf '896' ;;
        *"RTX 4090"*) printf '1008' ;;
        *) printf '1555' ;;
    esac
}

infer_peak_fp32_gflops() {
    local name="$1"
    case "${name}" in
        *"A100"*) printf '19500' ;;
        *"H100"*) printf '51000' ;;
        *"RTX 5070 Ti"*) printf '44000' ;;
        *"RTX 4090"*) printf '82580' ;;
        *) printf '19500' ;;
    esac
}

PEAK_BW_GBS="${PEAK_BW_GBS:-$(infer_peak_bw_gbs "${GPU_NAME}")}"
PEAK_FP32_GFLOPS="${PEAK_FP32_GFLOPS:-$(infer_peak_fp32_gflops "${GPU_NAME}")}"

CLOCK_LOCK_REQUEST="${GPU_CLOCK_LOCK:-}"
CLOCK_LOCK_STATUS="not-requested"
cleanup_gpu_clocks() {
    if [[ -n "${CLOCK_LOCK_REQUEST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -rgc >/dev/null 2>&1 || true
    fi
}
trap cleanup_gpu_clocks EXIT

if [[ -n "${CLOCK_LOCK_REQUEST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -pm 1 >/dev/null 2>&1 && nvidia-smi -lgc "${CLOCK_LOCK_REQUEST}" >/dev/null 2>&1; then
        CLOCK_LOCK_STATUS="locked:${CLOCK_LOCK_REQUEST}"
    else
        CLOCK_LOCK_STATUS="lock-failed:${CLOCK_LOCK_REQUEST}"
    fi
fi

echo "Node:  $(hostname)"
echo "GPU:   ${GPU_NAME}"
echo "CC:    ${GPU_CC}"
echo "Arch:  ${NVCC_ARCH_FLAG}"
echo "Start: $(date)"
echo "OUT:   ${OUT_ROOT}"
echo "Peak BW (GB/s): ${PEAK_BW_GBS}"
echo "Peak FP32 (GFLOP/s): ${PEAK_FP32_GFLOPS}"
echo "Clock lock: ${CLOCK_LOCK_STATUS}"
echo

{
    echo "gpu_name=${GPU_NAME}"
    echo "gpu_cc=${GPU_CC}"
    echo "nvcc_arch_flag=${NVCC_ARCH_FLAG}"
    echo "peak_bw_gbs=${PEAK_BW_GBS}"
    echo "peak_fp32_gflops=${PEAK_FP32_GFLOPS}"
    echo "clock_lock_status=${CLOCK_LOCK_STATUS}"
} >> "${ENV_TXT}"

echo "[Stage 1/4] Generate synthetic inputs"
GEN_BIN="${BIN_DIR}/generate_inputs"
g++ -O2 -std=c++17 -o "${GEN_BIN}" "${PROJECT_ROOT}/SyntheticData/generate_inputs.cpp"
mkdir -p "${INPUT_DIR}"
"${GEN_BIN}" "${INPUT_DIR}" | tee "${OUT_ROOT}/generate_inputs.log"

echo
echo "[Stage 2/4] Run sequential baseline and write references"
REF_BIN="${BIN_DIR}/run_reference"
g++ -O2 -std=c++17 -o "${REF_BIN}" "${PROJECT_ROOT}/SequentialBaseline/run_reference.cpp"
mkdir -p "${REF_DIR}"
"${REF_BIN}" "${INPUT_DIR}" "${REF_DIR}" | tee "${OUT_ROOT}/run_reference.log"

echo
echo "[Stage 3/4] Benchmark and profile GPU kernels"

QUERY_MODE=""
QUERY_OUTPUT=""
if QUERY_OUTPUT="$(ncu --query-metrics --chips ga100 --csv 2>/dev/null)"; then
    QUERY_MODE="--chips ga100"
elif QUERY_OUTPUT="$(ncu --query-metrics --chip sm_${GPU_CC} --csv 2>/dev/null)"; then
    QUERY_MODE="--chip sm_${GPU_CC}"
elif QUERY_OUTPUT="$(ncu --query-metrics --csv 2>/dev/null)"; then
    QUERY_MODE="default"
else
    echo "Failed to query Nsight Compute metrics."
    exit 1
fi

AVAIL_METRICS="$(printf '%s\n' "${QUERY_OUTPUT}" | awk -F, 'NR>1 {gsub(/"/,"",$1); print $1}')"
echo "Metric query mode: ${QUERY_MODE}"

BIN="${BIN_DIR}/profile_driver"
echo "Compiling profile driver (${NVCC_ARCH_FLAG})"
nvcc -O3 -std=c++17 "${NVCC_ARCH_FLAG}" --maxrregcount=64 -o "${BIN}" "${SCRIPT_DIR}/profile_driver.cu"

PROBE_KERNEL="${PROFILE_KERNELS[0]}"
PROBE_DTYPE="${DTYPE_LIST[0]}"
PROBE_D="${D_LIST[0]}"
PROBE_L="${L_LIST[0]}"
PROBE_INPUT_FILE="${INPUT_DIR}/input_B1_L${PROBE_L}_D${PROBE_D}.bin"
PROBE_REF_FILE="${REF_DIR}/ref_B1_L${PROBE_L}_D${PROBE_D}.bin"

echo "Validating Nsight metric candidates with ${PROBE_KERNEL} dtype=${PROBE_DTYPE} D=${PROBE_D} L=${PROBE_L}"

if [[ ! -f "${PROBE_INPUT_FILE}" || ! -f "${PROBE_REF_FILE}" ]]; then
    echo "Missing probe input/reference files for metric validation."
    exit 1
fi

declare -A METRIC_PROBE_CACHE

metric_family_listed() {
    local candidate="$1"
    local family="${candidate%%.*}"
    grep -Fxq "${candidate}" <<< "${AVAIL_METRICS}" || grep -Fxq "${family}" <<< "${AVAIL_METRICS}"
}

probe_metric() {
    local metric="$1"
    local cache_key="${metric}"
    local safe_metric
    local probe_err

    if [[ -n "${METRIC_PROBE_CACHE[$cache_key]+x}" ]]; then
        [[ "${METRIC_PROBE_CACHE[$cache_key]}" == "ok" ]]
        return
    fi

    safe_metric="$(printf '%s' "${metric}" | tr -c '[:alnum:]_' '_')"
    probe_err="${PROBE_DIR}/${safe_metric}.stderr"

    if ncu \
        --target-processes all \
        --clock-control base \
        --cache-control all \
        --replay-mode kernel \
        --csv \
        --page raw \
        --metrics "${metric}" \
        --force-overwrite \
        --log-file "${PROBE_DIR}/${safe_metric}.csv" \
        "${BIN}" \
            --kernel "${PROBE_KERNEL}" \
            --dtype "${PROBE_DTYPE}" \
            --D "${PROBE_D}" \
            --L "${PROBE_L}" \
            --input_dir "${INPUT_DIR}" \
            --ref_dir "${REF_DIR}" \
            --warmup 0 \
            --repeat 1 \
            --skip_check \
            --no_print >/dev/null 2>"${probe_err}"; then
        METRIC_PROBE_CACHE[$cache_key]="ok"
        return 0
    fi

    METRIC_PROBE_CACHE[$cache_key]="fail"
    return 1
}

pick_metric() {
    for candidate in "$@"; do
        if ! metric_family_listed "${candidate}"; then
            continue
        fi
        if probe_metric "${candidate}"; then
            echo "${candidate}"
            return 0
        fi
    done
    echo ""
    return 0
}

M_GPU_TIME="$(pick_metric gpu__time_duration.sum gpu__time_duration.avg gpu__time_duration)"
M_SM_UTIL="$(pick_metric sm__throughput.avg.pct_of_peak_sustained_elapsed sm__throughput.avg.pct_of_peak_sustained_active sm__throughput)"
M_DRAM_UTIL="$(pick_metric dram__throughput.avg.pct_of_peak_sustained_elapsed dram__throughput.avg.pct_of_peak_sustained_active dram__throughput gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed gpu__dram_throughput.avg.pct_of_peak_sustained_active gpu__dram_throughput)"
M_OCC="$(pick_metric sm__warps_active.avg.pct_of_peak_sustained_active sm__warps_active.avg.pct_of_peak_sustained_elapsed sm__warps_active)"
M_DRAM_BYTES="$(pick_metric dram__bytes.sum dram__bytes)"
M_DRAM_BYTES_READ="$(pick_metric dram__bytes_read.sum dram__bytes_read)"
M_DRAM_BYTES_WRITE="$(pick_metric dram__bytes_write.sum dram__bytes_write)"
M_FADD="$(pick_metric smsp__sass_thread_inst_executed_op_fadd_pred_on.sum sm__sass_thread_inst_executed_op_fadd_pred_on.sum smsp__sass_thread_inst_executed_op_fadd_pred_on sm__sass_thread_inst_executed_op_fadd_pred_on)"
M_FMUL="$(pick_metric smsp__sass_thread_inst_executed_op_fmul_pred_on.sum sm__sass_thread_inst_executed_op_fmul_pred_on.sum smsp__sass_thread_inst_executed_op_fmul_pred_on sm__sass_thread_inst_executed_op_fmul_pred_on)"
M_FFMA="$(pick_metric smsp__sass_thread_inst_executed_op_ffma_pred_on.sum sm__sass_thread_inst_executed_op_ffma_pred_on.sum smsp__sass_thread_inst_executed_op_ffma_pred_on sm__sass_thread_inst_executed_op_ffma_pred_on)"
M_REGS="$(pick_metric launch__registers_per_thread)"
M_SMEM_BLOCK="$(pick_metric launch__shared_mem_per_block_allocated launch__shared_mem_per_block_static)"
M_BLOCK_SIZE="$(pick_metric launch__block_size)"

if [[ -z "${M_GPU_TIME}" || -z "${M_SM_UTIL}" || -z "${M_DRAM_UTIL}" || -z "${M_OCC}" ]]; then
    echo "Failed to resolve required utilization metrics."
    echo "Probe logs: ${PROBE_DIR}"
    exit 1
fi

if [[ -z "${M_DRAM_BYTES}" && -z "${M_DRAM_BYTES_READ}" && -z "${M_DRAM_BYTES_WRITE}" ]]; then
    echo "Failed to resolve any DRAM byte metrics for roofline profiling."
    echo "Probe logs: ${PROBE_DIR}"
    exit 1
fi

if [[ -z "${M_FADD}" && -z "${M_FMUL}" && -z "${M_FFMA}" ]]; then
    echo "Failed to resolve any FP instruction metrics for roofline profiling."
    echo "Probe logs: ${PROBE_DIR}"
    exit 1
fi

UTIL_METRICS=()
for m in "${M_GPU_TIME}" "${M_SM_UTIL}" "${M_DRAM_UTIL}" "${M_OCC}" "${M_REGS}" "${M_SMEM_BLOCK}" "${M_BLOCK_SIZE}"; do
    [[ -n "${m}" ]] && UTIL_METRICS+=("${m}")
done

BYTES_METRICS=()
for m in "${M_GPU_TIME}" "${M_DRAM_BYTES}" "${M_DRAM_BYTES_READ}" "${M_DRAM_BYTES_WRITE}"; do
    [[ -n "${m}" ]] && BYTES_METRICS+=("${m}")
done

FLOPS_METRICS=()
for m in "${M_GPU_TIME}" "${M_FADD}" "${M_FMUL}" "${M_FFMA}"; do
    [[ -n "${m}" ]] && FLOPS_METRICS+=("${m}")
done

UTIL_METRIC_STR="$(IFS=,; echo "${UTIL_METRICS[*]}")"
BYTES_METRIC_STR="$(IFS=,; echo "${BYTES_METRICS[*]}")"
FLOPS_METRIC_STR="$(IFS=,; echo "${FLOPS_METRICS[*]}")"

profile_pass() {
    local metric_str="$1"
    local log_dir="$2"

    ncu \
        --target-processes all \
        --clock-control base \
        --cache-control all \
        --replay-mode kernel \
        --csv \
        --page raw \
        --metrics "${metric_str}" \
        --force-overwrite \
        --log-file "${log_dir}/${TAG}.csv" \
        "${BIN}" \
            --kernel "${KERNEL}" \
            --dtype "${DTYPE}" \
            --D "${D}" \
            --L "${L}" \
            --input_dir "${INPUT_DIR}" \
            --ref_dir "${REF_DIR}" \
            --warmup 0 \
            --repeat 1 \
            --skip_check \
            --no_print
}

TOTAL=$(( ${#D_LIST[@]} * ${#L_LIST[@]} * ${#PROFILE_KERNELS[@]} * ${#DTYPE_LIST[@]} ))
COUNT=0
for DTYPE in "${DTYPE_LIST[@]}"; do
    for D in "${D_LIST[@]}"; do
        for KERNEL in "${PROFILE_KERNELS[@]}"; do
            for L in "${L_LIST[@]}"; do
            COUNT=$((COUNT + 1))
            TAG="${KERNEL}_${DTYPE}_D${D}_L${L}"

            INPUT_FILE="${INPUT_DIR}/input_B1_L${L}_D${D}.bin"
            REF_FILE="${REF_DIR}/ref_B1_L${L}_D${D}.bin"
            if [[ ! -f "${INPUT_FILE}" || ! -f "${REF_FILE}" ]]; then
                echo "[${COUNT}/${TOTAL}] Skipping ${TAG} (missing input/ref file)"
                continue
            fi

            echo "[${COUNT}/${TOTAL}] Timing ${TAG}"
            "${BIN}" \
                --kernel "${KERNEL}" \
                --dtype "${DTYPE}" \
                --D "${D}" \
                --L "${L}" \
                --input_dir "${INPUT_DIR}" \
                --ref_dir "${REF_DIR}" \
                --warmup 10 \
                --repeat 50 \
                --csv_append "${TIMING_CSV}"

            echo "[${COUNT}/${TOTAL}] Profiling ${TAG} with Nsight Compute (util pass)"
            profile_pass "${UTIL_METRIC_STR}" "${RAW_UTIL_DIR}"

            echo "[${COUNT}/${TOTAL}] Profiling ${TAG} with Nsight Compute (bytes pass)"
            profile_pass "${BYTES_METRIC_STR}" "${RAW_BYTES_DIR}"

            echo "[${COUNT}/${TOTAL}] Profiling ${TAG} with Nsight Compute (flops pass)"
            profile_pass "${FLOPS_METRIC_STR}" "${RAW_FLOPS_DIR}"
            done
        done
    done
done

if command -v python3 >/dev/null 2>&1; then
    echo
    echo "[Stage 4/4] Optional selective_scan_cuda timing sweep"
    rm -f "${OUT_ROOT}/selective_scan_timing.log"
    for DTYPE in "${DTYPE_LIST[@]}"; do
        python3 "${PROJECT_ROOT}/Benchmarking/benchmark_selective_scan.py" \
            --input_dir "${INPUT_DIR}" \
            --ref_dir "${REF_DIR}" \
            --output_csv "${TIMING_CSV}" \
            --warmup 10 \
            --repeat 50 \
            --dtype "${DTYPE}" \
            --no_print | tee -a "${OUT_ROOT}/selective_scan_timing.log" || true
    done
fi

echo
echo "Running analysis..."
python3 "${SCRIPT_DIR}/analyze_metrics.py" \
    --timing_csv "${TIMING_CSV}" \
    --raw_dir "${RAW_UTIL_DIR}" \
    --raw_dir "${RAW_BYTES_DIR}" \
    --raw_dir "${RAW_FLOPS_DIR}" \
    --out_dir "${OUT_ROOT}" \
    --peak_bw_gbs "${PEAK_BW_GBS}" \
    --peak_fp32_gflops "${PEAK_FP32_GFLOPS}" | tee "${OUT_ROOT}/analysis_stdout.txt"

echo
echo "Finished: $(date)"
echo "All outputs saved in: ${OUT_ROOT}"
