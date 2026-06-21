#!/bin/bash

set -euo pipefail

OUT_PATH="${1:?usage: collect_env.sh <output-path>}"

mkdir -p "$(dirname "${OUT_PATH}")"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"

    if command -v git >/dev/null 2>&1; then
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
        echo "git_status_start"
        git status --short 2>/dev/null || true
        echo "git_status_end"
    fi

    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc_version_start"
        nvcc --version
        echo "nvcc_version_end"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi_summary_start"
        nvidia-smi -L || true
        nvidia-smi --query-gpu=name,driver_version,compute_cap,pstate,clocks.current.graphics,clocks.max.graphics,clocks.applications.graphics,memory.total --format=csv,noheader || true
        echo "nvidia_smi_summary_end"

        echo "nvidia_smi_clocks_start"
        nvidia-smi -q -d CLOCK || true
        echo "nvidia_smi_clocks_end"
    fi

    if command -v python3 >/dev/null 2>&1; then
        echo "python_version=$(python3 --version 2>&1)"
        python3 - <<'PY'
try:
    import torch
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_version={getattr(torch.version, 'cuda', 'NA')}")
except Exception as exc:
    print(f"torch_import_error={exc}")

try:
    import mamba_ssm
    print(f"mamba_ssm_version={getattr(mamba_ssm, '__version__', 'unknown')}")
except Exception as exc:
    print(f"mamba_ssm_import_error={exc}")
PY
    fi
} > "${OUT_PATH}"
