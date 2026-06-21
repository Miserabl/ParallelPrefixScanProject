#!/bin/bash
#SBATCH --job-name=probe_kernels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

module purge
module load gcc/13.3.0
module load cuda/12.6.3

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

echo "=========================================="
echo "WARP SHUFFLE PROBE"
echo "=========================================="
for D in 16 64 256 512; do
    echo "--- D=$D ---"
    nvcc -O2 -std=c++17 -arch=sm_80 -DD=$D --maxrregcount=64 -o probe_ws_D${D} probe_warp_shuffle.cu
    ./probe_ws_D${D}
    echo ""
done

echo "=========================================="
echo "BLELLOCH PROBE"
echo "=========================================="
for D in 16 64 256 512; do
    echo "--- D=$D ---"
    nvcc -O2 -std=c++17 -arch=sm_80 -DD=$D --maxrregcount=64 -o probe_bl_D${D} probe_blelloch.cu
    ./probe_bl_D${D}
    echo ""
done

echo "=========================================="
echo "HILLIS-STEELE PROBE"
echo "=========================================="
for D in 16 64 256 512; do
    echo "--- D=$D ---"
    nvcc -O2 -std=c++17 -arch=sm_80 -DD=$D --maxrregcount=64 -o probe_hs_D${D} probe_hillis_steele.cu
    ./probe_hs_D${D}
    echo ""
done

echo "End: $(date)"
