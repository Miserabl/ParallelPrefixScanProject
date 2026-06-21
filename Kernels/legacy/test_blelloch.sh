#!/bin/bash
#SBATCH --job-name=test_blelloch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=logs/test_blelloch_%j.out
#SBATCH --error=logs/test_blelloch_%j.err

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

for D in 16 64 256 512; do
    echo "=== D=$D ==="
    nvcc -O2 -std=c++17 -arch=sm_80 -DD=$D --maxrregcount=64 -o test_blelloch_D${D} test_blelloch.cu
    ./test_blelloch_D${D}
    echo ""
done

echo "End: $(date)"
