#!/bin/bash
#SBATCH --job-name=thalita-ablation-t0
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
echo "================================================================"
echo "Ablacao trial 0 | Job ${SLURM_JOB_ID} | Node $(hostname) | $(date)"
echo "================================================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita
cd ${SLURM_SUBMIT_DIR}
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

python scripts/analysis/ablation_trial0.py

echo
echo "Fim: $(date)"
