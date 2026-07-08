#!/bin/bash
#SBATCH --job-name=thalita-benchmark
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-49%4
##  array 0-49 = os 50 combos do grid (ver: python scripts/analysis/benchmark.py --list)
##  %4 = no maximo 4 tarefas simultaneas (ajustar conforme GPUs livres)

# =====================================================================
# Benchmark multi-seed: 1 tarefa do array = 1 combo (model, object, seed).
# Resumivel: cada tarefa pula se o CSV do run ja existe.
#
#     sbatch scripts/benchmark.sbatch
# Depois de tudo terminar:  python scripts/analysis/benchmark_summary.py
# =====================================================================

set -euo pipefail

echo "================================================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} | Job ${SLURM_JOB_ID} | Node $(hostname) | $(date)"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita
cd ${SLURM_SUBMIT_DIR}
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"

python scripts/analysis/benchmark.py --task-id ${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} finalizada: $(date)"
