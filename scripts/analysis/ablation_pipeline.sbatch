#!/bin/bash
#SBATCH --job-name=thalita-ablation
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =====================================================================
# Ablation controlado do pipeline (por que o flex e' ~2x pior que o baseline?).
# Mesma arquitetura, varia 1 fator de cada vez (Adam vs AdamW, dropout,
# early-stop, scalars). Na GPU da' pra usar o DATASET INTEIRO + 40 epocas.
#
#   sbatch --export=ALL,OBJ=LRG scripts/analysis/ablation_pipeline.sbatch
#   sbatch --export=ALL,OBJ=ELG,EPOCHS=50 scripts/analysis/ablation_pipeline.sbatch
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
N="${N:-200000}"      # >= tamanho do dataset => usa TODOS os espectros
EPOCHS="${EPOCHS:-40}"

echo "================================================================"
echo "Ablation pipeline | Job ${SLURM_JOB_ID} | Node $(hostname) | $(date)"
echo "OBJ=${OBJ}  N=${N} (>=dataset => full)  EPOCHS=${EPOCHS}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita
cd ${SLURM_SUBMIT_DIR}
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"

nvidia-smi --query-gpu=name,memory.free --format=csv 2>/dev/null || true
echo

# -u = saida sem buffer (progresso variante a variante aparece no log)
python -u scripts/analysis/ablation_pipeline.py --object ${OBJ} --n ${N} --epochs ${EPOCHS}

echo "================================================================"
echo "Ablation finalizado: $(date)"
echo "================================================================"
