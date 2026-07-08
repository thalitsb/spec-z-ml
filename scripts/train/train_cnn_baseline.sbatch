#!/bin/bash
#SBATCH --job-name=thalita-cnn-baseline
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --nodelist=nodenv2   # optional: nodenv1 = RTX 4070 Ti 16 GB, nodenv2 = RTX A6000 48 GB

# =====================================================================
# Treina a CNN baseline para um objeto (ELG, LRG ou QSO).
#
# Uso:
#     sbatch --export=OBJ=LRG scripts/train_cnn_baseline.sbatch
#     sbatch --export=OBJ=ELG scripts/train_cnn_baseline.sbatch
#
# Hiperparametros opcionais (defaults estao no script Python):
#     sbatch --export=OBJ=LRG,EPOCHS=80,BATCH=128 scripts/train_cnn_baseline.sbatch
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
LR="${LR:-3e-4}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio de submissao: ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "Objeto                : ${OBJ}"
echo "Epochs / Batch / LR   : ${EPOCHS} / ${BATCH} / ${LR}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs

# Inicializar o Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

cd ${SLURM_SUBMIT_DIR}

# Forca o config.py a usar data/ DENTRO do projeto, ignorando o
# /home/valerio/Thalita/dataset (versao antiga).
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"
echo "SPECZML_DATA = ${SPECZML_DATA}"

# Diagnostico de GPU
echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

echo "Iniciando treinamento CNN baseline (${OBJ}): $(date)"
python scripts/train/train_cnn_baseline.py \
    --object ${OBJ} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH} \
    --learning-rate ${LR}
echo "Treinamento finalizado: $(date)"

echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
