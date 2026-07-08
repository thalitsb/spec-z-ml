#!/bin/bash
#SBATCH --job-name=thalita-cnn-qso-linedet
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --exclude=nodenv3,nodenv4,nodenv5,nodenv6   # RTX 5090 (CC 12.0) sem suporte no TF

# =====================================================================
# Treina a CNN de DETECCAO MULTI-LINHA (QuasarNET-like) para QSO.
# 3 linhas: MgII 2799 + CIII] 1909 + CIV 1549; z = media ponderada por confianca.
#
# IMPORTANTE: submeter com --exclude dos nos RTX 5090 (TF nao suporta CC 12.0):
#     sbatch --exclude=nodenv3,nodenv4,nodenv5,nodenv6 scripts/train/train_cnn_qso_linedet.sbatch
#     sbatch --exclude=nodenv3,nodenv4,nodenv5,nodenv6 --export=ALL,EPOCHS=80,HMW=0.3 scripts/train/train_cnn_qso_linedet.sbatch
# =====================================================================

set -euo pipefail

OBJ="QSO"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
LR="${LR:-3e-4}"
HMW="${HMW:-0.2}"        # peso da perda de heatmap (por linha)
CONFW="${CONFW:-0.2}"    # peso da perda de confianca (por linha)
SIGMA="${SIGMA:-2.0}"    # largura do alvo Gaussiano (passos do feature map)

echo "================================================================"
echo "Job iniciado : $(date)  |  ID ${SLURM_JOB_ID}  |  Node $(hostname)"
echo "QSO multi-linha  Epochs=${EPOCHS} Batch=${BATCH} LR=${LR} heatmap_w=${HMW} conf_w=${CONFW} sigma=${SIGMA}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita
cd ${SLURM_SUBMIT_DIR}
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"

echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

python scripts/train/train_cnn_qso_linedet.py \
    --object ${OBJ} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH} \
    --learning-rate ${LR} \
    --heatmap-weight ${HMW} \
    --conf-weight ${CONFW} \
    --sigma-bins ${SIGMA}

echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
