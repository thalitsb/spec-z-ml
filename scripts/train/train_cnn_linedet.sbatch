#!/bin/bash
#SBATCH --job-name=thalita-cnn-linedet
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --nodelist=nodenv2   # nodenv1 = RTX 4070 Ti 16 GB, nodenv2 = RTX A6000 48 GB

# =====================================================================
# Treina a CNN de DETECCAO DE LINHA [OII] (cabeca tipo QuasarNET) para ELG.
# So' ELG: [OII] nao se aplica a LRG (sem emissao) nem QSO (linhas largas).
#
# Uso:
#     sbatch scripts/train_cnn_linedet.sbatch
#     sbatch --export=ALL,EPOCHS=80,HMW=0.3 scripts/train_cnn_linedet.sbatch
# =====================================================================

set -euo pipefail

OBJ="ELG"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
LR="${LR:-3e-4}"
HMW="${HMW:-0.2}"        # peso da perda de heatmap
SIGMA="${SIGMA:-2.0}"    # largura do alvo Gaussiano (passos do feature map)

echo "================================================================"
echo "Job iniciado : $(date)  |  ID ${SLURM_JOB_ID}  |  Node $(hostname)"
echo "ELG  Epochs=${EPOCHS} Batch=${BATCH} LR=${LR} heatmap_w=${HMW} sigma=${SIGMA}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita
cd ${SLURM_SUBMIT_DIR}
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"

echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

python scripts/train/train_cnn_linedet.py \
    --object ${OBJ} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH} \
    --learning-rate ${LR} \
    --heatmap-weight ${HMW} \
    --sigma-bins ${SIGMA}

echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
