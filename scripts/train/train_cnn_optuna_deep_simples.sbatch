#!/bin/bash
#SBATCH --job-name=thalita-cnn-optuna-deep
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --nodelist=nodenv2   # opcional: nodenv2 = RTX A6000 48GB

# =====================================================================
# Versao DEEP do simples (arquitetura igual ao baseline + Optuna).
#
# Diferencas pra versao "simples" rasa:
#   - 4 blocos (3x2 conv + 1 conv final) em vez de 3 blocos x 1 conv
#   - Dense head 256 -> 128 -> 64 -> 1
#   - Output: ScaledSoftplus (em vez de sigmoid + Rescaling)
#   - Loss: MSE (em vez de MAE)
#   - EPOCHS_PER_TRIAL=50 (em vez de 25)
#
# Por isso o walltime aqui eh 24h (vs 16h no simples raso).
#
# Uso (rode da raiz do projeto):
#     sbatch --export=ALL,OBJ=LRG scripts/train_cnn_optuna_deep_simples.sbatch
#     sbatch --export=ALL,OBJ=LRG,N_TRIALS=30 scripts/train_cnn_optuna_deep_simples.sbatch
#
# Notebook executado salvo em:
#     notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_deep_simples_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
SEED="${SEED:-1}"
N_TRIALS="${N_TRIALS:-50}"
EPOCHS_PER_TRIAL="${EPOCHS_PER_TRIAL:-50}"
FINAL_EPOCHS="${FINAL_EPOCHS:-100}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio submissao   : ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "Objeto                : ${OBJ}"
echo "Seed                  : ${SEED}"
echo "N_TRIALS              : ${N_TRIALS}"
echo "EPOCHS_PER_TRIAL      : ${EPOCHS_PER_TRIAL}"
echo "FINAL_EPOCHS          : ${FINAL_EPOCHS}"
echo "Notebook              : DEEP SIMPLES (igual baseline + Optuna)"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
mkdir -p ${SLURM_SUBMIT_DIR}/notebooks/07_cnn_optuna/runs

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

cd ${SLURM_SUBMIT_DIR}

export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"
echo "SPECZML_DATA = ${SPECZML_DATA}"

if ! python -c "import papermill" 2>/dev/null; then
    echo "ERRO: papermill nao instalado. Instale no login node:"
    echo "    conda activate thalita && pip install papermill nbformat"
    exit 1
fi

echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

OBJ_LOWER=$(echo "${OBJ}" | tr '[:upper:]' '[:lower:]')
NB_IN="notebooks/07_cnn_optuna/cnn_optuna_${OBJ_LOWER}_deep_simples.ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_deep_simples_${RUN_ID}.ipynb"

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo "Iniciando notebook DEEP via papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

papermill "${NB_IN}" "${NB_OUT}" \
    -p OBJECT_TYPE       "${OBJ}" \
    -p SEED              "${SEED}" \
    -p N_TRIALS          "${N_TRIALS}" \
    -p EPOCHS_PER_TRIAL  "${EPOCHS_PER_TRIAL}" \
    -p FINAL_EPOCHS      "${FINAL_EPOCHS}" \
    --log-output \
    --progress-bar

echo
echo "Notebook executado: ${NB_OUT}"
echo "Resultados em    : results/${OBJ}/cnn_optuna_simples/n${N_TRIALS}_s${SEED}/"
echo "Modelo em        : models/${OBJ}/cnn_optuna_simples/n${N_TRIALS}_s${SEED}/"
echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
