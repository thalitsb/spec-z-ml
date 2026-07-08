#!/bin/bash
#SBATCH --job-name=thalita-cnn-optuna
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --nodelist=nodenv2   # opcional: nodenv2 = RTX A6000 48GB (recomendado, HPO eh pesado)

# =====================================================================
# HPO CNN com Optuna + 5-fold CV + retreino final.
#
# Roda o notebook 07_cnn_optuna/cnn_optuna_${OBJ}.ipynb via papermill,
# injetando os parametros (OBJECT_TYPE, OBJECTIVE, SEED, etc.).
#
# Uso (rode da raiz do projeto):
#     sbatch --export=ALL,OBJ=LRG scripts/train_cnn_optuna.sbatch
#     sbatch --export=ALL,OBJ=ELG scripts/train_cnn_optuna.sbatch
#
# Override de hiperparametros do estudo (opcionais):
#     sbatch --export=ALL,OBJ=LRG,OBJECTIVE=mae,SEED=7 scripts/train_cnn_optuna.sbatch
#
# Notebook executado e salvo em:
#     notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
OBJECTIVE="${OBJECTIVE:-composite}"   # "mae" | "composite"
SEED="${SEED:-42}"
USE_Z_WEIGHTS="${USE_Z_WEIGHTS:-False}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio de submissao: ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "Objeto                : ${OBJ}"
echo "Objetivo HPO          : ${OBJECTIVE}"
echo "Seed                  : ${SEED}"
echo "Use z-weights         : ${USE_Z_WEIGHTS}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
mkdir -p ${SLURM_SUBMIT_DIR}/notebooks/07_cnn_optuna/runs

# Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

cd ${SLURM_SUBMIT_DIR}

# Forca o config.py a usar data/ DENTRO do projeto, ignorando dataset/ antigo
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"
echo "SPECZML_DATA = ${SPECZML_DATA}"

# Garante papermill instalado (instala se faltar)
python -c "import papermill" 2>/dev/null || pip install --quiet papermill nbformat

# Diagnostico de GPU
echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

# Caminhos do notebook
NB_IN="notebooks/07_cnn_optuna/cnn_optuna_$(echo "${OBJ}" | tr '[:upper:]' '[:lower:]').ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_${RUN_ID}.ipynb"

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo "Iniciando HPO via papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

papermill "${NB_IN}" "${NB_OUT}" \
    -p OBJECT_TYPE          "${OBJ}" \
    -p OBJECTIVE_PAPERMILL  "${OBJECTIVE}" \
    -p SEED                 "${SEED}" \
    -p USE_Z_WEIGHTS        "${USE_Z_WEIGHTS}" \
    --log-output \
    --progress-bar

echo
echo "HPO finalizado: $(date)"
echo "Notebook executado salvo em: ${NB_OUT}"
echo "Resultados em: results/${OBJ}/cnn_optuna/  e  models/${OBJ}/cnn_optuna/"
echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
