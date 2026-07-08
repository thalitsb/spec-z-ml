#!/bin/bash
#SBATCH --job-name=thalita-cnn-optuna-simples
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --nodelist=nodenv2   # opcional: nodenv2 = RTX A6000 48GB

# =====================================================================
# Versao SIMPLES da CNN com Optuna (notebook didatico, com papermill).
#
# O notebook tem uma celula tageada `parameters` no topo expondo:
#     OBJECT_TYPE, SEED, N_TRIALS, EPOCHS_PER_TRIAL, FINAL_EPOCHS
#
# Esses parametros sao injetados via papermill -p (defaults entre parenteses).
#
# Uso (rode da raiz do projeto):
#
#   # Default (LRG, 50 trials, defaults do notebook):
#       sbatch --export=ALL,OBJ=LRG scripts/train_cnn_optuna_simples.sbatch
#
#   # Override de qualquer parametro:
#       sbatch --export=ALL,OBJ=LRG,N_TRIALS=100,SEED=7 \
#           scripts/train_cnn_optuna_simples.sbatch
#
#       sbatch --export=ALL,OBJ=LRG,N_TRIALS=80,EPOCHS_PER_TRIAL=30,FINAL_EPOCHS=120 \
#           scripts/train_cnn_optuna_simples.sbatch
#
# Notebook executado salvo em:
#     notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_simples_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

# Variaveis com defaults (overridable via --export). Se nao setadas, papermill
# usa os defaults da celula `parameters` do notebook.
OBJ="${OBJ:-LRG}"
SEED="${SEED:-1}"
N_TRIALS="${N_TRIALS:-50}"
EPOCHS_PER_TRIAL="${EPOCHS_PER_TRIAL:-25}"
FINAL_EPOCHS="${FINAL_EPOCHS:-80}"

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
echo "Notebook              : SIMPLES (didatico)"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
mkdir -p ${SLURM_SUBMIT_DIR}/notebooks/07_cnn_optuna/runs

# Inicializar conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

cd ${SLURM_SUBMIT_DIR}

# Forca o config.py a usar data/ DENTRO do projeto
export SPECZML_DATA="${SLURM_SUBMIT_DIR}/data"
echo "SPECZML_DATA = ${SPECZML_DATA}"

# Confere papermill instalado
if ! python -c "import papermill" 2>/dev/null; then
    echo "ERRO: papermill nao instalado. Instale no login node:"
    echo "    conda activate thalita && pip install papermill nbformat"
    exit 1
fi

# Diagnostico de GPU
echo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo

# Caminhos do notebook
OBJ_LOWER=$(echo "${OBJ}" | tr '[:upper:]' '[:lower:]')
NB_IN="notebooks/07_cnn_optuna/cnn_optuna_${OBJ_LOWER}_simples.ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_simples_${RUN_ID}.ipynb"

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo "Iniciando notebook simples via papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

# papermill executa o notebook celula a celula injetando os parametros
# em uma celula `injected-parameters` logo apos a celula tageada `parameters`.
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
echo "Resultados em    : results/${OBJ}/cnn_optuna_simples/"
echo "Modelo em        : models/${OBJ}/cnn_optuna_simples/"
echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
