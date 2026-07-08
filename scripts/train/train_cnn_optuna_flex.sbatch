#!/bin/bash
#SBATCH --job-name=thalita-cnn-optuna-flex
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
# Versao FLEX da CNN com Optuna
#
# O Optuna escolhe a ARQUITETURA INTEIRA, nao so HPs numericos:
#   - n_conv_blocks  (1-4)
#   - n_dense_layers (1-4)
#   - n_units, kernel_size, activation, dropout, lr, weight_decay, batch_size
# Pipeline robusto: splits 3-way estratificados por z, composite objective,
# PercentilePruner, scalars globais concatenados (sem leak), AdamW.
#
# Resume safe: o study tem SQLite + load_if_exists=True. Se cair no time
# limit, basta re-submeter — continua de onde parou.
#
# Uso (rode da raiz do projeto):
#     sbatch --export=ALL,OBJ=ELG scripts/train_cnn_optuna_flex.sbatch
#     sbatch --export=ALL,OBJ=LRG,N_TRIALS=80,SEED=7 scripts/train_cnn_optuna_flex.sbatch
#     sbatch --export=ALL,OBJ=ELG,OBJECTIVE_MODE=mae scripts/train_cnn_optuna_flex.sbatch
#
# Variaveis (defaults entre parenteses, override via --export=ALL,...):
#     OBJ              = ELG | LRG | QSO          (LRG)
#     SEED             = inteiro                  (42)
#     N_TRIALS         = inteiro                  (50)
#     EPOCHS_PER_TRIAL = inteiro                  (40)
#     FINAL_EPOCHS     = inteiro                  (120)
#     OBJECTIVE_MODE   = mae | composite          (mae)
#     USE_Z_WEIGHTS    = True | False             (False)
#
# Notebook executado salvo em:
#     notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_flex_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-50}"
EPOCHS_PER_TRIAL="${EPOCHS_PER_TRIAL:-40}"
FINAL_EPOCHS="${FINAL_EPOCHS:-120}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-mae}"
USE_Z_WEIGHTS="${USE_Z_WEIGHTS:-False}"
# Alvo do loop de trials SEM afetar o config_hash (pra ESTENDER/resumir um study
# existente ate um alvo maior). 0 = usa N_TRIALS. Ex.: N_TRIALS=60 + OVERRIDE=100
# resume o study de 60 e roda ate 100 completos.
TARGET_TRIALS_OVERRIDE="${TARGET_TRIALS_OVERRIDE:-0}"

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
echo "OBJECTIVE_MODE        : ${OBJECTIVE_MODE}"
echo "USE_Z_WEIGHTS         : ${USE_Z_WEIGHTS}"
echo "TARGET_TRIALS_OVERRIDE: ${TARGET_TRIALS_OVERRIDE}  (0 = usa N_TRIALS)"
echo "Notebook              : FLEX (arquitetura variavel + scalars globais)"
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
# NB_IN pode ser sobrescrito via --export=ALL,NB_IN=... (ex.: variante com scalars).
# Se nao for passado, usa o padrao por objeto.
NB_IN="${NB_IN:-notebooks/07_cnn_optuna/cnn_optuna_${OBJ_LOWER}_flex.ipynb}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/07_cnn_optuna/runs/cnn_optuna_${OBJ}_flex_${RUN_ID}.ipynb"

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo "Iniciando notebook FLEX via papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

papermill "${NB_IN}" "${NB_OUT}" \
    -p OBJECT_TYPE       "${OBJ}" \
    -p SEED              "${SEED}" \
    -p N_TRIALS          "${N_TRIALS}" \
    -p EPOCHS_PER_TRIAL  "${EPOCHS_PER_TRIAL}" \
    -p FINAL_EPOCHS      "${FINAL_EPOCHS}" \
    -p OBJECTIVE_MODE    "${OBJECTIVE_MODE}" \
    -p USE_Z_WEIGHTS     "${USE_Z_WEIGHTS}" \
    -p TARGET_TRIALS_OVERRIDE "${TARGET_TRIALS_OVERRIDE}" \
    --log-output \
    --progress-bar

echo
echo "Notebook executado: ${NB_OUT}"
echo "Resultados em    : results/${OBJ}/cnn_optuna_flex/<RUN_TAG>/"
echo "Modelo em        : models/${OBJ}/cnn_optuna_flex/<RUN_TAG>/"
echo "(RUN_TAG inclui hash da config — confira o output do notebook)"
echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
