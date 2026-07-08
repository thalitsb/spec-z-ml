#!/bin/bash
#SBATCH --job-name=thalita-cnn-base-strat
#SBATCH --partition=gpu-nv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =====================================================================
# Diagnostico: CNN baseline (arquitetura PaddedSpectralCNN) treinada no
# MESMO split estratificado do flex, pra testar se o NMAD do baseline
# original (0.0038) esta otimista por causa do split aleatorio.
#
# Roda notebooks/06_cnn_baseline/cnn_lrg_stratified.ipynb via papermill.
#
# Uso (da raiz do projeto = SLURM_SUBMIT_DIR):
#     sbatch scripts/train_cnn_baseline_stratified.sbatch
#
#     # Override (ex: outro flex split / objeto):
#     sbatch --export=ALL,OBJ=LRG,FLEX_SPLIT_REL="cnn_optuna_flex/<run>/split_idx.npz" \
#         scripts/train_cnn_baseline_stratified.sbatch
#
# Notebook executado salvo em:
#     notebooks/06_cnn_baseline/runs/cnn_${OBJ}_stratified_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FLEX_SPLIT_REL="${FLEX_SPLIT_REL:-cnn_optuna_flex/flex-LRG-t50_e40_s42-mae_nozw-4e97b2e3/split_idx.npz}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio submissao   : ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "Objeto                : ${OBJ}"
echo "Seed                  : ${SEED}"
echo "Epochs                : ${EPOCHS}"
echo "Batch size            : ${BATCH_SIZE}"
echo "Flex split rel        : ${FLEX_SPLIT_REL}"
echo "================================================================"

mkdir -p ${SLURM_SUBMIT_DIR}/logs
mkdir -p ${SLURM_SUBMIT_DIR}/notebooks/06_cnn_baseline/runs

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
NB_IN="notebooks/06_cnn_baseline/cnn_${OBJ_LOWER}_stratified.ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/06_cnn_baseline/runs/cnn_${OBJ}_stratified_${RUN_ID}.ipynb"

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

# Confere que o split do flex existe (senao o notebook falha no assert)
if [ ! -f "results/${OBJ}/${FLEX_SPLIT_REL}" ]; then
    echo "ERRO: split do flex nao encontrado: results/${OBJ}/${FLEX_SPLIT_REL}"
    echo "      Rode o flex primeiro ou ajuste FLEX_SPLIT_REL."
    exit 1
fi

echo "Iniciando papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

papermill "${NB_IN}" "${NB_OUT}" \
    -p OBJECT_TYPE     "${OBJ}" \
    -p SEED            "${SEED}" \
    -p EPOCHS          "${EPOCHS}" \
    -p BATCH_SIZE      "${BATCH_SIZE}" \
    -p FLEX_SPLIT_REL  "${FLEX_SPLIT_REL}" \
    --log-output \
    --progress-bar

echo
echo "Notebook executado: ${NB_OUT}"
echo "Resultados em    : results/${OBJ}/cnn_baseline_stratified/"
echo "================================================================"
echo "Job finalizado: $(date)"
echo "================================================================"
