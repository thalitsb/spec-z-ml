#!/bin/bash
#SBATCH --job-name=thalita-xgb-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=z3-short
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/%x-slurm_job-%j.out
#SBATCH --error=logs/%x-slurm_job-%j.err

# =====================================================================
# XGBoost baseline CRU (features brutas, sem PCA) — CPU only, z3-short.
#
# Roda notebooks/04_xgboost_baseline/xgb_${OBJ_LOWER}.ipynb via papermill.
# Espelha a estrutura do sweep_xgb_pca.sbatch (scratch + SPECZML_DATA).
#
# Uso (rode da raiz do projeto = SLURM_SUBMIT_DIR):
#       sbatch --export=ALL,OBJ=QSO scripts/train_xgb_baseline.sbatch
#
# Resultados em:
#     results/${OBJ}/xgboost_baseline/{metrics.csv,predictions.npz}
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
SEED="${SEED:-42}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio submissao   : ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "CPUs                  : ${SLURM_CPUS_PER_TASK}"
echo "Objeto                : ${OBJ}"
echo "Seed                  : ${SEED}"
echo "================================================================"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

# Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

if ! python -c "import papermill" 2>/dev/null; then
    echo "ERRO: papermill nao instalado. conda activate thalita && pip install papermill nbformat"
    exit 1
fi

# =====================================================================
# Local scratch (rapido) — copia codigo + HDF5 do objeto, roda la,
# depois sincroniza results/ de volta pro home.
# =====================================================================
scratch_dir="/local-scratch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${scratch_dir}"
echo "Scratch dir: ${scratch_dir}"

echo
echo "[1/3] Copiando codigo pro scratch (sem /data/, /results/, /models/, /logs/)..."
rsync -a \
    --exclude='/data/' --exclude='/results/' --exclude='/models/' \
    --exclude='/logs/' --exclude='/.git/' \
    --exclude='__pycache__/' --exclude='.ipynb_checkpoints/' --exclude='*.pyc' \
    ${SLURM_SUBMIT_DIR}/ ${scratch_dir}/

echo
echo "[2/3] Copiando HDF5 ${OBJ}spectra_padded.h5 pro scratch..."
mkdir -p "${scratch_dir}/data/processed/${OBJ}"
cp ${SLURM_SUBMIT_DIR}/data/processed/${OBJ}/${OBJ}spectra_padded.h5 \
   ${scratch_dir}/data/processed/${OBJ}/
ls -lh ${scratch_dir}/data/processed/${OBJ}/

cd ${scratch_dir}
export SPECZML_DATA="${scratch_dir}/data"
echo "SPECZML_DATA=${SPECZML_DATA}"

OBJ_LOWER=$(echo "${OBJ}" | tr '[:upper:]' '[:lower:]')
NB_IN="notebooks/04_xgboost_baseline/xgb_${OBJ_LOWER}.ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/04_xgboost_baseline/runs/xgb_${OBJ}_baseline_${RUN_ID}.ipynb"
mkdir -p notebooks/04_xgboost_baseline/runs

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo
echo "[3/3] Rodando baseline via papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
# -k python3: forca o kernel que existe no cluster (os notebooks podem ter sido
# salvos com kernel 'spec_z_ml' do ambiente local, que nao existe aqui).
papermill "${NB_IN}" "${NB_OUT}" -k python3

echo
echo "Baseline finalizado: $(date)"

# =====================================================================
# Sincroniza resultados (e notebook executado) de volta pro home
# =====================================================================
echo
echo "Movendo resultados pro submit_dir..."
mkdir -p ${SLURM_SUBMIT_DIR}/results ${SLURM_SUBMIT_DIR}/notebooks/04_xgboost_baseline/runs
rsync -av ${scratch_dir}/results/ ${SLURM_SUBMIT_DIR}/results/ || true
rsync -av ${scratch_dir}/notebooks/04_xgboost_baseline/runs/ \
          ${SLURM_SUBMIT_DIR}/notebooks/04_xgboost_baseline/runs/ || true

echo "Limpando scratch..."
rm -rf "${scratch_dir}"

echo
echo "================================================================"
echo "Job finalizado: $(date)"
echo "Resultados em:"
echo "  ${SLURM_SUBMIT_DIR}/results/${OBJ}/xgboost_baseline/"
echo "================================================================"
