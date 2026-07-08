#!/bin/bash
#SBATCH --job-name=thalita-xgb-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=z3-short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x-slurm_job-%j.out
#SBATCH --error=logs/%x-slurm_job-%j.err

# =====================================================================
# XGBoost + PCA + Optuna no cluster (CPU only, z3-short).
#
# Roda o notebook notebooks/05_xgboost_optuna/xgb_optuna_pca_<obj>.ipynb
# via papermill, usando /local-scratch como working dir pra IO rapido.
#
# Uso (rode da raiz do projeto = SLURM_SUBMIT_DIR):
#
#   # Default (LRG, 50 trials, 300 PCs):
#       sbatch scripts/train_xgb_optuna_pca.sbatch
#
#   # Override de qualquer parametro:
#       sbatch --export=ALL,OBJ=LRG,N_TRIALS=100,N_COMPONENTS=400 \
#           scripts/train_xgb_optuna_pca.sbatch
#
# Notebook executado salvo em:
#     notebooks/05_xgboost_optuna/runs/xgb_optuna_pca_${OBJ}_${RUN_ID}.ipynb
# =====================================================================

set -euo pipefail

# Defaults overridable via --export=ALL,VAR=VALUE
OBJ="${OBJ:-LRG}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-50}"
N_COMPONENTS="${N_COMPONENTS:-300}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio submissao   : ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "CPUs                  : ${SLURM_CPUS_PER_TASK}"
echo "Objeto                : ${OBJ}"
echo "Seed                  : ${SEED}"
echo "N_TRIALS              : ${N_TRIALS}"
echo "N_COMPONENTS          : ${N_COMPONENTS}"
echo "================================================================"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

# Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

# Confere papermill
if ! python -c "import papermill" 2>/dev/null; then
    echo "ERRO: papermill nao instalado. Instale no login node:"
    echo "    conda activate thalita && pip install papermill nbformat"
    exit 1
fi

# =====================================================================
# Local scratch (rapido) — copia codigo + HDF5 do objeto, roda la,
# depois sincroniza models/ e results/ de volta pro home.
# =====================================================================
scratch_dir="/local-scratch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${scratch_dir}"
echo "Scratch dir: ${scratch_dir}"

echo
echo "[1/3] Copiando codigo pro scratch (sem /data/, /results/, /models/, /logs/)..."
# IMPORTANTE: excludes ancorados (/data/) so' batem na raiz; senao tirariam
# tambem src/data/, src/models/, etc. e o import quebra.
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

# Caminhos do notebook
OBJ_LOWER=$(echo "${OBJ}" | tr '[:upper:]' '[:lower:]')
NB_IN="notebooks/05_xgboost_optuna/xgb_optuna_pca_${OBJ_LOWER}.ipynb"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
NB_OUT="notebooks/05_xgboost_optuna/runs/xgb_optuna_pca_${OBJ}_${RUN_ID}.ipynb"
mkdir -p notebooks/05_xgboost_optuna/runs

if [ ! -f "${NB_IN}" ]; then
    echo "ERRO: notebook nao encontrado: ${NB_IN}"
    exit 1
fi

echo
echo "[3/3] Rodando papermill: $(date)"
echo "  Input  : ${NB_IN}"
echo "  Output : ${NB_OUT}"
echo

papermill "${NB_IN}" "${NB_OUT}" \
    -p OBJECT_TYPE   "${OBJ}" \
    -p SEED          "${SEED}" \
    -p N_TRIALS      "${N_TRIALS}" \
    -p N_COMPONENTS  "${N_COMPONENTS}" \
    --log-output --progress-bar

echo
echo "Papermill finalizado: $(date)"

# =====================================================================
# Sincroniza resultados de volta pro home (submit_dir)
# =====================================================================
echo
echo "Movendo resultados pro submit_dir..."
mkdir -p ${SLURM_SUBMIT_DIR}/results
mkdir -p ${SLURM_SUBMIT_DIR}/models
mkdir -p ${SLURM_SUBMIT_DIR}/notebooks/05_xgboost_optuna/runs

# results/ e models/ — ja contem subpasta xgb_optuna_pca/
rsync -av ${scratch_dir}/results/  ${SLURM_SUBMIT_DIR}/results/  || true
rsync -av ${scratch_dir}/models/   ${SLURM_SUBMIT_DIR}/models/   || true
cp ${scratch_dir}/${NB_OUT} ${SLURM_SUBMIT_DIR}/${NB_OUT}

# Limpa scratch
echo "Limpando scratch..."
rm -rf "${scratch_dir}"

echo
echo "================================================================"
echo "Job finalizado: $(date)"
echo "Resultados em:"
echo "  notebook  : ${SLURM_SUBMIT_DIR}/${NB_OUT}"
echo "  metricas  : ${SLURM_SUBMIT_DIR}/results/${OBJ}/xgb_optuna_pca/"
echo "  modelo    : ${SLURM_SUBMIT_DIR}/models/${OBJ}/xgb_optuna_pca/"
echo "================================================================"
