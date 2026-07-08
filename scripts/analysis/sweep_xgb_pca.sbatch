#!/bin/bash
#SBATCH --job-name=thalita-xgb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=z3-short
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/%x-slurm_job-%j.out
#SBATCH --error=logs/%x-slurm_job-%j.err

# =====================================================================
# Sweep de N_COMPONENTS pro XGBoost+PCA (CPU only, z3-short).
#
# Roda scripts/sweep_xgb_pca.py, que fita o PCA uma vez com o N maximo
# e treina um XGBoost (HPs manuais) pra cada N do sweep, gerando a
# curva NMAD vs N_COMPONENTS.
#
# Uso (rode da raiz do projeto = SLURM_SUBMIT_DIR):
#
#   # Default (LRG, components 300 500 800 1200 1912):
#       sbatch scripts/sweep_xgb_pca.sbatch
#
#   # Override:
#       sbatch --export=ALL,OBJ=LRG,COMPONENTS="300 600 900 1912" \
#           scripts/sweep_xgb_pca.sbatch
#
# Resultados em:
#     results/${OBJ}/xgb_pca_sweep/sweep_ncomponents_${OBJ}_s42.{csv,json,png}
# =====================================================================

set -euo pipefail

OBJ="${OBJ:-LRG}"
SEED="${SEED:-42}"
COMPONENTS="${COMPONENTS:-300 500 800 1200 1912}"

echo "================================================================"
echo "Job iniciado          : $(date)"
echo "Job ID                : ${SLURM_JOB_ID}"
echo "Job name              : ${SLURM_JOB_NAME}"
echo "Diretorio submissao   : ${SLURM_SUBMIT_DIR}"
echo "Node                  : $(hostname)"
echo "CPUs                  : ${SLURM_CPUS_PER_TASK}"
echo "Objeto                : ${OBJ}"
echo "Seed                  : ${SEED}"
echo "COMPONENTS            : ${COMPONENTS}"
echo "================================================================"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

# Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

# =====================================================================
# Local scratch (rapido) — copia codigo + HDF5 do objeto, roda la,
# depois sincroniza results/ de volta pro home.
# =====================================================================
scratch_dir="/local-scratch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${scratch_dir}"
echo "Scratch dir: ${scratch_dir}"

echo
echo "[1/3] Copiando codigo pro scratch (sem /data/, /results/, /models/, /logs/)..."
# Excludes ancorados (/data/) so' batem na raiz; senao tirariam src/data/ etc.
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

echo
echo "[3/3] Rodando sweep: $(date)"
python scripts/analysis/sweep_xgb_pca.py --object "${OBJ}" --seed "${SEED}" --components ${COMPONENTS}

echo
echo "Sweep finalizado: $(date)"

# =====================================================================
# Sincroniza resultados de volta pro home (submit_dir)
# =====================================================================
echo
echo "Movendo resultados pro submit_dir..."
mkdir -p ${SLURM_SUBMIT_DIR}/results
rsync -av ${scratch_dir}/results/ ${SLURM_SUBMIT_DIR}/results/ || true

echo "Limpando scratch..."
rm -rf "${scratch_dir}"

echo
echo "================================================================"
echo "Job finalizado: $(date)"
echo "Resultados em:"
echo "  ${SLURM_SUBMIT_DIR}/results/${OBJ}/xgb_pca_sweep/"
echo "================================================================"
