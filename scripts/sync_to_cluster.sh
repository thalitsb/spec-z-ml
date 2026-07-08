#!/usr/bin/env bash
# Sincroniza o projeto local com o cluster via rsync (incremental).
#
# Uso:
#     bash scripts/sync_to_cluster.sh           # codigo apenas (~700 KB)
#     bash scripts/sync_to_cluster.sh --data    # codigo + padded.h5 (~5 GB)
#     bash scripts/sync_to_cluster.sh --all     # codigo + processed/ inteiro (~24 GB)
#     bash scripts/sync_to_cluster.sh --dry     # mostra o que faria, sem transferir
#
# Pode combinar: --data --dry, etc.

set -euo pipefail

# =====================================================================
# Configuracao
# =====================================================================
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"
REMOTE_HOST="172.20.76.10"
REMOTE_USER="valerio"
REMOTE_DIR="/home/valerio/Thalita/spec_z_ml/"

# =====================================================================
# Parse argumentos
# =====================================================================
INCLUDE_DATA="none"   # none | padded | all
DRY=""
for arg in "$@"; do
  case "$arg" in
    --data) INCLUDE_DATA="padded" ;;
    --all)  INCLUDE_DATA="all"    ;;
    --dry)  DRY="--dry-run"       ;;
    *) echo "Argumento desconhecido: $arg"; exit 2 ;;
  esac
done

# =====================================================================
# Comandos rsync
# =====================================================================

# Sempre exclui (codigo e dado vao em duas chamadas separadas).
COMMON_EXCLUDES=(
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo'
  --exclude='.ipynb_checkpoints/' --exclude='.git/'
  --exclude='.venv/' --exclude='venv/'
  --exclude='.DS_Store'
)

echo "============================================================"
echo "[1/2] Subindo CODIGO (sem data/, models/, results/, logs/)"
echo "  ${LOCAL_DIR}  ->  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "============================================================"

rsync -avz --progress ${DRY} \
  --exclude='/data/' \
  --exclude='/results/' --exclude='/logs/' \
  --exclude='/archive/' --exclude='_backup_*/' \
  --exclude='models/**/*.keras' --exclude='models/**/*.pkl' \
  --exclude='models/**/*.h5'    --exclude='models/**/*.joblib' \
  "${COMMON_EXCLUDES[@]}" \
  "${LOCAL_DIR}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

if [[ "$INCLUDE_DATA" == "none" ]]; then
  echo
  echo "Concluido (so codigo). Para incluir os dados padded:"
  echo "    bash scripts/sync_to_cluster.sh --data"
  exit 0
fi

echo
echo "============================================================"
if [[ "$INCLUDE_DATA" == "padded" ]]; then
  echo "[2/2] Subindo SO os padded.h5 (~5 GB total, ELG + LRG)"
elif [[ "$INCLUDE_DATA" == "all" ]]; then
  echo "[2/2] Subindo processed/ inteiro (sem cache/) (~24 GB)"
fi
echo "============================================================"

DATA_EXCLUDES=(
  --exclude='processed/*/cache/'   # caches de processamento (70 GB, nao usados)
)

if [[ "$INCLUDE_DATA" == "padded" ]]; then
  # Sobe so os 2 padded.h5 (e estrutura minima de pastas).
  DATA_EXCLUDES+=(
    --include='processed/'
    --include='processed/ELG/' --include='processed/LRG/'
    --include='processed/ELG/ELGspectra_padded.h5'
    --include='processed/LRG/LRGspectra_padded.h5'
    --exclude='*'
  )
fi

rsync -avz --progress --partial ${DRY} \
  "${DATA_EXCLUDES[@]}" \
  "${COMMON_EXCLUDES[@]}" \
  "${LOCAL_DIR}data/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}data/"

echo
echo "Sync concluido."
