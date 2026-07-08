#!/usr/bin/env bash
# Sincroniza resultados do cluster -> maquina local via rsync.
#
# Uso:
#     bash scripts/sync_from_cluster.sh              # tudo (results + models + runs + logs)
#     bash scripts/sync_from_cluster.sh --logs-only  # so logs (leve, pra checar status)
#     bash scripts/sync_from_cluster.sh --results-only  # so results/ (sem modelos pesados)
#     bash scripts/sync_from_cluster.sh --dry        # preview sem transferir
#     bash scripts/sync_from_cluster.sh --delete-during  # remove local arquivos que sumiram remoto
#
# Sempre mostra um dry-run primeiro e pede confirmacao Y/n antes do sync real.

set -euo pipefail

# =====================================================================
# Configuracao do remoto (mesmo do sync_to_cluster.sh)
# =====================================================================
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"
REMOTE_HOST="172.20.76.10"
REMOTE_USER="valerio"
REMOTE_DIR="/home/valerio/Thalita/spec_z_ml/"
REMOTE_BASE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

# =====================================================================
# Parse argumentos
# =====================================================================
DELETE_FLAG=""
LOGS_ONLY=false
RESULTS_ONLY=false
DRY=""

for arg in "$@"; do
  case "$arg" in
    --logs-only)     LOGS_ONLY=true ;;
    --results-only)  RESULTS_ONLY=true ;;
    --delete-during) DELETE_FLAG="--delete-during" ;;
    --dry)           DRY="--dry-run" ;;
    *) echo "Argumento desconhecido: $arg"
       echo "Permitidos: --logs-only, --results-only, --delete-during, --dry"
       exit 1 ;;
  esac
done

if $LOGS_ONLY && $RESULTS_ONLY; then
  echo "ERRO: --logs-only e --results-only sao mutuamente exclusivos."
  exit 1
fi

# =====================================================================
# O que sincronizar
# =====================================================================
# Por padrao sincroniza:
#   - results/      (SQLite do Optuna, JSON, npz, plots/, csv)
#   - models/       (best_model.keras) — pode ser pesado
#   - notebooks/07_cnn_optuna/runs/  (notebooks executados c/ plots embutidos)
#   - logs/         (saidas do SLURM)
#
# --logs-only:    so logs/
# --results-only: results/ + runs/ (deixa models de fora pra economizar transferencia)

declare -a FOLDERS

if $LOGS_ONLY; then
  FOLDERS=("logs/")
elif $RESULTS_ONLY; then
  FOLDERS=("results/" "notebooks/07_cnn_optuna/runs/")
else
  FOLDERS=("results/" "models/" "notebooks/07_cnn_optuna/runs/" "logs/")
fi

COMMON_OPTS=(
  -avh
  --progress
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='.ipynb_checkpoints/'
)

# =====================================================================
# Confere existencia das pastas remotas
# =====================================================================
echo "============================================================"
echo "Conferindo pastas remotas..."
echo "  origem : ${REMOTE_BASE}"
echo "  destino: ${LOCAL_DIR}"
echo "============================================================"

declare -a VALID_FOLDERS
for f in "${FOLDERS[@]}"; do
  if ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d '${REMOTE_DIR}${f}'" 2>/dev/null; then
    VALID_FOLDERS+=("$f")
    echo "  [OK]      ${f}"
  else
    echo "  [SKIP]    ${f}  (nao existe no remoto)"
  fi
done

if [ "${#VALID_FOLDERS[@]}" -eq 0 ]; then
  echo
  echo "Nenhuma pasta remota encontrada. Nada a sincronizar."
  exit 0
fi

# Cria as pastas locais que faltarem
for f in "${VALID_FOLDERS[@]}"; do
  mkdir -p "${LOCAL_DIR}${f}"
done

# =====================================================================
# 1) Dry-run preview
# =====================================================================
echo
echo "============================================================"
echo "[1/2] DRY-RUN — preview do que seria sincronizado"
echo "============================================================"

for f in "${VALID_FOLDERS[@]}"; do
  echo
  echo "--- ${f} ---"
  rsync "${COMMON_OPTS[@]}" --dry-run $DELETE_FLAG \
    "${REMOTE_BASE}${f}" "${LOCAL_DIR}${f}"
done

if [ -n "$DRY" ]; then
  echo
  echo "Modo --dry: encerrando sem transferir."
  exit 0
fi

# =====================================================================
# 2) Confirmacao interativa
# =====================================================================
echo
read -r -p "Proceder com o sync real? [Y/n] " reply
reply="${reply:-Y}"
case "$reply" in
  [Yy]|[Yy][Ee][Ss]) ;;
  [Nn]|[Nn][Oo]) echo "Abortado."; exit 0 ;;
  *) echo "Resposta invalida. Abortado."; exit 1 ;;
esac

# =====================================================================
# 3) Sync real
# =====================================================================
echo
echo "============================================================"
echo "[2/2] SYNC REAL"
echo "============================================================"

for f in "${VALID_FOLDERS[@]}"; do
  echo
  echo "--- Sincronizando ${f} ---"
  rsync "${COMMON_OPTS[@]}" $DELETE_FLAG \
    "${REMOTE_BASE}${f}" "${LOCAL_DIR}${f}"
done

echo
echo "Sync concluido."
