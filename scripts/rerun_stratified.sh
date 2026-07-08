#!/bin/bash
# =====================================================================
# Arranque do processo NOVO no split canonico ESTRATIFICADO por z.
# Rode da raiz do projeto, no login node do cluster:
#     bash scripts/rerun_stratified.sh
#
# Snapshot do split antigo ja' arquivado em:
#     archive/pre_stratified_split_20260619/
# =====================================================================
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thalita

# Garante que le' o dataset CORRETO (PROJECT_ROOT/data), nunca o legado quebrado.
export SPECZML_DATA="$(pwd)/data"
echo "SPECZML_DATA = ${SPECZML_DATA}"

OBJS="LRG QSO ELG"

echo "================================================================"
echo "1) Gerando/validando splits/<OBJ>_split.npz (estratificado por z)"
echo "================================================================"
for OBJ in $OBJS; do
    python scripts/analysis/check_split.py --object "$OBJ"
done

echo
echo "================================================================"
echo "2) Re-rodando BASELINES (sem Optuna -> seguro durante a ablation)"
echo "   O 1o XGBoost por objeto ja' reusa o .npz gerado acima."
echo "================================================================"
for OBJ in $OBJS; do
    sbatch --export=ALL,OBJ="$OBJ" scripts/train/train_xgb_baseline.sbatch
    sbatch --export=ALL,OBJ="$OBJ" scripts/train/train_cnn_baseline.sbatch
done

# CNN linedet (so' ELG)
sbatch --export=ALL,OBJ=ELG scripts/train/train_cnn_linedet.sbatch

echo
echo "================================================================"
echo "3) PENDENTE — modelos com Optuna (segurar ate' a ablation decidir)"
echo "================================================================"
cat <<'EOF'
  Antes de re-rodar os modelos Optuna no split novo, LIMPE os studies antigos
  (senao load_if_exists resume trials do split antigo):

    # CNN flex: o study novo so' nao colide se use_batchnorm ja' foi aplicado
    # (_flex_changes.py muda o config_hash -> study_name novo). Se NAO aplicar,
    # apague o .db antes:
    #   rm results/<OBJ>/cnn_optuna_flex/optuna_study.db

    # XGBoost optuna PCA / tune_xgb: apague o .db pra comecar fresh:
    #   rm models/<OBJ>/xgb_optuna_pca/.../optuna_pca_*.db
    #   rm models/<OBJ>/xgboost/optuna_xgb_*.db

  Depois:
    for OBJ in LRG QSO ELG; do
      sbatch --export=ALL,OBJ=$OBJ scripts/train/train_cnn_optuna_flex.sbatch
      sbatch --export=ALL,OBJ=$OBJ scripts/train/train_xgb_optuna_pca.sbatch
    done

  (Lembrete: aplicar _flex_changes.py + bumpar FINAL_EPOCHS/N_TRIALS no sbatch
   ANTES de re-rodar o flex.)
EOF

echo
echo "Acompanhe: squeue -u valerio"
