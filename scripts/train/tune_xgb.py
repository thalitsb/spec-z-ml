"""Roda Optuna em XGBoost para um tipo de objeto.

Uso:
    python scripts/tune_xgb.py --object ELG --n-trials 100

Recomendado rodar no cluster (longa duracao). O estudo e salvo em SQLite,
pode ser pausado e continuado.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import paths_for, MODELS_DIR, SPLITS_DIR
from src.data import (
    load_spectral_dataset, normalize_spectra, make_or_load_split, make_split, apply_split,
)
from src.models import tune_xgboost_with_optuna


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, choices=["ELG", "LRG", "QSO"])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Limita N amostras (para debug)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    obj = args.object
    paths = paths_for(obj)
    hdf5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")

    print(f"[{obj}] Carregando: {hdf5_path}")
    X, y, _ = load_spectral_dataset(hdf5_path, n_samples=args.n_samples, seed=args.seed)
    X = normalize_spectra(X)

    # Split canonico estratificado por z (mesmo de todos os modelos).
    # Com --n-samples (debug) o N nao bate com o .npz salvo -> gera em memoria,
    # sem persistir, pra nao corromper o split canonico.
    if args.n_samples is not None:
        tr, va, te = make_split(y, random_state=args.seed)
    else:
        tr, va, te = make_or_load_split(obj, y, SPLITS_DIR)
    X_train, X_val, _ = apply_split(X, tr, va, te)
    y_train, y_val, _ = apply_split(y, tr, va, te)

    study_dir = MODELS_DIR / obj / "xgboost"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / f'optuna_xgb_{obj.lower()}.db'}"

    print(f"[{obj}] Iniciando Optuna ({args.n_trials} trials)")
    study = tune_xgboost_with_optuna(
        X_train, y_train, X_val, y_val,
        n_trials=args.n_trials,
        study_name=f"xgboost_{obj.lower()}",
        storage=storage,
        seed=args.seed,
    )

    print(f"\n[{obj}] Melhor RMSE: {study.best_value:.4f}")
    print(f"[{obj}] Melhores params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n[{obj}] Estudo salvo: {storage}")


if __name__ == "__main__":
    main()
