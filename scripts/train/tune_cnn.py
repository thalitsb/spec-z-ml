"""Roda Optuna em CNN para um tipo de objeto.

Uso:
    python scripts/tune_cnn.py --object ELG --n-trials 30

CNN trials sao caros (cada um treina uma rede), use n-trials moderado
e prefira rodar no cluster com GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import paths_for, MODELS_DIR
from src.data import load_spectral_dataset, normalize_spectra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, choices=["ELG", "LRG", "QSO"])
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    obj = args.object
    paths = paths_for(obj)
    hdf5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")

    print(f"[{obj}] Carregando: {hdf5_path}")
    X, y, n_wave = load_spectral_dataset(hdf5_path, n_samples=args.n_samples, seed=args.seed)
    X = normalize_spectra(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=args.seed)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=args.seed)

    # Importacoes pesadas dentro do main para nao falhar se TF nao estiver disponivel
    from src.models.cnn import PaddedSpectralCNN

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        # TODO: adicionar mais hiperparametros (kernel sizes, dropout, etc)

        cnn = PaddedSpectralCNN(n_wave=n_wave, learning_rate=lr)
        cnn.build()
        cnn.fit(X_train, y_train, X_val, y_val,
                epochs=20, batch_size=bs, verbose=0)
        results = cnn.evaluate(X_val, y_val)
        return results["rmse"]

    study_dir = MODELS_DIR / obj / "cnn"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / f'optuna_cnn_{obj.lower()}.db'}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"cnn_{obj.lower()}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\n[{obj}] Melhor RMSE: {study.best_value:.4f}")
    print(f"[{obj}] Melhores params: {study.best_params}")


if __name__ == "__main__":
    main()
