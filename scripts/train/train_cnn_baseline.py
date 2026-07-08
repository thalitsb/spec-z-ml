"""Treina a CNN baseline para um objeto. Versao CLI do notebook
notebooks/06_cnn_baseline/cnn_{ELG,LRG}.ipynb.

Uso:
    python scripts/train_cnn_baseline.py --object LRG
    python scripts/train_cnn_baseline.py --object ELG --epochs 80 --batch-size 128

Outputs (mesmos do notebook):
    models/{OBJ}/cnn_baseline/cnn_baseline.keras
    results/{OBJ}/cnn_baseline/predictions.npz
    results/{OBJ}/cnn_baseline/metrics.csv
    results/{OBJ}/cnn_baseline/training_and_scatter.png
    results/{OBJ}/cnn_baseline/residuals_by_z.png
"""
import argparse
import sys
from pathlib import Path

# Backend headless: necessario para nodes sem display (caso do cluster).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.style import set_science_style
set_science_style()

from config import paths_for, RESULTS_DIR, MODELS_DIR, SPLITS_DIR, print_info
from src.data import (
    load_spectral_dataset, normalize_spectra, make_or_load_split, apply_split,
)
from src.models import PaddedSpectralCNN
from src.evaluation import compute_redshift_metrics, metrics_by_redshift_bin


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", required=True, choices=["ELG", "LRG", "QSO"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--patience-es", type=int, default=20)
    p.add_argument("--patience-lr", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    obj = args.object

    print_info()
    paths = paths_for(obj)
    hdf5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")

    out_dir   = RESULTS_DIR / obj / "cnn_baseline"
    model_dir = MODELS_DIR  / obj / "cnn_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_science_style()

    # ---- 1. Dados ---------------------------------------------------------
    print(f"\n[1/4] Lendo {hdf5_path}")
    X, y, n_wave = load_spectral_dataset(hdf5_path)
    X = normalize_spectra(X)
    print(f"    X: {X.shape}  y: {y.shape}  n_wave: {n_wave}")

    train_idx, val_idx, test_idx = make_or_load_split(obj, y, splits_dir=SPLITS_DIR)
    X_train, X_val, X_test = apply_split(X, train_idx, val_idx, test_idx)
    y_train, y_val, y_test = apply_split(y, train_idx, val_idx, test_idx)
    print(f"    Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ---- 2. GPU + Treino --------------------------------------------------
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"    GPUs visiveis: {len(gpus)}  ({[g.name for g in gpus]})")

    print(f"\n[2/4] Construindo CNN (n_wave={n_wave})")
    cnn = PaddedSpectralCNN(n_wave=n_wave, learning_rate=args.learning_rate)
    cnn.build()
    cnn.model.summary(print_fn=print)

    print(f"\n[3/4] Treinando: epochs={args.epochs}, batch_size={args.batch_size}")
    cnn.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs, batch_size=args.batch_size,
        patience_es=args.patience_es, patience_lr=args.patience_lr,
        verbose=2,   # uma linha por epoca: ideal para logs de SLURM
    )

    # ---- 3. Avaliacao + plots --------------------------------------------
    y_pred = cnn.predict(X_test)
    results = compute_redshift_metrics(y_test, y_pred)
    results["y_pred"] = y_pred
    results["y_test"] = y_test
    print(f"\n[4/4] Resultados em TEST  (n={len(y_test):,}, train_time={cnn.train_time:.0f}s)")
    print(f"    RMSE       : {results['rmse']:.4f}")
    print(f"    MAE        : {results['mae']:.4f}")
    print(f"    R2         : {results['r2']:.4f}")
    print(f"    bias       : {results['bias']:+.4e}")
    print(f"    sigma_NMAD : {results['nmad']:.4e}")
    print(f"    outliers (|Δz/(1+z)|>0.05) : {results['outliers_0.05_pct']:.2f}%")
    print(f"    outliers (|Δz/(1+z)|>0.15) : {results['outliers_0.15_pct']:.2f}%")

    # ---- 4. Salvar PRIMEIRO (antes dos plots, pra nao perder o treino) ---
    cnn.save(model_dir / "cnn_baseline.keras")
    np.savez(
        out_dir / "predictions.npz",
        y_test=results["y_test"], y_pred=results["y_pred"],
        delta_z=results["delta_z"], test_idx=test_idx,
    )
    scalar_keys = ["rmse", "mae", "r2", "bias", "nmad",
                   "outliers_pct", "outliers_0.05_pct", "outliers_0.15_pct"]
    row = {"model": "cnn_baseline", "object": obj,
           "n_test": len(y_test), "train_time_s": cnn.train_time,
           "epochs_run": len(cnn.history.history["loss"]),
           "batch_size": args.batch_size, "learning_rate": args.learning_rate,
           **{k: results[k] for k in scalar_keys}}
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nSalvo:")
    print(f"  {model_dir / 'cnn_baseline.keras'}")
    print(f"  {out_dir / 'predictions.npz'}")
    print(f"  {out_dir / 'metrics.csv'}")

    # Plots por ultimo — se quebrar, modelo + predictions ja foram salvos
    try:
        _plot_training_and_scatter(cnn, results, y_test, obj, out_dir)
        _plot_residuals_by_z(results, y_test, obj, out_dir)
        print(f"  {out_dir / 'training_and_scatter.png'}")
        print(f"  {out_dir / 'residuals_by_z.png'}")
    except Exception as e:
        print(f"\nWARNING: plotagem falhou ({type(e).__name__}: {e}). "
              f"Model + predictions ja salvos — re-plotar a partir de predictions.npz.")


def _plot_training_and_scatter(cnn, results, y_test, obj, out_dir):
    history = cnn.history.history
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"CNN baseline — {obj}", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["loss"],     label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val",   lw=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["rmse"],     label="Train", lw=2)
    axes[1].plot(epochs, history["val_rmse"], label="Val",   lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE"); axes[1].legend()

    y_pred = results["y_pred"]
    sc = axes[2].scatter(y_test, y_pred, c=y_test, cmap="viridis", alpha=0.5, s=8)
    axes[2].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--", lw=1.5, label="Ideal")
    axes[2].set_xlabel("z_true"); axes[2].set_ylabel("z_pred")
    axes[2].set_title(f"R²={results['r2']:.4f}  σ_NMAD={results['nmad']:.4f}")
    axes[2].legend()
    plt.colorbar(sc, ax=axes[2], label="z_true")

    plt.tight_layout()
    plt.savefig(out_dir / "training_and_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_by_z(results, y_test, obj, out_dir):
    delta_z = results["delta_z"]
    bins = metrics_by_redshift_bin(y_test, results["y_pred"], n_bins=10)["bins"]
    if not bins:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Residuos — {obj}", fontsize=13, fontweight="bold")

    axes[0].hist(delta_z, bins=80, color="steelblue", alpha=0.8)
    for thr, c in [(0.05, "orange"), (0.15, "red")]:
        axes[0].axvline( thr, ls="--", c=c, label=f"|Δz/(1+z)|={thr}")
        axes[0].axvline(-thr, ls="--", c=c)
    axes[0].axvline(results["bias"], color="k", lw=1.5,
                    label=f"bias={results['bias']:+.2e}")
    axes[0].set_xlabel("Δz / (1+z)")
    axes[0].set_title("Distribuicao dos residuos")
    axes[0].legend()

    zc = [b["z_center"] for b in bins]
    nmad = [b["nmad"] for b in bins]
    width = (zc[1] - zc[0]) * 0.85 if len(zc) > 1 else 0.05
    axes[1].bar(zc, nmad, width=width, color="steelblue", alpha=0.8)
    axes[1].axhline(results["nmad"], ls="--", color="k",
                    label=f"global={results['nmad']:.2e}")
    axes[1].set_xlabel("z_true (centro do bin)"); axes[1].set_ylabel("σ_NMAD")
    axes[1].set_title("σ_NMAD por faixa de z"); axes[1].legend()

    out = [b["outliers_pct"] for b in bins]
    axes[2].bar(zc, out, width=width, color="salmon", alpha=0.8)
    axes[2].axhline(results["outliers_pct"], ls="--", color="k",
                    label=f"global={results['outliers_pct']:.2f}%")
    axes[2].set_xlabel("z_true (centro do bin)"); axes[2].set_ylabel("Outliers (%)")
    axes[2].set_title("% |Δz/(1+z)| > 0.15 por faixa de z"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "residuals_by_z.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
