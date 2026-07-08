"""Treina a CNN de DETECCAO DE LINHA [OII] (cabeca tipo QuasarNET) para ELG.

Usa o MESMO split do baseline (make_or_load_split) -> comparavel maca-com-maca
e direto no compare_model_vs_pipeline.py.

Uso:
    python scripts/train_cnn_linedet.py --object ELG
    python scripts/train_cnn_linedet.py --object ELG --epochs 80 --heatmap-weight 0.3

Outputs:
    models/{OBJ}/cnn_linedet/cnn_linedet.keras
    results/{OBJ}/cnn_linedet/predictions.npz   (y_test, y_pred, delta_z, test_idx)
    results/{OBJ}/cnn_linedet/metrics.csv
    results/{OBJ}/cnn_linedet/heatmap_examples.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.style import set_science_style
set_science_style()

from config import paths_for, RESULTS_DIR, MODELS_DIR, SPLITS_DIR, print_info
from src.data import (
    load_spectral_dataset, normalize_spectra, make_or_load_split, apply_split,
)
from src.models.cnn_linedet import OIILineDetectionCNN, OII_REST
from src.evaluation import compute_redshift_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", default="ELG", choices=["ELG"],
                   help="So' ELG: cabeca de [OII] nao se aplica a LRG (sem emissao) nem a QSO (linhas largas).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--heatmap-weight", type=float, default=0.2)
    p.add_argument("--sigma-bins", type=float, default=2.0)
    p.add_argument("--patience-es", type=int, default=20)
    p.add_argument("--patience-lr", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    obj = args.object
    print_info()
    paths = paths_for(obj)
    hdf5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")

    out_dir   = RESULTS_DIR / obj / "cnn_linedet"
    model_dir = MODELS_DIR  / obj / "cnn_linedet"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Dados + grade log-lambda -------------------------------------
    print(f"\n[1/4] Lendo {hdf5_path}")
    X, y, n_wave = load_spectral_dataset(hdf5_path)
    X = normalize_spectra(X)
    with h5py.File(hdf5_path, "r") as f:
        wave = f["wave_global"][:]
    loglam0 = float(np.log10(wave[0]))
    dloglam = float(np.median(np.diff(np.log10(wave))))
    print(f"    X: {X.shape}  loglam0={loglam0:.5f}  dloglam={dloglam:.2e}")

    train_idx, val_idx, test_idx = make_or_load_split(obj, y, splits_dir=SPLITS_DIR)
    X_train, X_val, X_test = apply_split(X, train_idx, val_idx, test_idx)
    y_train, y_val, y_test = apply_split(y, train_idx, val_idx, test_idx)
    print(f"    Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ---- 2. Treino -------------------------------------------------------
    print(f"\n[2/4] CNN de deteccao [OII]  (heatmap_weight={args.heatmap_weight}, sigma={args.sigma_bins})")
    cnn = OIILineDetectionCNN(
        n_wave=n_wave, loglam0=loglam0, dloglam=dloglam,
        learning_rate=args.learning_rate,
        heatmap_weight=args.heatmap_weight, sigma_bins=args.sigma_bins,
    )
    cnn.build()
    cnn.model.summary(print_fn=print)
    cnn.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            epochs=args.epochs, batch_size=args.batch_size,
            patience_es=args.patience_es, patience_lr=args.patience_lr, verbose=2)

    # ---- 3. Avaliacao ----------------------------------------------------
    y_pred = cnn.predict(X_test)
    results = compute_redshift_metrics(y_test, y_pred)
    results["y_pred"] = y_pred; results["y_test"] = y_test
    print(f"\n[3/4] TEST (n={len(y_test):,}, train_time={cnn.train_time:.0f}s)")
    print(f"    sigma_NMAD : {results['nmad']:.4e}   (~{results['nmad']*2.998e5:.0f} km/s)")
    print(f"    bias       : {results['bias']:+.4e}")
    print(f"    outliers (|Δz/(1+z)|>0.05) : {results['outliers_0.05_pct']:.2f}%")

    # ---- 4. Salvar -------------------------------------------------------
    cnn.save(model_dir / "cnn_linedet.keras")
    np.savez(out_dir / "predictions.npz",
             y_test=y_test, y_pred=y_pred, delta_z=results["delta_z"], test_idx=test_idx)
    row = {"model": "cnn_linedet", "object": obj, "n_test": len(y_test),
           "train_time_s": cnn.train_time, "epochs_run": len(cnn.history.history["loss"]),
           "heatmap_weight": args.heatmap_weight, "sigma_bins": args.sigma_bins,
           **{k: results[k] for k in ["rmse","mae","r2","bias","nmad",
                                       "outliers_pct","outliers_0.05_pct","outliers_0.15_pct"]}}
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(f"\n[4/4] Salvo em {out_dir}")

    try:
        _plot_heatmaps(cnn, X_test, y_test, wave, out_dir, obj)
        print(f"  {out_dir / 'heatmap_examples.png'}")
    except Exception as e:
        print(f"WARNING: plot do heatmap falhou ({type(e).__name__}: {e}) — modelo ja salvo.")


def _plot_heatmaps(cnn, X_test, y_test, wave, out_dir, obj, n=4):
    """Mostra o heatmap aprendido vs a posicao real do [OII] (interpretabilidade)."""
    z_pred, hm = cnn.predict(X_test[:n], return_heatmap=True)
    D = cnn.DOWNSAMPLE
    fm_to_wave = 10 ** (cnn.loglam0 + np.arange(cnn._L) * D * cnn.dloglam)

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n))
    for k in range(n):
        ax = axes[k]
        ax.plot(fm_to_wave, hm[k], color="steelblue", lw=1.3, label="heatmap (rede)")
        oii_true = OII_REST * (1 + y_test[k])
        oii_pred = OII_REST * (1 + z_pred[k])
        ax.axvline(oii_true, color="green", ls="-", lw=1.5, label=f"[OII] real (z={y_test[k]:.3f})")
        ax.axvline(oii_pred, color="red", ls="--", lw=1.5, label=f"[OII] previsto (z={z_pred[k]:.3f})")
        ax.set_ylabel("prob")
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Comprimento de onda observado (A)")
    fig.suptitle(f"Heatmap de deteccao do [OII] — {obj}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
