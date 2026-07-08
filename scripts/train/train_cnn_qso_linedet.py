"""Treina a CNN de DETECCAO MULTI-LINHA (cabeca tipo QuasarNET) para QSO.

3 linhas: MgII 2799 (ancora), CIII] 1909, CIV 1549. z final = media das linhas
ponderada pela confianca. MESMO split canonico dos outros modelos (comparavel).

Uso:
    python scripts/train/train_cnn_qso_linedet.py --object QSO
    python scripts/train/train_cnn_qso_linedet.py --object QSO --epochs 80 --heatmap-weight 0.3

Outputs:
    models/QSO/cnn_linedet/cnn_qso_linedet.keras
    results/QSO/cnn_linedet/predictions.npz   (y_test, y_pred, delta_z, test_idx, conf_<linha>)
    results/QSO/cnn_linedet/metrics.csv
    results/QSO/cnn_linedet/heatmap_examples.png
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
from src.models.cnn_qso_linedet import MultiLineDetectionCNN, QSO_LINES
from src.evaluation import compute_redshift_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", default="QSO", choices=["QSO"],
                   help="So' QSO: cabeca multi-linha larga (MgII/CIII]/CIV).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--heatmap-weight", type=float, default=0.2)
    p.add_argument("--conf-weight", type=float, default=0.2)
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
    wave_min, wave_max = float(wave[0]), float(wave[-1])
    print(f"    X: {X.shape}  janela=[{wave_min:.0f},{wave_max:.0f}]A  loglam0={loglam0:.5f}  dloglam={dloglam:.2e}")

    train_idx, val_idx, test_idx = make_or_load_split(obj, y, splits_dir=SPLITS_DIR)
    X_train, X_val, X_test = apply_split(X, train_idx, val_idx, test_idx)
    y_train, y_val, y_test = apply_split(y, train_idx, val_idx, test_idx)
    print(f"    Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    # cobertura das linhas no test (sanity)
    for name, rest in QSO_LINES.items():
        lam = rest * (1 + y_test)
        frac = np.mean((lam >= wave_min) & (lam <= wave_max)) * 100
        print(f"    {name:5s} ({rest:.0f}A) visivel em {frac:.0f}% do test")

    # ---- 2. Treino -------------------------------------------------------
    print(f"\n[2/4] CNN multi-linha QSO  (heatmap_w={args.heatmap_weight}, conf_w={args.conf_weight}, sigma={args.sigma_bins})")
    cnn = MultiLineDetectionCNN(
        n_wave=n_wave, loglam0=loglam0, dloglam=dloglam,
        wave_min=wave_min, wave_max=wave_max,
        learning_rate=args.learning_rate, heatmap_weight=args.heatmap_weight,
        conf_weight=args.conf_weight, sigma_bins=args.sigma_bins,
    )
    cnn.build()
    cnn.model.summary(print_fn=print)
    cnn.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            epochs=args.epochs, batch_size=args.batch_size,
            patience_es=args.patience_es, patience_lr=args.patience_lr, verbose=2)

    # ---- 3. Avaliacao ----------------------------------------------------
    y_pred, heatmaps, confs = cnn.predict(X_test, return_extras=True)
    results = compute_redshift_metrics(y_test, y_pred)
    results["y_pred"] = y_pred; results["y_test"] = y_test
    print(f"\n[3/4] TEST (n={len(y_test):,}, train_time={cnn.train_time:.0f}s)")
    print(f"    sigma_NMAD : {results['nmad']:.4e}   (~{results['nmad']*2.998e5:.0f} km/s)")
    print(f"    bias       : {results['bias']:+.4e}")
    print(f"    outliers (|Δz/(1+z)|>0.05) : {results['outliers_0.05_pct']:.2f}%")
    print(f"    confianca media por linha  : " +
          "  ".join(f"{n}={confs[n].mean():.2f}" for n in QSO_LINES))

    # ---- 4. Salvar -------------------------------------------------------
    cnn.save(model_dir / "cnn_qso_linedet.keras")
    np.savez(out_dir / "predictions.npz",
             y_test=y_test, y_pred=y_pred, delta_z=results["delta_z"], test_idx=test_idx,
             **{f"conf_{n}": confs[n] for n in QSO_LINES})
    row = {"model": "cnn_qso_linedet", "object": obj, "n_test": len(y_test),
           "train_time_s": cnn.train_time, "epochs_run": len(cnn.history.history["loss"]),
           "heatmap_weight": args.heatmap_weight, "conf_weight": args.conf_weight,
           "sigma_bins": args.sigma_bins,
           **{k: results[k] for k in ["rmse","mae","r2","bias","nmad",
                                       "outliers_pct","outliers_0.05_pct","outliers_0.15_pct"]}}
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(f"\n[4/4] Salvo em {out_dir}")

    try:
        _plot_heatmaps(cnn, X_test, y_test, out_dir, obj)
        print(f"  {out_dir / 'heatmap_examples.png'}")
    except Exception as e:
        print(f"WARNING: plot do heatmap falhou ({type(e).__name__}: {e}) — modelo ja salvo.")


def _plot_heatmaps(cnn, X_test, y_test, out_dir, obj, n=4):
    """Heatmaps das 3 linhas vs posicoes reais + confianca (interpretabilidade)."""
    z_pred, hm, cf = cnn.predict(X_test[:n], return_extras=True)
    D = cnn.DOWNSAMPLE
    fm_to_wave = 10 ** (cnn.loglam0 + np.arange(cnn._L) * D * cnn.dloglam)
    colors = {"MgII": "steelblue", "CIII": "darkorange", "CIV": "purple"}

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n))
    for k in range(n):
        ax = axes[k]
        for name, rest in QSO_LINES.items():
            ax.plot(fm_to_wave, hm[name][k], color=colors[name], lw=1.2,
                    label=f"{name} (conf={cf[name][k]:.2f})")
            lam_true = rest * (1 + y_test[k])
            if fm_to_wave[0] <= lam_true <= fm_to_wave[-1]:
                ax.axvline(lam_true, color=colors[name], ls=":", lw=1.0, alpha=0.7)
        ax.set_ylabel("prob")
        ax.set_title(f"z real={y_test[k]:.3f}  |  z previsto={z_pred[k]:.3f}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("Comprimento de onda observado (A)")
    fig.suptitle(f"Heatmaps de deteccao multi-linha — {obj}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
