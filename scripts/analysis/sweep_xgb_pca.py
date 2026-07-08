"""Sweep de N_COMPONENTS pro XGBoost + PCA — gera a curva NMAD vs n_components.

Objetivo: achar o "joelho" da curva acuracia x compressao. Mostra quanto da
informacao de redshift (posicao das linhas espectrais) o PCA perde ao comprimir.

Otimizacao: o PCA e' fitado UMA vez com o N maximo; pra cada N menor a gente
fatia os primeiros N componentes (PCs sao ordenados, entao P[:, :N] == PCA(N)).
Isso evita re-fitar o PCA pra cada ponto do sweep.

Uso:
    python scripts/sweep_xgb_pca.py --object LRG
    python scripts/sweep_xgb_pca.py --object LRG --components 300 500 800 1200 1912
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.style import set_science_style
set_science_style()

from config import paths_for, RESULTS_DIR
from src.data import load_spectral_dataset, normalize_spectra
from src.evaluation.metrics import delta_z_normalized, sigma_nmad

# Referencia: XGBoost com espectro cru (4674 features), sem PCA — por objeto.
# Valores de results/{OBJ}/xgboost_baseline/metrics.csv.
XGB_RAW_NMAD = {
    "LRG": 0.0180,
    "ELG": 0.0261,
    # "QSO": preencher quando tiver o baseline
}

# HPs manuais (mesma config do xgb_pca_didatico — sem Optuna, pra sweep rapido).
XGB_PARAMS = dict(
    n_estimators=10000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=50,
)


def extract_scalars(X_orig):
    X_abs = np.abs(X_orig)
    X_pos = np.where(X_orig > 0, X_orig, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_max = np.log10(X_abs.max(axis=1))
        log_median = np.log10(np.nanmedian(X_pos, axis=1))
        log_sum = np.log10(X_abs.sum(axis=1))
        log_p95 = np.log10(np.nanpercentile(X_pos, 95, axis=1))
    scalars = np.stack([log_max, log_median, log_sum, log_p95], axis=1)
    scalars = np.where(np.isfinite(scalars), scalars, -40.0)
    return scalars.astype(np.float32)


def make_strat_bins(y, n_splits=2, start_q=20, min_q=2):
    q = start_q
    while q >= min_q:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts()
        if (counts >= n_splits).all():
            return np.asarray(bins), q
        q -= 1
    raise ValueError("Nao consegui estratificar.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", default="LRG", choices=["ELG", "LRG", "QSO"])
    p.add_argument("--components", type=int, nargs="+",
                   default=[300, 500, 800, 1200, 1912])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    args = p.parse_args()

    obj = args.object
    seed = args.seed
    comps = sorted(args.components)
    n_max = max(comps)

    print(f"[{obj}] sweep N_COMPONENTS = {comps}  (PCA fit uma vez com {n_max})")

    # ---------- Dados ----------
    paths = paths_for(obj)
    hdf5 = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")
    print(f"[{obj}] Carregando {hdf5}")
    X_orig, y_all, _ = load_spectral_dataset(hdf5)
    scalars_all = extract_scalars(X_orig)
    X_spec = normalize_spectra(X_orig).astype(np.float32)
    y_all = y_all.astype(np.float32)
    del X_orig

    # ---------- Split canonico ESTRATIFICADO (mesmo .npz de TODOS os modelos) ----------
    from config import SPLITS_DIR
    from src.data import make_or_load_split, apply_split
    tr_idx, val_idx, test_idx = make_or_load_split(obj, y_all, SPLITS_DIR)
    X_train, X_val, X_test = apply_split(X_spec, tr_idx, val_idx, test_idx)
    S_train, S_val, S_test = apply_split(scalars_all, tr_idx, val_idx, test_idx)
    y_train, y_val, y_test = apply_split(y_all, tr_idx, val_idx, test_idx)
    print(f"[{obj}] train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")

    # ---------- PCA fit UMA vez com n_max ----------
    print(f"[{obj}] Fitando PCA({n_max}) no train ({len(X_train):,} espectros)...")
    t0 = time.time()
    pca = PCA(n_components=n_max, random_state=seed)
    P_train_full = pca.fit_transform(X_train)
    P_val_full = pca.transform(X_val)
    P_test_full = pca.transform(X_test)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"[{obj}] PCA fit em {time.time()-t0:.1f}s | var@{n_max} = {cumvar[-1]*100:.2f}%")

    # ---------- Loop: pra cada N, fatia os primeiros N PCs ----------
    rows = []
    for n in comps:
        print(f"\n[{obj}] === N_COMPONENTS = {n} (var = {cumvar[n-1]*100:.2f}%) ===")
        Xtr = np.concatenate([P_train_full[:, :n], S_train], axis=1)
        Xva = np.concatenate([P_val_full[:, :n], S_val], axis=1)
        Xte = np.concatenate([P_test_full[:, :n], S_test], axis=1)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)

        t0 = time.time()
        model = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        train_time = time.time() - t0

        y_pred = model.predict(Xte)
        dz = delta_z_normalized(y_test, y_pred)
        nmad = sigma_nmad(dz)
        row = {
            "n_components": n,
            "variance_retained": float(cumvar[n - 1]),
            "best_iteration": int(model.best_iteration),
            "nmad": nmad,
            "mae": float(np.mean(np.abs(y_pred - y_test))),
            "bias": float(np.median(dz)),
            "eta_outliers_0.05": float(100.0 * np.mean(np.abs(dz) > 0.05)),
            "eta_outliers_0.15": float(100.0 * np.mean(np.abs(dz) > 0.15)),
            "train_time_s": float(train_time),
        }
        rows.append(row)
        print(f"[{obj}] N={n}: NMAD={nmad:.4f}  var={cumvar[n-1]*100:.1f}%  "
              f"trees={model.best_iteration}  ({train_time:.0f}s)")

    df = pd.DataFrame(rows)

    # ---------- Salva ----------
    out_dir = RESULTS_DIR / obj / "xgb_pca_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"sweep_ncomponents_{obj}_s{seed}.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "object_type": obj,
        "seed": seed,
        "components": comps,
        "xgb_params": XGB_PARAMS,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "results": rows,
    }
    with open(out_dir / f"sweep_ncomponents_{obj}_s{seed}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["n_components"], df["nmad"], "o-", color="steelblue", lw=2, ms=7)
    raw_ref = XGB_RAW_NMAD.get(obj)
    if raw_ref is not None:
        ax.axhline(raw_ref, color="red", ls="--", lw=1.5,
                   label=f"xgb_{obj.lower()} cru (4674 feat) = {raw_ref}")
    for _, r in df.iterrows():
        ax.annotate(f'{r["nmad"]:.4f}', (r["n_components"], r["nmad"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("N_COMPONENTS (PCs)")
    ax.set_ylabel("sigma_NMAD (test)")
    ax.set_title(f"({obj}) NMAD vs N_COMPONENTS")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["n_components"], df["variance_retained"] * 100, "s-", color="darkorange", lw=2, ms=7)
    ax.axhline(95, color="gray", ls=":", lw=1, label="95%")
    ax.set_xlabel("N_COMPONENTS (PCs)")
    ax.set_ylabel("variancia retida (%)")
    ax.set_title(f"({obj}) Variancia retida vs N_COMPONENTS")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / f"sweep_ncomponents_{obj}_s{seed}.png"
    plt.savefig(png_path, dpi=130, bbox_inches="tight")

    # ---------- Resumo ----------
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETO ({obj})")
    print(f"{'='*60}")
    print(df.to_string(index=False,
                       formatters={"variance_retained": "{:.3f}".format,
                                   "nmad": "{:.4f}".format,
                                   "mae": "{:.4f}".format,
                                   "bias": "{:+.4f}".format,
                                   "eta_outliers_0.05": "{:.2f}".format,
                                   "eta_outliers_0.15": "{:.2f}".format,
                                   "train_time_s": "{:.0f}".format}))
    best = df.loc[df["nmad"].idxmin()]
    print(f"\nMelhor NMAD: {best['nmad']:.4f} com {int(best['n_components'])} PCs "
          f"({best['variance_retained']*100:.1f}% variancia)")
    print(f"\nSalvo em:")
    print(f"  CSV : {csv_path}")
    print(f"  PNG : {png_path}")


if __name__ == "__main__":
    main()
