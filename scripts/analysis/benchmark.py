"""Benchmark multi-seed — UMA execucao (modelo, objeto, seed) por chamada.

Regra de ouro: para um dado seed, TODOS os modelos usam o MESMO split
(make_split(y, random_state=seed), estratificado por z) -> comparacao pareada entre modelos.

Modelos: xgb_raw, xgb_pca, cnn_baseline, cnn_linedet (so' ELG).
Saida: 1 arquivo por run em results/benchmark/runs/{model}__{obj}__s{seed}.csv
       (resumivel: pula se ja existe; o summary agrega tudo).

Uso:
    python scripts/benchmark.py --model xgb_pca --object ELG --seed 42
    python scripts/benchmark.py --task-id 7              # pega o 7o combo do grid
    python scripts/benchmark.py --list                   # lista o grid completo
    python scripts/benchmark.py --model xgb_raw --object ELG --seed 42 --limit 3000  # smoke
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import paths_for, RESULTS_DIR
from src.data import load_spectral_dataset, normalize_spectra, make_split, apply_split
from src.evaluation import compute_redshift_metrics

SEEDS = [42, 43, 44, 45, 46]
OBJECTS = ["LRG", "ELG", "QSO"]
PCA_COMPONENTS = {"LRG": 800, "ELG": 1200, "QSO": 800}  # joelho por objeto (Conclusoes 1-2)
OUT_DIR = RESULTS_DIR / "benchmark" / "runs"


def grid():
    """Todos os combos (model, object, seed) do benchmark."""
    combos = []
    for s in SEEDS:
        for o in OBJECTS:
            combos.append(("xgb_raw", o, s))
            combos.append(("xgb_pca", o, s))
            combos.append(("cnn_baseline", o, s))
            if o == "ELG":
                combos.append(("cnn_linedet", o, s))
    return combos


def extract_scalars(X_raw):
    """4 features escalares do espectro cru (log do max/mediana/soma/p95)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = np.stack([
            np.log10(np.nanmax(X_raw, 1)),
            np.log10(np.nanmedian(X_raw, 1)),
            np.log10(np.nansum(np.abs(X_raw), 1)),
            np.log10(np.nanpercentile(X_raw, 95, axis=1)),
        ], axis=1)
    return np.where(np.isfinite(feats), feats, -40.0).astype(np.float32)


def train_eval(model, obj, seed, X_raw, y, wave, limit=None):
    """Treina o modelo e retorna o dict de metricas no test."""
    tr, va, te = make_split(y, random_state=seed)
    t0 = time.time()

    if model in ("xgb_raw", "xgb_pca"):
        from src.models import train_xgboost
        Xn = normalize_spectra(X_raw)
        if model == "xgb_raw":
            Xtr, Xva, Xte = apply_split(Xn, tr, va, te)
        else:  # xgb_pca: PCA + scalars + StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            ncomp = PCA_COMPONENTS[obj]
            sc = extract_scalars(X_raw)
            pca = PCA(n_components=ncomp, random_state=seed)
            Ptr = pca.fit_transform(Xn[tr]); Pva = pca.transform(Xn[va]); Pte = pca.transform(Xn[te])
            std = StandardScaler().fit(Ptr)
            Ptr, Pva, Pte = std.transform(Ptr), std.transform(Pva), std.transform(Pte)
            Xtr = np.hstack([Ptr, sc[tr]]); Xva = np.hstack([Pva, sc[va]]); Xte = np.hstack([Pte, sc[te]])
        ytr, yva, yte = apply_split(y, tr, va, te)
        m = train_xgboost(Xtr, ytr, Xva, yva, seed=seed)
        y_pred = m.predict(Xte)

    elif model == "cnn_baseline":
        from src.models import PaddedSpectralCNN
        Xn = normalize_spectra(X_raw)
        Xtr, Xva, Xte = apply_split(Xn, tr, va, te)
        ytr, yva, yte = apply_split(y, tr, va, te)
        cnn = PaddedSpectralCNN(n_wave=Xn.shape[1], learning_rate=3e-4)
        cnn.fit(Xtr, ytr, X_val=Xva, y_val=yva, epochs=50, batch_size=64, verbose=2)
        y_pred = cnn.predict(Xte)

    elif model == "cnn_linedet":
        from src.models.cnn_linedet import OIILineDetectionCNN
        Xn = normalize_spectra(X_raw)
        Xtr, Xva, Xte = apply_split(Xn, tr, va, te)
        ytr, yva, yte = apply_split(y, tr, va, te)
        loglam0 = float(np.log10(wave[0])); dloglam = float(np.median(np.diff(np.log10(wave))))
        cnn = OIILineDetectionCNN(n_wave=Xn.shape[1], loglam0=loglam0, dloglam=dloglam,
                                  learning_rate=3e-4, heatmap_weight=0.2, sigma_bins=2.0)
        cnn.fit(Xtr, ytr, X_val=Xva, y_val=yva, epochs=60, batch_size=64, verbose=2)
        y_pred = cnn.predict(Xte)
    else:
        raise ValueError(model)

    res = compute_redshift_metrics(yte, y_pred)
    return {"model": model, "object": obj, "seed": seed, "n_test": len(yte),
            "train_time_s": round(time.time() - t0, 1),
            **{k: res[k] for k in ["nmad", "rmse", "mae", "r2", "bias",
                                    "outliers_0.05_pct", "outliers_0.15_pct"]},
            "nmad_kms": round(res["nmad"] * 299792.458, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["xgb_raw", "xgb_pca", "cnn_baseline", "cnn_linedet"])
    ap.add_argument("--object", choices=OBJECTS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task-id", type=int, help="indice (0-based) no grid()")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, help="subamostra p/ smoke test")
    ap.add_argument("--force", action="store_true", help="re-roda mesmo se ja existe")
    args = ap.parse_args()

    if args.list:
        for i, c in enumerate(grid()):
            print(f"{i:3d}  {c[0]:14s} {c[1]:4s} seed={c[2]}")
        print(f"\nTotal: {len(grid())} runs")
        return

    if args.task_id is not None:
        model, obj, seed = grid()[args.task_id]
    else:
        model, obj, seed = args.model, args.object, args.seed
    assert model and obj and seed is not None, "informe --model/--object/--seed ou --task-id"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{model}__{obj}__s{seed}.csv"
    if out.exists() and not args.force:
        print(f"[skip] ja existe: {out}")
        return

    print(f"[run] {model} {obj} seed={seed}")
    h5 = paths_for(obj)["spectra_h5"].with_name(f"{obj}spectra_padded.h5")
    X, y, _ = load_spectral_dataset(h5, n_samples=args.limit, seed=seed)
    with h5py.File(h5, "r") as f:
        wave = f["wave_global"][:]

    row = train_eval(model, obj, seed, X, y, wave, limit=args.limit)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[ok] NMAD={row['nmad']:.4e} (~{row['nmad_kms']:.0f} km/s)  "
          f"eta05={row['outliers_0.05_pct']:.2f}%  -> {out}")


if __name__ == "__main__":
    main()
