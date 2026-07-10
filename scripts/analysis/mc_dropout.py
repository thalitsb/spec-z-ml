"""MC Dropout — incerteza por objeto na CNN.

Ideia: mantendo o dropout LIGADO na inferencia (training=True) e rodando T
passadas, cada objeto ganha uma DISTRIBUICAO de predicoes -> media (z previsto)
+ desvio (incerteza). O teste decisivo: o desvio previsto correlaciona com o
erro real? Se sim, a rede "sabe quando nao sabe" (util pra rejeitar/priorizar).

Uso (cluster, env thalita):
    python scripts/analysis/mc_dropout.py --object LRG --passes 30 --limit 3000
Saida: results/{OBJ}/uncertainty/mc_dropout_{OBJ}.npz + results/figures/fig_mc_dropout_{OBJ}.png
"""
import sys, argparse, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import SPLITS_DIR
from src.data import load_spectral_dataset, normalize_spectra, make_or_load_split, apply_split
from src.models.cnn import ScaledSoftplus  # registra a layer custom
try:
    from src.evaluation.style import set_science_style
    set_science_style()
except Exception:
    pass

import keras
from scipy.stats import spearmanr


def find_model(obj):
    for pat in [f"models/{obj}/cnn_baseline/*.keras",
                f"models/{obj}/cnn_baseline_stratified/*.keras"]:
        g = glob.glob(str(ROOT / pat))
        if g:
            return g[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="LRG")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--limit", type=int, default=3000, help="subamostra do test (CPU)")
    args = ap.parse_args()
    obj = args.object

    mp = find_model(obj)
    if mp is None:
        print(f"Sem modelo cnn_baseline para {obj}."); return
    print(f"Modelo: {mp}")
    model = keras.models.load_model(mp)

    # dados + split canonico -> test set
    h5 = glob.glob(str(ROOT / f"data/processed/{obj}/*padded*.h5"))[0]
    X, y, _ = load_spectral_dataset(h5, seed=42)
    X = normalize_spectra(X)
    _, _, te = make_or_load_split(obj, y, SPLITS_DIR)
    Xte, yte = X[te], y[te]

    if args.limit and args.limit < len(yte):
        idx = np.sort(np.random.default_rng(42).choice(len(yte), args.limit, replace=False))
        Xte, yte = Xte[idx], yte[idx]
    n_wave = model.input_shape[-2] if len(model.input_shape) == 3 else Xte.shape[1]
    Xte = Xte.reshape(-1, n_wave, 1).astype("float32")
    print(f"Test usado: {len(yte):,} objetos | {args.passes} passadas MC")

    # MC dropout: training=True mantem dropout ativo
    T = args.passes
    preds = np.zeros((T, len(yte)), dtype=np.float32)
    for t in range(T):
        out = []
        for i in range(0, len(yte), 512):
            out.append(np.asarray(model(Xte[i:i+512], training=True)).ravel())
        preds[t] = np.concatenate(out)
        print(f"  passada {t+1}/{T}", end="\r")
    print()

    z_mean = preds.mean(0)
    z_std = preds.std(0)                       # incerteza epistemica por objeto
    dz = np.abs(z_mean - yte) / (1 + yte)       # erro real |Δz|/(1+z)

    rho, _ = spearmanr(z_std, dz)
    print(f"Spearman(incerteza, |erro|) = {rho:+.3f}  (quanto maior, melhor calibrada)")

    out_dir = ROOT / "results" / obj / "uncertainty"; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"mc_dropout_{obj}.npz",
             y_test=yte, z_mean=z_mean, z_std=z_std, dz_abs=dz, spearman=rho, passes=T)

    # ---- figura: calibracao + scatter colorido pela incerteza ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) erro medio por bin de incerteza
    q = np.quantile(z_std, np.linspace(0, 1, 9))
    xc, ym = [], []
    for i in range(len(q) - 1):
        m = (z_std >= q[i]) & (z_std < q[i + 1] if i < len(q) - 2 else z_std <= q[i + 1])
        if m.sum() > 5:
            xc.append(np.median(z_std[m])); ym.append(np.median(dz[m]))
    a1.plot(xc, ym, "o-", color="#1f77b4")
    a1.set_xlabel("incerteza prevista (desvio MC)")
    a1.set_ylabel(r"|$\Delta z$|/(1+z) mediano")
    a1.set_title(f"Calibracao — Spearman={rho:+.2f}", fontweight="bold")
    a1.grid(alpha=0.3)
    # (b) z_pred vs z_true colorido pela incerteza
    sc = a2.scatter(yte, z_mean, c=z_std, s=6, cmap="plasma", alpha=0.6, rasterized=True)
    lim = [yte.min(), yte.max()]
    a2.plot(lim, lim, "k--", lw=1)
    a2.set_xlabel("z real"); a2.set_ylabel("z previsto (media MC)")
    a2.set_title(f"{obj} — cor = incerteza", fontweight="bold")
    fig.colorbar(sc, ax=a2, label="desvio MC")
    fig.suptitle(f"MC Dropout ({T} passadas) — {obj}", fontweight="bold")
    fig.tight_layout()
    figp = ROOT / "results" / "figures" / f"fig_mc_dropout_{obj}.png"
    fig.savefig(figp, dpi=200, bbox_inches="tight")
    print(f"figura: {figp}")


if __name__ == "__main__":
    main()
