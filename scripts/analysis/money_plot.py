"""Money plot — sigma_NMAD (km/s) multi-seed por modelo/objeto, com barras de erro.
Le results/benchmark/summary.csv e gera results/figures/summary_nmad.png
Consistente com paper_figures.py (mesmas cores/nomes; ordem LRG,ELG,QSO).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.style import set_science_style
    set_science_style()
except Exception:
    pass

C = 299792.458
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

NAME = {"xgb_raw": "XGBoost", "xgb_pca": "XGBoost+PCA",
        "cnn_baseline": "CNN baseline", "cnn_linedet": "CNN linedet"}
COLORS = {"XGBoost": "#d62728", "XGBoost+PCA": "#ff7f0e",
          "CNN baseline": "#1f77b4", "CNN linedet": "#9467bd"}
MODEL_ORDER = ["XGBoost", "XGBoost+PCA", "CNN baseline", "CNN linedet"]
OBJS = ["LRG", "ELG", "QSO"]

df = pd.read_csv(ROOT / "results" / "benchmark" / "summary.csv")
df["label"] = df["model"].map(NAME)
piv_m = df.pivot(index="object", columns="label", values="mean")
piv_s = df.pivot(index="object", columns="label", values="std")

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(OBJS))
nmod = len(MODEL_ORDER)
w = 0.8 / nmod
for i, model in enumerate(MODEL_ORDER):
    means = np.array([piv_m.loc[o, model] * C if (o in piv_m.index and model in piv_m.columns
                     and not np.isnan(piv_m.loc[o, model])) else np.nan for o in OBJS])
    errs = np.array([piv_s.loc[o, model] * C if (o in piv_s.index and model in piv_s.columns
                    and not np.isnan(piv_s.loc[o, model])) else np.nan for o in OBJS])
    pos = x + (i - (nmod - 1) / 2) * w
    ax.bar(pos, means, w, yerr=errs, capsize=3, color=COLORS[model],
           edgecolor="black", linewidth=0.5, label=model, alpha=0.9,
           error_kw=dict(lw=1.0))

ax.axhline(20, color="#2ca02c", ls=":", lw=1.5)
ax.text(len(OBJS) - 0.5, 22, "piso físico do ELG ($\\sim$20 km/s)",
        fontsize=7, color="#2ca02c", ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(OBJS)
ax.set_ylabel(r"$\sigma_{\mathrm{NMAD}}$ (km/s)")
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "summary_nmad.png", dpi=200, bbox_inches="tight")
print("salvo:", OUT / "summary_nmad.png")
