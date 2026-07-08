"""Grafico do experimento SCALARS (CNN flex com vs sem) por objeto.

Auto-descobre os run_info.json dos flex, classifica em com/sem scalars pelo flag
`scalars_config.used_in_model`, pega a melhor run (menor sigma_NMAD) por (obj, variante),
e plota barras agrupadas + a linha do baseline (so' espectro) como referencia.
Inclui QSO automaticamente quando as runs existirem.

Uso: python scripts/analysis/plot_scalars_experiment.py
Saida: results/scalars_experiment.png + .csv
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import RESULTS_DIR
from src.evaluation.style import set_science_style
set_science_style()

OBJS = ["LRG", "ELG", "QSO"]
C = 299792.458


def discover():
    """Melhor (menor sigma_NMAD) run flex por (objeto, variante)."""
    rows = []
    for obj in OBJS:
        for p in (RESULTS_DIR / obj / "cnn_optuna_flex").glob("*/run_info.json"):
            d = json.load(open(p))
            if d.get("n_trials_complete", 0) < 40:      # ignora runs de teste (t3)
                continue
            used = d.get("scalars_config", {}).get("used_in_model", True)
            variant = "sem scalars" if used is False else "com scalars"
            rows.append({"obj": obj, "variant": variant,
                         "nmad": d["test_metrics"]["sigma_nmad"],
                         "eta05": d["test_metrics"]["eta_outliers_0.05"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("nmad").groupby(["obj", "variant"], as_index=False).first()


def baseline_nmad(obj):
    f = RESULTS_DIR / obj / "cnn_baseline" / "metrics.csv"
    return pd.read_csv(f).iloc[0]["nmad"] if f.exists() else None


def main():
    df = discover()
    if df.empty:
        print("Nenhuma run flex completa encontrada."); return
    objs = [o for o in OBJS if o in df["obj"].values]
    df.to_csv(RESULTS_DIR / "scalars_experiment.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(1.9 * len(objs) + 1.5, 3.2))
    x = np.arange(len(objs)); w = 0.34
    colors = {"com scalars": "#2c7fb8", "sem scalars": "#d95f0e"}
    for k, variant in enumerate(["com scalars", "sem scalars"]):
        vals = [df[(df.obj == o) & (df.variant == variant)]["nmad"].squeeze()
                if not df[(df.obj == o) & (df.variant == variant)].empty else np.nan
                for o in objs]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=variant, color=colors[variant])
        for xi, v in zip(x + (k - 0.5) * w, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=6)
    # baseline (so' espectro) como tracinho de referencia por objeto
    for xi, o in zip(x, objs):
        b = baseline_nmad(o)
        if b is not None:
            ax.plot([xi - 0.5, xi + 0.5], [b, b], color="k", ls="--", lw=1.0,
                    label="baseline (só espectro)" if xi == 0 else None)
            ax.text(xi + 0.52, b, f"{b:.4f}", va="center", fontsize=5.5, color="k")
    ax.set_xticks(x); ax.set_xticklabels(objs)
    ax.set_ylabel(r"$\sigma_{\mathrm{NMAD}}$ (teste)")
    ax.set_title("CNN flex: efeito dos scalars (estatísticas de fluxo) vs baseline")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "scalars_experiment.png")
    print(f"\nSalvo: {RESULTS_DIR/'scalars_experiment.png'}")


if __name__ == "__main__":
    main()
