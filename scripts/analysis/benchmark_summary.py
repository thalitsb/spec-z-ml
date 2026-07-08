"""Agrega os runs do benchmark multi-seed -> tabela media +/- desvio + teste pareado.

Le results/benchmark/runs/*.csv (1 linha por model/object/seed) e gera:
  - results/benchmark/summary.csv          (media+/-std por model x object)
  - results/benchmark/summary_nmad.png     (barras com erro)
  - comparacao pareada (ELG): linedet vs baseline vs xgb, diferenca por seed.

Uso: python scripts/benchmark_summary.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.style import set_science_style
set_science_style()
from config import RESULTS_DIR

C = 299792.458
BASE = RESULTS_DIR / "benchmark"
MODEL_ORDER = ["xgb_raw", "xgb_pca", "cnn_baseline", "cnn_linedet"]


def main():
    files = sorted((BASE / "runs").glob("*.csv"))
    if not files:
        print("Nenhum run em results/benchmark/runs/. Rode o benchmark primeiro.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Runs encontrados: {len(df)}  (esperado 50 quando completo)")
    print("Por (model,object): contagem de seeds")
    print(df.groupby(["model", "object"]).size().unstack(fill_value=0))

    # ---- tabela media +/- std ----
    g = df.groupby(["model", "object"])["nmad"].agg(["mean", "std", "count"]).reset_index()
    g["nmad_kms_mean"] = g["mean"] * C
    g["fmt"] = g.apply(lambda r: f"{r['mean']:.4f}±{r['std']:.4f} (n={int(r['count'])})", axis=1)
    table = g.pivot(index="model", columns="object", values="fmt").reindex(MODEL_ORDER)
    print("\n===== σ_NMAD  media ± desvio  =====")
    print(table.to_string())

    g.to_csv(BASE / "summary.csv", index=False)

    # ---- comparacao pareada (ELG): diferenca por seed ----
    print("\n===== Comparacao PAREADA (ELG) — diferenca por seed =====")
    elg = df[df.object == "ELG"].pivot_table(index="seed", columns="model", values="nmad")
    pairs = [("cnn_baseline", "xgb_pca"), ("cnn_linedet", "cnn_baseline")]
    for a, b in pairs:
        if a in elg and b in elg:
            d = (elg[a] - elg[b]).dropna()
            if len(d) >= 2:
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                sig = "SIGNIFICATIVO" if abs(t) > 2.78 else "nao concl. (|t|<2.78, n<5?)"  # t_0.975, df=4
                print(f"  {a} - {b}: Δ={d.mean():+.4f} ± {d.std(ddof=1):.4f} (n={len(d)})  "
                      f"t={t:+.2f}  [{sig}]")

    # ---- plot barras com erro ----
    objs = sorted(df.object.unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    x = np.arange(len(objs))
    for k, model in enumerate([m for m in MODEL_ORDER if m in df.model.unique()]):
        means, stds = [], []
        for o in objs:
            sub = df[(df.model == model) & (df.object == o)]["nmad"]
            means.append(sub.mean() * C if len(sub) else np.nan)
            stds.append(sub.std() * C if len(sub) > 1 else 0)
        ax.bar(x + k * width, means, width, yerr=stds, capsize=3, label=model)
    ax.axhline(20, ls=":", color="green", lw=1, label="piso ELG ~20 km/s (Raichoor 2021)")
    ax.set_yscale("log")
    ax.set_xticks(x + 1.5 * width); ax.set_xticklabels(objs)
    ax.set_ylabel("σ_NMAD (km/s, log)"); ax.set_title("Benchmark multi-seed — σ_NMAD por modelo/objeto")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE / "summary_nmad.png", dpi=150, bbox_inches="tight")
    print(f"\nSalvo: {BASE/'summary.csv'} e {BASE/'summary_nmad.png'}")


if __name__ == "__main__":
    main()
