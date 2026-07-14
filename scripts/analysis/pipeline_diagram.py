"""Diagrama de fluxo da montagem do dataset (para o TCC).
Gera results/figures/fig_pipeline_dataset.png
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

steps = [
    ("Catálogos de\n\\itclustering\n(NGC + SGC)", "#4C72B0"),
    ("Cross-match 1\"\ncom catálogo completo\n(S/N, mags, cores)", "#55A868"),
    ("Cortes de qualidade\n($z \\geq 0$,\nZWARNING $= 0$)", "#C44E52"),
    ("Download dos espectros\n(cache + paralelo;\nURLs legacy / eBOSS)", "#8172B3"),
    ("HDF5 estruturado\n(fluxos + $z_{spec}$\n+ metadados)", "#CCB974"),
]

fig, ax = plt.subplots(figsize=(13, 2.6))
n = len(steps)
bw, bh, gap = 2.15, 1.6, 0.55
x = 0.0
centers = []
for label, color in steps:
    box = FancyBboxPatch((x, 0), bw, bh,
                         boxstyle="round,pad=0.08,rounding_size=0.15",
                         linewidth=1.3, edgecolor="black",
                         facecolor=color, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + bw/2, bh/2, label.replace("\\it", "").replace("$z_{spec}$", "z_spec")
            .replace("$z \\geq 0$", "z ≥ 0").replace("$", ""),
            ha="center", va="center", fontsize=8.5, color="white", weight="bold")
    centers.append(x + bw/2)
    x += bw + gap

for i in range(n - 1):
    a = FancyArrowPatch((centers[i] + bw/2, bh/2), (centers[i+1] - bw/2, bh/2),
                        arrowstyle="-|>", mutation_scale=18, lw=1.6, color="black")
    ax.add_patch(a)

ax.set_xlim(-0.3, x - gap + 0.3)
ax.set_ylim(-0.3, bh + 0.3)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "fig_pipeline_dataset.png", dpi=200, bbox_inches="tight")
print("salvo:", OUT / "fig_pipeline_dataset.png")
