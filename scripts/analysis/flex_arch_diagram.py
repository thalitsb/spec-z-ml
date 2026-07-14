"""Diagrama da arquitetura da CNN selecionada pela busca Optuna (flex LRG).
Gera results/figures/fig_flex_arch.png
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

# arquitetura escolhida pelo Optuna (best_params, flex LRG):
# n_conv_blocks=4, kernel 25 (-6/bloco), taper_last_block -> filtros 64,128,256,128
# n_dense_layers=4, n_units=576 (halving) -> 576,288,144,72 ; ELU, AdamW, BN
steps = [
    ("Espectro\n$(N_\\lambda \\times 1)$", "#4C72B0"),
    ("4 blocos\nconvolucionais\n64$\\to$128$\\to$256$\\to$128 filtros\nkernel 25$\\to$7 · BN · ELU", "#55A868"),
    ("Global\nAverage\nPooling", "#8172B3"),
    ("4 camadas densas\n576$\\to$288$\\to$144$\\to$72\ndropout (ramp)", "#C44E52"),
    ("ScaledSoftplus\n$z \\geq 0$", "#CCB974"),
]
fig, ax = plt.subplots(figsize=(13, 2.7))
bw, bh, gap = 2.2, 1.7, 0.55
x = 0.0; centers = []
for label, color in steps:
    ax.add_patch(FancyBboxPatch((x, 0), bw, bh,
                 boxstyle="round,pad=0.08,rounding_size=0.15",
                 lw=1.3, edgecolor="black", facecolor=color, alpha=0.88))
    ax.text(x + bw/2, bh/2, label, ha="center", va="center",
            fontsize=8.5, color="white", weight="bold")
    centers.append(x + bw/2); x += bw + gap
for i in range(len(steps) - 1):
    ax.add_patch(FancyArrowPatch((centers[i]+bw/2, bh/2), (centers[i+1]-bw/2, bh/2),
                 arrowstyle="-|>", mutation_scale=18, lw=1.6, color="black"))
ax.set_xlim(-0.3, x - gap + 0.3); ax.set_ylim(-0.3, bh + 0.3); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "fig_flex_arch.png", dpi=200, bbox_inches="tight")
print("salvo:", OUT / "fig_flex_arch.png")
