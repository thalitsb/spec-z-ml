"""Figuras exploratorias (EDA) do TCC — leem o catalogo rico do HDF5 de cada classe:
  A. fig_eda_stacked.png   — espectro MEDIANO empilhado no referencial de REPOUSO, por
                             classe, com as linhas caracteristicas marcadas.
  B. fig_eda_colorcolor.png— diagrama cor-cor (g-r x r-i) colorido por z, por classe.
  C. fig_eda_panorama.png  — distribuicao de z sobreposta + tamanhos das amostras.

Uso (no cluster, env thalita):
    python scripts/analysis/eda_figures.py
Saida: results/figures/
"""
import sys, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.style import set_science_style
    set_science_style()
except Exception as e:
    print(f"(aviso: set_science_style falhou: {e})")

OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
OBJECTS = ["LRG", "ELG", "QSO"]
COLOR = {"LRG": "#d62728", "ELG": "#2ca02c", "QSO": "#1f77b4"}

# linhas caracteristicas (repouso, Angstrom)
LINES = {
    "LRG": {"CaII K": 3933.7, "CaII H": 3968.5, "G-band": 4304, "Mg b": 5175},
    "ELG": {"[OII]": 3727, r"H$\beta$": 4861, "[OIII]": 5007, r"H$\alpha$": 6563},
    "QSO": {r"Ly$\alpha$": 1216, "CIV": 1549, "CIII]": 1909, "MgII": 2799},
}


def h5path(obj):
    g = glob.glob(str(ROOT / f"data/processed/{obj}/*padded*.h5"))
    return g[0] if g else None


def load_cat(obj, keys):
    p = h5path(obj)
    if not p:
        return None
    out = {}
    with h5py.File(p, "r") as h:
        for k in keys:
            dk = f"catalog/{k}"
            out[k] = h[dk][:] if dk in h else None
        out["z"] = h["ml_dataset/y"][:]
    return out


def sample_spectra(obj, n=3000, seed=42):
    """Amostra n espectros + z + grade de onda (observada)."""
    p = h5path(obj)
    if not p:
        return None
    with h5py.File(p, "r") as h:
        N = h["ml_dataset/y"].shape[0]
        idx = np.sort(np.random.default_rng(seed).choice(N, min(n, N), replace=False))
        X = h["ml_dataset/X_spec"][idx]
        z = h["ml_dataset/y"][idx]
        wave = h["wave_global"][:]
    return X, z, wave


# ---------------------------------------------------------------------------
# A — espectro mediano no referencial de repouso
# ---------------------------------------------------------------------------
def fig_stacked():
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    for ax, obj in zip(axes, OBJECTS):
        s = sample_spectra(obj)
        if s is None:
            ax.set_visible(False); continue
        X, z, wave = s
        rest_min = wave.min() / (1 + np.percentile(z, 99))
        rest_max = wave.max() / (1 + np.percentile(z, 1))
        grid = np.linspace(rest_min, rest_max, 1600)
        stack = np.full((len(z), len(grid)), np.nan)
        for i in range(len(z)):
            m = X[i] > 0
            if m.sum() < 50:
                continue
            f = X[i][m] / np.nanmedian(X[i][m])          # normaliza pela mediana
            stack[i] = np.interp(grid, wave[m] / (1 + z[i]), f, left=np.nan, right=np.nan)
        med = np.nanmedian(stack, axis=0)
        ax.plot(grid, med, color=COLOR[obj], lw=1.0)
        ymax = np.nanpercentile(med, 99.5)
        for name, lam in LINES[obj].items():
            if grid.min() < lam < grid.max():
                ax.axvline(lam, color="gray", ls=":", lw=0.8, alpha=0.7)
                ax.text(lam, ymax * 0.95, name, rotation=90, va="top", ha="right",
                        fontsize=6, color="dimgray")
        ax.set_ylabel("fluxo norm.")
        ax.set_title(f"{obj} — espectro mediano (referencial de repouso)", fontweight="bold")
        ax.set_ylim(0, ymax * 1.05)
        ax.set_xlim(grid.min(), grid.max())
    axes[-1].set_xlabel(r"comprimento de onda em repouso [$\AA$]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_stacked.png", dpi=200, bbox_inches="tight")
    plt.close(fig); return "fig_eda_stacked.png"


# ---------------------------------------------------------------------------
# B — diagrama cor-cor colorido por z
# ---------------------------------------------------------------------------
def fig_colorcolor():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, obj in zip(axes, OBJECTS):
        c = load_cat(obj, ["gr_color", "ri_color"])
        gr = c["gr_color"] if c else None
        if gr is None:
            ax.set_visible(False); continue
        gr = np.asarray(gr, dtype=float)
        z = np.asarray(c["z"], dtype=float)
        ri = c["ri_color"]
        if ri is not None:                                  # LRG/QSO: cor-cor
            ri = np.asarray(ri, dtype=float)
            m = np.isfinite(gr) & np.isfinite(ri) & (np.abs(gr) < 4) & (np.abs(ri) < 4)
            sc = ax.scatter(gr[m], ri[m], c=z[m], s=3, alpha=0.4, cmap="viridis",
                            rasterized=True)
            ax.set_xlabel("g - r"); ax.set_ylabel("r - i")
            fig.colorbar(sc, ax=ax, label="z")
        else:                                               # ELG: so g-r -> g-r vs z
            m = np.isfinite(gr) & (np.abs(gr) < 4)
            hb = ax.hexbin(z[m], gr[m], gridsize=45, cmap="viridis", mincnt=1, bins="log")
            ax.set_xlabel("z"); ax.set_ylabel("g - r")
            fig.colorbar(hb, ax=ax, label="log N")
        ax.set_title(obj, fontweight="bold")
    fig.suptitle("Cor x redshift (LRG/QSO: cor-cor; ELG: g-r vs z)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_colorcolor.png", dpi=200, bbox_inches="tight")
    plt.close(fig); return "fig_eda_colorcolor.png"


# ---------------------------------------------------------------------------
# C — panorama: distribuicao de z + tamanhos
# ---------------------------------------------------------------------------
def fig_panorama():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    sizes = {}
    for obj in OBJECTS:
        c = load_cat(obj, [])
        if c is None:
            continue
        z = c["z"]; sizes[obj] = len(z)
        ax1.hist(z, bins=60, histtype="step", lw=1.8, density=True,
                 color=COLOR[obj], label=f"{obj} (n={len(z):,})")
    ax1.set_xlabel("z"); ax1.set_ylabel("densidade")
    ax1.set_title("Distribuicao de redshift por classe", fontweight="bold")
    ax1.legend(fontsize=8)
    ax2.bar(list(sizes.keys()), list(sizes.values()),
            color=[COLOR[o] for o in sizes])
    for i, (o, n) in enumerate(sizes.items()):
        ax2.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("nº de objetos"); ax2.set_title("Tamanho da amostra", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_panorama.png", dpi=200, bbox_inches="tight")
    plt.close(fig); return "fig_eda_panorama.png"


def main():
    made = []
    for fn in (fig_panorama, fig_colorcolor, fig_stacked):
        try:
            made.append(fn()); print(f"  ok: {made[-1]}")
        except Exception as e:
            print(f"  FALHOU {fn.__name__}: {e}")
    print("=== figuras EDA em results/figures/ ===")
    for f in made: print(f"  {f}")


if __name__ == "__main__":
    main()
