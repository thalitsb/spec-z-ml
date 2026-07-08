"""Figuras comparativas do TCC (Fase 3) — leem as predicoes (y_test, y_pred) de
cada (objeto, modelo) e geram:

  1. fig_sigma_by_zbin.png   — sigma_NMAD por faixa de z (linha por modelo, por objeto).
                               Mostra que o desempenho e estavel em TODO o intervalo de z
                               (justifica visualmente o split estratificado).
  2. fig_residuals_vs_z.png  — residuos Δz/(1+z) vs z (hexbin) do melhor modelo, por objeto.
                               Revela vies sistematico / regioes problematicas.
  3. fig_deltaz_hist.png     — histograma de Δz/(1+z) sobreposto (todos os modelos), log-y.
                               Nucleo estreito vs cauda de catastrofes num olhar.

Arquivos que nao existirem sao pulados (o script roda com o que ja tem e melhora
conforme mais runs terminam). Saida em results/figures/.

Uso (no cluster, com o env thalita):
    python scripts/analysis/paper_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.style import set_science_style
    set_science_style()
except Exception as e:
    print(f"(aviso: set_science_style falhou: {e} — seguindo com estilo padrao)")

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_KMS = 299792.458
OBJECTS = ["LRG", "ELG", "QSO"]

# Registro (objeto -> {modelo: caminho do .npz}). Ajuste os run-tags do flex conforme
# os runs canonicos forem terminando; caminhos ausentes sao ignorados.
REG = {
    "LRG": {
        "XGBoost":      "results/LRG/xgboost_baseline/predictions.npz",
        "XGBoost+PCA":  "results/LRG/xgb_optuna_pca/xgb_optuna_pca-LRG-pc300_t50_s42/test_outputs.npz",
        "CNN baseline": "results/LRG/cnn_baseline/predictions.npz",
        "CNN flex":     "results/LRG/cnn_optuna_flex/flex-LRG-t100_e80_s42-mae_nozw-c0cfd800/test_outputs.npz",
    },
    "ELG": {
        "XGBoost":      "results/ELG/xgboost_baseline/predictions.npz",
        "CNN baseline": "results/ELG/cnn_baseline/predictions.npz",
        "CNN linedet":  "results/ELG/cnn_linedet/predictions.npz",
    },
    "QSO": {
        "XGBoost":      "results/QSO/xgboost_baseline/predictions.npz",
        "CNN baseline": "results/QSO/cnn_baseline/predictions.npz",
    },
}

# cor fixa por modelo (consistencia entre figuras)
COLORS = {
    "XGBoost": "#d62728", "XGBoost+PCA": "#ff7f0e",
    "CNN baseline": "#1f77b4", "CNN flex": "#2ca02c", "CNN linedet": "#9467bd",
}
# preferencia de "melhor modelo" por objeto (pro painel de residuos)
BEST = {"LRG": "CNN flex", "ELG": "CNN linedet", "QSO": "CNN baseline"}
# ordem canonica dos modelos na legenda
MODEL_ORDER = ["XGBoost", "XGBoost+PCA", "CNN baseline", "CNN flex", "CNN linedet"]


def _unified_legend(fig, axes):
    """Uma legenda unica pra figura, juntando os modelos de todos os paineis."""
    seen = {}
    for ax in np.atleast_1d(axes).ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labels = [m for m in MODEL_ORDER if m in seen]
    handles = [seen[m] for m in labels]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=True)


def load(path):
    """Retorna (z_true, dz_norm) ou None se o arquivo nao existir."""
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p)
    y, yp = np.asarray(d["y_test"]).ravel(), np.asarray(d["y_pred"]).ravel()
    dz = np.asarray(d["delta_z"]).ravel() if "delta_z" in d else (yp - y) / (1 + y)
    return y, dz


def sigma_nmad(dz):
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))


def load_object(obj):
    """{modelo: (z_true, dz)} apenas com os arquivos existentes."""
    out = {}
    for model, path in REG[obj].items():
        r = load(path)
        if r is not None:
            out[model] = r
    return out


# ---------------------------------------------------------------------------
# Fig 1 — sigma_NMAD por faixa de z
# ---------------------------------------------------------------------------
def fig_sigma_by_zbin(data, n_bins=8):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, obj in zip(axes, OBJECTS):
        for model, (z, dz) in data[obj].items():
            edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
            edges[-1] += 1e-9
            centers, sig = [], []
            for i in range(n_bins):
                m = (z >= edges[i]) & (z < edges[i + 1])
                if m.sum() < 20:
                    continue
                centers.append(0.5 * (edges[i] + edges[i + 1]))
                sig.append(sigma_nmad(dz[m]))
            ax.plot(centers, sig, marker="o", ms=3, lw=1.5,
                    color=COLORS.get(model), label=model)
        ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_title(obj, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{NMAD}}$")
    _unified_legend(fig, axes)
    fig.suptitle(r"$\sigma_{\mathrm{NMAD}}$ por faixa de redshift (menor = melhor)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sigma_by_zbin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "fig_sigma_by_zbin.png"


# ---------------------------------------------------------------------------
# Fig 2 — residuos Δz/(1+z) vs z (melhor modelo por objeto)
# ---------------------------------------------------------------------------
def fig_residuals_vs_z(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, obj in zip(axes, OBJECTS):
        model = BEST[obj] if BEST[obj] in data[obj] else next(iter(data[obj]), None)
        if model is None:
            ax.set_visible(False)
            continue
        z, dz = data[obj][model]
        dzc = np.clip(dz, -0.15, 0.15)
        hb = ax.hexbin(z, dzc, gridsize=45, cmap="magma", mincnt=1, bins="log")
        ax.axhline(0, color="cyan", lw=1, alpha=0.7)
        s = sigma_nmad(dz)
        ax.axhline(+s, color="white", lw=0.9, ls="--")
        ax.axhline(-s, color="white", lw=0.9, ls="--")
        ax.set_xlabel("z")
        ax.set_title(f"{obj} — {model}", fontweight="bold")
        ax.set_ylim(-0.15, 0.15)
        fig.colorbar(hb, ax=ax, label="log N")
    axes[0].set_ylabel(r"$\Delta z / (1+z)$")
    fig.suptitle("Residuos vs redshift (linha branca = ±$\\sigma_{\\mathrm{NMAD}}$)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_residuals_vs_z.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "fig_residuals_vs_z.png"


# ---------------------------------------------------------------------------
# Fig 3 — histograma de Δz/(1+z) sobreposto
# ---------------------------------------------------------------------------
def fig_deltaz_hist(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bins = np.linspace(-0.05, 0.05, 120)
    for ax, obj in zip(axes, OBJECTS):
        for model, (z, dz) in data[obj].items():
            ax.hist(np.clip(dz, -0.05, 0.05), bins=bins, histtype="step", lw=1.4,
                    density=True, color=COLORS.get(model), label=model)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta z / (1+z)$")
        ax.set_title(obj, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("densidade (log)")
    _unified_legend(fig, axes)
    fig.suptitle(r"Distribuicao de $\Delta z/(1+z)$ — nucleo estreito vs cauda",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_deltaz_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "fig_deltaz_hist.png"


def main():
    data = {obj: load_object(obj) for obj in OBJECTS}
    print("=== modelos carregados por objeto ===")
    for obj in OBJECTS:
        got = data[obj]
        print(f"  {obj}: " + (", ".join(f"{m} ({sigma_nmad(d):.4f})"
              for m, (z, d) in got.items()) or "(nenhum)"))
    made = []
    for obj in OBJECTS:
        if data[obj]:
            made += [fig_sigma_by_zbin(data), fig_residuals_vs_z(data), fig_deltaz_hist(data)]
            break
    print("\n=== figuras geradas em results/figures/ ===")
    for f in dict.fromkeys(made):
        print(f"  {f}")


if __name__ == "__main__":
    main()
