"""Figuras do estudo Optuna (busca de arquitetura) — o destaque do trabalho.
Reconstroi os plots manualmente (controle total do estilo/idioma):
  fig_optuna_history.png    — objetivo por trial + melhor-ate-agora (convergencia)
  fig_optuna_importance.png — importancia dos hiperparametros (fANOVA, seed fixa)
Imprime best_params (para o diagrama da arquitetura).

Uso: python scripts/analysis/optuna_figures.py [--db ...] [--study ...]
"""
import sys, argparse
from pathlib import Path
import numpy as np
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
import optuna
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/LRG/cnn_optuna_flex/optuna_study.db")
    ap.add_argument("--study", default="cnn_flex_LRG_mae_c0cfd800")
    a = ap.parse_args()
    study = optuna.load_study(study_name=a.study, storage=f"sqlite:///{a.db}")
    comp = [t for t in study.trials if t.value is not None]
    print(f"Estudo {a.study}: {len(study.trials)} trials ({len(comp)} completos), "
          f"melhor={study.best_value:.5f} (trial {study.best_trial.number})")

    # ---- 1. Historico de otimizacao ----
    nums = np.array([t.number for t in comp])
    vals = np.array([t.value for t in comp])
    order = np.argsort(nums)
    nums, vals = nums[order], vals[order]
    best_so_far = np.minimum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(nums, vals, s=18, alpha=0.5, color="#1f77b4", label="trial")
    ax.plot(nums, best_so_far, color="#d62728", lw=1.8, label="melhor até agora")
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1]*0.92, "trial 0 =\nbaseline", fontsize=7,
            color="dimgray", va="top")
    ax.set_xlabel("Trial"); ax.set_ylabel("objetivo (MAE de validação)")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_optuna_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  salvo: fig_optuna_history.png")

    # ---- 2. Importancia dos hiperparametros ----
    imp = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=42))
    names = list(imp.keys()); vimp = list(imp.values())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(names))
    ax.barh(y[::-1], vimp, color="#2ca02c", alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_yticks(y[::-1]); ax.set_yticklabels(names)
    for yi, v in zip(y[::-1], vimp):
        ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlabel("importância relativa (fANOVA)")
    ax.set_xlim(0, max(vimp) * 1.15); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "fig_optuna_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  salvo: fig_optuna_importance.png")

    print("\n=== best_params ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
