"""Plots padronizados para o paper."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_predicted_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray,
    title: str = "Predito vs Real",
    save_path: Path = None,
    ax=None,
):
    """Scatter de predito vs real com colorbar e linha 1:1."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(y_true, y_pred, c=y_true, cmap="viridis", alpha=0.5, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            "r--", lw=2, label="Ideal")
    ax.set_xlabel("True Redshift")
    ax.set_ylabel("Predicted Redshift")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="True z")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_delta_z_histogram(
    delta_z: np.ndarray,
    threshold: float = 0.01,
    save_path: Path = None,
    ax=None,
):
    """Histograma de Delta_z / (1+z) com linhas de threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(delta_z, bins=80, ax=ax, color="steelblue", alpha=0.7)
    ax.axvline(0,           color="red",   ls="--", lw=1.5)
    ax.axvline( threshold,  color="black", ls="--", lw=1.0)
    ax.axvline(-threshold,  color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"$\Delta z / (1 + z)$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribuicao de $\Delta z / (1+z)$", fontweight="bold")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_metrics_by_z_bin(
    bins: list,
    metric: str = "nmad",
    save_path: Path = None,
    ax=None,
):
    """Plota metrica por bin de redshift."""
    df = pd.DataFrame(bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(data=df, x="z_center", y=metric, marker="o", ax=ax)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por bin de z", fontweight="bold")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
