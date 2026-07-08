"""Metricas de qualidade para predicao de redshift espectroscopico.

Convencoes (usadas em todos os notebooks/baselines do projeto):

- Δz_norm = (z_pred − z_true) / (1 + z_true)
- bias    = mediana(Δz_norm)
- σ_NMAD  = NMAD_K * mediana(|Δz_norm − bias|), com NMAD_K = 1.4826
            (constante de Tukey: para amostra normal, σ_NMAD == σ)
- outlier = |Δz_norm| > threshold
            threshold padrao = 0.15 (convencao photo-z, e.g. Hildebrandt+2012)
            tambem reportamos 0.05 (mais estrito) em multi_threshold_outliers().
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constante de Tukey: NMAD scaled para coincidir com sigma de uma normal.
NMAD_K = 1.4826

# Threshold padrao de outlier em fotometria de redshift.
DEFAULT_OUTLIER_THR = 0.15


def delta_z_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Δz / (1 + z_true)."""
    return (y_pred - y_true) / (1.0 + y_true)


def sigma_nmad(delta_z: np.ndarray) -> float:
    """σ_NMAD robusto = 1.4826 * MAD(Δz)."""
    bias = float(np.median(delta_z))
    return float(NMAD_K * np.median(np.abs(delta_z - bias)))


def multi_threshold_outliers(
    delta_z: np.ndarray,
    thresholds=(0.05, 0.15),
) -> dict:
    """Fracao de outliers em varios thresholds simultaneos."""
    return {f"outliers_{t:.2f}_pct": float(100.0 * np.mean(np.abs(delta_z) > t))
            for t in thresholds}


def compute_redshift_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outlier_threshold: float = DEFAULT_OUTLIER_THR,
) -> dict:
    """Metricas completas para um conjunto (treino, val ou teste).

    Returns
    -------
    dict com:
        rmse, mae, r2     : regressao basica
        bias              : mediana(Δz_norm)
        nmad              : σ_NMAD = 1.4826 * MAD(Δz_norm)
        outliers_pct      : % com |Δz_norm| > outlier_threshold (default 0.15)
        outliers_0.05_pct : % com |Δz_norm| > 0.05 (alternativa estrita)
        outliers_0.15_pct : % com |Δz_norm| > 0.15
        delta_z           : array de Δz_norm
    """
    delta_z = delta_z_normalized(y_true, y_pred)
    bias = float(np.median(delta_z))

    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "bias": bias,
        "nmad": sigma_nmad(delta_z),
        "outliers_pct": float(100.0 * np.mean(np.abs(delta_z) > outlier_threshold)),
        "delta_z": delta_z,
    }
    out.update(multi_threshold_outliers(delta_z))
    return out


def metrics_by_redshift_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    outlier_threshold: float = DEFAULT_OUTLIER_THR,
) -> dict:
    """Metricas em bins de z. Identifica em que faixa o modelo falha mais."""
    z_min, z_max = float(y_true.min()), float(y_true.max())
    edges = np.linspace(z_min, z_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    delta_z = delta_z_normalized(y_true, y_pred)
    for i in range(n_bins):
        mask = (y_true >= edges[i]) & (y_true < edges[i + 1])
        if mask.sum() < 2:
            continue
        dz = delta_z[mask]
        rows.append({
            "z_center": float(centers[i]),
            "z_low":    float(edges[i]),
            "z_high":   float(edges[i + 1]),
            "n":        int(mask.sum()),
            "bias":     float(np.median(dz)),
            "nmad":     sigma_nmad(dz),
            "outliers_pct": float(100.0 * np.mean(np.abs(dz) > outlier_threshold)),
        })
    return {"bins": rows}
