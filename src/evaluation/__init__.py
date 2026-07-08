"""Modulos de avaliacao: metricas e plots."""
from .metrics import (
    compute_redshift_metrics, delta_z_normalized, metrics_by_redshift_bin,
    sigma_nmad, multi_threshold_outliers,
    NMAD_K, DEFAULT_OUTLIER_THR,
)
from .style import set_science_style
