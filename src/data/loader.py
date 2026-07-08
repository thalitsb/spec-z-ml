"""Carrega datasets HDF5 e aplica pre-processamento basico."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import h5py


def load_spectral_dataset(
    hdf5_path: Path,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Carrega X_spec e y de um HDF5 estruturado em /ml_dataset.

    Parameters
    ----------
    hdf5_path : Path
        Caminho do arquivo HDF5.
    n_samples : int, optional
        Se fornecido, carrega apenas N amostras aleatorias.
    seed : int
        Seed para sub-amostragem reprodutivel.

    Returns
    -------
    X_spec : (N, n_pts) array
    y : (N,) array (redshift)
    n_pts : int
        Numero de pontos do espectro.
    """
    with h5py.File(hdf5_path, "r") as f:
        total = f["ml_dataset/y"].shape[0]
        n_pts = f["ml_dataset/X_spec"].shape[1]

        if n_samples and n_samples < total:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, n_samples, replace=False))
            X_spec = f["ml_dataset/X_spec"][indices]
            y = f["ml_dataset/y"][indices]
        else:
            X_spec = f["ml_dataset/X_spec"][:]
            y = f["ml_dataset/y"][:]

    return X_spec, y, n_pts


def normalize_spectra(X_spec: np.ndarray) -> np.ndarray:
    """Normaliza cada espectro pelo seu max absoluto.

    Preserva a forma das linhas espectrais e e robusta a zeros
    introduzidos por padding.
    """
    norms = np.abs(X_spec).max(axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X_spec / norms
