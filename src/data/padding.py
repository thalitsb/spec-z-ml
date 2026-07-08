"""Aplica padding nos espectros para alinhar comprimentos."""
from pathlib import Path
from typing import Tuple

import numpy as np
import h5py


def pad_spectra_to_common_length(
    spectra_list: list,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Aplica padding (zeros) ate atingir target_length.

    Parameters
    ----------
    spectra_list : list of arrays
        Lista de espectros com comprimentos variaveis.
    target_length : int
        Comprimento alvo (ex: tamanho do maior espectro).
    pad_value : float
        Valor usado no padding (default 0).
    """
    n = len(spectra_list)
    out = np.full((n, target_length), pad_value, dtype=np.float32)
    for i, spec in enumerate(spectra_list):
        L = min(len(spec), target_length)
        out[i, :L] = spec[:L]
    return out


def build_padded_hdf5(
    raw_hdf5: Path,
    output_hdf5: Path,
    target_length: int = None,
) -> None:
    """Le um HDF5 com espectros de comprimento variavel e gera um padded.

    TODO: implementar de acordo com a estrutura do raw HDF5 do projeto.
    Estrutura esperada:
        raw_hdf5/spectra/{spec_id}/wave, flux, ivar
        raw_hdf5/catalog/{spec_id, redshift, ...}
    """
    raise NotImplementedError("Adapte para a estrutura especifica do seu HDF5 raw.")
