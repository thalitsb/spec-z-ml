"""Modulos de carga e processamento de dados."""
from .loader import load_spectral_dataset, normalize_spectra
from .splits import (
    make_split, save_split, load_split, make_or_load_split, apply_split,
    make_strat_bins, split_path,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
)
