"""Train/val/test split centralizado — ESTRATIFICADO POR z.

Politica unica do projeto (replica exatamente o que a CNN flex usava):
    1) OUTER: StratifiedShuffleSplit(test_size=0.15) sobre bins de z -> pool / test
    2) INNER: StratifiedShuffleSplit(test_size=0.15) sobre bins de z do pool -> train / val

Fracoes finais aproximadas: 72.25% train / 12.75% val / 15.0% test (estratificado por z).

A estratificacao por bins de z garante que a cauda de z (alto z) fique bem
representada em todos os subsets — importante pra spec-z. Os indices sao salvos
em disco (splits/<OBJ>_split.npz) para que XGBoost, CNN flex e qualquer notebook
comparem o **mesmo conjunto de teste** maca-com-maca.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Defaults — espelham o codigo da CNN flex.
TEST_SIZE   = 0.15   # fracao do total que vira test
VAL_SIZE    = 0.15   # fracao do POOL (train+val) que vira val
RANDOM_SEED = 42


def make_strat_bins(y, n_splits: int = 2, start_q: int = 20, min_q: int = 2):
    """Bins de z robustos a duplicatas. Baixa q se algum bin tiver < n_splits.

    pd.qcut: corta y em q quantis. Cada bin tem ~mesmo numero de amostras.
        labels=False: devolve indice numerico do bin (0..q-1), nao string.
        duplicates="drop": se algum quantil cai em valor repetido, dropa o bin
                           duplicado (em vez de levantar erro). Resulta em
                           menos bins do que q pedido.

    O while-loop garante que cada bin tem amostras suficientes pra o split
    estratificado funcionar. Sem isso, com y muito desbalanceado, qcut(q=20) pode
    dar um bin com 1 amostra -> split estratificado quebra.
    """
    q = start_q
    while q >= min_q:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts()
        if (counts >= n_splits).all():
            return np.asarray(bins), q
        q -= 1
    raise ValueError("Nao consegui estratificar (y muito desbalanceado).")


def make_split(
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera indices train/val/test estratificados por bins de z.

    Retorna indices no espaco ORIGINAL de y (0..len(y)-1), nao relativos ao pool.
    """
    y = np.asarray(y)
    n = len(y)
    indices = np.arange(n)

    # OUTER — pool / test (estratificado por z)
    z_bins_all, _q_outer = make_strat_bins(y, n_splits=2, start_q=20, min_q=2)
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    pool_rel, test_rel = next(sss_outer.split(indices, z_bins_all))
    pool_idx, test_idx = indices[pool_rel], indices[test_rel]

    # INNER — train / val dentro do pool (estratificado por z do pool)
    y_pool = y[pool_idx]
    z_bins_pool, _q_inner = make_strat_bins(y_pool, n_splits=2, start_q=20, min_q=2)
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state,
    )
    train_in_pool, val_in_pool = next(sss_inner.split(pool_idx, z_bins_pool))
    train_idx = pool_idx[train_in_pool]
    val_idx   = pool_idx[val_in_pool]

    return train_idx, val_idx, test_idx


def split_path(obj: str, splits_dir: Path) -> Path:
    return splits_dir / f"{obj.upper()}_split.npz"


def save_split(
    obj: str,
    splits_dir: Path,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Path:
    """Salva os indices + metadados de rastreio da politica (estratificada)."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    p = split_path(obj, splits_dir)
    # q efetivamente usado em cada nivel (pra rastreabilidade)
    _, q_outer = make_strat_bins(np.asarray(y), n_splits=2, start_q=20, min_q=2)
    _, q_inner = make_strat_bins(np.asarray(y)[np.concatenate([train_idx, val_idx])],
                                 n_splits=2, start_q=20, min_q=2)
    np.savez(
        p,
        train=train_idx, val=val_idx, test=test_idx,
        stratified=True, q_outer=q_outer, q_inner=q_inner,
        test_size=TEST_SIZE, val_size=VAL_SIZE, random_seed=RANDOM_SEED,
    )
    return p


def load_split(obj: str, splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = split_path(obj, splits_dir)
    if not p.exists():
        raise FileNotFoundError(f"Split nao existe: {p}. Rode make_or_load_split primeiro.")
    z = np.load(p)
    return z["train"], z["val"], z["test"]


def make_or_load_split(
    obj: str,
    y: np.ndarray,
    splits_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega o split do disco se existir; caso contrario, cria e salva.

    `y` e obrigatorio porque o split e estratificado por z.
    """
    n = len(y)
    p = split_path(obj, splits_dir)
    if p.exists():
        train, val, test = load_split(obj, splits_dir)
        if len(train) + len(val) + len(test) != n:
            raise ValueError(
                f"Split em {p} tem N={len(train)+len(val)+len(test)} mas dataset tem N={n}. "
                "Apague o arquivo se o dataset mudou."
            )
        return train, val, test
    train, val, test = make_split(y)
    save_split(obj, splits_dir, y, train, val, test)
    return train, val, test


def apply_split(
    array: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atalho para fatiar X ou y nos 3 subsets."""
    return array[train_idx], array[val_idx], array[test_idx]
