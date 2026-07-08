"""Sanity check do split canonico estratificado por z.

Roda no cluster (precisa do env thalita). Verifica:
  1) Particao exata sem overlap (sem leak train/val/test).
  2) Fracoes ~ 72.25 / 12.75 / 15.0 %.
  3) Estratificacao: z_mean parecido nos 3 subsets.
  4) Determinismo/persistencia: 2a chamada carrega do .npz e bate.

Uso:
    python scripts/analysis/check_split.py --object LRG
    python scripts/analysis/check_split.py --object QSO
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import paths_for, SPLITS_DIR
from src.data import load_spectral_dataset, make_or_load_split, split_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, choices=["ELG", "LRG", "QSO"])
    args = p.parse_args()
    obj = args.object

    from config import DATA_BASE
    paths = paths_for(obj)
    hdf5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")
    print(f"[{obj}] DATA_BASE: {DATA_BASE}")
    print(f"[{obj}] Carregando: {hdf5_path}")
    _, y, _ = load_spectral_dataset(hdf5_path)
    n = len(y)

    n_nan = int(np.isnan(y).sum())
    if n_nan:
        raise ValueError(
            f"[{obj}] y tem {n_nan}/{n} NaN ({n_nan/n*100:.1f}%). "
            f"Dataset errado? Confira DATA_BASE/{hdf5_path}."
        )

    tr, va, te = make_or_load_split(obj, y, SPLITS_DIR)

    # 1) particao exata
    assert len(set(tr) | set(va) | set(te)) == n, "LEAK ou cobertura incompleta!"
    assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0, "LEAK test!"
    assert len(set(tr) & set(va)) == 0, "LEAK train/val!"

    # 2) fracoes
    f_tr, f_va, f_te = len(tr) / n, len(va) / n, len(te) / n
    print(f"  N={n:,}  train={f_tr:.3%}  val={f_va:.3%}  test={f_te:.3%}")
    assert abs(f_te - 0.15) < 0.01, "fracao de test fora do esperado"

    # 3) estratificacao
    print(f"  z_mean  train={y[tr].mean():.4f}  val={y[va].mean():.4f}  test={y[te].mean():.4f}")
    print(f"  z_std   train={y[tr].std():.4f}  val={y[va].std():.4f}  test={y[te].std():.4f}")
    assert abs(y[tr].mean() - y[te].mean()) < 0.02, "z_mean muito diferente (estratificacao?)"

    # 4) determinismo / persistencia
    z = np.load(split_path(obj, SPLITS_DIR))
    print(f"  npz: stratified={bool(z['stratified'])}  q_outer={int(z['q_outer'])}  "
          f"q_inner={int(z['q_inner'])}  seed={int(z['random_seed'])}")
    tr2, va2, te2 = make_or_load_split(obj, y, SPLITS_DIR)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2), \
        "2a chamada nao bateu (persistencia quebrada)"

    print(f"[{obj}] OK — split estratificado salvo em {split_path(obj, SPLITS_DIR)}")


if __name__ == "__main__":
    main()
