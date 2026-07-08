"""Constroi os HDF5 (raw e padded) para ELG, LRG e QSO.

TODO: integrar a logica de download e construcao do HDF5 raw que esta
no notebook 01_build_datasets/build_elg.ipynb. Mover para src/data/build_dataset.py
e chamar daqui.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import paths_for, OBJECT_TYPES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+", default=OBJECT_TYPES,
                   help="Tipos a construir (default: todos)")
    args = p.parse_args()

    for obj in args.objects:
        print(f"\n=== {obj} ===")
        paths = paths_for(obj)
        print(f"  raw HDF5 : {paths['spectra_h5']}")
        # TODO: chamar src/data/build_dataset.py
        print("  [TODO] Implementar build do HDF5 raw a partir do FITS")
        # TODO: chamar src/data/padding.py
        print("  [TODO] Implementar padding e gerar HDF5 padded")


if __name__ == "__main__":
    main()
