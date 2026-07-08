"""Auditoria de duplicatas / vazamento train-test (rigor do split, Conclusao 3).

Para LRG e ELG:
  1. Duplicatas EXATAS: mesmo (plate, mjd, fiberid) aparecendo >1x.
  2. Duplicatas no CEU: mesma posicao (RA,DEC) < 1" (repeat observations, mesmo objeto).
  3. Vazamento: reproduz o split aleatorio do baseline (train_test_split seed 42,
     0.30 depois 0.50) e checa quantos pares duplicados caem em train E test.

Uso: python scripts/audit_duplicates.py
"""
import sys
from pathlib import Path

import numpy as np
import h5py
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def baseline_partition(n, seed=42):
    """Reproduz o split do baseline: idx 0..n-1 -> train/val/test."""
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    val, te = train_test_split(tmp, test_size=0.50, random_state=seed)
    return set(tr.tolist()), set(val.tolist()), set(te.tolist())


def audit(obj):
    h5 = ROOT / "data" / "processed" / obj / f"{obj}spectra_padded.h5"
    with h5py.File(h5, "r") as f:
        c = f["catalog"]
        plate = c["plate"][:]; mjd = c["mjd"][:]; fiber = c["fiberid"][:]
        ra = c["ra"][:]; dec = c["dec"][:]
    n = len(plate)
    print(f"\n===== {obj}  (N = {n:,}) =====")

    # 1. duplicatas exatas (plate, mjd, fiberid)
    keys = np.array([f"{p}-{m}-{fi}" for p, m, fi in zip(plate, mjd, fiber)])
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_dup_exact = int((counts > 1).sum())
    n_extra_exact = int((counts[counts > 1] - 1).sum())
    print(f"[1] Duplicatas EXATAS (plate,mjd,fiberid): {n_dup_exact} chaves repetidas, "
          f"{n_extra_exact} espectros extras ({100*n_extra_exact/n:.3f}% do dataset)")

    # 2. duplicatas no ceu (< 1 arcsec) — mesmo objeto observado 2x
    coo = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    i1, i2, sep, _ = search_around_sky(coo, coo, 1.0 * u.arcsec)
    m = i1 < i2  # pares unicos, exclui self (i==i)
    pairs = list(zip(i1[m].tolist(), i2[m].tolist()))
    print(f"[2] Pares no CEU < 1\": {len(pairs)} pares ({100*2*len(pairs)/n:.3f}% dos espectros envolvidos)")

    # 3. vazamento no split do baseline
    tr, val, te = baseline_partition(n)
    def side(i):
        return "tr" if i in tr else ("val" if i in val else "te")
    # exatas: pares dentro de cada grupo repetido
    leak_exact = 0
    for k in np.where(counts > 1)[0]:
        members = np.where(inv == k)[0]
        sides = {side(i) for i in members}
        if "tr" in sides and "te" in sides:
            leak_exact += 1
    # ceu:
    leak_sky = sum(1 for a, b in pairs if {side(a), side(b)} >= {"tr", "te"} or
                   ({side(a), side(b)} == {"tr", "te"}))
    leak_sky = sum(1 for a, b in pairs if (side(a) == "tr" and side(b) == "te") or
                   (side(a) == "te" and side(b) == "tr"))
    print(f"[3] VAZAMENTO train<->test no split aleatorio do baseline:")
    print(f"    exatas: {leak_exact} grupos atravessam train/test")
    print(f"    ceu<1\": {leak_sky} pares atravessam train/test")
    verdict = "LIMPO ✓" if (leak_exact == 0 and leak_sky == 0) else "TEM VAZAMENTO ⚠️"
    print(f"    -> {verdict}")
    return dict(obj=obj, n=n, dup_exact=n_extra_exact, sky_pairs=len(pairs),
                leak_exact=leak_exact, leak_sky=leak_sky)


def main():
    rows = [audit(o) for o in ["LRG", "ELG"]]
    print("\n===== RESUMO =====")
    for r in rows:
        print(f"  {r['obj']}: dup_exatas={r['dup_exact']}  pares_ceu={r['sky_pairs']}  "
              f"vazam(tr<->te) exatas={r['leak_exact']} ceu={r['leak_sky']}")


if __name__ == "__main__":
    main()
