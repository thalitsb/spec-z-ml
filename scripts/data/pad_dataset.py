"""Gera o HDF5 padded a partir do raw `{OBJ}spectra_all.h5`.

Replica a logica provada de `notebooks/02_padding.ipynb`, mas parametrizavel por
objeto (rodar so' o QSO sem precisar do raw de ELG/LRG no cluster).

Alinha cada espectro numa grade log-lambda global comum (por indice de loglam,
nao por posicao crua) -> lida corretamente com espectros legados SDSS-I/II do QSO,
que comecam/terminam em comprimentos de onda diferentes do eBOSS.

Uso (da raiz do projeto):
    python scripts/pad_dataset.py --objects QSO
    python scripts/pad_dataset.py --objects QSO --n-test 5000   # teste rapido
    python scripts/pad_dataset.py --objects ELG LRG QSO         # todos
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import paths_for, OBJECT_TYPES  # noqa: E402

# --- Parametros (iguais ao notebook 02_padding) ---
DLOGLAM      = 1e-4
TOL_DLOGLAM  = 1e-6
WAVE_MAX_MIN = 9000.0   # descarta espectros que nao chegam ate >= 9000 A

USE_QUALITY_CUTS = True
ZWARNING_OK      = 0
ZERR_REL_MAX     = 1e-3
DELTACHI2_MIN    = 9.0
SNR_MIN          = 0.0


def _to_str(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def calcular_dloglam(wave):
    return float(np.diff(np.log10(wave)).mean())


def aplicar_padding(wave, flux, wave_global, dloglam=DLOGLAM):
    """Mapeia cada pixel na grade log-lambda global. Bordas/buracos = 0."""
    loglam_start = np.log10(wave_global[0])
    out = np.zeros(len(wave_global), dtype=np.float32)
    idx = np.round((np.log10(wave) - loglam_start) / dloglam).astype(int)
    ok = (idx >= 0) & (idx < len(wave_global))
    out[idx[ok]] = flux[ok]
    return out


def quality_mask(cat):
    n = len(cat["redshift"])
    m = np.ones(n, dtype=bool)
    if not USE_QUALITY_CUTS:
        return m
    if "zwarning" in cat:
        m &= cat["zwarning"] == ZWARNING_OK
    if "zerr" in cat:
        rel = cat["zerr"] / (1 + cat["redshift"])
        m &= np.isfinite(rel) & (rel < ZERR_REL_MAX)
    if "deltachi2" in cat:
        m &= cat["deltachi2"] > DELTACHI2_MIN
    if "sn_median" in cat:
        m &= cat["sn_median"] > SNR_MIN
    return m


def build_padded_for_object(obj, n_test=None, seed=42):
    paths = paths_for(obj)
    raw = paths["spectra_h5"]
    out = raw.with_name(f"{obj}spectra_padded.h5")
    print(f"\n{'='*60}\n[{obj}] {raw}\n   -> {out}\n{'='*60}")
    if not raw.exists():
        print(f"[{obj}] raw nao encontrado, pulando: {raw}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw, "r") as f:
        cat = {k: f["catalog"][k][:] for k in f["catalog"].keys()}
        spec_keys_all = set(f["spectra"].keys())
    spec_ids_cat = np.array([_to_str(s) for s in cat["spec_id"]])

    qmask = quality_mask(cat)
    print(f"[{obj}] cortes de qualidade: {qmask.sum():,}/{len(qmask):,} "
          f"({qmask.mean()*100:.1f}%)")

    sel_idx = np.where(qmask)[0]
    if n_test is not None and n_test < len(sel_idx):
        rng = np.random.default_rng(seed)
        sel_idx = np.sort(rng.choice(sel_idx, n_test, replace=False))
        print(f"[{obj}] MODO TESTE: {len(sel_idx):,} aleatorios")

    keys_sel = spec_ids_cat[sel_idx]
    in_h5 = np.array([k in spec_keys_all for k in keys_sel])
    if not in_h5.all():
        print(f"[{obj}] AVISO: {(~in_h5).sum()} spec_id sem espectro em /spectra/. Removendo.")
    keys_sel = keys_sel[in_h5]
    sel_idx = sel_idx[in_h5]

    print(f"[{obj}] lendo metadados de {len(keys_sel):,} espectros...")
    t0 = time.time()
    wave_min = np.zeros(len(keys_sel))
    wave_max = np.zeros(len(keys_sel))
    dlog = np.zeros(len(keys_sel))
    with h5py.File(raw, "r") as f:
        for i, k in enumerate(tqdm(keys_sel, desc=f"{obj} meta")):
            w = f[f"spectra/{k}/wave"][:]
            wave_min[i] = w[0]
            wave_max[i] = w[-1]
            dlog[i] = calcular_dloglam(w)
    print(f"[{obj}] meta lido em {time.time()-t0:.1f}s")

    m_grade = np.abs(dlog - DLOGLAM) <= TOL_DLOGLAM
    m_range = wave_max >= WAVE_MAX_MIN
    m_final = m_grade & m_range
    print(f"[{obj}] dloglam padrao    : {m_grade.sum():,} ({m_grade.mean()*100:.1f}%)")
    print(f"[{obj}] wave_max>={WAVE_MAX_MIN:.0f} A : {m_range.sum():,} ({m_range.mean()*100:.1f}%)")
    print(f"[{obj}] aprovados (final) : {m_final.sum():,} ({m_final.mean()*100:.1f}%)")

    keys_final = keys_sel[m_final]
    cat_idx_final = sel_idx[m_final]
    N = len(keys_final)
    if N == 0:
        print(f"[{obj}] nenhum espectro aprovado. Pulando.")
        return

    loglam_start = np.log10(wave_min[m_final].min())
    loglam_end = np.log10(wave_max[m_final].max())
    grade = np.arange(loglam_start, loglam_end + DLOGLAM, DLOGLAM)
    wave_global = 10 ** grade
    L = len(wave_global)
    print(f"[{obj}] grade global: [{wave_global[0]:.2f}, {wave_global[-1]:.2f}] A  L={L}")
    print(f"[{obj}] shape final do dataset: ({N}, {L})")

    t0 = time.time()
    with h5py.File(raw, "r") as f_in, h5py.File(out, "w") as f_out:
        f_out.create_dataset("wave_global", data=wave_global)
        ds_X = f_out.create_dataset(
            "ml_dataset/X_spec", shape=(N, L), dtype="float32",
            compression="gzip", chunks=(64, L),
        )
        ds_y = f_out.create_dataset("ml_dataset/y", shape=(N,), dtype="float32")

        cat_g = f_out.create_group("catalog")
        for k, arr in cat.items():
            cat_g.create_dataset(k, data=arr[cat_idx_final])

        f_out.attrs["dloglam"] = DLOGLAM
        f_out.attrs["wave_max_min"] = WAVE_MAX_MIN
        f_out.attrs["quality_cuts"] = USE_QUALITY_CUTS
        f_out.attrs["snr_min"] = SNR_MIN
        f_out.attrs["deltachi2_min"] = DELTACHI2_MIN
        f_out.attrs["zerr_rel_max"] = ZERR_REL_MAX

        for i, k in enumerate(tqdm(keys_final, desc=f"{obj} pad")):
            wave = f_in[f"spectra/{k}/wave"][:]
            flux = f_in[f"spectra/{k}/flux"][:]
            ds_X[i] = aplicar_padding(wave, flux, wave_global)
            ds_y[i] = float(cat["redshift"][cat_idx_final[i]])

    print(f"[{obj}] gravado em {(time.time()-t0)/60:.1f} min  "
          f"({os.path.getsize(out)/1024**3:.2f} GB)")

    # Verificacao rapida
    with h5py.File(out, "r") as f:
        sample = f["ml_dataset/X_spec"][:200]
        zeros_pct = (sample == 0).mean() * 100
    print(f"[{obj}] VERIF: N={N:,}  L={L}  wave=[{wave_global[0]:.0f},{wave_global[-1]:.0f}] A  "
          f"zeros(amostra)={zeros_pct:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+", default=["QSO"],
                   help="Objetos a processar (default: QSO)")
    p.add_argument("--n-test", type=int, default=None,
                   help="Limita a N espectros aleatorios (teste rapido)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for obj in args.objects:
        if obj not in OBJECT_TYPES:
            print(f"[aviso] {obj} nao esta em OBJECT_TYPES={OBJECT_TYPES}")
        build_padded_for_object(obj, n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
