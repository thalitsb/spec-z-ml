"""Rejeicao por confianca do heatmap (cnn_linedet ELG) — o "money plot".

Carrega o modelo treinado, recalcula o heatmap no test set, mede a CONFIANCA
por objeto (largura/entropia/pico do heatmap) e mostra que rejeitar os de baixa
confianca remove os catastroficos -> recupera precisao E pureza.

Uso:
    python scripts/heatmap_confidence_elg.py

Saidas:
    results/ELG/cnn_linedet/confidence_rejection.{png,csv}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.style import set_science_style
set_science_style()

from config import paths_for, RESULTS_DIR, MODELS_DIR
from src.data import normalize_spectra
from src.models.cnn_linedet import OIILineDetectionCNN  # registra as camadas custom

C_KMS = 299792.458
NMAD_K = 1.4826


def nmad(dz):
    return NMAD_K * np.median(np.abs(dz - np.median(dz)))


def main():
    obj = "ELG"
    out_dir = RESULTS_DIR / obj / "cnn_linedet"
    paths = paths_for(obj)
    h5_path = paths["spectra_h5"].with_name(f"{obj}spectra_padded.h5")

    # 1. test set (mesmo split salvo no predictions.npz)
    pred = np.load(out_dir / "predictions.npz")
    test_idx = pred["test_idx"]
    y_test = pred["y_test"].astype(np.float64)

    with h5py.File(h5_path, "r") as f:
        X = f["ml_dataset/X_spec"][:]
    X = normalize_spectra(X)[test_idx]

    # 2. heatmap no test set
    print("Carregando modelo e recalculando heatmaps...")
    cnn = OIILineDetectionCNN.load(MODELS_DIR / obj / "cnn_linedet" / "cnn_linedet.keras")
    z_pred, hm = cnn.predict(X, return_heatmap=True)   # hm: (N, L)
    z_pred = z_pred.astype(np.float64)

    # 3. metricas de confianca por objeto (a partir da distribuicao hm)
    L = hm.shape[1]
    pos = np.arange(L)
    mean_pos = (hm * pos).sum(1)
    width = np.sqrt((hm * (pos - mean_pos[:, None]) ** 2).sum(1))   # desvio (bins) — MENOR = + confiante
    peak = hm.max(1)                                                # MAIOR = + confiante
    entropy = -(hm * np.log(hm + 1e-12)).sum(1)                     # MENOR = + confiante

    dz = (z_pred - y_test) / (1 + y_test)
    abs_dz = np.abs(dz)

    # 4. correlacao (Spearman) de cada metrica com |erro|
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]
    print("\nCorrelacao (Spearman) com |dz/(1+z)| — qual metrica melhor preve o erro:")
    print(f"   width   (largura):  {spearman(width, abs_dz):+.3f}")
    print(f"   entropy (entropia): {spearman(entropy, abs_dz):+.3f}")
    print(f"   peak    (pico):     {spearman(peak, abs_dz):+.3f}  (negativo esperado)")

    # usa a largura como confianca (menor width = mais confiante)
    conf_metric = width
    order = np.argsort(conf_metric)        # mais confiantes primeiro

    # 5. curva de rejeicao: mantem os top-f% mais confiantes
    n = len(dz)
    n_out_total = int((abs_dz > 0.05).sum())
    keep_fracs = np.array([1.0, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50])
    rows = []
    print(f"\n{'keep%':>6} {'NMAD':>9} {'km/s':>7} {'out>0.05%':>10} {'out>0.15%':>10} {'catastrof removidos':>20}")
    for fr in keep_fracs:
        k = max(10, int(round(fr * n)))
        sel = order[:k]
        nm = nmad(dz[sel])
        o05 = np.mean(np.abs(dz[sel]) > 0.05) * 100
        o15 = np.mean(np.abs(dz[sel]) > 0.15) * 100
        removed = n_out_total - int((np.abs(dz[sel]) > 0.05).sum())
        rows.append((fr * 100, nm, nm * C_KMS, o05, o15, 100 * removed / max(1, n_out_total)))
        print(f"{fr*100:6.0f} {nm:9.2e} {nm*C_KMS:7.0f} {o05:10.2f} {o15:10.2f} {100*removed/max(1,n_out_total):19.0f}%")

    import csv
    with open(out_dir / "confidence_rejection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["keep_pct", "nmad", "nmad_kms", "out_0.05_pct", "out_0.15_pct", "catastrophes_removed_pct"])
        w.writerows(rows)

    # 6. money plot
    fr_arr = np.array([r[0] for r in rows])
    nmad_arr = np.array([r[1] for r in rows])
    o05_arr = np.array([r[3] for r in rows])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(fr_arr, nmad_arr * C_KMS, "o-", color="steelblue", label="σ_NMAD (km/s)")
    ax1.axhline(20, ls=":", color="green", label="piso ELG ~20 km/s (Raichoor 2021)")
    ax1.axhline(100, ls=":", color="olive", label="eBOSS 95% < 100 km/s")
    ax1.set_xlabel("Completeza — % mantido (mais confiantes pela largura do heatmap)")
    ax1.set_ylabel("σ_NMAD (km/s)", color="steelblue")
    ax1.invert_xaxis()
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(fr_arr, o05_arr, "s--", color="crimson", label="outliers >0.05 (%)")
    ax2.set_ylabel("outliers |Δz/(1+z)|>0.05  (%)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    plt.title("cnn_linedet ELG — rejeicao por confianca do heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_rejection.png", dpi=150, bbox_inches="tight")
    print(f"\nSalvo: {out_dir/'confidence_rejection.png'} e .csv")


if __name__ == "__main__":
    main()
