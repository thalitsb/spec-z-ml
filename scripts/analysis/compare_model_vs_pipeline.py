"""
Compara o modelo (CNN/XGBoost) com o pipeline de redshift do SDSS (redrock/idlspec2d).

ATENCAO CONCEITUAL: o modelo foi TREINADO para reproduzir o Z do pipeline.
Logo NMAD(y_pred, Z) mede EMULACAO, nao superioridade. As comparacoes honestas sao:
  (1) emulacao   : NMAD do modelo vs Z
  (2) precisao   : residuo do modelo vs ZERR do pipeline (ZERR e' o piso de informacao)
  (3) vs verdade : modelo e pipeline vs Z_VI (so' QSO tem inspecao visual)
  + robustez     : outliers vs deltachi2 (confianca do pipeline) e vs S/N

Uso:
    python scripts/compare_model_vs_pipeline.py LRG results/LRG/cnn_baseline/predictions.npz
    python scripts/compare_model_vs_pipeline.py ELG results/ELG/cnn_baseline/predictions.npz
"""
import sys
import numpy as np
import h5py

NMAD_K = 1.4826   # sigma_NMAD = 1.4826 * mediana(|dz - mediana(dz)|) / (1+z)
C_KMS = 299792.458  # velocidade da luz, pra converter NMAD -> km/s


def nmad(dz_norm):
    return NMAD_K * np.median(np.abs(dz_norm - np.median(dz_norm)))


def main(obj, pred_path):
    # arquivo de espectros usado no treino (padded e' o que entra na CNN)
    h5_path = f"data/processed/{obj}/{obj}spectra_padded.h5"

    pred = np.load(pred_path, allow_pickle=True)
    y_pred = pred["y_pred"].astype(np.float64)
    y_test = pred["y_test"].astype(np.float64)
    test_idx = pred["test_idx"]

    with h5py.File(h5_path, "r") as f:
        c = f["catalog"]
        Z = c["redshift"][:][test_idx].astype(np.float64)           # = saida do redrock
        zerr = c["zerr"][:][test_idx].astype(np.float64) if "zerr" in c else None
        dchi2 = c["deltachi2"][:][test_idx].astype(np.float64) if "deltachi2" in c else None
        snr = c["sn_median"][:][test_idx].astype(np.float64) if "sn_median" in c else None
        zvi = c["z_vi"][:][test_idx].astype(np.float64) if "z_vi" in c else None

    # sanity: y_test do npz tem que bater com Z[test_idx]
    assert np.allclose(y_test, Z, atol=1e-4), "y_test != redshift[test_idx] (split desalinhado!)"

    res = y_pred - Z                  # residuo do modelo em reproduzir o pipeline
    dz_norm = res / (1.0 + Z)

    print(f"\n===== {obj}  (N_test = {len(Z):,}) =====")

    # (1) EMULACAO
    nm = nmad(dz_norm)
    print("\n[1] Emulacao do pipeline (modelo vs Z):")
    print(f"    sigma_NMAD       = {nm:.5f}   (~ {nm * C_KMS:.0f} km/s)")
    print(f"    bias (mediana)   = {np.median(dz_norm):+.5f}")
    print(f"    eta_0.05 (|dz/(1+z)|>0.05) = {np.mean(np.abs(dz_norm) > 0.05) * 100:.2f}%")

    # (2) PRECISAO vs ZERR — CUIDADO: ZERR e' o erro FORMAL do ajuste (curvatura do chi2),
    # otimista demais (ignora sistematicos). NAO e' um piso realista. A acuracia REAL
    # documentada e' ~300 km/s (Lyke 2020). Reportamos o ZERR so' pra evidenciar isso.
    if zerr is not None:
        ok = np.isfinite(zerr) & (zerr > 0)
        print("\n[2] Precisao vs ZERR (formal, OTIMISTA — nao e' piso realista):")
        print(f"    std(residuo)        = {np.std(res[ok]):.6f}  (~ {np.std(res[ok]) * C_KMS:.0f} km/s)")
        print(f"    mediana(ZERR)       = {np.median(zerr[ok]):.6f}  (~ {np.median(zerr[ok]) * C_KMS:.0f} km/s, erro FORMAL)")
        print(f"    acuracia real (Lyke2020) ~ 300 km/s  <- yardstick honesto")
        pull = res[ok] / zerr[ok]
        print(f"    std(pull=res/ZERR)  = {np.std(pull):.1f}   (alto pq ZERR e' otimista, NAO pq modelo e' ruim)")

    # (3) vs VERDADE INDEPENDENTE (Z_VI)
    if zvi is not None and np.sum(zvi > 0) > 100:
        v = zvi > 0
        nmad_pipe = nmad((Z[v] - zvi[v]) / (1 + zvi[v]))
        nmad_model = nmad((y_pred[v] - zvi[v]) / (1 + zvi[v]))
        print(f"\n[3] vs inspecao visual Z_VI (verdade independente, N={v.sum():,}):")
        print(f"    NMAD pipeline vs VI = {nmad_pipe:.5f}")
        print(f"    NMAD modelo   vs VI = {nmad_model:.5f}")
        print("    (modelo treina no Z, entao tende a herdar o pipeline; comparar, nao competir)")
    else:
        print("\n[3] Sem Z_VI suficiente nesta amostra (normal pra LRG/ELG).")

    # ROBUSTEZ: outliers vs deltachi2 (confianca do pipeline)
    if dchi2 is not None:
        out = np.abs(dz_norm) > 0.05
        good = np.isfinite(dchi2)
        print("\n[robustez] deltachi2 do pipeline (confianca) — outliers caem na baixa confianca?")
        print(f"    mediana(deltachi2) | acerto   = {np.median(dchi2[good & ~out]):.2f}")
        print(f"    mediana(deltachi2) | outlier  = {np.median(dchi2[good & out]):.2f}")

    # ROBUSTEZ: NMAD por bin de S/N
    if snr is not None:
        print("\n[robustez] NMAD por faixa de S/N (degradacao graciosa?):")
        edges = np.nanpercentile(snr, [0, 25, 50, 75, 100])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (snr >= lo) & (snr < hi) if hi != edges[-1] else (snr >= lo) & (snr <= hi)
            if m.sum() > 50:
                print(f"    S/N [{lo:5.1f},{hi:5.1f}): N={m.sum():6d}  NMAD={nmad(dz_norm[m]):.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
