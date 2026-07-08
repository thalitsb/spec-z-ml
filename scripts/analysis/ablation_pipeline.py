"""Ablation controlado: por que o pipeline do FLEX e' ~2x pior que o do BASELINE?

Mesma arquitetura (estilo baseline), variando UM fator de cada vez. Mesmo split.
Isola o culpado do gap (baseline ~0.004 vs flex ~0.007) entre:
  optimizer (Adam vs AdamW+wd), dropout (leve vs schedule pesado do flex),
  treino (epocas fixas vs early-stop), e scalars (sem vs com).

Subamostra + CPU = numeros ABSOLUTOS piores que o full, mas as DIFERENCAS
RELATIVAS entre variantes (o que queremos) se mantem.

Uso: python scripts/analysis/ablation_pipeline.py [--object LRG] [--n 15000] [--epochs 30]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import paths_for
from src.data import normalize_spectra, make_split, apply_split
from src.evaluation.metrics import sigma_nmad, delta_z_normalized


def extract_scalars(X_raw):
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.stack([np.log10(np.nanmax(X_raw, 1)), np.log10(np.nanmedian(X_raw, 1)),
                      np.log10(np.nansum(np.abs(X_raw), 1)),
                      np.log10(np.nanpercentile(X_raw, 95, axis=1))], 1)
    return np.where(np.isfinite(f), f, -40.0).astype(np.float32)


def build(n_wave, *, optimizer, weight_decay, dropout_mode, with_scalars, n_scalars=4):
    import tensorflow as tf
    from tensorflow import keras
    from src.models.cnn import ScaledSoftplus
    init = keras.initializers.GlorotUniform(seed=42)
    spec = keras.Input((n_wave, 1), name="spectrum")
    x = spec
    filters = [64, 128, 256, 128]
    kernels = [21, 15, 9, 5]
    for b in range(4):
        # dropout: 'light' = fixo baixo (baseline); 'heavy' = schedule crescente (flex)
        dr = (0.10 + 0.03 * b) if dropout_mode == "light" else 0.25 * (0.5 + 0.5 * (b + 1) / 4)
        for _ in range(2):
            x = keras.layers.Conv1D(filters[b], kernels[b], activation="relu",
                                    padding="same", kernel_initializer=init)(x)
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Dropout(dr, seed=42)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    inputs = [spec]
    if with_scalars:
        sc = keras.Input((n_scalars,), name="scalars")
        inputs.append(sc)
        x = keras.layers.Concatenate()([x, sc])

    for u in [256, 128, 64]:
        x = keras.layers.Dense(u, activation="relu", kernel_initializer=init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2 if dropout_mode == "light" else 0.3, seed=42)(x)
    out = ScaledSoftplus(beta=10.0, name="redshift")(keras.layers.Dense(1)(x))

    if optimizer == "adam":
        opt = keras.optimizers.Adam(3e-4, clipnorm=1.0)
    else:
        opt = keras.optimizers.AdamW(3e-4, weight_decay=weight_decay, clipnorm=1.0)
    m = keras.Model(inputs, out)
    m.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return m


def run_variant(name, cfg, data, epochs):
    import tensorflow as tf
    from tensorflow import keras
    Xtr, Str, ytr, Xva, Sva, yva, Xte, Ste, yte, n_wave = data
    tf.keras.backend.clear_session()
    cfg = dict(cfg)
    early = cfg.pop("early_stop", False)
    m = build(n_wave, **cfg)
    ws = cfg["with_scalars"]
    xin = lambda X, S: ({"spectrum": X.reshape(-1, n_wave, 1), "scalars": S} if ws
                        else X.reshape(-1, n_wave, 1))
    cbs = []
    if early:
        cbs.append(keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                                 monitor="val_loss"))
    t0 = time.time()
    m.fit(xin(Xtr, Str), ytr, validation_data=(xin(Xva, Sva), yva),
          epochs=epochs, batch_size=128, callbacks=cbs, verbose=0)
    yp = m.predict(xin(Xte, Ste), verbose=0).ravel()
    nm = sigma_nmad(delta_z_normalized(yte, yp))
    print(f"  {name:18s} sigma_NMAD={nm:.5f}  ({time.time()-t0:.0f}s)")
    return nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="LRG")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    h5 = paths_for(args.object)["spectra_h5"].with_name(f"{args.object}spectra_padded.h5")
    with h5py.File(h5, "r") as f:
        ntot = f["ml_dataset/X_spec"].shape[0]
        idx = np.sort(np.random.default_rng(42).choice(ntot, min(args.n, ntot), replace=False))
        X_raw = f["ml_dataset/X_spec"][idx]; y = f["catalog"]["redshift"][idx]
    S = extract_scalars(X_raw)
    # z-score dos scalars (fit no train depois); aqui normaliza espectro
    Xn = normalize_spectra(X_raw)
    n_wave = Xn.shape[1]
    tr, va, te = make_split(y, random_state=42)
    Xtr, Xva, Xte = apply_split(Xn, tr, va, te)
    Str, Sva, Ste = apply_split(S, tr, va, te)
    ytr, yva, yte = apply_split(y, tr, va, te)
    mu, sd = Str.mean(0), Str.std(0) + 1e-8
    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
    data = (Xtr, Str, ytr, Xva, Sva, yva, Xte, Ste, yte, n_wave)
    print(f"{args.object}  N={len(y)}  n_wave={n_wave}  epochs={args.epochs}  (subamostra CPU; ver DIFERENCAS relativas)")

    base = dict(optimizer="adam", weight_decay=0.0, dropout_mode="light",
                with_scalars=False, early_stop=False)
    variants = {
        "BASELINE-pipe":   base,
        "+AdamW(wd=1e-4)": {**base, "optimizer": "adamw", "weight_decay": 1e-4},
        "+dropout pesado": {**base, "dropout_mode": "heavy"},
        "+early-stop":     {**base, "early_stop": True},
        "+scalars":        {**base, "with_scalars": True},
        "FLEX-like(tudo)": dict(optimizer="adamw", weight_decay=1e-4, dropout_mode="heavy",
                                with_scalars=True, early_stop=True),
    }
    print("\nVariante            sigma_NMAD (menor=melhor)")
    res = {k: run_variant(k, v, data, args.epochs) for k, v in variants.items()}
    print("\n=== RESUMO (gap vs BASELINE-pipe) ===")
    b = res["BASELINE-pipe"]
    for k, v in res.items():
        print(f"  {k:18s} {v:.5f}   {'(ref)' if k=='BASELINE-pipe' else f'{(v/b-1)*100:+.0f}%'}")


if __name__ == "__main__":
    main()
