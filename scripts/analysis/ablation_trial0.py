"""Ablacao do TRIAL 0 (receita do baseline) no pipeline do flex — LRG.

Objetivo: descobrir qual DIFERENCA de pipeline/arquitetura ainda segura a
arquitetura do baseline (~0.006 val) quando rodada no pipeline do flex (~0.0107).

Testa, num job so, a receita do baseline com cada diferenca restante ligada/desligada
e compara o val_MAE. Reusa o MESMO loader/split/normalizacao do notebook flex.

Diferencas testadas (vs o baseline manual, src/models/cnn.py):
  - dropout dos blocos conv : flex flat=0.2   vs baseline 0.1/0.15/0.2
  - dropout do head         : flex flat=0.2   vs baseline 0.3/0.2/0.1
  - BN na ultima densa      : flex sempre poe vs baseline NAO poe
E roda o PaddedSpectralCNN.fit() de verdade como "chao" (REF-baseline).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys, time, random
from pathlib import Path

import numpy as np

# ---- PROJECT_ROOT (onde tem config.py) ----
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf
import keras

from config import paths_for, SPLITS_DIR
from src.data import load_spectral_dataset, normalize_spectra, make_or_load_split
from src.models.cnn import ScaledSoftplus, PaddedSpectralCNN

OBJECT_TYPE = "LRG"
SEED = 42

# receita do baseline (= o seed do trial 0)
LR = 3e-4
BATCH = 64
EPOCHS = 80          # como no flex atual (early-stop corta)
BASELINE_CONV_DR = [0.1, 0.15, 0.2, 0.0]   # por bloco (bloco 4 = taper, sem dropout)
BASELINE_DENSE_DR = [0.3, 0.2, 0.1]


def set_reproducibility(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    keras.utils.set_random_seed(seed); tf.random.set_seed(seed)


def build_model(n_wave, seed=42, conv_dropout="flat", dense_dropout="flat",
                bn_last_dense=True):
    """Receita do baseline (4 conv + taper, head 256->128->64, relu, BN) com
    knobs pros levers restantes. conv/dense_dropout in {'flat','baseline'}."""
    init = keras.initializers.GlorotUniform(seed=seed)
    n_conv, n_dense, n_units, ksize, act, drop = 4, 3, 256, 21, "relu", 0.2
    inp = keras.Input(shape=(n_wave, 1), name="spectrum")
    x = inp
    for b in range(n_conv):
        k = max(3, ksize - 6 * b)
        if b == n_conv - 1:                       # bloco de afunilamento (taper)
            nf = min(64 * (2 ** b), 256) // 2
            x = keras.layers.Conv1D(nf, k, activation=act, padding="same",
                                    kernel_initializer=init)(x)
            x = keras.layers.BatchNormalization()(x)
            continue
        nf = min(64 * (2 ** b), 256)
        dr = BASELINE_CONV_DR[b] if conv_dropout == "baseline" else drop
        x = keras.layers.Conv1D(nf, k, activation=act, padding="same",
                                kernel_initializer=init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(nf, k, activation=act, padding="same",
                                kernel_initializer=init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Dropout(dr, seed=seed)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    units = n_units
    for d in range(n_dense):
        x = keras.layers.Dense(units, activation=act, kernel_initializer=init)(x)
        is_last = (d == n_dense - 1)
        if not (is_last and not bn_last_dense):   # baseline NAO poe BN na ultima
            x = keras.layers.BatchNormalization()(x)
        dr = BASELINE_DENSE_DR[d] if dense_dropout == "baseline" else drop
        x = keras.layers.Dropout(dr, seed=seed)(x)
        units = max(64, units // 2)
    x = keras.layers.Dense(1)(x)
    out = ScaledSoftplus(beta=10.0, name="redshift")(x)
    m = keras.Model(inp, out, name="ablation")
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LR, beta_1=0.9,
              beta_2=0.999, clipnorm=1.0), loss="mse", metrics=["mae"])
    return m


def train_flex(model, Xtr, ytr, Xva, yva):
    """Pipeline de treino do flex (ja corrigido): min_delta=0 + ReduceLR alinhado."""
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                      restore_best_weights=True, min_delta=0.0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=10, min_lr=1e-8),
    ]
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                  batch_size=BATCH, callbacks=cbs, verbose=0)
    yp = model.predict(Xva, verbose=0).ravel()
    val_mae = float(np.mean(np.abs(yp - yva)))
    best_ep = int(np.argmin(h.history["val_loss"]) + 1)
    return val_mae, best_ep


def main():
    print(f"=== ABLACAO TRIAL 0 — {OBJECT_TYPE} ===", flush=True)
    paths = paths_for(OBJECT_TYPE)
    h5 = paths["spectra_h5"].with_name(f"{OBJECT_TYPE}spectra_padded.h5")
    print(f"HDF5: {h5}", flush=True)
    X_orig, y_all, n_wave = load_spectral_dataset(h5)
    X_spec = normalize_spectra(X_orig).astype(np.float32)
    y_all = y_all.astype(np.float32)
    del X_orig

    tr, va, te = make_or_load_split(OBJECT_TYPE, y_all, SPLITS_DIR)
    Xtr = X_spec[tr].reshape(-1, n_wave, 1); ytr = y_all[tr]
    Xva = X_spec[va].reshape(-1, n_wave, 1); yva = y_all[va]
    print(f"n_wave={n_wave}  train={len(ytr)}  val={len(yva)}", flush=True)

    # (nome, kwargs do build_model)  — REF_baseline tratado a parte
    variants = [
        ("REF_flex (=trial 0)",       dict(conv_dropout="flat",     dense_dropout="flat",     bn_last_dense=True)),
        ("+conv_dropout baseline",    dict(conv_dropout="baseline", dense_dropout="flat",     bn_last_dense=True)),
        ("+dense_dropout baseline",   dict(conv_dropout="flat",     dense_dropout="baseline", bn_last_dense=True)),
        ("-BN na ultima densa",       dict(conv_dropout="flat",     dense_dropout="flat",     bn_last_dense=False)),
        ("ALL baseline-like",         dict(conv_dropout="baseline", dense_dropout="baseline", bn_last_dense=False)),
    ]

    results = []
    for name, kw in variants:
        keras.backend.clear_session()
        set_reproducibility(SEED)
        t0 = time.time()
        model = build_model(n_wave, seed=SEED, **kw)
        val_mae, best_ep = train_flex(model, Xtr, ytr, Xva, yva)
        dt = time.time() - t0
        results.append((name, val_mae, best_ep, dt))
        print(f"[{name:28s}] val_MAE={val_mae:.5f}  best_ep={best_ep:3d}  ({dt/60:.1f} min)", flush=True)

    # REF-baseline: PaddedSpectralCNN.fit() de verdade (chao real)
    keras.backend.clear_session()
    set_reproducibility(SEED)
    t0 = time.time()
    cnn = PaddedSpectralCNN(n_wave=n_wave, learning_rate=LR)
    cnn.build()
    cnn.fit(Xtr, ytr, X_val=Xva, y_val=yva, epochs=50, batch_size=BATCH,
            patience_es=20, patience_lr=10, verbose=0)
    yp = cnn.predict(Xva)
    val_mae = float(np.mean(np.abs(yp - yva)))
    dt = time.time() - t0
    results.append(("REF_baseline (fit real)", val_mae, -1, dt))
    print(f"[{'REF_baseline (fit real)':28s}] val_MAE={val_mae:.5f}  ({dt/60:.1f} min)", flush=True)

    print("\n==================== RESUMO ====================", flush=True)
    print(f"{'variante':30s} {'val_MAE':>9s} {'best_ep':>8s}", flush=True)
    for name, v, ep, dt in results:
        print(f"{name:30s} {v:9.5f} {ep:8d}", flush=True)
    print("================================================", flush=True)


if __name__ == "__main__":
    main()
