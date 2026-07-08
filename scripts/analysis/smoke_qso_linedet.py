"""Smoke test do MultiLineDetectionCNN (QSO multi-linha) — dado sintetico pequeno.

Verifica: build (multi-output compila), alvos batem com as saidas, fit 2 epocas,
predict (z + extras), save/load com camadas custom. Roda no cluster (precisa TF).

    python scripts/analysis/smoke_qso_linedet.py
"""
import sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models import MultiLineDetectionCNN, QSO_LINES

# Grade log-uniforme sintetica 3600-9800 A
N_WAVE = 1024
WMIN, WMAX = 3600.0, 9800.0
loglam0 = np.log10(WMIN)
dloglam = (np.log10(WMAX) - np.log10(WMIN)) / (N_WAVE - 1)

# Dado fake: z em [0.8, 2.2] (exercita presenca das 3 linhas), espectros aleatorios
rng = np.random.default_rng(0)
N = 200
y = rng.uniform(0.8, 2.2, size=N).astype(np.float32)
X = rng.normal(size=(N, N_WAVE)).astype(np.float32)

print("=== build ===")
cnn = MultiLineDetectionCNN(n_wave=N_WAVE, loglam0=loglam0, dloglam=dloglam,
                            wave_min=WMIN, wave_max=WMAX)
cnn.build()
print("  L (feature map):", cnn._L)
print("  outputs:", list(cnn.model.output_names) if hasattr(cnn.model, "output_names")
      else [o.name for o in cnn.model.outputs])

print("=== alvos batem com saidas? ===")
yt = cnn._targets(y[:10])
print("  target keys:", sorted(yt.keys()))
# presenca: MgII deve ser ~todo 1; CIV so' em z alto
for n in QSO_LINES:
    print(f"  conf_{n} presente em {int(yt[f'conf_{n}'].sum())}/10")

print("=== fit 2 epocas (tiny) ===")
cnn.fit(X[:160], y[:160], X_val=X[160:], y_val=y[160:],
        epochs=2, batch_size=32, patience_es=5, patience_lr=3, verbose=2)

print("=== predict ===")
z = cnn.predict(X[:5])
print("  z shape:", z.shape, "| z:", np.round(z, 3))
z2, hm, cf = cnn.predict(X[:5], return_extras=True)
print("  heatmaps:", {k: v.shape for k, v in hm.items()})
print("  confs:", {k: np.round(v, 2).tolist() for k, v in cf.items()})

print("=== save/load ===")
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "qso_linedet.keras"
    cnn.save(p)
    cnn2 = MultiLineDetectionCNN.load(p, wave_min=WMIN, wave_max=WMAX)
    z3 = cnn2.predict(X[:5])
    assert np.allclose(z, z3, atol=1e-4), "z divergiu apos load!"
    print("  load OK, z bate (atol 1e-4)")

print("\nSMOKE OK ✓")
