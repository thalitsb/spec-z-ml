"""Estilo unico de figuras do TCC: SciencePlots (github.com/garrettj403/SciencePlots).

Use `set_science_style()` no inicio de qualquer script/notebook que gere figura,
DEPOIS de `matplotlib.use("Agg")` e de `sys.path` apontar pra raiz do projeto.
"""
import matplotlib.pyplot as plt

_APPLIED = False


def set_science_style(force: bool = False):
    """Aplica o estilo SciencePlots ['science','no-latex','grid'].

    - `no-latex` usa mathtext (robusto, nao depende de LaTeX nem escapar '%').
    - Defensivo: se o pacote `scienceplots` nao estiver instalado (ex.: env do
      cluster), cai num fallback limpo sem quebrar o job.
    - Idempotente: so' aplica uma vez (use force=True pra reaplicar).
    """
    global _APPLIED
    if _APPLIED and not force:
        return
    try:
        import scienceplots  # noqa: F401  (registra os estilos)
        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        for fallback in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
            if fallback in plt.style.available:
                plt.style.use(fallback)
                break
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "legend.fontsize": 7,
        "axes.titlesize": 10,
    })
    _APPLIED = True
