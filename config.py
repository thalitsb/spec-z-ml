"""
Configuração central do projeto spec-z-ml.
Todos os caminhos e parâmetros globais ficam aqui.

USO:
    from config import paths_for, PROCESSED_DIR
    p = paths_for("ELG")
    print(p["spectra_h5"])
"""
from pathlib import Path
import os

# ============================================================
# RAIZ DO PROJETO (detectada automaticamente)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# BASE DE DADOS
#
# O dataset correto vive SEMPRE em PROJECT_ROOT/data (cluster e local).
# Prioridade:
#   1. SPECZML_DATA (env explicita) — usada pelos sbatch.
#   2. data/ do projeto, se tiver processed/.
#   3. legado /home/valerio/Thalita/dataset — APENAS fallback; o ml_dataset/y
#      la' esta' QUEBRADO (LRG todo NaN, QSO zerado). Nao usar.
#
# Para forçar um caminho:
#   export SPECZML_DATA="/outro/caminho"
# ============================================================
_CLUSTER_PATH = Path("/home/valerio/Thalita/dataset")   # LEGADO/quebrado — nao usar
_LOCAL_PATH = PROJECT_ROOT / "data"

if os.environ.get("SPECZML_DATA"):
    DATA_BASE = Path(os.environ["SPECZML_DATA"])
elif (_LOCAL_PATH / "processed").exists():
    DATA_BASE = _LOCAL_PATH
elif _CLUSTER_PATH.exists():
    DATA_BASE = _CLUSTER_PATH
else:
    DATA_BASE = _LOCAL_PATH

# ============================================================
# DIRETÓRIOS DE DADOS
# ============================================================
RAW_DIR       = DATA_BASE / "raw"
PROCESSED_DIR = DATA_BASE / "processed"
FILTERED_DIR  = DATA_BASE / "filtered"
CATALOGS_DIR  = DATA_BASE / "catalogs"

# Estes ficam DENTRO do projeto (não nos dados)
LOGS_DIR    = PROJECT_ROOT / "logs"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR  = PROJECT_ROOT / "splits"

# ============================================================
# TIPOS DE OBJETOS
# ============================================================
OBJECT_TYPES = ["ELG", "LRG", "QSO"]  # QSO: 305k apos build de 2026-06-02 (sem corte IS_QSO_FINAL)


def paths_for(obj_type: str) -> dict:
    """
    Retorna todos os caminhos para um tipo de objeto.

    Exemplo:
        p = paths_for("ELG")
        p["ngc_fits"]      # .../raw/eBOSS_ELG_clustering_data-NGC-vDR16.fits
        p["spectra_h5"]    # .../processed/ELG/ELGspectra_all.h5
        p["model_dir"]     # .../models/ELG/
    """
    obj = obj_type.upper()
    return {
        # FITS brutos
        "ngc_fits":      RAW_DIR / f"eBOSS_{obj}_clustering_data-NGC-vDR16.fits",
        "sgc_fits":      RAW_DIR / f"eBOSS_{obj}_clustering_data-SGC-vDR16.fits",
        "full_fits":     RAW_DIR / f"eBOSS_{obj}_full_ALLdata-vDR16.fits",
        "combined_fits": RAW_DIR / f"eBOSS_{obj}_clustering_combined-vDR16.fits",
        # Filtrados e processados
        "filtered_fits": FILTERED_DIR / f"filtered_eBOSS_{obj}.fits",
        "spectra_h5":    PROCESSED_DIR / obj / f"{obj}spectra_all.h5",
        "cache_dir":     PROCESSED_DIR / obj / "cache",
        # Metadados
        "metadata_csv":  CATALOGS_DIR / obj / f"{obj}spectra_metadata.csv",
        # Modelos e resultados
        "model_dir":     MODELS_DIR / obj,
        "results_dir":   RESULTS_DIR / obj,
        # Log
        "log":           LOGS_DIR / f"job{obj}.log",
    }


# ============================================================
# PARÂMETROS DO DATASET
# ============================================================
SDSS_RELEASE       = "DR17"
REDSHIFT_RANGE     = (0.0, 2.0)
WAVELENGTH_RANGE   = (3600, 9800)   # Angstroms
N_WAVELENGTH_POINTS = 1000          # Grid de reamostragem

# ============================================================
# PARÂMETROS DE TREINAMENTO (defaults)
# ============================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.20

BATCH_SIZE              = 64
EPOCHS                  = 100
LEARNING_RATE           = 1e-3
EARLY_STOPPING_PATIENCE = 10


# ============================================================
# CRIAÇÃO DE DIRETÓRIOS (chamar explicitamente quando necessário)
# ============================================================
def create_dirs():
    """Cria toda a árvore de diretórios do projeto."""
    for obj in OBJECT_TYPES:
        p = paths_for(obj)
        p["cache_dir"].mkdir(parents=True, exist_ok=True)
        p["metadata_csv"].parent.mkdir(parents=True, exist_ok=True)
        p["model_dir"].mkdir(parents=True, exist_ok=True)
        p["results_dir"].mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INFO (para debug)
# ============================================================
def print_info():
    """Mostra a configuração atual (útil para verificar no cluster)."""
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_BASE:    {DATA_BASE}")
    print(f"  → raw:       {RAW_DIR}")
    print(f"  → processed: {PROCESSED_DIR}")
    print(f"  → filtered:  {FILTERED_DIR}")
    print(f"LOGS_DIR:     {LOGS_DIR}")
    print(f"MODELS_DIR:   {MODELS_DIR}")
    print(f"RESULTS_DIR:  {RESULTS_DIR}")
    cluster = _CLUSTER_PATH.exists()
    print(f"Ambiente:     {'CLUSTER' if cluster else 'LOCAL'}")


if __name__ == "__main__":
    print_info()
