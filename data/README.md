# Dados

Os dados **não** são versionados no git (dezenas de GB). Esta pasta é o ponto de montagem
esperado pelo código.

## Fonte

- **SDSS DR17 / eBOSS** — https://www.sdss.org/
- Catálogos de *clustering* (LRG, QSO, ELG) do DR16:
  `https://data.sdss.org/datamodel/files/EBOSS_LSS/catalogs/DR16/`
- Espectros (*lite*):
  `https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/`

## Como obter

1. Baixar os catálogos de *clustering* limpos das três classes.
2. Selecionar os alvos (cortes: `ZWARNING = 0`, $z \ge 0$) e montar a lista
   `PLATE`/`MJD`/`FIBERID`.
3. Baixar os espectros correspondentes via `specutils`.
4. Construir os HDF5 por classe com os notebooks `notebooks/01_build_datasets/`.

## Onde apontar

O código encontra os dados por:

```bash
export SPECZML_DATA="/caminho/para/os/dados"
```

ou pela detecção automática em `config.py`. Estrutura esperada:

```
data/
├── raw/         # FITS originais (nunca modificar)
├── processed/   # HDF5 prontos para treino (ELG/ LRG/ QSO/)
├── filtered/    # catálogos filtrados
└── catalogs/    # metadados
```
