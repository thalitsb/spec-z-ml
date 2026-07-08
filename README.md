# spec-z-ml

Estimativa de **redshift espectroscópico** (spec-z) diretamente de espectros do
**SDSS/eBOSS**, usando aprendizado de máquina — de XGBoost a redes neurais
convolucionais 1D.

Código do Trabalho de Conclusão de Curso de **Thalita Beninca** (Física — UFES).
A monografia (LaTeX) fica em um repositório separado.

---

## Objetivo

Prever o redshift espectroscópico ($z_\mathrm{spec}$) a partir do espectro completo do
objeto, combinando a **eficiência** dos métodos fotométricos com a **precisão** da
espectroscopia. Comparamos, sobre o **mesmo split canônico estratificado por $z$**, cinco
abordagens em três classes de alvos do eBOSS: galáxias vermelhas luminosas (**LRG**),
galáxias com linhas de emissão (**ELG**) e quasares (**QSO**).

## Resultados principais

Métrica: $\sigma_\mathrm{NMAD}$ (menor = melhor), sobre $\Delta z/(1+z)$.

| Modelo | LRG | ELG | QSO | Papel |
|---|---|---|---|---|
| XGBoost (espectro cru) | 0.0180 | 0.0261 | 0.0551 | baseline alto |
| XGBoost + PCA | 0.0250 | 0.0352 | ~0.045 | baseline justificador |
| **CNN baseline** | **0.00376** | **0.00435** | **0.00743** | modelo principal |
| **CNN linedet [O II]** | — | **0.000288** | — | contribuição nova (só ELG) |

- As CNNs superam os modelos de árvore por **~5–15×**.
- No ELG, a **detecção de linha** ([O II]) bate a regressão por **~15×**, chegando a
  ~86 km/s (vs ~20 km/s do piso físico).
- **QSO** é o objeto mais difícil; a PCA só ajuda nele (efeito de *denoising*).

Detalhes, catástrofes ($\eta$), viés e leitura física em [docs/resultados.md](docs/resultados.md).

## Estrutura do repositório

```
spec-z-ml/
├── config.py             # caminhos e parâmetros (detecta cluster vs local)
├── environment.yml       # ambiente conda
├── src/                  # código reutilizável
│   ├── data/             #   loader, padding, splits (split canônico estratificado)
│   ├── models/           #   cnn, cnn_linedet, xgboost_model
│   └── evaluation/       #   metrics, plots, style
├── notebooks/            # pipeline 01 → 19 (ver docs/pipeline.md)
├── scripts/              # jobs do cluster (sbatch) + análises
├── results/
│   ├── metrics/          # métricas por experimento (json/csv) — versionadas
│   └── figures/          # figuras do TCC — versionadas
├── docs/                 # metodologia, resultados, pipeline
└── data/                 # dados NÃO versionados (ver data/README.md)
```

## Como reproduzir

```bash
# 1. Ambiente (conda)
conda env create -f environment.yml
conda activate thalita

# 2. Dados — não vêm no repo (pesados). Ver data/README.md
#    Depois, aponte o caminho:
export SPECZML_DATA="/caminho/para/os/dados"

# 3. Conferir configuração
python config.py

# 4. Rodar o pipeline pelos notebooks (ordem numérica) ou os jobs em scripts/
```

O `config.py` detecta automaticamente se está rodando no cluster ou local; o caminho dos
dados também pode ser fixado via `SPECZML_DATA`.

## Documentação

- **[docs/metodologia.md](docs/metodologia.md)** — dados, cortes, split, modelos e métricas
- **[docs/resultados.md](docs/resultados.md)** — tabela consolidada + discussão de robustez
- **[docs/pipeline.md](docs/pipeline.md)** — o que cada notebook faz (01 → 19)

## Dados e modelos

`data/` (~93 GB) e `models/` (~245 MB) **não** são versionados. Os espectros vêm do
SDSS DR17 / eBOSS — instruções de download em [data/README.md](data/README.md). As
métricas e figuras dos resultados estão versionadas em `results/`.
