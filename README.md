# spec-z-ml

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

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
Salvo indicação, os valores são **média $\pm$ desvio-padrão sobre 5 sementes**
(protocolo multissemente, mesmo split canônico por semente → comparação pareada).

| Modelo | LRG | ELG | QSO | Papel |
|---|---|---|---|---|
| XGBoost (espectro cru) | 0.0177±0.0001 | 0.0264±0.0003 | 0.0551±0.0001 | baseline alto |
| XGBoost + PCA | 0.0283±0.0003 | 0.0392±0.0004 | 0.0528±0.0005 | baseline justificador |
| **CNN baseline** | **0.0032±0.0001** | **0.0038±0.0002** | **0.0073±0.0003** | modelo principal |
| CNN + Optuna (flex) † | 0.0016 | 0.0026 | 0.0069 | busca automática de arquitetura |
| **CNN linedet [O II]** | — | **0.0003±0.0001** | — | contribuição nova (só ELG) |

† Valor de semente única — a busca de arquitetura é cara e o multi-semente do flex
fica como trabalho futuro; ainda assim, supera o *baseline* nos três alvos.

- As CNNs superam os modelos de árvore por **~5–15×** (ex.: LRG, ~5.3k → ~0.97k km/s).
- A **busca de arquitetura com Optuna** melhora ainda mais sobre a CNN *baseline*,
  com maior ganho no LRG (~2×) e menor no QSO (~1.05×).
- No ELG, a **detecção de linha** ([O II]) bate a regressão por **~14×**, chegando a
  ~81 km/s (vs ~20 km/s do piso físico do instrumento).
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

## Licença

Distribuído sob a licença [MIT](LICENSE) — livre para uso, com atribuição.
