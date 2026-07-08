# Metodologia

Resumo dos procedimentos. A premissa central é a **comparabilidade**: todos os modelos
são treinados e avaliados sobre a **mesma partição** dos dados, de modo que a métrica
final de cada arquitetura se refira exatamente ao mesmo conjunto de teste.

## 1. Dados

- **Fonte:** SDSS DR17 / eBOSS (Telescópio Sloan 2,5 m, Apache Point).
- **Alvos:** três classes, cada uma com sua física espectral que ancora o redshift:
  - **LRG** — galáxias vermelhas luminosas; contínuo vermelho + quebra em 4000 Å + linhas
    de absorção (Ca II H&K, banda G).
  - **ELG** — galáxias com linhas de emissão; ancoradas no dubleto [O II] λ3727.
  - **QSO** — quasares; linhas largas (Mg II, C III], C IV) e ampla cobertura em $z$
    ($0{,}8 \le z \le 2{,}2$).
- **Grandeza-alvo:** $z_\mathrm{spec}$ (variável `Z`), medida pelo *pipeline* oficial do
  eBOSS por ajuste de *templates*.
- **Tamanho após cortes:** ~130 952 LRG, ~166 801 ELG, ~306 005 QSO.

## 2. Seleção e qualidade

- Catálogos **limpos** de *clustering* (só alvos com medições completas e confiáveis).
- Cortes: `ZWARNING = 0` (remove redshift duvidoso) e $z_\mathrm{spec} \ge 0$.
- Aquisição dos espectros via `specutils` (identificador `PLATE`/`MJD`/`FIBERID`).

## 3. Pré-processamento

- Reamostragem em grade de comprimento de onda em **escala logarítmica**
  (passo fixo $\Delta\log_{10}\lambda = 10^{-4}$, padrão do SDSS). Cada classe tem sua
  própria grade e comprimento $N_\lambda$ (~4674 LRG, ~4635 ELG, ~4678 QSO).
- **Zero-padding** dos espectros mais curtos até o comprimento máximo da classe.
- **Normalização** individual (max-abs) para remover a escala global de fluxo.
- Ajustes (PCA, escalonamento) estimados **só no treino** e aplicados a val/teste
  (sem vazamento).

## 4. Split canônico estratificado

Um único split, **compartilhado por todos os modelos e classes**, **estratificado por
faixas de $z$** (bins por quantis) — as distribuições de $z$ de treino/val/teste ficam
estatisticamente idênticas. Esquema em dois níveis com `StratifiedShuffleSplit`
(*seed* 42): 72,25 % treino / 12,75 % validação / **15 % teste**. Índices persistidos em
disco (reprodutível). Código: [`src/data/splits.py`](../src/data/splits.py).

## 5. Modelos comparados

Em ordem crescente de especialização:

1. **XGBoost sobre o espectro cru** — *gradient boosting* de árvores sobre o vetor de
   fluxo completo. Baseline não especializado.
2. **XGBoost + PCA** — mesma ideia sobre uma representação de dimensionalidade reduzida;
   quantifica o custo da **compressão linear** (*baseline justificador*).
3. **CNN baseline** — CNN 1D sobre o espectro (arquitetura ajustada manualmente).
   **Modelo principal.**
4. **CNN + Optuna (flex)** — busca bayesiana (Optuna) escolhe **toda** a arquitetura e os
   hiperparâmetros. Duas variantes: **sem** e **com** *scalars* globais de escala.
5. **CNN linedet** — inspirada na QuasarNET: **detecta linhas espectrais** e infere $z$ da
   posição. Aplicada ao [O II] do ELG (multi-linha para QSO em desenvolvimento).

## 6. Métricas

Sobre o resíduo normalizado $\Delta z/(1+z)$, com $\Delta z = z_\mathrm{pred}-z_\mathrm{spec}$:

- **$\sigma_\mathrm{NMAD}$** (principal): $1{,}4826 \cdot \mathrm{mediana}(|\Delta z/(1+z) - \mathrm{mediana}|)$
  — dispersão robusta a *outliers* (Brammer et al. 2008).
- **$\eta$** — fração de *outliers* com $|\Delta z|/(1+z) > \delta$, para $\delta = 0{,}05$
  e $0{,}15$ (catástrofes).
- **viés** — mediana de $\Delta z/(1+z)$.
- **RMSE / $R^2$** — só como conferência (não-robustos; **não** usar como principal).

Tradução física: $\Delta v \simeq c\,\sigma_\mathrm{NMAD}$ ($c = 299792{,}458$ km/s),
para comparar com o piso imposto pela largura das linhas.

Código das métricas: [`src/evaluation/metrics.py`](../src/evaluation/metrics.py).
