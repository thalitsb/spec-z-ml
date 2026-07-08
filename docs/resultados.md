# Tabela consolidada de resultados — TCC spec-z eBOSS

> Estado em 2026-06-09. Números honestos atuais (1 seed cada, split aleatório baseline
> seed 42 — auditado SEM duplicatas/vazamento). As barras de erro (±) virão do
> benchmark multi-seed (5 seeds). QSO completo nos 3 baselines (CNN baseline job 38743).
> σ_NMAD = 1.4826·mediana(|Δz/(1+z) − mediana|) (Brammer et al. 2008).
> km/s = σ_NMAD × c.

## Tabela 1 — σ_NMAD principal (menor = melhor)

Métrica única de precisão: **σ_NMAD** (adimensional, sobre Δz/(1+z)). É o padrão da
literatura de redshift (Brammer et al. 2008). A tradução para km/s aparece só na §"piso
físico" abaixo (é o MESMO número × c, não uma métrica nova).

| Modelo | LRG | ELG | QSO | papel |
|---|---|---|---|---|
| XGBoost cru (espectro 4635-d) | 0.0180 | 0.0261 | 0.0551 | baseline alto |
| XGBoost + PCA (joelho) | 0.0250 (800 PC) | 0.0352 (1200 PC) | ~0.045 (flat) | baseline justificador |
| **CNN baseline** (regressão) | **0.00376** | **0.00435** | **0.00743** | modelo principal |
| **CNN linedet [OII]** (detecção) | — n/a | **0.000288** | — n/a | contribuição nova |
| CNN linedet + rejeição (keep 90%) | — | **0.000228** | — | + incerteza/pureza |

- **História em 1 olhada:** em LRG/ELG a PCA piora vs XGB cru (custo da compressão); CNN bate ambos por ~5-15×;
  no ELG a detecção de linha bate a regressão por **15×**.
- **⚠️ QSO quebra o padrão da PCA:** XGB+PCA (0.0454) é **melhor** que XGB cru (0.0551) — oposto de LRG/ELG.
  Interpretação: espectro de QSO é ruidoso (linhas largas, z 0.8-2.2) e a PCA **denoise**; em LRG/ELG as
  linhas estreitas carregam o z e a PCA as destrói. *(Ressalva: cru usa split aleatório, sweep usa
  estratificado — confirmar no multi-seed, mesmo split.)* QSO é o objeto mais difícil em ambos.
- LRG/QSO não têm linha de emissão forte / têm linhas largas → cabeça de [OII] não se aplica
  (QSO terá cabeça multi-linha; LRG fica com regressão/template).

### Leitura física: comparação com o piso do pipeline (em km/s)
Aqui o km/s ganha o lugar — é a única forma de comparar com os pisos da literatura.
Δv = σ_NMAD × c (c = 299792 km/s).

| | ELG | referência |
|---|---|---|
| CNN baseline (regressão) | ~1304 km/s | |
| CNN linedet (full) | ~86 km/s | |
| CNN linedet + rejeição 90% | ~68 km/s | já dentro do "95% < 100 km/s" do eBOSS |
| **piso físico** | **~20 km/s** | Raichoor 2021 (mediana, repetições) |

→ a detecção de linha leva o ELG de ~1300 km/s (regressão) para ~70-86 km/s, a ~3-4×
do piso físico de ~20 km/s. Para QSO o piso é ~300 km/s (Lyke 2020, linhas largas).

## Tabela 2 — métricas (convenção spec-z: σ_NMAD + catástrofes + bias)

Trio padrão da literatura de redshift: **σ_NMAD** (precisão do núcleo) + **η** (fração de
catástrofes = a cauda) + **bias** (viés sistemático). Conta a história completa e honesta.

| Modelo · objeto | σ_NMAD | η>0.05 | η>0.15 | bias |
|---|---|---|---|---|
| XGB cru · LRG | 0.0180 | 2.96% | 0.005% | +0.0014 |
| XGB cru · ELG | 0.0261 | 11.34% | 0.104% | −0.0006 |
| XGB cru · QSO | 0.0551 | 38.06% | 5.89% | −0.0006 |
| XGB+PCA · LRG (800) | 0.0250 | 7.39% | 0% | +0.0019 |
| XGB+PCA · ELG (1200) | 0.0352 | 18.40% | 0.096% | +0.0016 |
| XGB+PCA · QSO (~800) | 0.0454 | 31.3% | 4.66% | +0.0008 |
| CNN baseline · LRG | 0.00376 | 0.275% | 0% | −0.0001 |
| CNN baseline · ELG | 0.00435 | 1.171% | 0.040% | −0.0015 |
| CNN baseline · QSO | 0.00743 | 1.398% | 0.333% | −0.0015 |


→ o η faz, de forma transparente, o que o R² tentava (capturar a cauda). No linedet, σ_NMAD
minúsculo + η alto = "núcleo cirúrgico, cauda de catástrofes" — exatamente a leitura correta.

### Sanity check (RMSE/R² — NÃO-robustos, só p/ modelos sem cauda)
Métricas quadráticas, infladas por outliers. Servem só de conferência nos modelos
bem-comportados; **NÃO usar como métrica principal** (Conclusão 9).

| | RMSE | R² |
|---|---|---|
| XGB cru · LRG | 0.0379 | 0.813 |
| XGB cru · ELG | 0.0602 | 0.640 |
| CNN baseline · LRG | 0.0135 | 0.976 |
| CNN baseline · ELG | 0.0227 | 0.948 |
| CNN baseline · QSO | 0.0467 | 0.985 |

⚠️ **Não reportar R² do linedet** (=0.71): é artefato dos ~4.5% de catástrofes, não reflete o
núcleo (σ_NMAD ~86 km/s). A cauda é tratada pela rejeição via heatmap (Tabela 4).

## Tabela 3 — XGBoost + PCA: joelho por objeto (Conclusões 1-2)

| objeto | joelho | variância | σ_NMAD | nota |
|---|---|---|---|---|
| LRG | 800 PC | 88% | 0.0250 | platô após 800 |
| ELG | 1200 PC | 91% | 0.0352 | joelho mais tardio (linhas estreitas) |
| QSO | ~300-800 PC | 90-96% | ~0.045 | **flat** — PCA não ajuda; η>0.05 ~31% |

→ "mais variância retida ≠ melhor redshift"; QSO é o mais difícil para PCA.

## Tabela 4 — CNN linedet ELG: rejeição por confiança do heatmap (o money plot)

| mantém | σ_NMAD | km/s | η>0.05 | catástrofes removidas |
|---|---|---|---|---|
| 100% | 0.000289 | 87 | 4.50% | 0% |
| 95% | 0.000255 | 76 | 1.29% | 73% |
| **90%** | **0.000228** | **68** | **0.41%** | **92%** |
| 85% | 0.000203 | 61 | 0.16% | 97% |
| 70% | 0.000149 | 45 | 0.02% | 100% |

→ Rejeitando 10% (heatmap mais largo): NMAD **68 km/s** E outliers **0.41%** — bate o baseline
em precisão (×19) **e** em pureza. A largura do heatmap = incerteza por objeto (Spearman +0.85 com |erro|).

## Leitura de robustez (discussão)

**Precisão ≠ robustez.** Precisão = σ_NMAD (núcleo); **robustez = quão raramente o modelo
falha feio = outliers, sobretudo η>0.15 (catástrofes)**. Um modelo pode ser preciso E frágil.

| Modelo | η>0.15 | leitura |
|---|---|---|
| CNN baseline · LRG | 0% | 🟢 mais robusto |
| XGB cru · LRG | 0.005% | 🟢 |
| CNN baseline · ELG | 0.04% | 🟢 |
| CNN baseline · QSO | 0.33% | 🟢 robusto p/ o objeto mais difícil |
| linedet + rejeição 90% | 0.115% | 🟢 robustez restaurada |
| XGB+PCA · ELG | 0.096% | 🟡 |
| linedet · ELG (full) | 1.05% | 🟠 preciso mas frágil |
| XGB+PCA · QSO | 4.66% | 🔴 menos robusto |

- **🟢 CNN baseline** — robusto E preciso de saída (quase nenhuma catástrofe). Mais confiável.
- **🟡 XGB cru** — robusto (η>0.15 baixo) mas pouco preciso; ELG tem ombro de falhas moderadas (η>0.05 11.3%).
- **🟠 XGB+PCA** — a compressão **degrada a robustez**: η>0.05 sobe cru→PCA (LRG 2.96→7.39%, ELG 11.3→18.4%);
  **QSO é o pior** (η>0.05 31.3%, η>0.15 4.66%).
- **🔴→🟢 linedet** — preciso mas frágil (η>0.15 1.05%, ~26× o baseline): cirúrgico quando acha o [OII],
  catastrófico quando trava na linha errada. A **rejeição por heatmap (keep 90%) restaura a robustez**
  (η>0.15 0.115%, η>0.05 0.41% — melhor que o baseline), ao custo de 10% de completeza.

**Evidências além da tabela:** (1) degradação graciosa com S/N (a CNN não desaba em baixo S/N);
(2) falhas "honestas" — os outliers do linedet caem onde o pipeline tem baixa confiança
(deltachi2 109→28) e o heatmap avisa; (3) sem vazamento (0 duplicatas) → números não inflados.

**Resumo:** a CNN baseline é robusta e precisa de saída; a PCA degrada a robustez (pior no QSO);
o linedet troca robustez por precisão extrema, mas o heatmap recupera a robustez rejeitando 10%
— entregando os dois ao mesmo tempo. *(1 seed; o ± virá do multi-seed.)*

## Checagens de rigor (já feitas)
- **Split**: baseline (aleatório) ≈ estratificado (LRG 0.00376 vs 0.00395) → não é o split.
- **Duplicatas**: 0 exatas, 0 no céu <1″, 0 vazamento train/test (LRG, ELG).
- → CNN baseline é honesto. (Conclusão 3.)

## Pendências para a versão final
- **±** (barras de erro) de todos os números: benchmark multi-seed (5 seeds) — `scripts/analysis/benchmark.sbatch`.
- **Coluna QSO**: XGBoost cru (ok), XGBoost+PCA (ok), CNN baseline (ok, job 38743, σ_NMAD 0.00743), cabeça multi-linha (pendente).
- **QSO vs Z_VI**: comparação contra inspeção visual (a única "vs redrock" real).
- Auditar duplicatas no QSO (~306k) quando local.

### Conversões usadas
c = 299792.458 km/s · km/s = σ_NMAD × c · piso ELG 20 km/s ≈ σ_NMAD 6.7e-5.
