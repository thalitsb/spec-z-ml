# Pipeline — o que cada notebook faz

Os notebooks em `notebooks/` seguem uma ordem numérica, dos dados brutos à comparação
final. Todos importam o código de `src/` e usam o **split canônico** de `src/data/splits.py`.

## Dados e preparação

| Notebook | O que faz |
|---|---|
| `01_build_datasets/build_{lrg,elg,qso}.ipynb` | Constrói os datasets HDF5 por classe (download + reamostragem + metadados). |
| `02_padding.ipynb` | Zero-padding para grade de comprimento uniforme por classe. |
| `02b_data_quality.ipynb` | Checagens de qualidade (fluxos negativos, tamanhos, duplicatas). |
| `03_eda/eda_{lrg,elg,qso}.ipynb` | Análise exploratória (distribuições de $z$, espectros típicos). |
| `03_eda/analise_negativos_e_tamanho.ipynb` | Investigação de fluxos negativos e comprimentos. |

## Modelos

| Notebook | Modelo |
|---|---|
| `04_xgboost_baseline/xgb_{lrg,elg,qso}.ipynb` | XGBoost sobre o espectro cru. |
| `04_xgboost_baseline/xgb_pca_*.ipynb` | XGBoost + PCA (varredura de componentes). |
| `05_xgboost_optuna/xgb_optuna_*.ipynb` | XGBoost com Optuna (inclui variante PCA). |
| `06_cnn_baseline/cnn_{lrg,elg,qso}.ipynb` | **CNN baseline** (arquitetura manual). |
| `06_cnn_baseline/cnn_lrg_stratified.ipynb` | CNN baseline no split estratificado (checagem). |
| `07_cnn_optuna/cnn_optuna_{lrg,elg,qso}_flex.ipynb` | **CNN + Optuna** (busca de arquitetura), **sem** *scalars*. |
| `07_cnn_optuna/cnn_optuna_{lrg,elg,qso}_flex_scalars.ipynb` | Idem, **com** *scalars* globais concatenados. |
| `07_cnn_optuna/monitor_optuna.ipynb` | Monitoramento dos estudos Optuna. |

> O **CNN linedet** (detector de linhas) é treinado por script:
> `scripts/train/train_cnn_linedet.py` / `.sbatch`.

## Análise e discussão

| Notebook | O que faz |
|---|---|
| `08_error_analysis.ipynb` | Análise geral de erros. |
| `09_catastrophic_outliers.ipynb` | Investigação das catástrofes ($\eta > 0{,}15$). |
| `10_cnn_interpretability.ipynb` | Mapas de saliência (a CNN aprendeu a física das linhas). |
| `11_learning_curves.ipynb` | Curvas de aprendizado (desempenho vs tamanho de treino). |
| `12_cross_validation.ipynb` | Validação cruzada / estabilidade. |
| `13_ablation.ipynb` | Ablação de componentes da arquitetura. |
| `15_pipeline_comparison.ipynb` | Comparação modelo vs pipeline oficial. |
| `16_speed_benchmark.ipynb` | Benchmark de velocidade (inferência). |
| `17_low_snr_robustness.ipynb` | Robustez em baixo sinal-ruído. |
| `18_zwarning_recovery.ipynb` | Recuperação de objetos com `ZWARNING`. |
| `19_final_comparison.ipynb` | Comparação final consolidada (figuras do TCC). |

*(Não há notebook 14.)*
