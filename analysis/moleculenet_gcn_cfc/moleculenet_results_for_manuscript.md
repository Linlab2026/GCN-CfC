# MoleculeNet GCN-CfC Benchmark Results

Metrics are reported as mean +/- SD over random seeds. Tox21 and ClinTox use mean ROC-AUC; ESOL, FreeSolv, and Lipophilicity use RMSE. Splits are scaffold-based 80/10/10. Reference baselines are included for context and are not strict head-to-head comparisons because implementations and splits can differ.

| Dataset | Task | Metric | GCN-CfC | GraphConv ref. | Weave ref. | MPNN ref. |
|---|---|---|---:|---:|---:|---:|
| ClinTox | classification | ROC-AUC | 0.875 +/- 0.068 | 0.845 | 0.832 | 0.879 |
| ESOL | regression | RMSE | 0.779 +/- 0.070 | 0.681 | 0.610 | 0.580 |
| FreeSolv | regression | RMSE | 1.794 +/- 0.055 | 1.150 | 1.220 | 1.150 |
| Lipophilicity | regression | RMSE | 0.721 +/- 0.007 | 0.655 | 0.715 | 0.719 |
| Tox21 | classification | ROC-AUC | 0.804 +/- 0.023 | 0.829 | 0.820 | 0.808 |

## Manuscript-Ready Text

To test whether GCN-CfC generalizes beyond CPI and PAD4 screening, we evaluated the molecule-only version of the architecture on five MoleculeNet benchmarks spanning toxicity classification and physicochemical property regression. The same graph encoder and CfC refinement hyperparameters were used across tasks, with only the output head adapted to the number and type of labels. GCN-CfC was competitive on Tox21, ClinTox, and Lipophilicity relative to the reference MoleculeNet baselines, while ESOL and FreeSolv showed weaker RMSE than the strongest reference models. These results support the use of GCN-CfC as a transferable molecular representation learner, but do not imply universal state-of-the-art performance across all molecular property endpoints.
