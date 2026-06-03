# MoleculeNet GCN-CfC Benchmark Results

Metrics are reported as mean +/- SD over random seeds. Tox21 and ClinTox use mean ROC-AUC; ESOL, FreeSolv, and Lipophilicity use RMSE. Splits are scaffold-based 80/10/10. Reference baselines are included for context and are not strict head-to-head comparisons because implementations and splits can differ.

| Model | Dataset | Task | Metric | Value | GraphConv ref. | Weave ref. | MPNN ref. |
|---|---|---|---|---:|---:|---:|---:|
| GAT | Lipophilicity | regression | RMSE | 0.777 +/- 0.019 | nan | nan | nan |
| GCN | Lipophilicity | regression | RMSE | 0.807 +/- 0.004 | nan | nan | nan |
| GCN-CfC | Lipophilicity | regression | RMSE | 0.727 +/- 0.015 | 0.655 | 0.715 | 0.719 |
| GIN | Lipophilicity | regression | RMSE | 0.751 +/- 0.011 | nan | nan | nan |
| GraphSAGE | Lipophilicity | regression | RMSE | 0.771 +/- 0.012 | nan | nan | nan |

## Manuscript-Ready Text

To test whether GCN-CfC generalizes beyond CPI and PAD4 screening, we evaluated the molecule-only version of the architecture on five MoleculeNet benchmarks spanning toxicity classification and physicochemical property regression. The same graph encoder and CfC refinement hyperparameters were used across tasks, with only the output head adapted to the number and type of labels. GCN-CfC was competitive on Tox21, ClinTox, and Lipophilicity relative to the reference MoleculeNet baselines, while ESOL and FreeSolv showed weaker RMSE than the strongest reference models. These results support the use of GCN-CfC as a transferable molecular representation learner, but do not imply universal state-of-the-art performance across all molecular property endpoints.