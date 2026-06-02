# GCN-CfC Over-Smoothing Diagnostics

This folder contains the scripts and manuscript text needed to support the over-smoothing analysis. The implementation is intentionally conservative: it does not install, upgrade, downgrade, or overwrite any package in the existing environment.

## Smoke Test

From the repository root:

```powershell
python classification\oversmoothing_analysis.py --dataset PAD4_007_08 --max-samples 8 --batch-size 4 --epochs 0 --depths 3 --seeds 7
```

This verifies dataset loading, node-embedding extraction, graph embedding extraction, plotting, and the TensorFlow-missing fallback path.

## Full Plain-GCN Diagnostic

Train Plain-GCN baselines at depths 3, 5, and 7 with three seeds:

```powershell
python classification\oversmoothing_analysis.py --dataset PAD4_007_08 --depths 3,5,7 --seeds 7,17,37 --epochs 100 --batch-size 512 --device cuda:0
```

The script writes:

- `environment_versions.json`
- `plain_gcn_training_history.csv`
- `node_similarity.csv`
- `node_similarity_summary.csv`
- `node_similarity_curve.png`
- `selected_analogue_pairs.csv`
- `pair_selection_thresholds.json`
- `latent_coordinates.csv`
- `latent_pairs_gcn_space.png`
- `latent_metrics.json`
- `pair_distance_metrics.csv`
- `pair_distance_comparison.png`

`selected_analogue_pairs.csv` includes label-discordant analogue pairs, Tanimoto similarity, atom-count difference, and SMARTS-based electrophilic/covalent-warhead annotations. If the default Tanimoto threshold does not return enough pairs, the script relaxes thresholds in predefined steps and records the actual selection rule in `pair_selection_thresholds.json`.

## GCN-CfC Encoder Checkpoint

To use a trained PyTorch encoder checkpoint for the GCN-CfC side of the node-similarity diagnostic:

```powershell
python classification\oversmoothing_analysis.py --dataset PAD4_007_08 --epochs 0 --depths 3,5,7 --seeds 7,17,37 --gcn-cfc-checkpoint "classification\FinalModel\1\model_1.pt"
```

If matching Plain-GCN checkpoints already exist in the output checkpoint folder, `--epochs 0` reuses them.

## Optional CfC Final Latent Space

The current checked environment does not have TensorFlow installed. The script therefore skips post-CfC embeddings unless it is run in an existing TensorFlow-capable environment that can load the original `.h5` CfC weights.

## Local Environment Isolation

If TensorFlow or another missing dependency must be installed later, keep it isolated inside this repository:

```powershell
cd D:\gcn_cfc
python -m venv .venv_cfc_tf
.\.venv_cfc_tf\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install any required packages only after activating `.venv_cfc_tf`, and always use `python -m pip ...` from that activated environment. Do not use global `pip`, `conda install`, `conda update`, or package commands from another environment.

To run the analysis from the isolated environment:

```powershell
cd D:\gcn_cfc
.\.venv_cfc_tf\Scripts\Activate.ps1
python classification\oversmoothing_analysis.py --dataset PAD4_007_08 --cfc-model-path "cfc_part\save\model\...\model.h5"
```

When TensorFlow is available:

```powershell
python classification\oversmoothing_analysis.py --dataset PAD4_007_08 --epochs 0 --cfc-model-path "cfc_part\save\model\...\model.h5"
```

Do not install or change TensorFlow in the existing global/conda environment just for this script. If a TensorFlow-capable environment is required, create it inside this repository folder, for example `D:\gcn_cfc\.venv_cfc_tf`, and install only into that local environment.
