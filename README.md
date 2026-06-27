# CGC

## Project Overview

CGC is a molecular representation learning framework for compound-protein interaction prediction and PAD4 inhibitor screening. The workflow first converts molecules into graph objects using RDKit atom and bond descriptors. A graph convolutional network (GCN) extracts molecular graph embeddings, and a closed-form continuous-time (CfC) module further models latent molecular representations for binary classification and virtual screening.

The repository is organized as follows:

- `classification/`: GCN preprocessing, training, testing, prediction, and embedding export scripts.
- `cfc_part/`: CfC model definitions and training scripts for latent embedding analysis.
- `analysis/`: MoleculeNet benchmark scripts, oversmoothing analysis, exported embeddings, and result files.

## Environment Setup

The code was developed for Python with PyTorch, PyTorch Geometric, RDKit, and TensorFlow. A clean virtual environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Required Python packages are listed in `requirements.txt`:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
networkx
rdkit
torch
torch-geometric
tensorflow
bokeh
gradio
```

For GPU runs, install the PyTorch and PyTorch Geometric wheels that match the local CUDA version before running the training scripts.

## Data

The project uses public compound-protein interaction and molecular property datasets.

- **BindingDB**: compound-protein interaction records are from the public BindingDB database. The processed BindingDB-style files are used for external CPI model training and evaluation.
- **PAD4 dataset**: PAD4 activity records are curated from public PAD4 bioactivity sources, including PubChem PAD4 records and high-throughput screening records. The PAD4 data are used for noncovalent PAD4 inhibitor classification and screening.
- **MoleculeNet datasets**: benchmark datasets such as Tox21, ClinTox, ESOL, FreeSolv, and Lipophilicity are loaded through the MoleculeNet/PyTorch Geometric workflow in `analysis/moleculenet_gcn_cfc/`.

For the GCN scripts, each dataset should be placed under:

```text
classification/data/<dataset_name>/raw/data.csv
```

The expected CSV format is:

```text
compound_iso_smiles,target_sequence,affinity
```

where `compound_iso_smiles` is the ligand SMILES string, `target_sequence` is the protein sequence, and `affinity` is the binary class label.

## Usage

Run commands from the repository root unless otherwise noted.

### 1. Preprocess a CPI or PAD4 dataset

The preprocessing script uses the dataset name written at the bottom of `classification/preprocessing.py`. Before running, update these lines to match the dataset folder:

```python
data_split_train_val_test(data_root="data", data_set="PAD4_007_08")
GNNDataset(root="data/PAD4_007_08")
```

Then run:

```powershell
cd classification
python preprocessing.py
```

If the dataset has not been split yet, place `data.csv` in `classification/data/<dataset_name>/raw/` before preprocessing.

### 2. Train the GCN classifier

```powershell
cd classification
python train.py --dataset PAD4_007_08 --save_model --lr 5e-4 --batch_size 512
```

Model checkpoints and logs are saved under:

```text
classification/save/
classification/log/
```

### 3. Test a trained GCN model

```powershell
cd classification
python test.py --dataset PAD4_007_08 --model_path save/<run_name>/model/<checkpoint>.pt
```

Replace `<run_name>` and `<checkpoint>.pt` with the actual saved model directory and checkpoint file.

### 4. Run inference or screening

The automatic screening script currently reads several paths from the top of the script. Before running, edit these variables in `classification/get_predictions_auto.py`:

```python
FinalModel_1 = "1"
source_file = "data.csv"
data_dir_2 = "PAD4_033_auto_test"
```

Then place the screening CSV at `classification/data.csv` or update `source_file`, and run:

```powershell
cd classification
python get_predictions_auto.py
```

Prediction outputs are written to:

```text
classification/save_predictions/<data_dir_2>/
```

### 5. Export GCN embeddings for CfC analysis

```powershell
python classification/export_plain_gcn_embeddings.py `
  --dataset PAD4_007_08 `
  --checkpoint analysis/oversmoothing/PAD4_007_08/checkpoints/plain_gcn_depth3_seed7.pt `
  --depth 3 `
  --batch-size 512 `
  --device cuda:0 `
  --output-dir analysis/oversmoothing/PAD4_007_08/plain_gcn3_embeddings
```

Use `--device cpu` if CUDA is not available.

### 6. Train CfC on exported embeddings

```powershell
python cfc_part/train_cfc_on_embeddings.py `
  --embedding-dir analysis/oversmoothing/PAD4_007_08/plain_gcn3_embeddings `
  --pairs-csv analysis/oversmoothing/PAD4_007_08/selected_analogue_pairs.csv `
  --output-dir analysis/oversmoothing/PAD4_007_08/cfc_embeddings `
  --epochs 50 `
  --seed 7
```

### 7. Reproduce MoleculeNet benchmark runs

```powershell
python analysis/moleculenet_gcn_cfc/run_moleculenet_gcn_cfc.py `
  --datasets Tox21,ClinTox,ESOL,FreeSolv,Lipophilicity `
  --seeds 7,17,37 `
  --models GCN-CfC `
  --epochs 100 `
  --patience 20 `
  --batch-size 128 `
  --device cuda:0 `
  --output-dir analysis/moleculenet_gcn_cfc
```

The benchmark outputs are saved under:

```text
analysis/moleculenet_gcn_cfc/
```
