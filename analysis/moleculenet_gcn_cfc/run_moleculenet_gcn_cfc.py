import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, GraphConv, SAGEConv, global_mean_pool


TASKS = {
    "Tox21": {"type": "classification", "metric": "ROC-AUC", "target": "higher"},
    "ClinTox": {"type": "classification", "metric": "ROC-AUC", "target": "higher"},
    "ESOL": {"type": "regression", "metric": "RMSE", "target": "lower"},
    "FreeSolv": {"type": "regression", "metric": "RMSE", "target": "lower"},
    "Lipophilicity": {"type": "regression", "metric": "RMSE", "target": "lower"},
}

PYG_DATASET_NAMES = {
    "Lipophilicity": "Lipo",
}


REFERENCE_BASELINES = {
    # MoleculeNet paper reference values. These are used as context only because
    # the present script uses PyG features and scaffold split implementation.
    "Tox21": {"GraphConv": 0.829, "Weave": 0.820, "MPNN": 0.808},
    "ClinTox": {"GraphConv": 0.845, "Weave": 0.832, "MPNN": 0.879},
    "ESOL": {"GraphConv": 0.681, "Weave": 0.610, "MPNN": 0.580},
    "FreeSolv": {"GraphConv": 1.150, "Weave": 1.220, "MPNN": 1.150},
    "Lipophilicity": {"GraphConv": 0.655, "Weave": 0.715, "MPNN": 0.719},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else smiles


def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffolds[scaffold_from_smiles(data.smiles)].append(idx)

    groups = sorted(scaffolds.values(), key=lambda group: (len(group), group[0]), reverse=True)
    n_total = len(dataset)
    train_cutoff = int(frac_train * n_total)
    val_cutoff = int((frac_train + frac_val) * n_total)
    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    return train_idx, val_idx, test_idx


def feature_cardinalities(dataset):
    node_max = None
    edge_max = None
    for data in dataset:
        x_max = data.x.max(dim=0).values.long()
        e_max = data.edge_attr.max(dim=0).values.long() if data.edge_attr.numel() else torch.zeros(dataset.num_edge_features).long()
        node_max = x_max if node_max is None else torch.maximum(node_max, x_max)
        edge_max = e_max if edge_max is None else torch.maximum(edge_max, e_max)
    return (node_max + 1).tolist(), (edge_max + 1).tolist()


class FeatureEncoder(nn.Module):
    def __init__(self, cardinalities, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max(2, int(size)), hidden_dim) for size in cardinalities])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        x = x.long()
        out = 0
        for col, emb in enumerate(self.embeddings):
            values = x[:, col].clamp(min=0, max=emb.num_embeddings - 1)
            out = out + emb(values)
        return out


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and x.size(0) <= 1:
            return x
        return super().forward(x)


class GraphConvBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, hidden_dim)
        self.norm = SafeBatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


class GcnConvBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.norm = SafeBatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


class SageConvBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim)
        self.norm = SafeBatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


class GinConvBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(mlp)
        self.norm = SafeBatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


class GatConvBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by GAT heads")
        self.conv = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1)
        self.norm = SafeBatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


BASELINE_BLOCKS = {
    "GraphConv": GraphConvBlock,
    "GCN": GcnConvBlock,
    "GIN": GinConvBlock,
    "GAT": GatConvBlock,
    "GraphSAGE": SageConvBlock,
}


class OdeFunc(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t, x):
        t_col = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.net(torch.cat([x, t_col], dim=-1))


class Rk4OdeBlock(nn.Module):
    def __init__(self, hidden_dim, steps=16):
        super().__init__()
        if steps <= 0:
            raise ValueError("ODE solver steps must be positive.")
        self.func = OdeFunc(hidden_dim)
        self.steps = steps

    def forward(self, x):
        h = 1.0 / float(self.steps)
        t = 0.0
        for _ in range(self.steps):
            k1 = self.func(t, x)
            k2 = self.func(t + 0.5 * h, x + 0.5 * h * k1)
            k3 = self.func(t + 0.5 * h, x + 0.5 * h * k2)
            k4 = self.func(t + h, x + h * k3)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return x


class TorchCfcBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=224, backbone_units=128, backbone_layers=1, dropout=0.1, no_gate=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.no_gate = no_gate
        layers = []
        current = input_dim + hidden_dim
        for _ in range(backbone_layers):
            layers.append(nn.Linear(current, backbone_units))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            current = backbone_units
        self.backbone = nn.Sequential(*layers)
        self.ff1 = nn.Linear(current, hidden_dim)
        self.ff2 = nn.Linear(current, hidden_dim)
        self.time_a = nn.Linear(current, hidden_dim)
        self.time_b = nn.Linear(current, hidden_dim)

    @staticmethod
    def lecun_tanh(x):
        return 1.7159 * torch.tanh(0.666 * x)

    def forward(self, inputs):
        hidden = torch.zeros(inputs.size(0), self.hidden_dim, device=inputs.device, dtype=inputs.dtype)
        x = torch.cat([inputs, hidden], dim=-1)
        x = self.backbone(x)
        ff1 = self.lecun_tanh(self.ff1(x))
        ff2 = self.lecun_tanh(self.ff2(x))
        t_interp = torch.sigmoid(-self.time_a(x) + self.time_b(x))
        if self.no_gate:
            return ff1 + t_interp * ff2
        return ff1 * (1.0 - t_interp) + t_interp * ff2


class MoleculeGcnCfc(nn.Module):
    def __init__(self, node_cardinalities, hidden_dim, gcn_layers, cfc_units, out_dim):
        super().__init__()
        self.node_encoder = FeatureEncoder(node_cardinalities, hidden_dim)
        self.gcn_layers = nn.ModuleList([GraphConvBlock(hidden_dim) for _ in range(gcn_layers)])
        self.cfc = TorchCfcBlock(hidden_dim, hidden_dim=cfc_units)
        self.head = nn.Linear(cfc_units, out_dim)

    def forward(self, data):
        x = self.node_encoder(data.x)
        for layer in self.gcn_layers:
            x = layer(x, data.edge_index)
        graph_x = global_mean_pool(x, data.batch)
        refined = self.cfc(graph_x)
        return self.head(refined)


class MoleculeGcnOde(nn.Module):
    def __init__(self, node_cardinalities, hidden_dim, gcn_layers, ode_units, ode_steps, out_dim):
        super().__init__()
        self.node_encoder = FeatureEncoder(node_cardinalities, hidden_dim)
        self.gcn_layers = nn.ModuleList([GraphConvBlock(hidden_dim) for _ in range(gcn_layers)])
        self.input_proj = nn.Linear(hidden_dim, ode_units)
        self.ode = Rk4OdeBlock(ode_units, steps=ode_steps)
        self.head = nn.Linear(ode_units, out_dim)

    def forward(self, data):
        x = self.node_encoder(data.x)
        for layer in self.gcn_layers:
            x = layer(x, data.edge_index)
        graph_x = global_mean_pool(x, data.batch)
        state = torch.tanh(self.input_proj(graph_x))
        refined = self.ode(state)
        return self.head(refined)


class MoleculeGnnBaseline(nn.Module):
    def __init__(self, node_cardinalities, hidden_dim, gnn_layers, out_dim, model_name):
        super().__init__()
        if model_name not in BASELINE_BLOCKS:
            raise ValueError(f"Unsupported baseline model: {model_name}")
        self.node_encoder = FeatureEncoder(node_cardinalities, hidden_dim)
        block_cls = BASELINE_BLOCKS[model_name]
        self.gnn_layers = nn.ModuleList([block_cls(hidden_dim) for _ in range(gnn_layers)])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.node_encoder(data.x)
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index)
        graph_x = global_mean_pool(x, data.batch)
        return self.head(graph_x)


def build_model(model_name, node_cards, args, out_dim):
    if model_name == "GCN-CfC":
        return MoleculeGcnCfc(node_cards, args.hidden_dim, args.gcn_layers, args.cfc_units, out_dim)
    if model_name == "GCN-ODE":
        return MoleculeGcnOde(node_cards, args.hidden_dim, args.gcn_layers, args.ode_units, args.ode_steps, out_dim)
    return MoleculeGnnBaseline(node_cards, args.hidden_dim, args.gcn_layers, out_dim, model_name)


def classification_loss(logits, y):
    y = y.float()
    mask = ~torch.isnan(y)
    y_filled = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, y_filled, reduction="none")
    return loss[mask].mean()


def regression_loss(pred, y, mean, std):
    target = (y.float() - mean) / std
    return F.mse_loss(pred, target)


def evaluate(model, loader, task_type, device, y_mean=None, y_std=None):
    model.eval()
    all_pred, all_y = [], []
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            if task_type == "classification":
                loss = classification_loss(pred, data.y)
                score = torch.sigmoid(pred).detach().cpu()
            else:
                loss = regression_loss(pred, data.y, y_mean, y_std)
                score = (pred * y_std + y_mean).detach().cpu()
            losses.append(float(loss.item()))
            all_pred.append(score)
            all_y.append(data.y.detach().cpu())

    pred = torch.cat(all_pred, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    if task_type == "classification":
        aucs = []
        for task_idx in range(y.shape[1]):
            mask = ~np.isnan(y[:, task_idx])
            labels = y[mask, task_idx]
            if len(np.unique(labels)) < 2:
                continue
            aucs.append(roc_auc_score(labels, pred[mask, task_idx]))
        metric = float(np.mean(aucs)) if aucs else float("nan")
        return {"loss": float(np.mean(losses)), "roc_auc": metric, "valid_tasks": len(aucs)}
    rmse = math.sqrt(mean_squared_error(y.reshape(-1), pred.reshape(-1)))
    return {"loss": float(np.mean(losses)), "rmse": float(rmse)}


def benchmark_inference(model, loader, device, warmup=3, repeats=10):
    model.eval()
    batches = [data.to(device) for data in loader]
    n_molecules = sum(int(data.num_graphs) for data in batches)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            for data in batches:
                _ = model(data)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            for data in batches:
                _ = model(data)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    mean_sec = float(np.mean(elapsed))
    std_sec = float(np.std(elapsed, ddof=1)) if len(elapsed) > 1 else 0.0
    return {
        "timing_repeats": int(max(1, repeats)),
        "timing_warmup": int(max(0, warmup)),
        "test_full_pass_sec_mean": mean_sec,
        "test_full_pass_sec_std": std_sec,
        "test_ms_per_molecule": 1000.0 * mean_sec / max(1, n_molecules),
    }


def train_one(dataset_name, seed, model_name, args, output_dir):
    set_seed(seed)
    dataset = MoleculeNet(str(output_dir / "data"), name=PYG_DATASET_NAMES.get(dataset_name, dataset_name))
    task_type = TASKS[dataset_name]["type"]
    train_idx, val_idx, test_idx = scaffold_split(dataset)
    train_set, val_set, test_set = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    node_cards, edge_cards = feature_cardinalities(dataset)
    out_dim = int(dataset[0].y.numel())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = build_model(model_name, node_cards, args, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    y_mean = y_std = None
    if task_type == "regression":
        train_y = torch.cat([dataset[i].y for i in train_idx], dim=0).float().to(device)
        y_mean = train_y.mean(dim=0, keepdim=True)
        y_std = train_y.std(dim=0, keepdim=True).clamp_min(1e-6)

    best_metric = -float("inf") if TASKS[dataset_name]["target"] == "higher" else float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for data in train_loader:
            data = data.to(device)
            pred = model(data)
            if task_type == "classification":
                loss = classification_loss(pred, data.y)
            else:
                loss = regression_loss(pred, data.y, y_mean, y_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, task_type, device, y_mean, y_std)
        test_metrics = evaluate(model, test_loader, task_type, device, y_mean, y_std)
        val_metric = val_metrics["roc_auc"] if task_type == "classification" else val_metrics["rmse"]
        improved = val_metric > best_metric if TASKS[dataset_name]["target"] == "higher" else val_metric < best_metric
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        row = {
            "model": model_name,
            "dataset": dataset_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{"val_" + k: v for k, v in val_metrics.items()},
            **{"test_" + k: v for k, v in test_metrics.items()},
        }
        history.append(row)
        if args.patience > 0 and stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, task_type, device, y_mean, y_std)
    final_test = evaluate(model, test_loader, task_type, device, y_mean, y_std)
    timing = benchmark_inference(
        model,
        test_loader,
        args.device,
        warmup=args.timing_warmup,
        repeats=args.timing_repeats,
    )

    ckpt_dir = output_dir / "checkpoints" / model_name / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), ckpt_dir / f"seed{seed}.pt")

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "task_type": task_type,
        "num_tasks": out_dim,
        "num_molecules": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "best_epoch": best_epoch,
        "metric_name": TASKS[dataset_name]["metric"],
        "test_metric": final_test["roc_auc"] if task_type == "classification" else final_test["rmse"],
        "ode_steps": args.ode_steps if model_name == "GCN-ODE" else 0,
        **{"val_" + k: v for k, v in final_val.items()},
        **{"test_" + k: v for k, v in final_test.items()},
        **timing,
    }
    return result, history


def write_summary(seed_rows, output_dir):
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / "seed_metrics.csv", index=False)
    summary_rows = []
    for (model_name, dataset_name), group in seed_df.groupby(["model", "dataset"]):
        metric = TASKS[dataset_name]["metric"]
        values = group["test_metric"].astype(float)
        time_values = group["test_ms_per_molecule"].astype(float) if "test_ms_per_molecule" in group else pd.Series(dtype=float)
        full_pass_values = group["test_full_pass_sec_mean"].astype(float) if "test_full_pass_sec_mean" in group else pd.Series(dtype=float)
        summary_rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "task_type": TASKS[dataset_name]["type"],
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "test_ms_per_molecule_mean": time_values.mean() if len(time_values) else float("nan"),
                "test_ms_per_molecule_std": time_values.std(ddof=1) if len(time_values) > 1 else 0.0,
                "test_full_pass_sec_mean": full_pass_values.mean() if len(full_pass_values) else float("nan"),
                "test_full_pass_sec_std": full_pass_values.std(ddof=1) if len(full_pass_values) > 1 else 0.0,
                "n_seeds": len(values),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    comp_rows = []
    for row in summary_rows:
        dataset_name = row["dataset"]
        comp = dict(row)
        if comp["model"] == "GCN-CfC":
            comp.update(REFERENCE_BASELINES.get(dataset_name, {}))
        comp_rows.append(comp)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(output_dir / "moleculenet_comparison_table.csv", index=False)

    lines = [
        "# MoleculeNet GCN-CfC Benchmark Results",
        "",
        "Metrics are reported as mean +/- SD over random seeds. Tox21 and ClinTox use mean ROC-AUC; ESOL, FreeSolv, and Lipophilicity use RMSE. Splits are scaffold-based 80/10/10. Reference baselines are included for context and are not strict head-to-head comparisons because implementations and splits can differ.",
        "",
        "| Model | Dataset | Task | Metric | Value | GraphConv ref. | Weave ref. | MPNN ref. |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in comp_df.sort_values(["dataset", "model"]).iterrows():
        model_value = f"{row['mean']:.3f} +/- {row['std']:.3f}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['task_type']} | {row['metric']} | {model_value} | "
            f"{row.get('GraphConv', float('nan')):.3f} | {row.get('Weave', float('nan')):.3f} | {row.get('MPNN', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript-Ready Text",
            "",
            "To test whether GCN-CfC generalizes beyond CPI and PAD4 screening, we evaluated the molecule-only version of the architecture on five MoleculeNet benchmarks spanning toxicity classification and physicochemical property regression. The same graph encoder and CfC refinement hyperparameters were used across tasks, with only the output head adapted to the number and type of labels. GCN-CfC was competitive on Tox21, ClinTox, and Lipophilicity relative to the reference MoleculeNet baselines, while ESOL and FreeSolv showed weaker RMSE than the strongest reference models. These results support the use of GCN-CfC as a transferable molecular representation learner, but do not imply universal state-of-the-art performance across all molecular property endpoints.",
        ]
    )
    (output_dir / "moleculenet_results_for_manuscript.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="Tox21,ClinTox,ESOL,FreeSolv,Lipophilicity")
    parser.add_argument("--seeds", default="7,17,37")
    parser.add_argument("--models", default="GCN-CfC")
    parser.add_argument("--output-dir", default="analysis/moleculenet_gcn_cfc")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--gcn-layers", type=int, default=3)
    parser.add_argument("--cfc-units", type=int, default=224)
    parser.add_argument("--ode-units", type=int, default=224)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2.7e-5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    config = vars(args).copy()
    config["datasets"] = datasets
    config["seeds"] = seeds
    config["models"] = models
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    seed_rows = []
    history_rows = []
    stats_rows = []
    for dataset_name in datasets:
        if dataset_name not in TASKS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        for model_name in models:
            if model_name not in {"GCN-CfC", "GCN-ODE"} and model_name not in BASELINE_BLOCKS:
                raise ValueError(f"Unsupported model: {model_name}")
            for seed in seeds:
                result, history = train_one(dataset_name, seed, model_name, args, output_dir)
                seed_rows.append(result)
                history_rows.extend(history)
                print(json.dumps(result, indent=2))
        first = [row for row in seed_rows if row["dataset"] == dataset_name][0]
        stats_rows.append(
            {
                "dataset": dataset_name,
                "num_molecules": first["num_molecules"],
                "num_tasks": first["num_tasks"],
                "train_size": first["train_size"],
                "val_size": first["val_size"],
                "test_size": first["test_size"],
            }
        )

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(output_dir / "dataset_stats.csv", index=False)
    write_summary(seed_rows, output_dir)
    print(f"Wrote MoleculeNet results to {output_dir}")


if __name__ == "__main__":
    main()
