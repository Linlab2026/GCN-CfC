import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool

from dataset import GNNDataset
from model import GCN, NodeLevelBatchNorm, TargetRepresentation


DEFAULT_DATASET = "PAD4_007_08"
DEFAULT_OUTPUT_DIR = Path("analysis") / "oversmoothing"


WARHEAD_SMARTS = [
    ("acrylamide_or_michael_acceptor", "[C,c]=[C,c]-C(=O)-[N,O,S]"),
    ("alpha_beta_unsaturated_carbonyl", "[C,c]=[C,c]-C(=O)"),
    ("chloroacetamide", "ClCC(=O)N"),
    ("haloacetamide", "[Cl,Br,I]CC(=O)N"),
    ("nitrile", "C#N"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("epoxide", "C1OC1"),
    ("sulfonyl_fluoride", "S(=O)(=O)F"),
    ("isothiocyanate", "N=C=S"),
    ("activated_ester", "C(=O)O[N,O]"),
]


def import_version(module_name):
    try:
        module = __import__(module_name)
    except Exception as exc:
        return "MISSING_OR_ERROR: {}".format(exc)
    if module_name == "rdkit":
        return rdBase.rdkitVersion
    return getattr(module, "__version__", "unknown")


def write_environment_versions(out_dir):
    versions = {
        "python": sys.version.replace("\n", " "),
        "torch": import_version("torch"),
        "torch_geometric": import_version("torch_geometric"),
        "tensorflow": import_version("tensorflow"),
        "rdkit": import_version("rdkit"),
        "numpy": import_version("numpy"),
        "sklearn": import_version("sklearn"),
        "pandas": import_version("pandas"),
        "matplotlib": import_version("matplotlib"),
    }
    path = out_dir / "environment_versions.json"
    path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return versions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict_compatible(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index)))


class PlainGraphEncoder(nn.Module):
    def __init__(self, in_channels=87, hidden_channels=96, depth=3):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(depth):
            layers.append(GraphConvBn(current, hidden_channels))
            current = hidden_channels
        self.layers = nn.ModuleList(layers)
        self.out_dim = hidden_channels

    def forward_nodes(self, data):
        x = data.x
        for layer in self.layers:
            x = layer(x, data.edge_index)
        return x

    def forward(self, data):
        nodes = self.forward_nodes(data)
        return global_mean_pool(nodes, data.batch), nodes


class PlainGCNClassifier(nn.Module):
    def __init__(
        self,
        depth,
        block_num=3,
        vocab_protein_size=26,
        embedding_size=128,
        hidden_channels=96,
        out_dim=2,
    ):
        super().__init__()
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        self.ligand_encoder = PlainGraphEncoder(87, hidden_channels, depth)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def encode(self, data):
        protein_x = self.protein_encoder(data.target)
        ligand_x, node_x = self.ligand_encoder(data)
        return torch.cat([protein_x, ligand_x], dim=-1), ligand_x, node_x

    def forward(self, data):
        graph_x, _, _ = self.encode(data)
        return self.classifier(graph_x)


def clone_batch(data):
    # PyG Data objects are mutated by the legacy model, so always operate on a clone.
    return data.clone()


def build_loaders(data_root, dataset, batch_size, max_samples=None):
    fpath = Path(data_root) / dataset
    train_set = GNNDataset(str(fpath), types="train")
    val_set = GNNDataset(str(fpath), types="val")
    test_set = GNNDataset(str(fpath), types="test")
    if max_samples is not None:
        train_set = train_set[:max_samples]
        val_set = val_set[:max_samples]
        test_set = test_set[:max_samples]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def evaluate_classifier(model, loader, device):
    model.eval()
    labels = []
    probs = []
    preds = []
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in loader:
            data = clone_batch(data).to(device)
            y = data.y.long()
            logits = model(data)
            loss = criterion(logits, y)
            prob = F.softmax(logits, dim=-1)[:, 1]
            pred = torch.argmax(logits, dim=-1)
            labels.extend(y.detach().cpu().numpy().tolist())
            probs.extend(prob.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu().item()))
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)) if labels else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)) if labels else float("nan"),
    }
    if len(set(labels)) == 2:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auc"] = float("nan")
    return metrics


def train_plain_model(depth, seed, loaders, device, epochs, lr, checkpoint_dir):
    set_seed(seed)
    model = PlainGCNClassifier(depth=depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = loaders
    best_auc = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for data in train_loader:
            data = clone_batch(data).to(device)
            y = data.y.long()
            logits = model(data)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate_classifier(model, val_loader, device)
        test_metrics = evaluate_classifier(model, test_loader, device)
        row = {
            "depth": depth,
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_f1": test_metrics["f1"],
        }
        history.append(row)
        if not math.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "plain_gcn_depth{}_seed{}.pt".format(depth, seed)
    torch.save(model.state_dict(), str(checkpoint_path))
    return model, history, checkpoint_path


def load_plain_model(depth, seed, checkpoint_dir, device):
    checkpoint_path = checkpoint_dir / "plain_gcn_depth{}_seed{}.pt".format(depth, seed)
    if not checkpoint_path.exists():
        return None
    model = PlainGCNClassifier(depth=depth).to(device)
    state = load_state_dict_compatible(str(checkpoint_path), device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_gcn_cfc_encoder(checkpoint_path, device):
    model = GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)
    if checkpoint_path:
        state = load_state_dict_compatible(checkpoint_path, device)
        model.load_state_dict(state)
    model.eval()
    return model


def encode_gcn_cfc_torch(model, data):
    data = clone_batch(data)
    protein_x = model.protein_encoder(data.target)
    ligand_x = model.ligand_encoder(data)
    graph_x = torch.cat([protein_x, ligand_x], dim=-1)
    return graph_x, ligand_x, data.x


def mean_pairwise_cosine(node_x):
    if node_x.size(0) < 2:
        return None
    z = F.normalize(node_x, p=2, dim=1, eps=1e-12)
    sim = torch.mm(z, z.t())
    n = sim.size(0)
    off_diag_sum = sim.sum() - sim.diag().sum()
    return float((off_diag_sum / (n * (n - 1))).detach().cpu().item())


def node_similarity_rows(model_name, depth_or_time, model, loader, device, raw_df, encoder_kind):
    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            batch_size = int(data.num_graphs)
            data = clone_batch(data).to(device)
            if encoder_kind == "plain":
                graph_x, ligand_x, node_x = model.encode(data)
            elif encoder_kind == "gcn_cfc":
                graph_x, ligand_x, node_x = encode_gcn_cfc_torch(model, data)
            else:
                raise ValueError("Unknown encoder kind: {}".format(encoder_kind))
            for graph_idx in range(batch_size):
                mask = data.batch == graph_idx
                nodes = node_x[mask]
                score = mean_pairwise_cosine(nodes)
                raw_idx = offset + graph_idx
                if score is None:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "depth_or_time": depth_or_time,
                        "sample_index": raw_idx,
                        "smiles": raw_df.iloc[raw_idx]["compound_iso_smiles"],
                        "label": int(raw_df.iloc[raw_idx]["affinity"]),
                        "num_nodes": int(nodes.size(0)),
                        "mean_node_cosine": score,
                        "node_dispersion": 1.0 - score,
                    }
                )
            offset += batch_size
    return rows


def summarize_node_similarity(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = (
        df.groupby(["model", "depth_or_time"])["mean_node_cosine"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["ci95_low"] = grouped["mean"] - 1.96 * grouped["sem"]
    grouped["ci95_high"] = grouped["mean"] + 1.96 * grouped["sem"]
    grouped["mean_dispersion"] = 1.0 - grouped["mean"]
    return grouped


def plot_node_similarity(summary_df, out_path):
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model_name, model_df in summary_df.groupby("model"):
        model_df = model_df.sort_values("depth_or_time")
        ax.plot(
            model_df["depth_or_time"],
            model_df["mean"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
        ax.fill_between(
            model_df["depth_or_time"].astype(float),
            model_df["ci95_low"].astype(float),
            model_df["ci95_high"].astype(float),
            alpha=0.18,
        )
    ax.set_xlabel("GCN depth (GCN-CfC shown at encoder depth 3)")
    ax.set_ylabel("Mean within-molecule node cosine similarity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def molecule_fingerprints(raw_df):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    atom_counts = []
    valid = []
    for _, row in raw_df.iterrows():
        mol = Chem.MolFromSmiles(row["compound_iso_smiles"])
        if mol is None:
            fps.append(None)
            atom_counts.append(0)
            valid.append(False)
            continue
        fps.append(generator.GetFingerprint(mol))
        atom_counts.append(mol.GetNumAtoms())
        valid.append(True)
    return fps, np.asarray(atom_counts), np.asarray(valid)


def compile_warhead_queries():
    queries = []
    for name, smarts in WARHEAD_SMARTS:
        mol = Chem.MolFromSmarts(smarts)
        if mol is not None:
            queries.append((name, mol))
    return queries


def covalent_warhead_flags(smiles, queries):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ""
    hits = [name for name, query in queries if mol.HasSubstructMatch(query)]
    return bool(hits), ";".join(hits)


def pair_candidates(raw_df, min_tanimoto, max_atom_delta):
    fps, atom_counts, valid = molecule_fingerprints(raw_df)
    warhead_queries = compile_warhead_queries()
    warhead_info = [
        covalent_warhead_flags(row["compound_iso_smiles"], warhead_queries)
        for _, row in raw_df.iterrows()
    ]
    candidates = []
    labels = raw_df["affinity"].astype(int).values
    for i in range(len(raw_df)):
        if not valid[i]:
            continue
        for j in range(i + 1, len(raw_df)):
            if not valid[j] or labels[i] == labels[j]:
                continue
            denom = max(atom_counts[i], atom_counts[j], 1)
            atom_delta = abs(atom_counts[i] - atom_counts[j]) / float(denom)
            if atom_delta > max_atom_delta:
                continue
            tanimoto = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if tanimoto >= min_tanimoto:
                candidates.append(
                    {
                        "pair_id": len(candidates),
                        "sample_i": i,
                        "sample_j": j,
                        "label_i": int(labels[i]),
                        "label_j": int(labels[j]),
                        "smiles_i": raw_df.iloc[i]["compound_iso_smiles"],
                        "smiles_j": raw_df.iloc[j]["compound_iso_smiles"],
                        "warhead_flag_i": bool(warhead_info[i][0]),
                        "warhead_flag_j": bool(warhead_info[j][0]),
                        "warhead_matches_i": warhead_info[i][1],
                        "warhead_matches_j": warhead_info[j][1],
                        "tanimoto": float(tanimoto),
                        "atom_delta_fraction": float(atom_delta),
                    }
                )
    candidates.sort(key=lambda row: row["tanimoto"], reverse=True)
    return candidates


def select_analogue_pairs(raw_df, min_tanimoto, max_atom_delta, top_k):
    tanimoto_thresholds = [min_tanimoto, 0.60, 0.55, 0.50, 0.45]
    atom_delta_thresholds = [max_atom_delta, 0.35, 0.50]
    tried = []
    best = []
    selected_thresholds = {"min_tanimoto": min_tanimoto, "max_atom_delta": max_atom_delta}
    for tanimoto_threshold in tanimoto_thresholds:
        for atom_delta_threshold in atom_delta_thresholds:
            candidates = pair_candidates(raw_df, tanimoto_threshold, atom_delta_threshold)
            tried.append(
                {
                    "min_tanimoto": tanimoto_threshold,
                    "max_atom_delta": atom_delta_threshold,
                    "candidate_count": len(candidates),
                }
            )
            if len(candidates) > len(best):
                best = candidates
                selected_thresholds = {
                    "min_tanimoto": tanimoto_threshold,
                    "max_atom_delta": atom_delta_threshold,
                }
            if len(candidates) >= top_k:
                best = candidates
                selected_thresholds = {
                    "min_tanimoto": tanimoto_threshold,
                    "max_atom_delta": atom_delta_threshold,
                }
                break
        if len(best) >= top_k:
            break
    selected = best[:top_k]
    for pair_id, row in enumerate(selected):
        row["pair_id"] = pair_id
        row["selection_min_tanimoto"] = selected_thresholds["min_tanimoto"]
        row["selection_max_atom_delta"] = selected_thresholds["max_atom_delta"]
    return selected, {"selected_thresholds": selected_thresholds, "tried_thresholds": tried}


def collect_graph_embeddings(model, loader, device, encoder_kind, raw_df):
    rows = []
    embeddings = []
    labels = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            batch_size = int(data.num_graphs)
            data = clone_batch(data).to(device)
            if encoder_kind == "plain":
                graph_x, _, _ = model.encode(data)
            elif encoder_kind == "gcn_cfc":
                graph_x, _, _ = encode_gcn_cfc_torch(model, data)
            else:
                raise ValueError("Unknown encoder kind: {}".format(encoder_kind))
            graph_np = graph_x.detach().cpu().numpy()
            for graph_idx in range(batch_size):
                raw_idx = offset + graph_idx
                embeddings.append(graph_np[graph_idx])
                label = int(raw_df.iloc[raw_idx]["affinity"])
                labels.append(label)
                rows.append(
                    {
                        "sample_index": raw_idx,
                        "smiles": raw_df.iloc[raw_idx]["compound_iso_smiles"],
                        "label": label,
                    }
                )
            offset += batch_size
    return np.asarray(embeddings), np.asarray(labels), rows


def tf_cfc_embeddings_if_available(input_embeddings, cfc_model_path, config_name):
    if not cfc_model_path:
        return None, "No --cfc-model-path supplied."
    try:
        import tensorflow as tf
    except Exception as exc:
        return None, "TensorFlow unavailable in this environment: {}".format(exc)

    sys.path.append(str(Path(__file__).resolve().parents[1] / "cfc_part"))
    try:
        from tf_cfc_2 import CfcCell, MixedCfcCell
    except Exception:
        from tf_cfc import CfcCell, MixedCfcCell

    configs = {
        "BEST_NO_GATE": {
            "clipnorm": 5,
            "optimizer": "rmsprop",
            "batch_size": 128,
            "size": 224,
            "embed_dim": 192,
            "embed_dr": 0.2,
            "epochs": 50,
            "base_lr": 0.0005,
            "decay_lr": 0.8,
            "backbone_activation": "silu",
            "backbone_dr": 0.1,
            "backbone_units": 128,
            "backbone_layers": 1,
            "weight_decay": 2.7e-05,
            "use_mixed": False,
            "no_gate": True,
            "minimal": False,
        }
    }
    config = configs[config_name]
    cell = MixedCfcCell(config["size"], config) if config["use_mixed"] else CfcCell(config["size"], config)
    inputs = tf.keras.layers.Input(shape=(input_embeddings.shape[1],))
    cell_input = tf.expand_dims(inputs, axis=1)
    states = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False, name="cfc_state")(cell_input)
    logits = tf.keras.layers.Dense(10, name="class_logits")(states)
    model = tf.keras.Model(inputs, logits)
    embedding_model = tf.keras.Model(inputs, states)
    model.load_weights(cfc_model_path)
    return embedding_model.predict(input_embeddings, verbose=0), None


def distance_metrics(embeddings, labels, pairs):
    rows = []
    if embeddings is None or len(pairs) == 0:
        return rows
    for pair in pairs:
        i = pair["sample_i"]
        j = pair["sample_j"]
        dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "sample_i": i,
                "sample_j": j,
                "label_i": pair["label_i"],
                "label_j": pair["label_j"],
                "tanimoto": pair["tanimoto"],
                "warhead_flag_i": pair["warhead_flag_i"],
                "warhead_flag_j": pair["warhead_flag_j"],
                "warhead_matches_i": pair["warhead_matches_i"],
                "warhead_matches_j": pair["warhead_matches_j"],
                "distance": dist,
            }
        )
    return rows


def plot_pair_distances(pair_distance_df, out_path):
    if pair_distance_df.empty or "distance_gcn_space" not in pair_distance_df.columns:
        return
    plot_df = pair_distance_df.sort_values("pair_id")
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(max(6.0, len(plot_df) * 0.55), 4.2))
    width = 0.36
    ax.bar(x - width / 2, plot_df["distance_gcn_space"], width=width, label="GCN space", color="#0072b2")
    if "distance_cfc_space" in plot_df.columns:
        ax.bar(x + width / 2, plot_df["distance_cfc_space"], width=width, label="GCN-CfC final", color="#d55e00")
    ax.set_xlabel("Activity-discordant analogue pair")
    ax.set_ylabel("Euclidean distance")
    ax.set_xticks(x)
    ax.set_xticklabels(["P{}".format(int(v)) for v in plot_df["pair_id"]], rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def nearest_opposite_label_distance(embeddings, labels):
    values = []
    for idx in range(len(embeddings)):
        mask = labels != labels[idx]
        if not np.any(mask):
            continue
        distances = np.linalg.norm(embeddings[mask] - embeddings[idx], axis=1)
        values.append(float(np.min(distances)))
    return float(np.mean(values)) if values else float("nan")


def latent_summary(embeddings, labels):
    if embeddings is None:
        return {}
    summary = {"nearest_opposite_label_distance": nearest_opposite_label_distance(embeddings, labels)}
    if len(set(labels.tolist())) > 1 and len(labels) > 2:
        try:
            summary["silhouette_score"] = float(silhouette_score(embeddings, labels))
        except Exception:
            summary["silhouette_score"] = float("nan")
    else:
        summary["silhouette_score"] = float("nan")
    return summary


def tsne_coordinates(embeddings, seed):
    n = len(embeddings)
    if n < 3:
        return np.zeros((n, 2), dtype=float)
    perplexity = min(30, max(2, n // 3))
    return TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(embeddings)


def plot_latent_pairs(coords, labels, pairs, title, out_path):
    if coords is None or len(coords) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = np.where(labels == 1, "#d55e00", "#0072b2")
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.55, linewidths=0)
    for pair in pairs:
        i = pair["sample_i"]
        j = pair["sample_j"]
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#333333", alpha=0.75, linewidth=1)
        ax.scatter([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], c=[colors[i], colors[j]], s=55, edgecolors="black", linewidths=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GCN/GNC-CfC over-smoothing diagnostics.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depths", default="3,5,7")
    parser.add_argument("--seeds", default="7,17,37")
    parser.add_argument("--epochs", type=int, default=0, help="Set >0 to train Plain-GCN checkpoints.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gcn-cfc-checkpoint", default=None, help="PyTorch GCN encoder checkpoint.")
    parser.add_argument("--cfc-model-path", default=None, help="Optional TensorFlow CfC .h5 weights for final embeddings.")
    parser.add_argument("--cfc-config", default="BEST_NO_GATE")
    parser.add_argument("--max-samples", type=int, default=None, help="Use a small subset for smoke testing.")
    parser.add_argument("--min-tanimoto", type=float, default=0.65)
    parser.add_argument("--max-atom-delta", type=float, default=0.25)
    parser.add_argument("--top-pairs", type=int, default=10)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    out_dir = output_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = write_environment_versions(out_dir)
    print("Environment versions written to {}".format(out_dir / "environment_versions.json"))
    if str(versions["tensorflow"]).startswith("MISSING"):
        print("TensorFlow is not available; CfC final latent embeddings will be skipped unless run in a TF-enabled environment.")

    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    device = torch.device(args.device)
    raw_path = data_root / args.dataset / "raw" / "data_test.csv"
    raw_df = pd.read_csv(raw_path)
    if args.max_samples is not None:
        raw_df = raw_df.iloc[: args.max_samples].reset_index(drop=True)

    loaders = build_loaders(str(data_root), args.dataset, args.batch_size, args.max_samples)
    _, _, test_loader = loaders
    checkpoint_dir = out_dir / "checkpoints"

    all_history = []
    similarity_rows = []
    representative_plain = None
    for depth in depths:
        for seed in seeds:
            model = None
            if args.epochs > 0:
                model, history, checkpoint_path = train_plain_model(depth, seed, loaders, device, args.epochs, args.lr, checkpoint_dir)
                all_history.extend(history)
                print("Saved {}".format(checkpoint_path))
            else:
                model = load_plain_model(depth, seed, checkpoint_dir, device)
            if model is None:
                print("Skipping Plain-GCN depth {} seed {}; no checkpoint and --epochs=0.".format(depth, seed))
                continue
            if representative_plain is None:
                representative_plain = model
            similarity_rows.extend(
                node_similarity_rows(
                    "Plain-GCN",
                    depth,
                    model,
                    test_loader,
                    device,
                    raw_df,
                    "plain",
                )
            )

    if all_history:
        pd.DataFrame(all_history).to_csv(out_dir / "plain_gcn_training_history.csv", index=False)

    gcn_cfc_model = load_gcn_cfc_encoder(args.gcn_cfc_checkpoint, device)
    similarity_rows.extend(
        node_similarity_rows(
            "GCN-CfC encoder",
            3,
            gcn_cfc_model,
            test_loader,
            device,
            raw_df,
            "gcn_cfc",
        )
    )
    pd.DataFrame(similarity_rows).to_csv(out_dir / "node_similarity.csv", index=False)
    summary = summarize_node_similarity(similarity_rows)
    summary.to_csv(out_dir / "node_similarity_summary.csv", index=False)
    plot_node_similarity(summary, out_dir / "node_similarity_curve.png")

    pairs, pair_selection_meta = select_analogue_pairs(raw_df, args.min_tanimoto, args.max_atom_delta, args.top_pairs)
    pd.DataFrame(pairs).to_csv(out_dir / "selected_analogue_pairs.csv", index=False)
    (out_dir / "pair_selection_thresholds.json").write_text(
        json.dumps(pair_selection_meta, indent=2),
        encoding="utf-8",
    )

    if representative_plain is not None:
        gcn_embeddings, labels, base_rows = collect_graph_embeddings(representative_plain, test_loader, device, "plain", raw_df)
        gcn_space_name = "Plain-GCN"
    else:
        gcn_embeddings, labels, base_rows = collect_graph_embeddings(gcn_cfc_model, test_loader, device, "gcn_cfc", raw_df)
        gcn_space_name = "GCN-CfC pre-CfC h_G"

    cfc_embeddings, cfc_skip_reason = tf_cfc_embeddings_if_available(gcn_embeddings, args.cfc_model_path, args.cfc_config)

    latent_metrics = {
        gcn_space_name: latent_summary(gcn_embeddings, labels),
    }
    pair_distance_df = pd.DataFrame(distance_metrics(gcn_embeddings, labels, pairs))
    pair_distance_df = pair_distance_df.rename(columns={"distance": "distance_gcn_space"})
    if cfc_embeddings is not None:
        latent_metrics["GCN-CfC final"] = latent_summary(cfc_embeddings, labels)
        cfc_distance_df = pd.DataFrame(distance_metrics(cfc_embeddings, labels, pairs)).rename(columns={"distance": "distance_cfc_space"})
        if not pair_distance_df.empty and not cfc_distance_df.empty:
            cfc_distance_df = cfc_distance_df[["pair_id", "distance_cfc_space"]]
            pair_distance_df = pair_distance_df.merge(cfc_distance_df, on="pair_id", how="left")
            pair_distance_df["distance_ratio_cfc_over_gcn"] = pair_distance_df["distance_cfc_space"] / pair_distance_df["distance_gcn_space"]
    else:
        latent_metrics["GCN-CfC final"] = {"skipped": cfc_skip_reason}
    (out_dir / "latent_metrics.json").write_text(json.dumps(latent_metrics, indent=2), encoding="utf-8")
    pair_distance_df.to_csv(out_dir / "pair_distance_metrics.csv", index=False)
    plot_pair_distances(pair_distance_df, out_dir / "pair_distance_comparison.png")

    coords = tsne_coordinates(gcn_embeddings, seeds[0] if seeds else 7)
    coords_rows = []
    for row, coord in zip(base_rows, coords):
        coords_rows.append({**row, "space": gcn_space_name, "x": float(coord[0]), "y": float(coord[1])})
    plot_latent_pairs(coords, labels, pairs, "{} t-SNE".format(gcn_space_name), out_dir / "latent_pairs_gcn_space.png")

    if cfc_embeddings is not None:
        cfc_coords = tsne_coordinates(cfc_embeddings, seeds[0] if seeds else 7)
        for row, coord in zip(base_rows, cfc_coords):
            coords_rows.append({**row, "space": "GCN-CfC final", "x": float(coord[0]), "y": float(coord[1])})
        plot_latent_pairs(cfc_coords, labels, pairs, "GCN-CfC final t-SNE", out_dir / "latent_pairs_cfc_space.png")

    pd.DataFrame(coords_rows).to_csv(out_dir / "latent_coordinates.csv", index=False)
    print("Analysis outputs written to {}".format(out_dir))


if __name__ == "__main__":
    main()
