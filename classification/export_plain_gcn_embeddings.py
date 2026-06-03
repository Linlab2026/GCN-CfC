import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader

from dataset import GNNDataset
from oversmoothing_analysis import PlainGCNClassifier, clone_batch, load_state_dict_compatible


def export_split(model, dataset_root, split, batch_size, device, output_path):
    data_set = GNNDataset(str(dataset_root), types=split)
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_path = dataset_root / "raw" / "data_{}.csv".format(split)
    raw_df = pd.read_csv(raw_path)

    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            batch_size_actual = int(data.num_graphs)
            data = clone_batch(data).to(device)
            graph_x, ligand_x, node_x = model.encode(data)
            graph_np = graph_x.detach().cpu().numpy()
            for i in range(batch_size_actual):
                raw_idx = offset + i
                row = {
                    "sample_index": raw_idx,
                    "smiles": raw_df.iloc[raw_idx]["compound_iso_smiles"],
                    "label": int(raw_df.iloc[raw_idx]["affinity"]),
                }
                for j, value in enumerate(graph_np[i]):
                    row["f{}".format(j)] = float(value)
                rows.append(row)
            offset += batch_size_actual
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PAD4_007_08")
    parser.add_argument("--data-root", default="classification/data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="analysis/oversmoothing/PAD4_007_08/plain_gcn3_embeddings")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    dataset_root = data_root / args.dataset
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = PlainGCNClassifier(depth=args.depth).to(device)
    state = load_state_dict_compatible(args.checkpoint, device)
    model.load_state_dict(state)

    for split in ["train", "val", "test"]:
        export_split(
            model=model,
            dataset_root=dataset_root,
            split=split,
            batch_size=args.batch_size,
            device=device,
            output_path=output_dir / "{}.csv".format(split),
        )
    print("Exported embeddings to {}".format(output_dir))


if __name__ == "__main__":
    main()
