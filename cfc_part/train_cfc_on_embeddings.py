import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score

BEST_NO_GATE = {
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


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


class CfcCell(tf.keras.layers.Layer):
    def __init__(self, units, hparams, **kwargs):
        super(CfcCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.hparams = hparams
        self._no_gate = bool(hparams.get("no_gate", False))
        self._minimal = bool(hparams.get("minimal", False))

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        activation_name = self.hparams.get("backbone_activation")
        if activation_name == "silu":
            backbone_activation = tf.nn.silu
        elif activation_name == "relu":
            backbone_activation = tf.nn.relu
        elif activation_name == "tanh":
            backbone_activation = tf.nn.tanh
        elif activation_name == "gelu":
            backbone_activation = tf.nn.gelu
        elif activation_name == "lecun":
            backbone_activation = lecun_tanh
        elif activation_name == "softplus":
            backbone_activation = tf.nn.softplus
        else:
            raise ValueError("Unknown backbone activation: {}".format(activation_name))

        layers = []
        for layer_idx in range(self.hparams["backbone_layers"]):
            layers.append(
                tf.keras.layers.Dense(
                    self.hparams["backbone_units"],
                    activation=backbone_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                    name="backbone_{}_dense".format(layer_idx),
                )
            )
            layers.append(
                tf.keras.layers.Dropout(
                    self.hparams["backbone_dr"],
                    name="backbone_{}_dropout".format(layer_idx),
                )
            )
        self.backbone = tf.keras.Sequential(layers, name="backbone")

        if self._minimal:
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                name="ff1",
            )
            self.w_tau = self.add_weight(
                name="w_tau",
                shape=(1, self.units),
                initializer=tf.keras.initializers.Zeros(),
            )
            self.A = self.add_weight(
                name="A",
                shape=(1, self.units),
                initializer=tf.keras.initializers.Ones(),
            )
        else:
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                activation=lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                name="ff1",
            )
            self.ff2 = tf.keras.layers.Dense(
                self.units,
                activation=lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                name="ff2",
            )
            self.time_a = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                name="time_a",
            )
            self.time_b = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams["weight_decay"]),
                name="time_b",
            )
        super(CfcCell, self).build(input_shape)

    def call(self, inputs, states, **kwargs):
        hidden_state = states[0]
        t = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            t = tf.reshape(elapsed, [-1, 1])
            inputs = inputs[0]

        x = tf.concat([inputs, hidden_state], axis=-1)
        x = self.backbone(x)
        ff1 = self.ff1(x)
        if self._minimal:
            new_hidden = -self.A * tf.math.exp(-t * (tf.math.abs(self.w_tau) + tf.math.abs(ff1))) * ff1 + self.A
        else:
            ff2 = self.ff2(x)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = tf.nn.sigmoid(-t_a * t + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, [new_hidden]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_embedding_csv(path):
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col.startswith("f")]
    feature_cols = sorted(feature_cols, key=lambda value: int(value[1:]))
    x = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("int64")
    return df, x, y


def build_model(input_dim, config):
    if config["use_mixed"]:
        raise ValueError("This standalone analysis script supports CfcCell only; use_mixed must be False.")
    cell = CfcCell(units=config["size"], hparams=config)

    inputs = tf.keras.layers.Input(shape=(input_dim,), name="graph_embedding")
    cell_input = tf.keras.layers.Reshape((1, input_dim), name="as_single_step")(inputs)
    states = tf.keras.layers.RNN(cell, return_sequences=False, name="cfc_state")(cell_input)
    logits = tf.keras.layers.Dense(2, name="class_logits")(states)
    model = tf.keras.Model(inputs, logits)
    embedding_model = tf.keras.Model(inputs, states)
    return model, embedding_model


def evaluate(model, x, y):
    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
    pred = np.argmax(logits, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, probs)) if len(set(y.tolist())) == 2 else float("nan"),
    }
    return metrics, probs, pred


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
    summary = {
        "nearest_opposite_label_distance": nearest_opposite_label_distance(embeddings, labels),
    }
    if len(set(labels.tolist())) > 1 and len(labels) > 2:
        summary["silhouette_score"] = float(silhouette_score(embeddings, labels))
    else:
        summary["silhouette_score"] = float("nan")
    return summary


def pair_distances(embeddings, pairs):
    rows = []
    for _, pair in pairs.iterrows():
        i = int(pair["sample_i"])
        j = int(pair["sample_j"])
        rows.append(
            {
                "pair_id": int(pair["pair_id"]),
                "distance": float(np.linalg.norm(embeddings[i] - embeddings[j])),
            }
        )
    return pd.DataFrame(rows)


def tsne_coordinates(embeddings, seed):
    n = len(embeddings)
    perplexity = min(30, max(2, n // 3))
    return TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(embeddings)


def plot_latent_pairs(coords, labels, pairs, title, out_path):
    colors = np.where(labels == 1, "#d55e00", "#0072b2")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.55, linewidths=0)
    for _, pair in pairs.iterrows():
        i = int(pair["sample_i"])
        j = int(pair["sample_j"])
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#333333", alpha=0.75, linewidth=1)
        ax.scatter([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], c=[colors[i], colors[j]], s=55, edgecolors="black", linewidths=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pair_distances(df, out_path):
    if df.empty:
        return
    df = df.sort_values("pair_id")
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.0, len(df) * 0.55), 4.2))
    ax.bar(x - width / 2, df["distance_gcn_space"], width=width, label="GCN h_G", color="#0072b2")
    ax.bar(x + width / 2, df["distance_cfc_space"], width=width, label="GCN-CfC final", color="#d55e00")
    ax.set_xlabel("Activity-discordant analogue pair")
    ax.set_ylabel("Euclidean distance")
    ax.set_xticks(x)
    ax.set_xticklabels(["P{}".format(int(v)) for v in df["pair_id"]])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_x, train_y = read_embedding_csv(embedding_dir / "train.csv")
    val_df, val_x, val_y = read_embedding_csv(embedding_dir / "val.csv")
    test_df, test_x, test_y = read_embedding_csv(embedding_dir / "test.csv")
    pairs = pd.read_csv(args.pairs_csv)

    config = dict(BEST_NO_GATE)
    config["epochs"] = args.epochs
    config["batch_size"] = min(config["batch_size"], len(train_x))

    model, embedding_model = build_model(train_x.shape[1], config)
    train_steps = max(1, train_x.shape[0] // config["batch_size"])
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        config["base_lr"],
        train_steps,
        config["decay_lr"],
    )
    optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        train_x,
        train_y,
        batch_size=config["batch_size"],
        epochs=args.epochs,
        validation_data=(val_x, val_y),
        verbose=2,
    )

    test_metrics, test_probs, test_pred = evaluate(model, test_x, test_y)
    pre_embeddings = test_x
    cfc_embeddings = embedding_model.predict(test_x, verbose=0)

    metrics = {
        "test_metrics": test_metrics,
        "pre_cfc_latent": latent_summary(pre_embeddings, test_y),
        "post_cfc_latent": latent_summary(cfc_embeddings, test_y),
        "seed": args.seed,
        "epochs": args.epochs,
        "input_dim": int(train_x.shape[1]),
        "cfc_units": int(config["size"]),
    }
    (output_dir / "cfc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(output_dir / "cfc_training_history.csv", index=False)

    pred_df = test_df[["sample_index", "smiles", "label"]].copy()
    pred_df["cfc_prob_positive"] = test_probs
    pred_df["cfc_pred"] = test_pred
    pred_df.to_csv(output_dir / "cfc_test_predictions.csv", index=False)

    emb_df = test_df[["sample_index", "smiles", "label"]].copy()
    for i in range(cfc_embeddings.shape[1]):
        emb_df["cfc_f{}".format(i)] = cfc_embeddings[:, i]
    emb_df.to_csv(output_dir / "cfc_test_embeddings.csv", index=False)

    gcn_dist = pair_distances(pre_embeddings, pairs).rename(columns={"distance": "distance_gcn_space"})
    cfc_dist = pair_distances(cfc_embeddings, pairs).rename(columns={"distance": "distance_cfc_space"})
    pair_dist = pairs.merge(gcn_dist, on="pair_id", how="left").merge(cfc_dist, on="pair_id", how="left")
    pair_dist["distance_ratio_cfc_over_gcn"] = pair_dist["distance_cfc_space"] / pair_dist["distance_gcn_space"]
    pair_dist.to_csv(output_dir / "pair_distance_metrics_with_cfc.csv", index=False)
    plot_pair_distances(pair_dist, output_dir / "pair_distance_comparison_with_cfc.png")

    pre_coords = tsne_coordinates(pre_embeddings, args.seed)
    post_coords = tsne_coordinates(cfc_embeddings, args.seed)
    plot_latent_pairs(pre_coords, test_y, pairs, "GCN h_G t-SNE", output_dir / "latent_pairs_pre_cfc.png")
    plot_latent_pairs(post_coords, test_y, pairs, "GCN-CfC final t-SNE", output_dir / "latent_pairs_post_cfc.png")

    coord_rows = []
    for row_idx, row in test_df.iterrows():
        coord_rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "smiles": row["smiles"],
                "label": int(row["label"]),
                "space": "GCN h_G",
                "x": float(pre_coords[row_idx, 0]),
                "y": float(pre_coords[row_idx, 1]),
            }
        )
        coord_rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "smiles": row["smiles"],
                "label": int(row["label"]),
                "space": "GCN-CfC final",
                "x": float(post_coords[row_idx, 0]),
                "y": float(post_coords[row_idx, 1]),
            }
        )
    pd.DataFrame(coord_rows).to_csv(output_dir / "latent_coordinates_pre_post_cfc.csv", index=False)
    print("Wrote CfC outputs to {}".format(output_dir))


if __name__ == "__main__":
    main()
