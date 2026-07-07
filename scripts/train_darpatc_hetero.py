"""
train_darpatc_hetero.py

Trains HeteroAPTGNN on the HeteroData object produced by
graph_construction.py --split train (e.g. cadets_train_hetero.pt).

Objective: each node predicts its own behavioral cluster label (data[nt].y,
assigned by KMeans in graph_construction.py) from its edge-type-aware
neighborhood embedding. This mirrors ThreaTrace's own self-supervised setup
(predict something about yourself from local structure), just with a
per-node-type cluster id instead of the now-degenerate entity-type label.

Training is full-batch and transductive over the WHOLE training graph --
same scale ThreaTrace itself operates at (train_mask/test_mask = all True
in their MyDataset). No windowing yet (deferred, same as memory).

A small held-out validation slice per node type is carved out purely to
monitor generalization during training -- ThreaTrace doesn't do this itself,
but it's useful to sanity-check the model isn't just memorizing before we
wire in real anomaly evaluation against groundtruth/ later.

Usage:
    python train_darpatc_hetero.py --scene cadets --epochs 30
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hetero_aptgnn_model import HeteroAPTGNN


def build_model_from_data(data, hidden_channels=32, num_layers=2):
    node_types = data.node_types
    edge_types = data.edge_types
    in_channels_dict = {nt: data[nt].x.size(-1) for nt in node_types}
    # Read actual cluster count per type from the data itself -- some types
    # (e.g. small ones like FILE_OBJECT_DIR) may have fewer than k clusters
    # if graph_construction.py's min(k, n_samples) kicked in.
    num_classes_dict = {nt: int(data[nt].y.max().item()) + 1 for nt in node_types}

    model_kwargs = dict(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict=in_channels_dict,
        num_classes_dict=num_classes_dict,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    model = HeteroAPTGNN(**model_kwargs)
    return model, model_kwargs


def make_train_val_masks(data, val_frac=0.1, seed=42):
    g = torch.Generator().manual_seed(seed)
    train_mask, val_mask = {}, {}
    for nt in data.node_types:
        n = data[nt].num_nodes
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(n * val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        tm = torch.zeros(n, dtype=torch.bool)
        vm = torch.zeros(n, dtype=torch.bool)
        tm[train_idx] = True
        vm[val_idx] = True
        train_mask[nt] = tm
        val_mask[nt] = vm
    return train_mask, val_mask


def compute_loss(out_dict, data, mask_dict=None):
    total_loss = 0.0
    total_nodes = 0
    for nt, out in out_dict.items():
        y = data[nt].y
        if mask_dict is not None:
            m = mask_dict[nt]
            out_m, y_m = out[m], y[m]
        else:
            out_m, y_m = out, y
        if y_m.numel() == 0:
            continue
        total_loss = total_loss + F.cross_entropy(out_m, y_m, reduction="sum")
        total_nodes += y_m.numel()
    return total_loss / max(total_nodes, 1)


def compute_accuracy(out_dict, data, mask_dict=None):
    correct, total = 0, 0
    for nt, out in out_dict.items():
        y = data[nt].y
        if mask_dict is not None:
            m = mask_dict[nt]
            out_m, y_m = out[m], y[m]
        else:
            out_m, y_m = out, y
        if y_m.numel() == 0:
            continue
        pred = out_m.argmax(dim=-1)
        correct += (pred == y_m).sum().item()
        total += y_m.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="cadets")
    parser.add_argument("--data_dir", default="../graph_data/darpatc")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_path = Path(args.data_dir) / f"{args.scene}_train_hetero.pt"
    print(f"Loading {data_path} ...")
    data = torch.load(data_path, weights_only=False)
    print(data)

    model, model_kwargs = build_model_from_data(
        data, hidden_channels=args.hidden_channels, num_layers=args.num_layers
    )
    print(f"\nModel classes per node type: {model_kwargs['num_classes_dict']}")

    train_mask, val_mask = make_train_val_masks(data, val_frac=args.val_frac)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    print(f"\nTraining for {args.epochs} epochs (full-batch, transductive)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        optimizer.zero_grad()
        out_dict = model(x_dict, edge_index_dict)
        loss = compute_loss(out_dict, data, mask_dict=train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_dict = model(x_dict, edge_index_dict)
            train_acc = compute_accuracy(out_dict, data, mask_dict=train_mask)
            val_acc = compute_accuracy(out_dict, data, mask_dict=val_mask)

        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}  loss {loss.item():.4f}  "
            f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  ({dt:.1f}s)"
        )

    out_path = Path(args.out) if args.out else Path(f"models/aptgnn_{args.scene}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "model_kwargs": model_kwargs}
    torch.save(checkpoint, out_path)
    print(f"\nSaved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()