"""
test_darpatc_hetero.py

Tests a trained HeteroAPTGNN checkpoint against a HeteroData graph built by
graph_construction.py. No cross-window state -- windowing is deferred, same
as GRU memory.

Label scheme (updated): each node's ground-truth target is its OWN behavioral
cluster label, stored per-node in data[nt].y (assigned by KMeans in
graph_construction.py) -- NOT a constant "true type per node type" the way
an earlier version of this script assumed. Anomaly signal: a node is flagged
if the model's confidence in that node's own cluster label falls below
`threshold`.

Usage:
    python test_darpatc_hetero.py --scene cadets
"""

import argparse
import pickle
from pathlib import Path

import torch

from hetero_aptgnn_model import HeteroAPTGNN


def test(model, data, threshold=0.5, device="cpu"):
    model.eval()
    fp_ids = set()

    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    with torch.no_grad():
        out_dict = model(x_dict, edge_index_dict)

        total_fp = 0
        total_nodes = 0
        for nt in data.node_types:
            logits = out_dict[nt]
            probs = torch.softmax(logits, dim=-1)
            y = data[nt].y.to(device)

            # Confidence the model assigns to each node's OWN true cluster
            # label -- gather along dim=1 using each node's own y, not a
            # shared per-type constant.
            confidence = probs.gather(1, y.unsqueeze(1)).squeeze(1)

            anomalous_mask = confidence < threshold
            total_nodes += logits.size(0)
            total_fp += anomalous_mask.sum().item()

            node_ids = data[nt].node_id  # plain list of raw uuids
            for idx in anomalous_mask.nonzero(as_tuple=True)[0].tolist():
                fp_ids.add((nt, node_ids[idx]))

            print(
                f"  {nt}: {anomalous_mask.sum().item()} / {logits.size(0)} "
                f"flagged anomalous"
            )

    acc = 1.0 - (total_fp / total_nodes if total_nodes else 0.0)
    print(f"\noverall acc:{acc:.4f}  total_fp:{total_fp}  total_nodes:{total_nodes}")
    return fp_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="cadets")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_dir", default="../graph_data/darpatc")
    parser.add_argument(
        "--checkpoint", default=None, help="defaults to models/aptgnn_<scene>.pt"
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint or f"models/aptgnn_{args.scene}.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = HeteroAPTGNN(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    data_path = Path(args.data_dir) / f"{args.scene}_test_hetero.pt"
    print(f"Loading {data_path} ...")
    data = torch.load(data_path, weights_only=False)
    print(data)

    fp_ids = test(model, data, threshold=args.threshold)

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/{args.scene}_hetero_fp_ids.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(fp_ids, f)
    print(f"\nSaved flagged node ids -> {out_path}")