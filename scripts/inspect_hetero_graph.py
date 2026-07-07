"""
inspect_hetero_graph.py

Run this against the actual HeteroData object produced by graph_construction.py
to answer the open questions before hetero_aptgnn_model.py can be correctly
wired to real data:

  1. What are the real node types and edge types?
  2. What feature dim does each node type have (for input_proj)?
  3. Does each node type carry a ground-truth label (.y)? What's the label
     cardinality per type (for the classifier head output dim)?
  4. Do edges carry timestamps (needed for windowing later)?

Usage:
    python inspect_hetero_graph.py --path /path/to/your/graph.pt
"""

import argparse
import torch


def inspect(data):
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    node_types, edge_types = data.metadata()
    print(f"Node types: {node_types}")
    print(f"Edge types: {edge_types}")

    print("\n" + "=" * 60)
    print("NODE STORES")
    print("=" * 60)
    for nt in node_types:
        store = data[nt]
        print(f"\n[{nt}]")
        print(f"  num_nodes: {store.num_nodes}")
        if hasattr(store, "x"):
            print(f"  x.shape: {tuple(store.x.shape)}  dtype: {store.x.dtype}")
        else:
            print("  x: NOT PRESENT")

        if hasattr(store, "y"):
            y = store.y
            unique_labels = torch.unique(y)
            print(f"  y.shape: {tuple(y.shape)}  dtype: {y.dtype}")
            print(f"  y unique label count: {unique_labels.numel()}")
            print(f"  y sample values: {unique_labels[:10].tolist()}"
                  f"{' ...' if unique_labels.numel() > 10 else ''}")
        else:
            print("  y: NOT PRESENT (no ground-truth label attached)")

        if hasattr(store, "node_id"):
            nid = store.node_id
            if isinstance(nid, list):
                print(f"  node_id present: list of {len(nid)} raw ids "
                      f"(sample: {nid[:3]})")
            else:
                print(f"  node_id present, dtype: {nid.dtype}")
        else:
            print("  node_id: NOT PRESENT")

    print("\n" + "=" * 60)
    print("EDGE STORES")
    print("=" * 60)
    for et in edge_types:
        store = data[et]
        print(f"\n{et}")
        print(f"  edge_index.shape: {tuple(store.edge_index.shape)}")
        if hasattr(store, "edge_attr"):
            print(f"  edge_attr.shape: {tuple(store.edge_attr.shape)}")
        else:
            print("  edge_attr: NOT PRESENT")
        if hasattr(store, "timestamp") or hasattr(store, "time") or hasattr(store, "ts"):
            for attr in ("timestamp", "time", "ts"):
                if hasattr(store, attr):
                    t = getattr(store, attr)
                    print(f"  {attr}.shape: {tuple(t.shape)}  "
                          f"min: {t.min().item()}  max: {t.max().item()}")
        else:
            print("  no per-edge timestamp attribute found "
                  "(checked: timestamp, time, ts)")

    print("\n" + "=" * 60)
    print("SUMMARY FOR MODEL WIRING")
    print("=" * 60)
    in_channels_dict = {
        nt: data[nt].x.size(-1) for nt in node_types if hasattr(data[nt], "x")
    }
    print(f"in_channels_dict = {in_channels_dict}")

    label_info = {}
    for nt in node_types:
        if hasattr(data[nt], "y"):
            label_info[nt] = torch.unique(data[nt].y).numel()
    print(f"per-type label cardinality = {label_info}")
    if len(set(label_info.values())) > 1:
        print("  -> label spaces DIFFER across node types: "
              "you likely need a SEPARATE classifier head per node type "
              "with its own output dim (not one shared num_classes).")
    elif label_info:
        print(f"  -> label space appears SHARED across node types "
              f"({list(label_info.values())[0]} classes) -- "
              f"a single shared num_classes may be valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to the saved HeteroData .pt file")
    args = parser.parse_args()

    # weights_only=False: safe here because this file was created by our own
    # graph_construction.py, not downloaded from an untrusted source. PyTorch
    # 2.6+ defaults to weights_only=True, which blocks unpickling HeteroData.
    data = torch.load(args.path, weights_only=False)
    inspect(data)