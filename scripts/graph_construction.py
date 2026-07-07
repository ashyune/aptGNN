"""
graph_construction.py

Builds a PyG HeteroData object from ThreaTrace's raw parsed edge-list files
(the ones parse_darpatc.py produces: cadets_train.txt, cadets_test.txt, ...).

Raw line format (tab-separated, confirmed from parse_darpatc.py output):
    srcId  srcType  dstId  dstType  edgeType  timestamp

e.g.
    72FB0406-...  SUBJECT_PROCESS  69DDB3CA-...  FILE_OBJECT_UNIX_SOCKET  EVENT_CLOSE  1522828474810632615

FEATURES (kept identical in spirit to ThreaTrace's MyDataset):
    Each node gets a histogram vector of length 2 * num_edge_types:
      - first half:  count of each edge type where this node is the SOURCE
      - second half: count of each edge type where this node is the DESTINATION
    This is exactly ThreaTrace's x_list construction, just partitioned by
    node type into separate HeteroData stores.

LABEL (the actual design change from ThreaTrace, see conversation):
    ThreaTrace predicts each node's own coarse entity type as the
    self-supervised target. That's degenerate once nodes are split into
    per-type HeteroData stores (the type is then implicit in the store).
    Instead: within each node type (e.g. all SUBJECT_PROCESS nodes), cluster
    nodes by their histogram feature into k behavioral sub-types via KMeans.
    The cluster id becomes y. Anomaly signal = node's neighborhood profile
    no longer matching its usual behavioral cluster.

TRAIN vs TEST consistency (mirrors ThreaTrace's own label.txt/feature.txt
reuse pattern):
    - On the TRAIN graph: fit the edge-type vocabulary and per-node-type
      KMeans models, save them to disk.
    - On the TEST graph: reuse the SAME edge-type vocabulary (unseen edge
      types in test are dropped, not added as new columns) and the SAME
      fitted KMeans models (predict, don't re-fit) so cluster ids mean the
      same thing across train and test.

Usage:
    # Step 1 -- build from the training graph, fit vocab + clustering:
    python graph_construction.py --scene cadets --split train \\
        --input ../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt \\
        --k 5

    # Step 2 -- build the test graph using the SAME fitted artifacts:
    python graph_construction.py --scene cadets --split test \\
        --input ../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
from sklearn.cluster import KMeans
from torch_geometric.data import HeteroData


def parse_raw_edges(path):
    """Read ThreaTrace's raw tab-separated edge-list file."""
    edges = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) != 6:
                continue
            src_id, src_type, dst_id, dst_type, edge_type, ts = parts
            edges.append((src_id, src_type, dst_id, dst_type, edge_type, int(ts)))
    return edges


def build_node_and_edge_vocab(edges, edge_type_map=None):
    """
    Assign a local (per-type) integer index to every node, and build/reuse
    a global edge-type vocabulary.

    node_registry: dict[node_type -> dict[raw_uuid -> local_index]]
    edge_type_map: dict[edge_type_str -> column_index]  (reused if provided,
                   for test-split consistency with the train split)
    """
    node_registry = {}      # node_type -> {uuid: local_idx}
    node_type_of = {}       # uuid -> node_type (for quick lookup while building edges)

    fit_edge_vocab = edge_type_map is None
    if fit_edge_vocab:
        edge_type_map = {}

    for src_id, src_type, dst_id, dst_type, edge_type, ts in edges:
        for node_id, node_type in ((src_id, src_type), (dst_id, dst_type)):
            if node_type not in node_registry:
                node_registry[node_type] = {}
            if node_id not in node_registry[node_type]:
                node_registry[node_type][node_id] = len(node_registry[node_type])
            node_type_of[node_id] = node_type

        if fit_edge_vocab and edge_type not in edge_type_map:
            edge_type_map[edge_type] = len(edge_type_map)

    return node_registry, node_type_of, edge_type_map


def build_features(edges, node_registry, node_type_of, edge_type_map):
    """
    ThreaTrace-style histogram features: for each node, count outgoing edges
    per edge type (first half) and incoming edges per edge type (second
    half). Edge types not in edge_type_map (only possible on the test split,
    if a new event type appears that the train split never saw) are dropped
    rather than added as a new column, to keep feature dimensionality
    consistent with the trained model.
    """
    num_edge_types = len(edge_type_map)
    feat_dim = 2 * num_edge_types

    features = {
        nt: torch.zeros(len(ids), feat_dim, dtype=torch.float)
        for nt, ids in node_registry.items()
    }

    dropped_edge_types = set()
    for src_id, src_type, dst_id, dst_type, edge_type, ts in edges:
        if edge_type not in edge_type_map:
            dropped_edge_types.add(edge_type)
            continue
        col = edge_type_map[edge_type]

        src_local = node_registry[src_type][src_id]
        features[src_type][src_local, col] += 1.0

        dst_local = node_registry[dst_type][dst_id]
        features[dst_type][dst_local, num_edge_types + col] += 1.0

    if dropped_edge_types:
        print(
            f"warning: {len(dropped_edge_types)} edge type(s) not in trained "
            f"vocabulary were dropped from feature construction: "
            f"{sorted(dropped_edge_types)}"
        )

    return features


def fit_or_load_clusters(features, k, cluster_models=None):
    """
    TRAIN split (cluster_models=None): fit a KMeans(k) per node type on that
    type's feature histograms, return {node_type: (labels_tensor, fitted_model)}.

    TEST split (cluster_models given): reuse the fitted models to predict
    cluster assignments for the new nodes -- do NOT refit, or cluster ids
    won't mean the same thing across train/test.
    """
    fitting = cluster_models is None
    if fitting:
        cluster_models = {}

    labels = {}
    for nt, x in features.items():
        x_np = x.numpy()

        if fitting:
            n_samples = x_np.shape[0]
            this_k = min(k, max(1, n_samples))  # can't have more clusters than points
            model = KMeans(n_clusters=this_k, n_init=10, random_state=42)
            y = model.fit_predict(x_np)
            cluster_models[nt] = model
        else:
            if nt not in cluster_models:
                print(f"warning: node type '{nt}' unseen in training clusters, "
                      f"assigning all nodes to cluster 0")
                y = [0] * x_np.shape[0]
            else:
                y = cluster_models[nt].predict(x_np)

        labels[nt] = torch.tensor(y, dtype=torch.long)

    return labels, cluster_models


def build_hetero_data(edges, node_registry, node_type_of, edge_type_map, features, labels):
    data = HeteroData()

    for nt, ids in node_registry.items():
        data[nt].x = features[nt]
        data[nt].y = labels[nt]
        # node_id: maps local index back to the original raw uuid, needed to
        # translate flagged anomalies back to real entities for evaluation
        # against groundtruth/<scene>.txt
        inv = [None] * len(ids)
        for uuid, local_idx in ids.items():
            inv[local_idx] = uuid
        data[nt].node_id = inv  # kept as a plain list of strings, not a tensor

    # Group edges by (src_type, edge_type, dst_type) triple, converting
    # global uuids to local per-type indices
    edge_buckets = {}
    for src_id, src_type, dst_id, dst_type, edge_type, ts in edges:
        key = (src_type, edge_type, dst_type)
        if key not in edge_buckets:
            edge_buckets[key] = ([], [])
        src_local = node_registry[src_type][src_id]
        dst_local = node_registry[dst_type][dst_id]
        edge_buckets[key][0].append(src_local)
        edge_buckets[key][1].append(dst_local)

    for key, (src_list, dst_list) in edge_buckets.items():
        data[key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="e.g. cadets")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--input", required=True, help="path to raw edge-list .txt file")
    parser.add_argument("--k", type=int, default=5,
                         help="number of behavioral clusters per node type (train split only)")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--output_dir", default="../graph_data/darpatc")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing raw edges from {args.input} ...")
    edges = parse_raw_edges(args.input)
    print(f"  {len(edges)} edges parsed")

    edge_vocab_path = artifacts_dir / f"{args.scene}_edge_type_map.json"
    cluster_path = artifacts_dir / f"{args.scene}_cluster_models.pkl"

    if args.split == "train":
        node_registry, node_type_of, edge_type_map = build_node_and_edge_vocab(edges)
        features = build_features(edges, node_registry, node_type_of, edge_type_map)
        labels, cluster_models = fit_or_load_clusters(features, args.k, cluster_models=None)

        with open(edge_vocab_path, "w") as f:
            json.dump(edge_type_map, f, indent=2)
        with open(cluster_path, "wb") as f:
            pickle.dump(cluster_models, f)
        print(f"Saved edge vocabulary ({len(edge_type_map)} types) -> {edge_vocab_path}")
        print(f"Saved cluster models -> {cluster_path}")

    else:  # test
        if not edge_vocab_path.exists() or not cluster_path.exists():
            raise FileNotFoundError(
                f"Missing training artifacts ({edge_vocab_path}, {cluster_path}). "
                f"Run --split train first for scene '{args.scene}'."
            )
        with open(edge_vocab_path) as f:
            edge_type_map = json.load(f)
        with open(cluster_path, "rb") as f:
            cluster_models = pickle.load(f)

        node_registry, node_type_of, edge_type_map = build_node_and_edge_vocab(
            edges, edge_type_map=edge_type_map
        )
        features = build_features(edges, node_registry, node_type_of, edge_type_map)
        labels, _ = fit_or_load_clusters(features, k=None, cluster_models=cluster_models)

    data = build_hetero_data(edges, node_registry, node_type_of, edge_type_map, features, labels)

    print("\nConstructed HeteroData:")
    print(data)
    print("\nNode types and cluster/class counts:")
    for nt in data.node_types:
        print(f"  {nt}: {data[nt].num_nodes} nodes, "
              f"{torch.unique(data[nt].y).numel()} classes, "
              f"feature dim {data[nt].x.size(-1)}")

    out_path = output_dir / f"{args.scene}_{args.split}_hetero.pt"
    torch.save(data, out_path)
    print(f"\nSaved HeteroData -> {out_path}")


if __name__ == "__main__":
    main()