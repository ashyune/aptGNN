# APTGNN Heterogeneous Pipeline — README

Edge-type-aware heterogeneous GNN extension of the ThreaTrace baseline, built
on PyG's `HeteroConv`/`HeteroData`. Cross-window GRU memory is deliberately
deferred (this is ablation config 2: heterogeneous, no-memory).

## Files in this pipeline

| File | Role |
|---|---|
| `graph_construction.py` | Builds `HeteroData` graphs from ThreaTrace's raw parsed edge-list files |
| `hetero_aptgnn_model.py` | `HeteroAPTGNN` model definition (edge-type-aware `HeteroConv` stack) |
| `train_darpatc_hetero.py` | Trains the model on the train-split graph, saves a checkpoint |
| `test_darpatc_hetero.py` | Runs the trained model on the test-split graph, flags anomalous nodes |
| `evaluate_darpatc_hetero.py` | Computes Precision/Recall/F-Score against ground truth |
| `inspect_hetero_graph.py` | Utility: prints the schema of any saved `HeteroData` `.pt` file |

`hetero_aptgnn_model.py` is imported by the other scripts — it isn't run
directly, but must sit in the same directory as the rest.

## Setup

All commands assume `heterogeneous_env` is activated and you're inside
`~/Desktop/AptGNN/threaTrace/scripts/`.

```bash
source ~/heterogeneous_env/bin/activate  
cd ~/Desktop/AptGNN/threaTrace/scripts
```

Confirmed working versions in this env: `torch 2.8.0+cpu`,
`torch_geometric 2.6.1`, `scikit-learn 1.6.1`. No `torch-scatter` /
`torch-sparse` / `pyg-lib` needed — `HeteroConv` and `SAGEConv` don't require
them.

## Run order

### 1. Build the training graph

Reads ThreaTrace's raw parsed edge-list, builds per-node-type feature
histograms (ThreaTrace-style in/out edge-type counts), fits a KMeans
behavioral-cluster label per node type, and saves everything to disk.

```bash
python graph_construction.py --scene cadets --split train \
    --input ../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt \
    --k 5
```

Produces:
- `../graph_data/darpatc/cadets_train_hetero.pt` — the `HeteroData` graph
- `artifacts/cadets_edge_type_map.json` — fitted edge-type vocabulary
- `artifacts/cadets_cluster_models.pkl` — fitted KMeans models per node type

### 2. (Optional) Inspect the graph

Sanity-check the schema before training — node/edge types, feature dims,
label cardinality.

```bash
python inspect_hetero_graph.py --path ../graph_data/darpatc/cadets_train_hetero.pt
```

### 3. Train

Full-batch, transductive training over the whole graph (same scale
ThreaTrace itself trains at). Holds out a small validation slice per node
type just to monitor generalization.

```bash
python train_darpatc_hetero.py --scene cadets --epochs 30
```

Produces `models/aptgnn_cadets.pt` (weights + model config, used by the test
script).

Useful flags: `--lr`, `--hidden_channels`, `--num_layers`, `--val_frac`,
`--out` (custom checkpoint path).

### 4. Build the test graph

Reuses the vocabulary and cluster models fit in step 1 — does **not** refit,
so cluster ids mean the same thing across train and test.

```bash
python graph_construction.py --scene cadets --split test \
    --input ../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt
```

Produces `../graph_data/darpatc/cadets_test_hetero.pt`.

### 5. Test

Runs the trained checkpoint on the test graph. A node is flagged anomalous
if the model's confidence in that node's own cluster label falls below
`--threshold`.

```bash
python test_darpatc_hetero.py --scene cadets --threshold 0.5
```

Produces `results/cadets_hetero_fp_ids.pkl` (flagged `(node_type, uuid)`
pairs).

### 6. Evaluate

Compares flagged nodes against ThreaTrace's own ground truth file.

```bash
python Evaluate_hetero_darpatc.py --scene cadets
```

Reads `../groundtruth/cadets.txt` by default (ThreaTrace's own convention —
one malicious node uuid per line). Prints `tp fp fn` then Precision / Recall
/ F-Score, in the same format as the ThreaTrace baseline, for direct
comparison.

## Full sequence, copy-pasteable

```bash
python graph_construction.py --scene cadets --split train \
    --input ../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt --k 5

python train_darpatc_hetero.py --scene cadets --epochs 30

python graph_construction.py --scene cadets --split test \
    --input ../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt

python test_darpatc_hetero.py --scene cadets

python evaluate_darpatc_hetero.py --scene cadets
```

## Design notes / known deviations from ThreaTrace

- **Label scheme differs from ThreaTrace.** ThreaTrace predicts each node's
  coarse entity type as its self-supervised target. That's degenerate once
  nodes are split into per-type `HeteroData` stores (the type becomes
  implicit in which store the node lives in). Instead, each node type's
  nodes are clustered via KMeans into `k` finer-grained behavioral
  sub-types, and the cluster id is the prediction target.
- **`--k 5` is a starting guess, not validated.** Worth a proper elbow /
  silhouette sweep per node type later rather than treating it as final.
- **No windowing yet.** Training and testing run on the whole graph as one
  shot, unlike ThreaTrace's 22-round iterative test loop. Windowing is
  deferred alongside GRU cross-window memory (ablation configs 3/4).
- **`weights_only=False`** is used in all `torch.load()` calls in this
  pipeline. Safe here because every `.pt` file involved was produced by
  these same scripts — never do this for files from an untrusted source.