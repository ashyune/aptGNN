"""Stage 6 (revised): node-level anomaly scoring from the frozen checkpoint.

Produces the per-node score artifact that evaluate_windowed.py turns into
AUPRC / AUROC / precision@k, plus operating-point P/R/F1/FPR. This replaces
the earlier alarm-file + whole-file 2-hop-credit design, which the
evaluation-design review found to be topology-dominated: a single hub's
2-hop neighbourhood reached almost every ground-truth node, pinning recall
at 1.0 regardless of detector quality, so the metric could not distinguish
the detector from a random alarm set.

Protocol (final)
-----------------
- Node-level scoring, NO neighborhood credit. A node counts only for
  itself -- being near a labeled node earns nothing.
- Continuous anomaly score = negative log-probability of the node's true
  CDM type (higher = the model was more surprised = more anomalous),
  aggregated across the windows a node appears in by MAX (a node's most
  anomalous moment -- see the design review for why max, not mean/sum/last).
- A fixed operating point is retained, but it is now derived from the
  score itself: a node is flagged iff its aggregated max score is >= the
  --flag-quantile quantile (default 0.999, nearest-rank) of the max-scores
  the SAME frozen checkpoint produces over the TRAINING windows, scored
  under the identical cold-start protocol. The threshold is therefore
  chosen on benign training data only. The previous operating point (the
  inherited confidence-ratio flag OR-ed across every window a node
  appeared in) is retired: OR-ing a per-window wrong-or-uncertain test
  inverted the baseline cascade's survive-every-model semantics and
  flagged entire CDM types wholesale (100% of UnnamedPipeObjects, ~55% of
  NetFlowObjects in the 2026-07 investigation). flag_nodes()/THRE_MAP
  remain defined below only because d6.py imports them.
- Ground truth defaults to ../groundtruth/<scene>.txt -- the immutable
  per-scene source file -- NOT scripts/groundtruth_uuid.txt, which
  train_darpatc.py silently overwrites with whichever scene the baseline
  was last trained on. Resolved to global_id as before; ThreaTrace's
  published (2-hop-expanded) label set stays the default target for
  comparability, with --groundtruth available for the raw entity set.

Unchanged from the previous version: the memory threading, the cold-start
NodeMemory (Decision 4), and evaluating the single frozen checkpoint. Only
the scoring artifact and how it is produced have changed.

build_whole_file_adjacency and two_hop_neighbors remain defined in this
module because the older diagnostic scripts still import them, but they are
no longer on the scoring path and main() no longer calls them.

Output (two files in --out-dir)
--------------------------------
- scores_windowed.txt        : one line per distinct test node,
                               '<global_id> <max_score> <flag>'
                               (flag is 0/1 at the fixed operating point)
- groundtruth_global_id.txt  : one resolved ground-truth global_id per line
"""

import argparse
import math
import os
import os.path as osp
import time

import torch
import torch.nn.functional as F

from node_vocab import load_node_vocab
from windowing import build_type_vocab
from windowed_data import build_windowed_dataset
from node_memory import NodeMemory
from train_windowed import SAGENetWithMemory

THRE_MAP = {"cadets": 1.5, "trace": 1.0, "theia": 1.5, "fivedirections": 1.0}


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


# ---------------------------------------------------------------------------
# Whole-file, global_id-keyed 2-hop adjacency.
#
# No longer used by the scoring path (the node-level protocol takes no
# neighborhood credit). Kept because diagnose_ground_truth_footprint.py,
# diagnostic3_random_baseline.py, and d3-1.py import these two functions.
# ---------------------------------------------------------------------------

def build_whole_file_adjacency(test_path, node_vocab):
    """Build adj/adj2 over every edge in *test_path*, pooled regardless of
    window boundaries, keyed by Stage 1's global_id.

    adj[dst] = incoming/predecessor global ids, adj2[src] = outgoing/
    successor global ids. Rows referencing a uuid outside node_vocab are
    skipped, matching the codebase's convention for unrecognised entries.
    """
    adj = {}
    adj2 = {}
    with open(test_path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            src_uuid, dst_uuid = temp[0], temp[2]
            if src_uuid not in node_vocab or dst_uuid not in node_vocab:
                continue
            src = node_vocab[src_uuid]
            dst = node_vocab[dst_uuid]
            adj.setdefault(dst, []).append(src)
            adj2.setdefault(src, []).append(dst)
    return adj, adj2


def two_hop_neighbors(node_id, adj, adj2):
    """Symmetric 2-hop neighbourhood of *node_id* (both adj and adj2)."""
    neighbours = set()
    for j in adj.get(node_id, []):
        neighbours.add(j)
        for k in adj.get(j, []):
            neighbours.add(k)
    for j in adj2.get(node_id, []):
        neighbours.add(j)
        for k in adj2.get(j, []):
            neighbours.add(k)
    return neighbours


# ---------------------------------------------------------------------------
# LEGACY: inherited confidence-ratio flag. No longer on the scoring path
# (see module docstring -- the operating point is now a train-quantile
# threshold on the continuous score). Kept because d6.py imports
# flag_nodes and THRE_MAP.
# ---------------------------------------------------------------------------

def flag_nodes(out, y_true, thre):
    """LEGACY -- boolean tensor: True where a node would have been flagged.

    Flagged if EITHER the argmax prediction is wrong, OR the top-1/top-2
    softmax ratio falls below `thre`. Matches the original _predict_batch
    exactly. `out` is F.log_softmax output; F.softmax(out) recovers the
    per-class probabilities (exp(log_softmax) already sums to 1), so the
    softmax-of-log-softmax is a value-preserving no-op, mirroring the
    original.

    Not called by main() anymore -- retained only for d6.py.
    """
    pred = out.argmax(dim=1)
    pro = F.softmax(out, dim=1)
    top1 = pro.max(dim=1)
    pro_rest = pro.clone()
    pro_rest[torch.arange(pro.size(0), device=out.device), top1.indices] = -1
    top2 = pro_rest.max(dim=1)

    wrong = pred != y_true
    uncertain = (top1.values / top2.values) < thre
    return wrong | uncertain


# ---------------------------------------------------------------------------
# Continuous anomaly score + cross-window aggregation + output.
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_dataset(model, dataset, node_memory):
    """Score one windowed dataset under the standard protocol.

    Cold-start memory threading over the window sequence in chronological
    order, per-node MAX aggregation across windows. Returns
    {global_id: max_score}. Used twice by main(): for the test windows
    (the scores that get written out) and for the training windows (the
    benign score distribution the operating-point threshold is drawn
    from). One implementation so the two distributions are produced by
    identical code -- any protocol drift between them would silently bias
    the threshold.

    Memory always updates, anomalous or not -- a node's history keeps
    accumulating regardless of any window's anomaly call.
    """
    score_by_gid = {}
    for window_idx, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        data = data.to(node_memory.device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, hidden = model(data.x, data.edge_index, memory)
        scores = score_nodes(out, data.y).cpu().tolist()
        gids = data.global_id.cpu().tolist()
        update_max_scores(score_by_gid, gids, scores)
        node_memory.update(data.global_id, hidden.detach(), window_idx)
    return score_by_gid


def nearest_rank_quantile(values, q):
    """Nearest-rank quantile: the smallest value v such that at least
    ceil(q * n) of the n values are <= v.

    No interpolation on purpose: the benign max-score distribution is
    heavily tied (the 2026-07 investigation measured ~81% of nodes at
    exactly 0.0), and interpolating definitions give threshold values
    that no actual node attains. q must be in (0, 1]. Note that any q at
    or below the tied mass's cumulative fraction returns the tied value
    itself -- main() warns when the resulting threshold is 0.0, since
    'flag everything above the floor' is unlikely to be an intended
    operating point.
    """
    if not values:
        raise ValueError('nearest_rank_quantile: empty values')
    if not 0 < q <= 1:
        raise ValueError(f'nearest_rank_quantile: q must be in (0, 1], got {q}')
    ordered = sorted(values)
    idx = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def score_nodes(out, y_true):
    """Per-node anomaly score = negative log-probability of the true type.

    `out` is F.log_softmax output [N, C]; out[i, y_true[i]] is the log
    probability the model assigned to node i's actual CDM type, so
    -out[i, y_true[i]] is the model's surprise at node i. Higher means the
    model thought the node's real type was unlikely = more anomalous.
    Returned as a 1-D tensor on the same device as `out`.
    """
    idx = torch.arange(out.size(0), device=out.device)
    return -out[idx, y_true]


def update_max_scores(score_by_gid, gids, scores):
    """Fold one window's per-node scores into the running per-node maximum.

    gids and scores are equal-length plain sequences (a window's global
    ids and their anomaly scores). For each node keep the maximum score
    seen across every window it appears in: "was this node ever anomalous"
    is a max over windows, not a mean/sum/last (see the design review).
    """
    for gid, score in zip(gids, scores):
        prev = score_by_gid.get(gid)
        if prev is None or score > prev:
            score_by_gid[gid] = score


def write_scores_file(path, score_by_gid, flagged_gids):
    """Write one line per distinct node: '<global_id> <max_score> <flag>'.

    flag is 1 if the node was flagged (flag_nodes True) in any window,
    else 0. Every node that appeared in any window is written, flagged or
    not -- AUPRC/AUROC need a score for the negatives too, so filtering to
    flagged nodes here would corrupt the denominators downstream.
    """
    with open(path, 'w') as fw:
        for gid, score in score_by_gid.items():
            flag = 1 if gid in flagged_gids else 0
            fw.write(f'{gid} {score} {flag}\n')


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 6 (revised): node-level anomaly scoring from the '
                     'frozen checkpoint -- produces scores_windowed.txt.'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Defaults to ../models/windowed_<scene>/best_model.pt')
    parser.add_argument('--window-size', type=int, default=50000,
                        help='Must match the value the checkpoint was trained with.')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Must match the frozen training config.')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Must match the frozen training config.')
    parser.add_argument('--max-norm', type=float, default=100.0,
                        help='Must match the frozen training config.')
    parser.add_argument('--flag-quantile', type=float, default=0.999,
                        help='Operating point: a test node is flagged iff '
                             'its aggregated max score is >= this '
                             'nearest-rank quantile of the training-window '
                             'max-score distribution under the same frozen '
                             'checkpoint and protocol. Affects the flag '
                             'column only, never the continuous score or '
                             'the ranking metrics.')
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--groundtruth', type=str, default=None,
                        help='Ground-truth UUID file, one per line. '
                             'Defaults to ../groundtruth/<scene>.txt (the '
                             'immutable per-scene source) rather than '
                             'scripts/groundtruth_uuid.txt, which '
                             'train_darpatc.py silently overwrites with '
                             'whichever scene it last ran.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to ../models/windowed_<scene>/eval/')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    checkpoint_path = args.checkpoint or f'../models/windowed_{args.scene}/best_model.pt'
    out_dir = args.out_dir or f'../models/windowed_{args.scene}/eval'
    groundtruth_path = args.groundtruth or f'../groundtruth/{args.scene}.txt'

    for p in (train_path, test_path, node_vocab_path, checkpoint_path, groundtruth_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')
    os.makedirs(out_dir, exist_ok=True)

    show(f'Scene: {args.scene} | operating point = train quantile '
         f'{args.flag_quantile} (affects the flag column only, not the '
         f'score or ranking metrics)')

    show(f'Loading node vocab: {node_vocab_path}')
    node_vocab = load_node_vocab(node_vocab_path)
    show(f'{len(node_vocab):,} global node identities')

    feature_map, label_map = build_type_vocab(train_path)
    in_channels = len(feature_map) * 2
    out_channels = len(label_map)

    ground_truth_uuids = set()
    with open(groundtruth_path, 'r') as f:
        for line in f:
            uuid = line.strip('\n')
            if uuid:
                ground_truth_uuids.add(uuid)

    ground_truth_global_ids = set()
    unresolved = 0
    for uuid in ground_truth_uuids:
        if uuid in node_vocab:
            ground_truth_global_ids.add(node_vocab[uuid])
        else:
            unresolved += 1
    if unresolved:
        show(f'WARNING: {unresolved} ground-truth UUIDs not found in '
             f'node_vocab -- they cannot be scored. Check that '
             f'{groundtruth_path} matches this scene\'s test file.')
    show(f'{len(ground_truth_global_ids)} ground-truth nodes resolved to global_id')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Device: {device}')

    model = SAGENetWithMemory(in_channels, out_channels, args.hidden_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    show(f'Loaded frozen checkpoint: {checkpoint_path}')

    # --- Calibration pass: benign max-score distribution from the TRAIN
    # file, same frozen checkpoint, same cold-start protocol, same
    # aggregation. Its only output is the operating-point threshold; the
    # scores themselves are discarded.
    show('Building windowed train dataset (operating-point calibration)')
    train_dataset = build_windowed_dataset(
        train_path, node_vocab, feature_map, label_map, args.window_size)
    show(f'{len(train_dataset)} train windows')
    calib_memory = NodeMemory(len(node_vocab), args.hidden_dim, args.decay_rate,
                              device=device, max_norm=args.max_norm)
    train_scores = score_dataset(model, train_dataset, calib_memory)
    threshold = nearest_rank_quantile(list(train_scores.values()),
                                      args.flag_quantile)
    n_train_at_or_above = sum(1 for s in train_scores.values() if s >= threshold)
    show(f'Operating-point threshold: score >= {threshold:.6f} '
         f'(train quantile {args.flag_quantile}; {n_train_at_or_above:,} of '
         f'{len(train_scores):,} train nodes at or above it)')
    if threshold <= 0.0:
        show('WARNING: threshold is 0.0 -- the flag quantile falls inside '
             'the saturated-at-zero mass of the benign score distribution, '
             'so everything above the floor will be flagged. Raise '
             '--flag-quantile.')
    del train_dataset, train_scores, calib_memory

    # --- Scoring pass: the test file ---
    show('Building windowed test dataset')
    test_dataset = build_windowed_dataset(
        test_path, node_vocab, feature_map, label_map, args.window_size)
    show(f'{len(test_dataset)} test windows')

    # Cold-start (Decision 4): fresh memory, no training-window replay.
    node_memory = NodeMemory(len(node_vocab), args.hidden_dim, args.decay_rate,
                             device=device, max_norm=args.max_norm)

    # Per-node running MAX anomaly score; score_by_gid's keys are exactly
    # the distinct test nodes. The flag column is a pure threshold on that
    # aggregated score -- no per-window flag state exists anymore.
    score_by_gid = score_dataset(model, test_dataset, node_memory)
    flagged_gids = {gid for gid, score in score_by_gid.items()
                    if score >= threshold}

    total_nodes = len(score_by_gid)
    show(f'{total_nodes:,} distinct nodes scored across the test sequence')
    if total_nodes > 0:
        show(f'{len(flagged_gids):,} nodes flagged at the fixed operating '
             f'point ({100 * len(flagged_gids) / total_nodes:.2f}% of test nodes)')

    scores_path = osp.join(out_dir, 'scores_windowed.txt')
    write_scores_file(scores_path, score_by_gid, flagged_gids)
    show(f'Wrote {scores_path}')

    gt_path = osp.join(out_dir, 'groundtruth_global_id.txt')
    with open(gt_path, 'w') as fw:
        for gid in ground_truth_global_ids:
            fw.write(f'{gid}\n')
    show(f'Wrote {gt_path}')

    show('Done. Run evaluate_windowed.py against these two files for '
         'AUPRC/AUROC and operating-point P/R/F1.')


if __name__ == '__main__':
    main()