"""Aggregation-variant behavior scoring (2026-07-16 experiment).

Tests one idea from the operating-point diagnosis: the per-window MEAN
per-event NLL dilutes a few anomalous events inside a mostly-normal
window. This scorer re-runs inference from a frozen BehaviorNet
checkpoint (no retraining) and produces, in ONE pass, three per-window
per-node scores from the same forward output:

  mean -- -(x . log p) / sum(x): the existing headline score
          (train_behavior.profile_nll), recomputed here as a
          determinism/faithfulness regression check against the
          arm's existing scores_windowed.txt.
  max  -- max over the node's observed event categories of -log p_k:
          the single most surprising event in the window.
  p95  -- nearest-rank 95th percentile of the window's per-event NLL
          multiset (category k contributes x_k copies of -log p_k):
          a tail statistic less brittle than the pure max.

Everything else is byte-identical to test_behavior.py's protocol:
chronological cold-start memory threading, per-node MAX aggregation
across windows, calibration pass over the train windows under the same
frozen checkpoint, per-aggregation operating-point threshold =
--flag-quantile nearest-rank quantile of that aggregation's train
max-score distribution. Pipeline files are untouched; this is a
separate additive scorer (workflow-gating rule).

Outputs in --out-dir:
  scores_windowed_aggmean.txt   (regression check vs existing mean file)
  scores_windowed_aggmax.txt
  scores_windowed_aggp95.txt
"""

import argparse
import os
import os.path as osp
import time

import torch

from node_vocab import load_node_vocab
from windowing import build_type_vocab
from windowed_data import build_windowed_dataset
from node_memory import NodeMemory
from train_behavior import (BehaviorNet, profile_nll, make_writeback,
                            type_onehot)
from test_windowed import (nearest_rank_quantile, update_max_scores,
                           write_scores_file)

AGGS = ('mean', 'max', 'p95')


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}', flush=True)


def per_event_nll_aggs(log_probs, x, percentile=0.95):
    """Per-window per-node {mean, max, p95} per-event NLL, each [N].

    The window's per-event NLL multiset for a node is category k
    repeated x_k times with value -log p_k. mean is exactly
    profile_nll; max is the largest -log p_k among observed
    categories; p95 is the nearest-rank percentile of the multiset
    (smallest value covering >= ceil(q * total) events in ascending
    order). Nodes with an all-zero profile (cannot happen upstream;
    clamped defensively like profile_nll) score 0 under all three.
    """
    nll = -log_probs
    total = x.sum(dim=1)
    observed = x > 0
    any_events = total > 0

    mean = profile_nll(log_probs, x)

    mx = nll.masked_fill(~observed, float('-inf')).max(dim=1).values
    mx = torch.where(any_events, mx, torch.zeros_like(mx))

    vals, order = nll.sort(dim=1)
    weights = x.gather(1, order)
    cum = weights.cumsum(dim=1)
    target = (percentile * total).ceil().clamp(min=1.0).unsqueeze(1)
    pos = (cum < target).sum(dim=1, keepdim=True)
    pos = pos.clamp(max=vals.size(1) - 1)
    p95 = vals.gather(1, pos).squeeze(1)
    p95 = torch.where(any_events, p95, torch.zeros_like(p95))

    return {'mean': mean, 'max': mx, 'p95': p95}


@torch.no_grad()
def score_dataset_aggs(model, dataset, node_memory, num_types):
    """One pass over a windowed dataset -> {agg: score_by_gid}.

    Mirrors test_behavior.score_dataset (cold-start memory threading,
    per-node MAX across windows, memory always updates); the only
    difference is that three window statistics are folded instead of
    one.
    """
    score_by_gid = {agg: {} for agg in AGGS}
    for window_idx, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        data = data.to(node_memory.device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, hidden = model(type_onehot(data.y, num_types), memory,
                            data.edge_index, data.edge_type)
        window_scores = per_event_nll_aggs(out, data.x)
        gids = data.global_id.cpu().tolist()
        for agg in AGGS:
            update_max_scores(score_by_gid[agg], gids,
                              window_scores[agg].cpu().tolist())
        node_memory.update(data.global_id, make_writeback(hidden, data.x),
                           window_idx)
    return score_by_gid


def regression_check(ref_path, mean_scores):
    """Compare recomputed mean scores against an existing scores file."""
    ref = {}
    with open(ref_path) as f:
        for line in f:
            gid, score, _flag = line.split(' ')
            ref[int(gid)] = float(score)
    if set(ref) != set(mean_scores):
        show(f'REGRESSION CHECK FAILED: node sets differ '
             f'(ref {len(ref):,} vs new {len(mean_scores):,})')
        return False
    worst = max(abs(ref[g] - mean_scores[g]) for g in ref)
    exact = sum(1 for g in ref if ref[g] == mean_scores[g])
    show(f'Regression check vs {ref_path}: max |diff| = {worst:.3e}, '
         f'{exact:,}/{len(ref):,} exact float matches')
    return worst < 1e-6


def main():
    parser = argparse.ArgumentParser(
        description='Aggregation-variant (mean/max/p95 per-event NLL) '
                    'scoring from a frozen BehaviorNet checkpoint.')
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=5000)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--decay-rate', type=float, default=0.1)
    parser.add_argument('--max-norm', type=float, default=100.0)
    parser.add_argument('--relational', action='store_true')
    parser.add_argument('--rel-bases', type=int, default=8)
    parser.add_argument('--flag-quantile', type=float, default=0.999)
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--ref-scores', type=str, default=None,
                        help='Existing mean-based scores_windowed.txt for '
                             'the same checkpoint; recomputed mean scores '
                             'must match it (faithfulness check).')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'

    for p in (train_path, test_path, node_vocab_path, args.checkpoint):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')
    os.makedirs(args.out_dir, exist_ok=True)

    node_vocab = load_node_vocab(node_vocab_path)
    feature_map, label_map = build_type_vocab(train_path)
    profile_dim = len(feature_map) * 2
    num_types = len(label_map)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_relations = len(feature_map) + 1 if args.relational else None
    model = BehaviorNet(num_types, profile_dim, args.hidden_dim,
                        num_relations=num_relations,
                        num_bases=args.rel_bases).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    show(f'Loaded frozen checkpoint: {args.checkpoint} '
         f'(relational={args.relational})')

    memory_dim = args.hidden_dim + profile_dim

    show('Calibration pass (train windows)')
    train_dataset = build_windowed_dataset(
        train_path, node_vocab, feature_map, label_map, args.window_size)
    calib_memory = NodeMemory(len(node_vocab), memory_dim, args.decay_rate,
                              device=device, max_norm=args.max_norm)
    train_scores = score_dataset_aggs(model, train_dataset, calib_memory,
                                      num_types)
    thresholds = {}
    for agg in AGGS:
        thresholds[agg] = nearest_rank_quantile(
            list(train_scores[agg].values()), args.flag_quantile)
        show(f'Operating-point threshold [{agg}]: score >= '
             f'{thresholds[agg]:.6f} (train quantile {args.flag_quantile})')
    del train_dataset, train_scores, calib_memory

    show('Scoring pass (test windows)')
    test_dataset = build_windowed_dataset(
        test_path, node_vocab, feature_map, label_map, args.window_size)
    node_memory = NodeMemory(len(node_vocab), memory_dim, args.decay_rate,
                             device=device, max_norm=args.max_norm)
    score_by_gid = score_dataset_aggs(model, test_dataset, node_memory,
                                      num_types)

    ok = True
    if args.ref_scores:
        ok = regression_check(args.ref_scores, score_by_gid['mean'])

    for agg in AGGS:
        flagged = {gid for gid, score in score_by_gid[agg].items()
                   if score >= thresholds[agg]}
        path = osp.join(args.out_dir, f'scores_windowed_agg{agg}.txt')
        write_scores_file(path, score_by_gid[agg], flagged)
        show(f'Wrote {path} ({len(score_by_gid[agg]):,} nodes, '
             f'{len(flagged):,} flagged)')

    if not ok:
        raise SystemExit('Regression check against --ref-scores FAILED; '
                         'do not use the agg score files.')
    show('Done.')


if __name__ == '__main__':
    main()
