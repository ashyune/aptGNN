"""Stage 6 (revised): score a per-node score file against ground truth.

Reads the two artifacts test_windowed.py now produces -- scores_windowed.txt
and groundtruth_global_id.txt -- and reports:

  Headline (threshold-free ranking):
    AUPRC   -- primary; imbalance-appropriate, no threshold needed
    AUROC   -- secondary; comparable to MAGIC/KAIROS, but flattering under
               heavy class imbalance, so read it next to AUPRC
    P@|GT|  -- precision among the |GT| highest-scoring nodes; an
               interpretable operating point that needs no threshold

  Operating point (from the flag column):
    TP/FP/FN/TN and Precision/Recall/F1/FPR at the single fixed
    confidence-ratio flag that test_windowed.py wrote per node.

No neighborhood credit anywhere: a node counts only for itself. This
replaces the earlier 2-hop alarm-credit scoring that the design review
found topology-dominated (a single hub reached ~all ground truth, pinning
recall at 1.0).

Ground-truth coverage
---------------------
A ground-truth node that never appears in any scored window has no score.
By default (include_unscored_gt=True) such nodes are:
  - counted as false negatives at the operating point (recall stays honest
    about coverage: its denominator is the full |GT|), and
  - added to the ranking population at the lowest possible score, i.e.
    ranked last, since the detector never surfaced them.
The count of such nodes is always reported so this is never silent. Pass
include_unscored_gt=False to score only over nodes the detector saw.

Zero dependencies beyond the standard library plus ranking_metrics.py --
no torch/torch_geometric anywhere in this file, matching evaluate_darpatc.py.

The legacy score_alarm_file() (the old 2-hop-credit scorer) is retained
UNCHANGED at the bottom of this file because diagnostic2_alarm_footprint.py
and diagnostic3_random_baseline.py still import it. It is no longer used by
main() and is not the reported metric.
"""

import argparse

from ranking_metrics import auprc, auroc, precision_at_k


def score_node_file(scores_path, gt_path, include_unscored_gt=True):
    """Score scores_windowed.txt against groundtruth_global_id.txt.

    Parameters
    ----------
    scores_path : str
        One line per distinct test node: '<global_id> <score> <flag>'.
    gt_path : str
        One ground-truth global_id per line.
    include_unscored_gt : bool
        See module docstring. Default True (honest about coverage).

    Returns
    -------
    dict with keys:
        auprc, auroc, precision_at_k          (ranking; may be None)
        tp, fp, fn, tn                        (operating point)
        precision, recall, f1, fpr            (operating point)
        n_scored, n_gt, n_gt_unscored, prevalence  (bookkeeping)
    """
    eps = 1e-10

    gt = set()
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line:
                gt.add(int(line))

    gids = []
    scores = []
    flags = []
    with open(scores_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            gid_str, score_str, flag_str = line.split(' ')
            gids.append(int(gid_str))
            scores.append(float(score_str))
            flags.append(int(flag_str))

    scored_set = set(gids)
    unscored_gt = gt - scored_set

    # --- Operating point, straight from the flag column ---
    tp = sum(1 for g, fl in zip(gids, flags) if fl == 1 and g in gt)
    fp = sum(1 for g, fl in zip(gids, flags) if fl == 1 and g not in gt)
    tn = sum(1 for g, fl in zip(gids, flags) if fl == 0 and g not in gt)
    fn = sum(1 for g, fl in zip(gids, flags) if fl == 0 and g in gt)
    if include_unscored_gt:
        fn += len(unscored_gt)  # never-seen GT nodes are missed detections

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)

    # --- Ranking population ---
    rank_scores = list(scores)
    rank_labels = [1 if g in gt else 0 for g in gids]
    if include_unscored_gt and unscored_gt:
        # Rank never-surfaced GT nodes at the very bottom.
        for _ in unscored_gt:
            rank_scores.append(float('-inf'))
            rank_labels.append(1)

    n_pos = len(gt)
    n_rank = len(rank_scores)
    prevalence = n_pos / n_rank if n_rank > 0 else None

    return {
        'auprc': auprc(rank_scores, rank_labels),
        'auroc': auroc(rank_scores, rank_labels),
        'precision_at_k': precision_at_k(rank_scores, rank_labels, n_pos),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr,
        'n_scored': len(gids), 'n_gt': len(gt),
        'n_gt_unscored': len(unscored_gt), 'prevalence': prevalence,
    }


def _fmt(x):
    return f'{x:.4f}' if x is not None else 'undefined'


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6 (revised): score scores_windowed.txt against '
                     'ground truth -- AUPRC/AUROC + operating-point P/R/F1.'
    )
    parser.add_argument('--scores-file', type=str, default=None)
    parser.add_argument('--groundtruth-file', type=str, default=None)
    parser.add_argument('--out-dir', type=str,
                        default='../models/windowed_cadets/eval',
                        help='Used only to build defaults for the two '
                             'paths above.')
    parser.add_argument('--exclude-unscored-gt', action='store_true',
                        help='Score only over nodes the detector actually '
                             'saw, rather than counting never-seen '
                             'ground-truth nodes as missed. Off by default.')
    args = parser.parse_args()

    scores_path = args.scores_file or f'{args.out_dir}/scores_windowed.txt'
    gt_path = args.groundtruth_file or f'{args.out_dir}/groundtruth_global_id.txt'

    r = score_node_file(scores_path, gt_path,
                        include_unscored_gt=not args.exclude_unscored_gt)

    print('=== Ranking (threshold-free) ===')
    base = f' (random baseline ~ {r["prevalence"]:.4f})' if r['prevalence'] is not None else ''
    print(f'AUPRC : {_fmt(r["auprc"])}{base}')
    print(f'AUROC : {_fmt(r["auroc"])}')
    print(f'P@|GT|: {_fmt(r["precision_at_k"])}   (k = {r["n_gt"]})')
    print()
    print('=== Operating point (flag column) ===')
    print(f'TP={r["tp"]} FP={r["fp"]} FN={r["fn"]} TN={r["tn"]}')
    print(f'Precision: {r["precision"]:.4f} | Recall: {r["recall"]:.4f} | '
          f'F1: {r["f1"]:.4f} | FPR: {r["fpr"]:.4f}')
    print()
    print(f'Nodes scored: {r["n_scored"]:,} | Ground truth: {r["n_gt"]:,} | '
          f'GT never scored: {r["n_gt_unscored"]:,}')


# ===========================================================================
# LEGACY -- old 2-hop-credit scorer. No longer used by main(); retained only
# because diagnostic2_alarm_footprint.py and diagnostic3_random_baseline.py
# import score_alarm_file. Not the reported metric. Left unchanged.
# ===========================================================================

def score_alarm_file(alarm_path, gt_path):
    """DEPRECATED 2-hop-credit scorer (see module docstring). Unchanged."""
    eps = 1e-10

    gt = set()
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line:
                gt.add(int(line))

    ans = {}
    total_nodes = None
    with open(alarm_path, 'r') as f:
        for line in f:
            if line == '\n':
                continue
            if ':' not in line:
                total_nodes = int(line.strip('\n'))
                continue

            line = line.strip('\n')
            gid_str, neighbours_str = line.split(':')
            a = int(gid_str)
            neighbours = [int(x) for x in neighbours_str.strip(' ').split(' ') if x != '']

            flag = 0
            for b in neighbours:
                if b in gt:
                    ans[b] = 'tp'
                    flag = 1

            if a in gt:
                ans[a] = 'tp'
            elif flag == 0:
                ans[a] = 'fp'

    for g in gt:
        if g not in ans:
            ans[g] = 'fn'

    tp = sum(1 for v in ans.values() if v == 'tp')
    fp = sum(1 for v in ans.values() if v == 'fp')
    fn = sum(1 for v in ans.values() if v == 'fn')
    tn = (total_nodes - len(ans)) if total_nodes is not None else None

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total_nodes': total_nodes,
        'precision': precision, 'recall': recall, 'fscore': fscore,
    }


if __name__ == '__main__':
    main()