"""Full comparison table for the aggregation-scoring experiment.

For every arm (E1/E2 x seed) and every aggregation (mean from the
original seedrun files, max/p95 from aggscoring outputs), on raw 46-node
GT: AUPRC, AUROC, P@46 (via scripts/ranking_metrics.py — same verified
code as the headline reports), flag operating point (TP/FP, precision,
MCC), top-46 operating point (TP, precision, MCC), and the GT rank
distribution (median rank + rank of the 24th GT node = what must be
<=46 for Orthrus-level precision at a 46-alarm budget).
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/tetsuya/aptGNN/scripts')
from ranking_metrics import auroc, auprc, precision_at_k

RAWGT = '/home/tetsuya/aptGNN/models/windowed_cadets/groundtruth_raw_global_id.txt'
AGGDIR = '/home/tetsuya/aptGNN/models/behavior_cadets/aggscoring'
ORIG = {
    'E1': '/home/tetsuya/aptGNN/models/behavior_cadets/seedrun/s{s}/mem/eval/scores_windowed.txt',
    'E2': '/home/tetsuya/aptGNN/models/behavior_cadets/e2seedrun/s{s}/eval/scores_windowed.txt',
}
SEEDS = [101, 202, 303]

gt = {int(l) for l in open(RAWGT) if l.strip()}
K = len(gt)


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def analyze(path, label):
    gids, scores, flags = [], [], []
    with open(path) as f:
        for line in f:
            g, sc, fl = line.split()
            gids.append(int(g)); scores.append(float(sc)); flags.append(int(fl))
    n = len(gids)
    labels = [1 if g in gt else 0 for g in gids]
    assert sum(labels) == K, path

    a_pr = auprc(scores, labels)
    a_roc = auroc(scores, labels)
    p_at_k = precision_at_k(scores, labels, K)

    # flag operating point
    tp = sum(1 for l, fl in zip(labels, flags) if fl and l)
    fp = sum(1 for l, fl in zip(labels, flags) if fl and not l)
    fn, tn = K - tp, n - K - (sum(flags) - tp)
    m_flag = mcc(tp, fp, fn, tn)
    p_flag = tp / (tp + fp) if tp + fp else 0.0
    flag_str = f'{tp}/{fp}'

    # top-46 operating point + GT ranks (desc by score, gid tie-break)
    order = sorted(range(n), key=lambda i: (-scores[i], gids[i]))
    ranks = [r for r, i in enumerate(order, 1) if labels[i] == 1]
    tp46 = sum(1 for i in order[:K] if labels[i] == 1)
    m46 = mcc(tp46, K - tp46, K - tp46, n - 2 * K + tp46)
    med = statistics.median(ranks)
    r24 = ranks[23]

    print(f'{label:14s} AUPRC={a_pr:.4f} AUROC={a_roc:.4f} P@46={p_at_k:.4f} | '
          f'flag TP/FP={flag_str:9s} P={p_flag:.4f} MCC={m_flag:.4f} | '
          f'top46 TP={tp46:2d} P={tp46/K:.4f} MCC={m46:.4f} | '
          f'GTrank med={med:>8.0f} 24th={r24:>6d} min={ranks[0]}')


for arm in ('E1', 'E2'):
    for s in SEEDS:
        analyze(ORIG[arm].format(s=s), f'{arm} s{s} mean')
        for agg in ('max', 'p95'):
            analyze(f'{AGGDIR}/{arm.lower()}_s{s}/scores_windowed_agg{agg}.txt',
                    f'{arm} s{s} {agg}')
        print()
