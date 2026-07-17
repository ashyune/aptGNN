"""Full comparison table for the E3 feature-enrichment experiment.

For every arm (E1 / E2 from the existing seedrun score files, E3 from
this run), on raw 46-node GT: AUPRC, AUROC, P@46 (via
scripts/ranking_metrics.py -- same verified code as the headline
reports), flag operating point (TP/FP, precision, MCC), top-46 operating
point (TP, precision, MCC), and the GT rank distribution (median rank +
rank of the 24th GT node + best rank). Also prints, per E3 seed, WHICH
GT uuids sit in the top 46, so the survey's prediction (new hits should
come from the feature-extreme set, complementary to E1's) is checkable
directly.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/tetsuya/aptGNN/scripts')
from ranking_metrics import auroc, auprc, precision_at_k

RAWGT = '/home/tetsuya/aptGNN/models/windowed_cadets/groundtruth_raw_global_id.txt'
VOCAB = '/home/tetsuya/aptGNN/models/cadets_node_vocab.txt'
ARMS = {
    'E1': '/home/tetsuya/aptGNN/models/behavior_cadets/seedrun/s{s}/mem/eval/scores_windowed.txt',
    'E2': '/home/tetsuya/aptGNN/models/behavior_cadets/e2seedrun/s{s}/eval/scores_windowed.txt',
    'E3': '/home/tetsuya/aptGNN/models/behavior_cadets/e3seedrun/s{s}/eval/scores_windowed.txt',
}
SEEDS = [101, 202, 303]

gt = {int(l) for l in open(RAWGT) if l.strip()}
K = len(gt)
gid_to_uuid = {}
with open(VOCAB) as f:
    for line in f:
        u, g = line.rstrip('\n').split('\t')
        gid_to_uuid[int(g)] = u


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def analyze(path, label, show_hits=False):
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

    tp = sum(1 for l, fl in zip(labels, flags) if fl and l)
    fp = sum(1 for l, fl in zip(labels, flags) if fl and not l)
    fn, tn = K - tp, n - K - (sum(flags) - tp)
    m_flag = mcc(tp, fp, fn, tn)
    p_flag = tp / (tp + fp) if tp + fp else 0.0

    order = sorted(range(n), key=lambda i: (-scores[i], gids[i]))
    ranks = [r for r, i in enumerate(order, 1) if labels[i] == 1]
    tp46 = sum(1 for i in order[:K] if labels[i] == 1)
    m46 = mcc(tp46, K - tp46, K - tp46, n - 2 * K + tp46)
    med = statistics.median(ranks)
    r24 = ranks[23]

    print(f'{label:8s} AUPRC={a_pr:.4f} AUROC={a_roc:.4f} P@46={p_at_k:.4f} | '
          f'flag TP/FP={tp}/{fp:<5d} P={p_flag:.4f} MCC={m_flag:.4f} | '
          f'top46 TP={tp46:2d} P={tp46/K:.4f} MCC={m46:.4f} | '
          f'GTrank med={med:>8.0f} 24th={r24:>6d} min={ranks[0]}')
    if show_hits:
        hits = [(r, gid_to_uuid[gids[i]]) for r, i in
                enumerate(order[:K], 1) if labels[i] == 1]
        print(f'         top-46 GT hits: '
              f'{[(r, u[:8]) for r, u in hits]}')


for s in SEEDS:
    for arm in ('E1', 'E2', 'E3'):
        analyze(ARMS[arm].format(s=s), f'{arm} s{s}', show_hits=(arm == 'E3'))
    print()
