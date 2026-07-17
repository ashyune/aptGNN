"""Diagnostic (2026-07-17, read-only): would cheap per-node per-window
statistical features -- computable from the already-parsed 6-column TSV,
with no re-parse and no model -- separate the 46 raw-GT CADETS nodes from
the benign test population?

Motivation: the 2026-07-16 aggregation test showed the 10x operating-point
gap to Orthrus is representational (the 56-dim edge-type profile carries
too little information), not a scoring-rule problem. Before building any
feature enrichment, this probe measures each candidate feature's marginal
separation directly, so the build decision rests on data instead of
guesses.

Method: exact pipeline windowing (windowing.generate_windows, 5000 rows,
time-sorted) and the exact build_window_data row filter (drop rows whose
src/dst/edge type is missing from the train-file vocab). Per (node,
window) we compute the features below, aggregate per node with MAX over
windows (matching the pipeline's per-node aggregation), and rank the 46
GT nodes inside the 357,174-node test population on each raw feature
alone. Ranks are tie-aware: optimistic = 1 + #strictly-greater,
pessimistic = #greater-or-equal (a feature whose top value is a large tie
block, e.g. span_frac == 1.0, must not fake a top-46 hit).

Features (all from the six parsed fields: src/dst uuid+type, edge type,
timestamp):
  n_events   total events the node participates in (either role)
  n_distinct distinct counterpart uuids (either role)
  rep_ratio  n_events / n_distinct (partner re-use)
  fan_out    distinct counterparts where node is src
  fan_in     distinct counterparts where node is dst
  n_etypes   distinct (edge_type, role) categories
  burst_cv   coefficient of variation of inter-event gaps (>= 3 events
             and mean gap > 0, else 0)
  span_frac  (node last ts - node first ts) / window span

This script only reads data files and prints; it writes nothing and is
imported by nothing in the pipeline.
"""
import bisect
from collections import Counter, defaultdict

from windowing import generate_windows, build_type_vocab, show

BASE = '../graphchi-cpp-master/graph_data/darpatc/'
VOCAB = '../models/cadets_node_vocab.txt'
GT = '../models/windowed_cadets/groundtruth_raw_global_id.txt'
WINDOW = 5000
FEATS = ['n_events', 'n_distinct', 'rep_ratio', 'fan_out', 'fan_in',
         'n_etypes', 'burst_cv', 'span_frac']


def profile_test_nodes(feature_map, label_map):
    """Per-node MAX over windows of each feature, plus node type."""
    node_max = defaultdict(lambda: [0.0] * len(FEATS))
    node_type = {}
    for rows in generate_windows(BASE + 'cadets_test.txt', WINDOW):
        ev = defaultdict(int)
        partners = defaultdict(set)
        outs = defaultdict(set)
        ins = defaultdict(set)
        etypes = defaultdict(set)
        ts_list = defaultdict(list)
        w_first, w_last = None, None
        for src, st, dst, dt, et, ts in rows:
            if st not in label_map or dt not in label_map \
                    or et not in feature_map:
                continue
            if w_first is None:
                w_first = ts
            w_last = ts
            ev[src] += 1
            ev[dst] += 1
            partners[src].add(dst)
            partners[dst].add(src)
            outs[src].add(dst)
            ins[dst].add(src)
            etypes[src].add((et, 0))
            etypes[dst].add((et, 1))
            ts_list[src].append(ts)
            ts_list[dst].append(ts)
            node_type[src] = st
            node_type[dst] = dt
        wspan = (w_last - w_first) \
            if (w_first is not None and w_last > w_first) else 0
        for n, cnt in ev.items():
            nd = len(partners[n])
            tl = ts_list[n]
            bcv = 0.0
            if len(tl) >= 3:
                gaps = [tl[i + 1] - tl[i] for i in range(len(tl) - 1)]
                m = sum(gaps) / len(gaps)
                if m > 0:
                    var = sum((g - m) ** 2 for g in gaps) / len(gaps)
                    bcv = var ** 0.5 / m
            sf = (tl[-1] - tl[0]) / wspan if wspan > 0 else 0.0
            vals = [float(cnt), float(nd), cnt / nd, float(len(outs[n])),
                    float(len(ins[n])), float(len(etypes[n])), bcv, sf]
            mx = node_max[n]
            for i, v in enumerate(vals):
                if v > mx[i]:
                    mx[i] = v
    return dict(node_max), node_type


def main():
    feature_map, label_map = build_type_vocab(BASE + 'cadets_train.txt')

    gid_to_uuid = {}
    with open(VOCAB) as f:
        for line in f:
            u, g = line.rstrip('\n').split('\t')
            gid_to_uuid[int(g)] = u
    gt_uuids = set()
    with open(GT) as f:
        for line in f:
            gt_uuids.add(gid_to_uuid[int(line.strip())])
    show(f'{len(gt_uuids)} GT uuids resolved')

    node_max, node_type = profile_test_nodes(feature_map, label_map)
    show(f'{len(node_max):,} test nodes profiled')

    gt_in = [u for u in gt_uuids if u in node_max]
    comp = Counter(node_type[u] for u in gt_in)
    print(f'\nGT nodes present in test windows: {len(gt_in)}/{len(gt_uuids)}')
    print('GT composition by node type:', dict(comp))

    N = len(node_max)
    for fi, fname in enumerate(FEATS):
        asc = sorted(node_max[n][fi] for n in node_max)
        ranked = []
        for u in gt_in:
            v = node_max[u][fi]
            opt = N - bisect.bisect_right(asc, v) + 1
            pes = N - bisect.bisect_left(asc, v)
            ranked.append((opt, pes, v, node_type[u], u))
        ranked.sort()
        t46o = sum(1 for r in ranked if r[0] <= 46)
        t46p = sum(1 for r in ranked if r[1] <= 46)
        med = ranked[len(ranked) // 2][0]
        print(f'\n{fname}: GT median opt rank {med:,} | '
              f'top-46 optimistic={t46o} pessimistic={t46p}')
        for opt, pes, v, t, u in ranked[:8]:
            print(f'  opt {opt:>7,} pes {pes:>9,} val {v:12.4f} {t} {u}')

    print('\nWithin-type median GT percentile '
          '(among same-type test nodes; higher = more extreme):')
    by_type = defaultdict(list)
    for n in node_max:
        by_type[node_type[n]].append(n)
    for t, members in by_type.items():
        gt_t = [u for u in gt_in if node_type[u] == t]
        if not gt_t:
            continue
        line = f'  {t} ({len(gt_t)} GT / {len(members):,} nodes):'
        for fi, fname in enumerate(FEATS):
            asc = sorted(node_max[n][fi] for n in members)
            pcts = sorted(
                bisect.bisect_left(asc, node_max[u][fi]) / len(asc)
                for u in gt_t)
            line += f' {fname}={pcts[len(pcts) // 2] * 100:.1f}%'
        print(line)


if __name__ == '__main__':
    main()
