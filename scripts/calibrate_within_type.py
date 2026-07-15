"""Within-node-type quantile calibration of scores_windowed.txt (reporting-only).

Motivation (D3 follow-up, 2026-07-14): the raw max-NLL score is not
comparable across CDM node types -- each type has its own prediction
difficulty, so a global ranking lets the hardest *type* dominate the top
of the list rather than the most anomalous *node within its type*. On the
old w50000 checkpoint, replacing each raw score with its within-type
quantile lifted raw-GT AUPRC ~4x with no model change. This script makes
that transformation a reproducible reporting step instead of a scratchpad
analysis.

What it does -- and deliberately nothing else:
- Reads a scores_windowed.txt ('<global_id> <score> <flag>').
- Assigns each scored node its CDM node type by streaming the raw test
  file once (uuid -> type from the src/dst columns) and mapping uuids
  through the Stage 1 node vocabulary. Rows with types outside the
  6-type vocabulary were already dropped during scoring, so every scored
  gid is expected to resolve; unresolved gids are kept in an 'UNKNOWN'
  group and counted loudly rather than silently dropped.
- Replaces each score with its mid-rank quantile within its type group:
      cal = (n_strictly_less + 0.5 * n_equal) / n_group
  Mid-rank keeps ties tied (no fabricated order within a saturated
  group) while remaining a proper quantile in (0, 1].
- Writes '<global_id> <calibrated_score> <flag>' with the flag column
  COPIED THROUGH UNCHANGED: the operating point stays defined on the raw
  score's train-quantile threshold; calibration affects ranking metrics
  only. Evaluate the output with evaluate_windowed.py exactly as usual.

The evaluation protocol itself is untouched: still node-level, still no
neighborhood credit, still both ground-truth variants reported.
"""

import argparse
from collections import Counter

from node_vocab import load_node_vocab


def load_scores(scores_path):
    """Return list of (gid, score, flag) preserving file order."""
    rows = []
    with open(scores_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            gid_str, score_str, flag_str = line.split(' ')
            rows.append((int(gid_str), float(score_str), flag_str))
    return rows


def build_gid_type_map(test_path, node_vocab, wanted_gids):
    """Map global_id -> node-type string for every gid in *wanted_gids*.

    First type seen wins; conflicting later observations are counted and
    reported by the caller (CADETS uuids are not expected to change type,
    but silence would hide it if one ever did).
    """
    gid_type = {}
    conflicts = 0
    with open(test_path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            for uuid, ntype in ((temp[0], temp[1]), (temp[2], temp[3])):
                gid = node_vocab.get(uuid)
                if gid is None or gid not in wanted_gids:
                    continue
                prev = gid_type.get(gid)
                if prev is None:
                    gid_type[gid] = ntype
                elif prev != ntype:
                    conflicts += 1
    return gid_type, conflicts


def within_type_quantiles(rows, gid_type):
    """Return {gid: mid-rank quantile of its score within its type group}."""
    by_type = {}
    for gid, score, _flag in rows:
        by_type.setdefault(gid_type.get(gid, 'UNKNOWN'), []).append(score)

    # Per type: sorted scores + tie counts, then mid-rank quantile lookup.
    lookup = {}
    for ntype, scores in by_type.items():
        scores.sort()
        n = len(scores)
        counts = Counter(scores)
        quantile = {}
        n_less = 0
        for s in sorted(counts):
            c = counts[s]
            quantile[s] = (n_less + 0.5 * c) / n
            n_less += c
        lookup[ntype] = quantile

    return {
        gid: lookup[gid_type.get(gid, 'UNKNOWN')][score]
        for gid, score, _flag in rows
    }


def main():
    parser = argparse.ArgumentParser(
        description='Within-node-type quantile calibration of a '
                    'scores_windowed.txt (ranking metrics only; flag '
                    'column passes through unchanged).')
    parser.add_argument('--scores-file', type=str, required=True)
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--test-file', type=str, default=None,
                        help='Raw provenance test file used to recover '
                             'each scored node\'s CDM type. Defaults to '
                             '../graphchi-cpp-master/graph_data/darpatc/'
                             '<scene>_test.txt')
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Defaults to ../models/<scene>_node_vocab.txt')
    parser.add_argument('--out', type=str, default=None,
                        help='Defaults to <scores-file dir>/'
                             'scores_windowed_caltype.txt')
    args = parser.parse_args()

    test_path = args.test_file or (
        f'../graphchi-cpp-master/graph_data/darpatc/{args.scene}_test.txt')
    vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    out_path = args.out or args.scores_file.rsplit('/', 1)[0] + \
        '/scores_windowed_caltype.txt'

    rows = load_scores(args.scores_file)
    node_vocab = load_node_vocab(vocab_path)
    wanted = {gid for gid, _s, _f in rows}

    gid_type, conflicts = build_gid_type_map(test_path, node_vocab, wanted)
    unknown = len(wanted) - len(gid_type)
    calibrated = within_type_quantiles(rows, gid_type)

    with open(out_path, 'w') as fw:
        for gid, _score, flag in rows:
            fw.write(f'{gid} {calibrated[gid]} {flag}\n')

    group_sizes = Counter(gid_type.get(gid, 'UNKNOWN') for gid in wanted)
    print(f'Calibrated {len(rows):,} nodes -> {out_path}')
    for ntype, n in group_sizes.most_common():
        print(f'  {ntype}: {n:,}')
    print(f'Type conflicts: {conflicts} | unresolved gids (UNKNOWN group): '
          f'{unknown}')


if __name__ == '__main__':
    main()
