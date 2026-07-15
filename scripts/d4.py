"""
Diagnostic 4 -- is the REAL alarm set's near-total footprint (99.51%)
and perfect recall (1.0000) attributable to a small number of extreme
per-node ground-truth reach values -- mirroring what Trial 9's random
sample stumbled onto (node 608335: 12,835 2-hop neighbours, of which
12,834 are ground truth) -- or is coverage distributed across many
alarm nodes with moderate reach?

Single question answered:
    For every node actually in ever_flagged (not a random sample), how
    many ground-truth nodes fall within {node} union its own stored
    2-hop neighbours? Ranked descending: is 608335 present? Does a
    small number of top-ranked real alarm nodes already account for
    (close to) the full 12,852, the way one node did in Trial 9?

No recomputation of adjacency or 2-hop neighbours: alarm_windowed.txt
already stores each real alarm node's neighbour list from the
completed Stage 6 run -- the same file Diagnostic 2 read. No model
loading, no inference, no NodeMemory, no torch/torch_geometric import
-- same zero-dependency profile as evaluate_windowed.py itself.

Explicitly NOT done here: any claim about WHY a node has extreme
reach; degree distribution over the whole graph; anything about the
original (non-windowed) baseline. Candidates for later, contingent on
what comes back here.
"""

import argparse

WATCH_ID = 608335  # identified in Diagnostic 3's Trial 9 audit


def per_node_gt_reach(alarm_path, gt_path):
    gt = set()
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line:
                gt.add(int(line))

    total_nodes = None
    entries = []  # (gid, gt_reach, neighbour_count, hit_set)

    with open(alarm_path, 'r') as f:
        for line in f:
            if line == '\n':
                continue
            if ':' not in line:
                total_nodes = int(line.strip('\n'))
                continue
            line = line.strip('\n')
            gid_str, neighbours_str = line.split(':')
            gid = int(gid_str)
            neighbours = [int(x) for x in neighbours_str.strip(' ').split(' ') if x != '']

            hit = set(neighbours) & gt
            if gid in gt:
                hit.add(gid)
            entries.append((gid, len(hit), len(neighbours), hit))

    return total_nodes, gt, entries


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 4: per-node ground-truth reach within '
                    'the real alarm set.')
    parser.add_argument('--alarm-file', type=str, default=None)
    parser.add_argument('--groundtruth-file', type=str, default=None)
    parser.add_argument('--out-dir', type=str,
                        default='../models/windowed_cadets/eval')
    parser.add_argument('--top-k', type=int, default=15)
    args = parser.parse_args()

    alarm_path = args.alarm_file or f'{args.out_dir}/alarm_windowed.txt'
    gt_path = args.groundtruth_file or f'{args.out_dir}/groundtruth_global_id.txt'

    total_nodes, gt, entries = per_node_gt_reach(alarm_path, gt_path)
    entries_sorted = sorted(entries, key=lambda e: e[1], reverse=True)

    print(f"Real alarm set: {len(entries):,} entries | ground truth: {len(gt):,}\n")

    watch = next((e for e in entries if e[0] == WATCH_ID), None)
    if watch:
        print(f"Watch node {WATCH_ID}: PRESENT in ever_flagged -- "
              f"2-hop neighbours={watch[2]:,}, GT reached={watch[1]:,}\n")
    else:
        print(f"Watch node {WATCH_ID}: NOT in ever_flagged.\n")

    print("Cumulative GT reach as top-ranked real alarm nodes are added:")
    cumulative = set()
    for rank, (gid, gt_reach, deg, hit) in enumerate(entries_sorted[:args.top_k], start=1):
        cumulative |= hit
        print(f"  #{rank:>2} node {gid}: alone reaches {gt_reach:,} GT nodes "
              f"(2-hop size {deg:,}) | cumulative: {len(cumulative):,} of "
              f"{len(gt):,} ({100*len(cumulative)/len(gt):.2f}%)")

    zero_reach = sum(1 for e in entries if e[1] == 0)
    print(f"\nReal alarm entries with gt_reach == 0 (candidate-fp, isolated "
          f"from ground truth): {zero_reach:,} of {len(entries):,}")
    print("(cross-check: should be close to the ~24,190-24,200 fp floor "
          "derived from Diagnostic 2's P=0.3469, R=1.0000)")


if __name__ == '__main__':
    main()