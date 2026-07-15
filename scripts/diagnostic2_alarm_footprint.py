"""
Diagnostic 2 -- structural footprint of the PREDICTION side.

Single question answered:
    How large is the alarm set the completed Stage 6 test_windowed.py
    run produced -- raw count, and 2-hop reach -- relative to the whole
    test graph? Direct counterpart to Diagnostic 1, computed for
    predictions instead of ground truth.

Explicitly NOT computed here (candidates for later, not this one):
overlap with Diagnostic 1's ground-truth footprint, per-node degree /
neighbour-list-size distribution, or a breakdown of alarm nodes into
direct-hit / near-miss-unpenalised / isolated-fp.
"""

import argparse
from evaluate_windowed import score_alarm_file


def alarm_footprint_stats(alarm_path):
    """Parse alarm_windowed.txt directly (same format score_alarm_file
    reads). Returns the two sizes Diagnostic 1 reported for ground
    truth, computed the same way for predictions.
    """
    total_nodes = None
    alarm_nodes = set()
    footprint = set()

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

            alarm_nodes.add(gid)
            footprint.add(gid)
            footprint.update(neighbours)

    if total_nodes is None:
        raise ValueError(f'{alarm_path} is missing its total-node header line')

    return {
        'total_nodes': total_nodes,
        'raw_alarm_count': len(alarm_nodes),
        'footprint_count': len(footprint),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 2: structural footprint of the alarm set.')
    parser.add_argument('--alarm-file', type=str, default=None)
    parser.add_argument('--groundtruth-file', type=str, default=None)
    parser.add_argument('--out-dir', type=str,
                        default='../models/windowed_cadets/eval')
    args = parser.parse_args()

    alarm_path = args.alarm_file or f'{args.out_dir}/alarm_windowed.txt'
    gt_path = args.groundtruth_file or f'{args.out_dir}/groundtruth_global_id.txt'

    # Confirm this is the same alarm file the reported numbers came
    # from, before trusting anything new read from it.
    scored = score_alarm_file(alarm_path, gt_path)
    print(f"Cross-check: P={scored['precision']:.4f} "
          f"R={scored['recall']:.4f} F={scored['fscore']:.4f} "
          f"(expect ~0.3469 / 1.0000 / 0.5152 -- if this doesn't match, "
          f"stop: wrong alarm file)\n")

    stats = alarm_footprint_stats(alarm_path)
    total = stats['total_nodes']

    print(f"Whole test sequence:            {total:,} nodes "
          f"(Diagnostic 1: 357,174 -- should match)")
    print(f"Raw alarm set (ever_flagged):    {stats['raw_alarm_count']:,} nodes "
          f"({100 * stats['raw_alarm_count'] / total:.2f}% of test nodes)")
    print(f"Predicted structural footprint:  {stats['footprint_count']:,} nodes "
          f"({100 * stats['footprint_count'] / total:.2f}% of test nodes)")
    print(f"\nFor reference (Diagnostic 1, ground truth):")
    print(f"  Ground-truth nodes: 12,852")
    print(f"  Ground-truth structural footprint: 38,444 nodes (10.76%)")


if __name__ == '__main__':
    main()