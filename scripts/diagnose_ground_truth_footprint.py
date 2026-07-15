"""Diagnostic 1: ground-truth 2-hop structural footprint.

Tests the specific hypothesis raised after the first real Stage 6 run:
whether Recall = 1.0000 (12,852/12,852 ground-truth nodes credited) is
substantially explained by how much of the test graph structurally lies
within reach of the ground-truth set under Decision 3's whole-file,
symmetric 2-hop credit rule -- independent of anything the model
actually predicted.

This is diagnosis only. It:
- reuses build_whole_file_adjacency() and two_hop_neighbors() from
  test_windowed.py EXACTLY as already implemented and tested -- no new
  adjacency or 2-hop logic is introduced here, and neither function is
  modified or duplicated;
- reuses node_vocab.load_node_vocab() the same way;
- loads the ALREADY-RESOLVED ground-truth ids from
  groundtruth_global_id.txt (the artifact test_windowed.py already wrote
  during the real run), rather than re-deriving the uuid -> global_id
  resolution -- so this is exactly the resolved set the real evaluation
  used, with zero duplication of that resolution logic either;
- does NOT load the model, run inference, or touch NodeMemory at all --
  this is a pure graph-structure question, answerable from the
  provenance file and the node vocabulary alone.

One thing worth knowing before reading the output: build_whole_file_
adjacency() resolves purely by UUID membership in node_vocab -- it does
not apply the node-type / edge-type vocabulary filtering that
windowed_data.py's window construction applies for actual inference.
So the "distinct nodes spanned by the whole-file adjacency" printed
below for context is not guaranteed to be identical to the 357,174-node
population the real evaluation ran over -- if the two are close, that
gap doesn't matter; if they diverge substantially, that's worth knowing
too. This script reports both rather than assuming.

Run directly:
    python diagnose_ground_truth_footprint.py --scene cadets

Needs torch AND torch_geometric to import at all, since it imports from
test_windowed.py, which imports SAGENetWithMemory from train_windowed.py
at module level -- even though nothing in this script touches a tensor.
Same limitation as every other test/diagnostic file in this project.
"""

import argparse
import os.path as osp
import time

from node_vocab import load_node_vocab
from test_windowed import build_whole_file_adjacency, two_hop_neighbors


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


def percentile(sorted_values, pct):
    """Linear-interpolation percentile with no numpy/statistics version
    dependency, so behaviour doesn't depend on the Python version this
    happens to run under."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 1: how much of the test graph lies '
                     'within 0-2 hops of the ground-truth set, under '
                     'the existing whole-file symmetric 2-hop rule.'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Defaults to ../models/<scene>_node_vocab.txt')
    parser.add_argument('--groundtruth-global-id', type=str, default=None,
                        help='Defaults to '
                             '../models/windowed_<scene>/eval/groundtruth_global_id.txt '
                             '-- the artifact test_windowed.py already wrote.')
    parser.add_argument('--distinct-test-nodes', type=int, default=357174,
                        help='The already-established distinct-node count '
                             'from the real test_windowed.py run -- used '
                             'only as the reporting denominator, not '
                             're-derived here.')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    gt_path = (args.groundtruth_global_id
              or f'../models/windowed_{args.scene}/eval/groundtruth_global_id.txt')

    for p in (test_path, node_vocab_path, gt_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')

    # Step 1: load the already-resolved ground-truth global ids.
    ground_truth_ids = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line:
                ground_truth_ids.append(int(line))
    show(f'Loaded {len(ground_truth_ids):,} resolved ground-truth global ids '
         f'from {gt_path}')

    show(f'Loading node vocab: {node_vocab_path}')
    node_vocab = load_node_vocab(node_vocab_path)
    show(f'{len(node_vocab):,} global node identities')

    # Step 2: build the whole-file adjacency -- the exact existing,
    # already-tested function, unchanged and unduplicated.
    show('Building whole-file adjacency (build_whole_file_adjacency, unmodified)')
    t0 = time.time()
    adj, adj2 = build_whole_file_adjacency(test_path, node_vocab)
    show(f'Adjacency built in {time.time() - t0:.1f}s')

    # Context only, not one of the required report numbers: how many
    # distinct nodes does the whole-file adjacency itself span, before
    # restricting to ground truth at all. See the module docstring for
    # why this can differ from the 357,174-node windowed-inference count.
    adjacency_span = len(set(adj.keys()) | set(adj2.keys()))
    show(f'For context: whole-file adjacency spans {adjacency_span:,} '
         f'distinct nodes with at least one resolved edge (compare to '
         f'{args.distinct_test_nodes:,} nodes in the actual windowed '
         f'inference run)')

    # Step 3 + 4: for every resolved ground-truth node, compute its
    # existing symmetric 2-hop neighborhood (two_hop_neighbors, unchanged)
    # and fold it into the running union, together with the ground-truth
    # node itself -- two_hop_neighbors returns neighbors only, exactly
    # matching how test_windowed.py's own alarm-writing loop adds the
    # node itself separately from its neighbor list.
    show(f'Computing 2-hop neighborhoods for {len(ground_truth_ids):,} '
         f'ground-truth nodes -- this was the slow step for the '
         f'31,890-node alarm set before, so expect a comparable wait '
         f'here too (fewer nodes, so proportionally less time)')
    t0 = time.time()

    footprint = set()
    neighborhood_sizes = []
    found_in_graph = 0
    progress_every = 2000

    for i, gid in enumerate(ground_truth_ids, start=1):
        footprint.add(gid)
        neighbors = two_hop_neighbors(gid, adj, adj2)
        footprint.update(neighbors)
        neighborhood_sizes.append(len(neighbors))
        if gid in adj or gid in adj2:
            found_in_graph += 1
        if i % progress_every == 0:
            show(f'  ... {i:,}/{len(ground_truth_ids):,} ground-truth '
                 f'nodes processed')

    show(f'2-hop expansion completed in {time.time() - t0:.1f}s')

    # Step 5: the required report.
    footprint_size = len(footprint)
    pct_of_test = 100 * footprint_size / args.distinct_test_nodes
    pct_found = 100 * found_in_graph / len(ground_truth_ids) if ground_truth_ids else 0.0

    print()
    print('=' * 70)
    print('Diagnostic 1: ground-truth 2-hop structural footprint')
    print('=' * 70)
    print(f'Resolved ground-truth nodes processed:       {len(ground_truth_ids):,}')
    print(f'Ground-truth nodes found in the test graph:  {found_in_graph:,} '
          f'({pct_found:.2f}%)')
    print(f'Distinct nodes in the 0-2-hop union:          {footprint_size:,}')
    print(f'As a percentage of {args.distinct_test_nodes:,} distinct test nodes: '
          f'{pct_of_test:.2f}%')

    # Step 6: per-node neighborhood size statistics -- cheap, since sizes
    # were already recorded during the loop above; no extra traversal.
    if neighborhood_sizes:
        sizes_sorted = sorted(neighborhood_sizes)
        n = len(sizes_sorted)
        print()
        print('Per-node 2-hop neighborhood size (neighbors only, not '
              'counting the ground-truth node itself):')
        print(f'  min:    {sizes_sorted[0]:,}')
        print(f'  p25:    {percentile(sizes_sorted, 25):.1f}')
        print(f'  median: {percentile(sizes_sorted, 50):.1f}')
        print(f'  mean:   {sum(sizes_sorted) / n:.1f}')
        print(f'  p75:    {percentile(sizes_sorted, 75):.1f}')
        print(f'  p90:    {percentile(sizes_sorted, 90):.1f}')
        print(f'  p95:    {percentile(sizes_sorted, 95):.1f}')
        print(f'  p99:    {percentile(sizes_sorted, 99):.1f}')
        print(f'  max:    {sizes_sorted[-1]:,}')
    print('=' * 70)


if __name__ == '__main__':
    main()