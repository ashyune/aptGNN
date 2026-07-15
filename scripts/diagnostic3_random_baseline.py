"""
Diagnostic 3 -- is the 99.51% predicted footprint (and recall = 1.0000)
explained by the RAW SIZE of the alarm set and the graph's topology
alone, or does it depend on which specific 32,120 nodes were flagged?

Single question answered:
    Does a uniformly random sample of 32,120 nodes -- same size as the
    real ever_flagged set, drawn from the same whole-file node
    population -- produce a similar footprint and recall/precision
    against the 12,852 ground-truth nodes, scored through the exact
    same, unmodified credit rule?
"""

import argparse
import os
import random
import tempfile

from node_vocab import load_node_vocab
from test_windowed import build_whole_file_adjacency, two_hop_neighbors
from evaluate_windowed import score_alarm_file


def run_random_trial(population, adj, adj2, k, total_nodes, gt_path, rng, tmp_path):
    sample = rng.sample(population, k)
    footprint = set()

    with open(tmp_path, 'w') as fw:
        fw.write(f'{total_nodes}\n')
        for gid in sample:
            neighbours = two_hop_neighbors(gid, adj, adj2)
            footprint.add(gid)
            footprint.update(neighbours)
            fw.write('\n')
            fw.write(f'{gid}:')
            for nb in neighbours:
                fw.write(f' {nb}')

    scored = score_alarm_file(tmp_path, gt_path)
    return len(footprint), scored


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 3: random-sample baseline for the '
                     'predicted structural footprint.')
    parser.add_argument('--scene', type=str, default='cadets')
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--groundtruth-file', type=str,
                        default='../models/windowed_cadets/eval/groundtruth_global_id.txt')
    parser.add_argument('--sample-size', type=int, default=32120,
                        help="Match Diagnostic 2's observed |ever_flagged|.")
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'

    node_vocab = load_node_vocab(node_vocab_path)
    adj, adj2 = build_whole_file_adjacency(test_path, node_vocab)
    population = list(set(adj) | set(adj2))
    total_nodes = len(population)
    print(f"Whole-file node population: {total_nodes:,} "
          f"(Diagnostic 1/2: 357,174 -- should match)\n")

    rng = random.Random(args.seed)
    footprint_results, scored_results = [], []

    with tempfile.TemporaryDirectory() as d:
        tmp_path = os.path.join(d, 'alarm_random.txt')
        for t in range(args.trials):
            footprint_size, scored = run_random_trial(
                population, adj, adj2, args.sample_size, total_nodes,
                args.groundtruth_file, rng, tmp_path)
            footprint_results.append(footprint_size)
            scored_results.append(scored)
            print(f"Trial {t}: footprint={footprint_size:,} "
                  f"({100*footprint_size/total_nodes:.2f}%) | "
                  f"P={scored['precision']:.4f} R={scored['recall']:.4f} "
                  f"F={scored['fscore']:.4f}")

    fp_pcts = [100 * f / total_nodes for f in footprint_results]
    recalls = [s['recall'] for s in scored_results]
    precisions = [s['precision'] for s in scored_results]

    print(f"\nRandom sample (size={args.sample_size:,}), {args.trials} trials:")
    print(f"  Footprint %: min={min(fp_pcts):.2f} mean={sum(fp_pcts)/len(fp_pcts):.2f} max={max(fp_pcts):.2f}")
    print(f"  Recall:      min={min(recalls):.4f} mean={sum(recalls)/len(recalls):.4f} max={max(recalls):.4f}")
    print(f"  Precision:   min={min(precisions):.4f} mean={sum(precisions)/len(precisions):.4f} max={max(precisions):.4f}")
    print(f"\nActual (model-driven) alarm set: footprint=99.51%, Precision=0.3469, Recall=1.0000")


if __name__ == '__main__':
    main()