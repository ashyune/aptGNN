"""
Verification pass on Diagnostic 3, trial 9 -- NOT a new hypothesis test.

Trial 9's actual sample was never saved to disk (it lived in a
tempfile.TemporaryDirectory() that no longer exists), but is fully
reproducible: replaying the same sequence of random.Random(SEED)
.sample() calls against the same population regenerates it exactly,
since int hashing is deterministic and unaffected by PYTHONHASHSEED.

If you ran Diagnostic 3 with a non-default --seed or --trials, update
SEED/TRIALS below to match, or this won't reproduce trial 9.
"""

import math
import random

from node_vocab import load_node_vocab
from test_windowed import build_whole_file_adjacency, two_hop_neighbors

SCENE = 'cadets'
SEED = 0
SAMPLE_SIZE = 32120
TRIALS = 10

base = '../graphchi-cpp-master/graph_data/darpatc/'
test_path = base + SCENE + '_test.txt'
node_vocab = load_node_vocab(f'../models/{SCENE}_node_vocab.txt')
adj, adj2 = build_whole_file_adjacency(test_path, node_vocab)
population = list(set(adj) | set(adj2))

gt = set()
with open('../models/windowed_cadets/eval/groundtruth_global_id.txt') as f:
    for line in f:
        line = line.strip('\n')
        if line:
            gt.add(int(line))

# Replay the exact same draw sequence Diagnostic 3 made.
rng = random.Random(SEED)
samples = [rng.sample(population, SAMPLE_SIZE) for _ in range(TRIALS)]

# --- Check 1: direct overlap, trial 9 vs trial 0 vs chance ---
N, K, n = len(population), len(gt), SAMPLE_SIZE
mean = n * K / N
var = n * (K / N) * (1 - K / N) * (N - n) / (N - 1)
print(f"Hypergeometric expectation for direct overlap: {mean:.1f} (std ~{math.sqrt(var):.1f})\n")
for t in (0, 9):
    direct = len(set(samples[t]) & gt)
    print(f"Trial {t}: direct GT overlap = {direct}")

# --- Check 2: per-node reach within trial 9's sample ---
per_node = []
for gid in samples[9]:
    nbs = two_hop_neighbors(gid, adj, adj2)
    gt_hit = len(nbs & gt) + (1 if gid in gt else 0)
    per_node.append((gid, len(nbs), gt_hit))

per_node.sort(key=lambda x: x[2], reverse=True)
print("\nTop 10 nodes in trial 9's sample by GT nodes reached:")
for gid, deg, gt_hit in per_node[:10]:
    print(f"  node {gid}: 2-hop neighbours={deg:,}, GT nodes reached={gt_hit:,}")

top_gt_union = set()
for gid, _, _ in per_node[:10]:
    top_gt_union |= (two_hop_neighbors(gid, adj, adj2) & gt)
print(f"\nGT nodes reached by just these top 10: {len(top_gt_union):,} of {len(gt):,} total")