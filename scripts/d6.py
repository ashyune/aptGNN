"""
Diagnostic 6 -- per-window flagging rate vs. the cumulative
(ever_flagged) rate.

PRIMARY question:
    Is the 8.99% cumulative alarm rate (32,120 / 357,174) already
    present in a typical single window, or is it much smaller per
    window and only reaches 8.99% because Decision 5 keeps a node
    flagged forever after a single flag in ANY of however many windows
    it appears in?

First diagnostic in this investigation requiring a real inference
replay -- no artifact on disk records per-window detail. Nothing about
the decision logic is reimplemented: flag_nodes(), thre, the frozen
checkpoint, NodeMemory's update order, and the windowed dataset
construction are imported and used exactly as test_windowed.py uses
them. Only the recording step is new.

SECONDARY, lower-priority breakdown, free once this replay is
happening anyway: for each window, how many flagged nodes were seen
for the first time ever (cold, zero prior memory) vs. had appeared in
an earlier window (warm). A DIFFERENT candidate mechanism (Decision
4's cold start) from the primary question -- reported separately, not
this diagnostic's headline conclusion either way.

Explicitly NOT done here: no change to thre, architecture, or
training; no claim about WHY any window's rate is what it is (class
imbalance, feature quality, etc.) -- contingent follow-up.
"""

import argparse
import collections

import torch

from node_vocab import load_node_vocab
from windowing import build_type_vocab
from windowed_data import build_windowed_dataset
from node_memory import NodeMemory
from train_windowed import SAGENetWithMemory
from test_windowed import flag_nodes, THRE_MAP


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 6: per-window vs. cumulative flag rate.')
    parser.add_argument('--scene', type=str, default='cadets')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--window-size', type=int, default=50000)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--decay-rate', type=float, default=0.1)
    parser.add_argument('--max-norm', type=float, default=100.0)
    parser.add_argument('--thre', type=float, default=None)
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--alarm-file', type=str, default=None,
                        help='Existing alarm file to cross-check against.')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    checkpoint_path = args.checkpoint or f'../models/windowed_{args.scene}/best_model.pt'
    alarm_path = args.alarm_file or f'../models/windowed_{args.scene}/eval/alarm_windowed.txt'
    thre = args.thre if args.thre is not None else THRE_MAP[args.scene]

    node_vocab = load_node_vocab(node_vocab_path)
    feature_map, label_map = build_type_vocab(train_path)
    in_channels = len(feature_map) * 2
    out_channels = len(label_map)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGENetWithMemory(in_channels, out_channels, args.hidden_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    test_dataset = build_windowed_dataset(
        test_path, node_vocab, feature_map, label_map, args.window_size)

    node_memory = NodeMemory(len(node_vocab), args.hidden_dim, args.decay_rate,
                             device=device, max_norm=args.max_norm)

    ever_flagged = set()
    seen_before = set()
    appearance_counts = collections.Counter()
    per_window = []  # (idx, n_nodes, n_flagged, n_new, cum, n_cold, n_warm)

    with torch.no_grad():
        for window_idx, data in enumerate(test_dataset):
            if data.num_nodes == 0:
                per_window.append((window_idx, 0, 0, 0, len(ever_flagged), 0, 0))
                continue
            data = data.to(device)
            gids = data.global_id.tolist()
            appearance_counts.update(gids)

            memory = node_memory.get_decayed(data.global_id, window_idx)
            out, hidden = model(data.x, data.edge_index, memory)
            flagged_mask = flag_nodes(out, data.y, thre)
            flagged_gids = set(data.global_id[flagged_mask].tolist())

            n_cold = sum(1 for g in flagged_gids if g not in seen_before)
            n_warm = len(flagged_gids) - n_cold

            newly = flagged_gids - ever_flagged
            ever_flagged.update(flagged_gids)
            seen_before.update(gids)

            per_window.append((window_idx, data.num_nodes, len(flagged_gids),
                                len(newly), len(ever_flagged), n_cold, n_warm))

            node_memory.update(data.global_id, hidden.detach(), window_idx)

    total_nodes = len(appearance_counts)

    # Fidelity check FIRST -- confirm this replay matches the original
    # run before trusting anything computed from it.
    on_disk = set()
    with open(alarm_path, 'r') as f:
        for line in f:
            if line == '\n' or ':' not in line:
                continue
            on_disk.add(int(line.strip('\n').split(':')[0]))

    print(f"Replay |ever_flagged|:      {len(ever_flagged):,}")
    print(f"On-disk alarm_windowed.txt: {len(on_disk):,} (expect 32,120)")
    print(f"Sets identical: {ever_flagged == on_disk}\n")
    if ever_flagged != on_disk:
        print("STOP: this replay diverged from the original run -- check "
              "checkpoint / hyperparameters / thre before trusting anything below.\n")

    print(f"{len(test_dataset)} windows | {total_nodes:,} distinct test nodes\n")
    print(f"{'win':>4} {'n_nodes':>8} {'n_flag':>8} {'rate%':>7} "
          f"{'new':>7} {'cum%':>6} {'cold':>6} {'warm':>6}")
    for w, n, f, new, cum, cold, warm in per_window:
        rate = 100 * f / n if n else 0.0
        cum_pct = 100 * cum / total_nodes if total_nodes else 0.0
        print(f"{w:>4} {n:>8,} {f:>8,} {rate:>6.2f}% {new:>7,} {cum_pct:>5.2f}% {cold:>6,} {warm:>6,}")

    rates = [100 * f / n for _, n, f, _, _, _, _ in per_window if n > 0]
    print(f"\nPer-window flag rate: min={min(rates):.2f}% "
          f"mean={sum(rates)/len(rates):.2f}% max={max(rates):.2f}%")
    print(f"Cumulative rate at end: {100*len(ever_flagged)/total_nodes:.2f}%")

    counts = list(appearance_counts.values())
    print(f"\nWindow-appearances per node: min={min(counts)} "
          f"mean={sum(counts)/len(counts):.2f} max={max(counts)}")
    multi = sum(1 for c in counts if c > 1)
    print(f"Nodes appearing in >1 window: {multi:,} ({100*multi/total_nodes:.2f}%)")

    total_cold = sum(c for _, _, _, _, _, c, _ in per_window)
    total_warm = sum(w for _, _, _, _, _, _, w in per_window)
    print(f"\n(secondary) Flagged instances -- cold (first-ever appearance): "
          f"{total_cold:,} | warm (had prior memory): {total_warm:,}")


if __name__ == '__main__':
    main()