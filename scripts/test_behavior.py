"""Stage 6-B (D1-revised): behavior-surprise scoring from a frozen
BehaviorNet checkpoint -- produces scores_windowed.txt.

Protocol is IDENTICAL to test_windowed.py (node-level, no neighborhood
credit, per-node MAX aggregation across windows, cold-start memory,
operating-point threshold = --flag-quantile nearest-rank quantile of the
train-window max-score distribution under the same frozen checkpoint,
ground truth resolved from ../groundtruth/<scene>.txt, output evaluated
by evaluate_windowed.py unchanged). Only the score-producing model
differs: per-node per-window score = mean-per-event NLL of the node's
observed edge-type profile under the history-conditioned prediction
(train_behavior.profile_nll), instead of v1's NLL-of-true-type.

The shared protocol helpers (nearest_rank_quantile, update_max_scores,
write_scores_file) are imported from test_windowed.py rather than
duplicated, so the two scorers cannot silently drift apart.
"""

import argparse
import os
import os.path as osp
import time

import torch

from node_vocab import load_node_vocab
from windowing import build_type_vocab
from windowed_data import build_windowed_dataset
from node_memory import NodeMemory
from train_behavior import (BehaviorNet, profile_nll, make_writeback,
                            type_onehot)
from test_windowed import (nearest_rank_quantile, update_max_scores,
                           write_scores_file)


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


@torch.no_grad()
def score_dataset(model, dataset, node_memory, num_types):
    """Score one windowed dataset under the standard protocol.

    Mirrors test_windowed.score_dataset: chronological cold-start memory
    threading, per-node MAX aggregation, memory always updates. Used for
    both the test windows and the train-window calibration pass so the
    two distributions come from identical code.
    """
    score_by_gid = {}
    for window_idx, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        data = data.to(node_memory.device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, hidden = model(type_onehot(data.y, num_types), memory,
                            data.edge_index, data.edge_type)
        scores = profile_nll(out, data.x).cpu().tolist()
        gids = data.global_id.cpu().tolist()
        update_max_scores(score_by_gid, gids, scores)
        node_memory.update(data.global_id, make_writeback(hidden, data.x),
                           window_idx)
    return score_by_gid


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6-B (D1-revised): history-conditioned behavior '
                    'scoring from the frozen BehaviorNet checkpoint.'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Defaults to ../models/behavior_<scene>/best_model.pt')
    parser.add_argument('--window-size', type=int, default=5000,
                        help='Must match the value the checkpoint was trained with.')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Must match the frozen training config.')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Must match the frozen training config.')
    parser.add_argument('--max-norm', type=float, default=100.0,
                        help='Must match the frozen training config. 0 = '
                             'memory-off ablation (type-prior baseline).')
    parser.add_argument('--relational', action='store_true',
                        help='Extension 2 checkpoint: relation-aware '
                             'memory routing. Must match the frozen '
                             'training config (state_dict load fails '
                             'loudly on a mismatch).')
    parser.add_argument('--rel-bases', type=int, default=8,
                        help='Must match the frozen training config.')
    parser.add_argument('--flag-quantile', type=float, default=0.999,
                        help='Operating point on the train max-score '
                             'distribution; affects the flag column only.')
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--groundtruth', type=str, default=None,
                        help='Defaults to ../groundtruth/<scene>.txt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to ../models/behavior_<scene>/eval/')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    checkpoint_path = args.checkpoint or f'../models/behavior_{args.scene}/best_model.pt'
    out_dir = args.out_dir or f'../models/behavior_{args.scene}/eval'
    groundtruth_path = args.groundtruth or f'../groundtruth/{args.scene}.txt'

    for p in (train_path, test_path, node_vocab_path, checkpoint_path,
              groundtruth_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')
    os.makedirs(out_dir, exist_ok=True)

    show(f'Scene: {args.scene} | operating point = train quantile '
         f'{args.flag_quantile}')

    show(f'Loading node vocab: {node_vocab_path}')
    node_vocab = load_node_vocab(node_vocab_path)
    show(f'{len(node_vocab):,} global node identities')

    feature_map, label_map = build_type_vocab(train_path)
    profile_dim = len(feature_map) * 2
    num_types = len(label_map)

    ground_truth_global_ids = set()
    unresolved = 0
    with open(groundtruth_path, 'r') as f:
        for line in f:
            uuid = line.strip('\n')
            if not uuid:
                continue
            if uuid in node_vocab:
                ground_truth_global_ids.add(node_vocab[uuid])
            else:
                unresolved += 1
    if unresolved:
        show(f'WARNING: {unresolved} ground-truth UUIDs not found in '
             f'node_vocab -- they cannot be scored.')
    show(f'{len(ground_truth_global_ids)} ground-truth nodes resolved to '
         f'global_id')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Device: {device}')

    num_relations = len(feature_map) + 1 if args.relational else None
    if args.relational:
        show(f'Extension 2 ON: relational memory routing, '
             f'{num_relations} relations, {args.rel_bases} bases')
    model = BehaviorNet(num_types, profile_dim, args.hidden_dim,
                        num_relations=num_relations,
                        num_bases=args.rel_bases).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    show(f'Loaded frozen checkpoint: {checkpoint_path}')

    memory_dim = args.hidden_dim + profile_dim

    # Calibration pass: benign max-score distribution over the TRAIN file.
    show('Building windowed train dataset (operating-point calibration)')
    train_dataset = build_windowed_dataset(
        train_path, node_vocab, feature_map, label_map, args.window_size)
    show(f'{len(train_dataset)} train windows')
    calib_memory = NodeMemory(len(node_vocab), memory_dim, args.decay_rate,
                              device=device, max_norm=args.max_norm)
    train_scores = score_dataset(model, train_dataset, calib_memory, num_types)
    threshold = nearest_rank_quantile(list(train_scores.values()),
                                      args.flag_quantile)
    show(f'Operating-point threshold: score >= {threshold:.6f} '
         f'(train quantile {args.flag_quantile})')
    del train_dataset, train_scores, calib_memory

    # Scoring pass: the test file, cold-start memory.
    show('Building windowed test dataset')
    test_dataset = build_windowed_dataset(
        test_path, node_vocab, feature_map, label_map, args.window_size)
    show(f'{len(test_dataset)} test windows')

    node_memory = NodeMemory(len(node_vocab), memory_dim, args.decay_rate,
                             device=device, max_norm=args.max_norm)
    score_by_gid = score_dataset(model, test_dataset, node_memory, num_types)
    flagged_gids = {gid for gid, score in score_by_gid.items()
                    if score >= threshold}

    total_nodes = len(score_by_gid)
    show(f'{total_nodes:,} distinct nodes scored across the test sequence')
    if total_nodes > 0:
        show(f'{len(flagged_gids):,} nodes flagged at the fixed operating '
             f'point ({100 * len(flagged_gids) / total_nodes:.2f}% of test '
             f'nodes)')

    scores_path = osp.join(out_dir, 'scores_windowed.txt')
    write_scores_file(scores_path, score_by_gid, flagged_gids)
    show(f'Wrote {scores_path}')

    gt_path = osp.join(out_dir, 'groundtruth_global_id.txt')
    with open(gt_path, 'w') as fw:
        for gid in ground_truth_global_ids:
            fw.write(f'{gid}\n')
    show(f'Wrote {gt_path}')
    show('Done. Run evaluate_windowed.py against these two files.')


if __name__ == '__main__':
    main()
