"""Per-window PyG Data construction for cross-window memory.

Turns the time-sorted, chunked rows produced by windowing.generate_windows
into a sequence of torch_geometric.data.Data objects, one per window,
using the stable global node ids from Stage 1's node_vocab so a node's
identity is consistent across the whole sequence.

Design: each window's Data object uses LOCAL, dense, window-scoped node
indexing for x / y / edge_index -- exactly like every other Data object in
this codebase, and exactly like the local batch indexing NeighborLoader
already produces via batch.n_id. A parallel `global_id` tensor maps each
local index back to its Stage 1 global node id; anything that needs to
persist information across windows (Stage 3's memory manager) reads and
writes through global_id, never through the local index directly, since
the local index is only meaningful within one window's Data object.

An alternative considered and rejected: size x / edge_index directly in
the full global id space (~696k nodes for the cadets scene), so no
local<->global mapping is needed anywhere. Rejected because most nodes
are inactive in any given window -- a dense per-window tensor over the
full global vocab would be almost entirely zero rows, wasting memory and
slowing down NeighborLoader, for every single window.
"""

import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from windowing import generate_windows, build_type_vocab, show
from node_vocab import load_node_vocab


def build_window_data(window_rows, node_vocab, feature_map, label_map):
    """Build one window's Data object.

    Parameters
    ----------
    window_rows : list of tuple
        One window's rows, as produced by windowing.generate_windows --
        (src_uuid, src_type, dst_uuid, dst_type, edge_type, timestamp).
    node_vocab : dict[str, int]
        Global UUID -> global node id, as produced by Stage 1's
        node_vocab.build_node_vocab / loaded via node_vocab.load_node_vocab.
    feature_map : dict[str, int]
        Edge-type vocabulary (mirrors feature.txt).
    label_map : dict[str, int]
        Node-type vocabulary (mirrors label.txt).

    Returns
    -------
    torch_geometric.data.Data
        x          : [num_local_nodes, feature_num * 2] edge-type
                     histogram, exactly like data_process_train.py /
                     data_process_test.py, counted only over this
                     window's edges.
        y          : [num_local_nodes] node-type label.
        edge_index : [2, num_local_edges (+ self-loops)], LOCAL indices.
        global_id  : [num_local_nodes] LongTensor; global_id[i] is the
                     Stage 1 global node id for local node i. This is
                     the field Stage 3's memory manager keys on.

    Rows whose type isn't in feature_map / label_map are skipped entirely,
    exactly matching MyDatasetA's existing behaviour in
    data_process_test.py -- this function is used for both train-file and
    test-file windows, and both should drop unrecognised types the same
    way the baseline already does.
    """
    feature_num = len(feature_map)

    local_id = {}
    global_id_list = []
    edge_s, edge_e, edge_feat = [], [], []
    node_label = {}

    for src_uuid, src_type, dst_uuid, dst_type, edge_type, _ts in window_rows:
        if src_type not in label_map:
            continue
        if dst_type not in label_map:
            continue
        if edge_type not in feature_map:
            continue

        if src_uuid not in local_id:
            if src_uuid not in node_vocab:
                raise KeyError(
                    f'{src_uuid} not found in node_vocab -- was this file '
                    'included when Stage 1 built the vocab?'
                )
            local_id[src_uuid] = len(global_id_list)
            global_id_list.append(node_vocab[src_uuid])
        src_local = local_id[src_uuid]
        node_label[src_local] = label_map[src_type]

        if dst_uuid not in local_id:
            if dst_uuid not in node_vocab:
                raise KeyError(
                    f'{dst_uuid} not found in node_vocab -- was this file '
                    'included when Stage 1 built the vocab?'
                )
            local_id[dst_uuid] = len(global_id_list)
            global_id_list.append(node_vocab[dst_uuid])
        dst_local = local_id[dst_uuid]
        node_label[dst_local] = label_map[dst_type]

        edge_s.append(src_local)
        edge_e.append(dst_local)
        edge_feat.append(feature_map[edge_type])

    num_nodes = len(global_id_list)
    x = torch.zeros((num_nodes, feature_num * 2), dtype=torch.float)
    y = torch.zeros(num_nodes, dtype=torch.long)

    for src_local, dst_local, feat in zip(edge_s, edge_e, edge_feat):
        x[src_local, feat] += 1
        x[dst_local, feat + feature_num] += 1

    for local_idx, label in node_label.items():
        y[local_idx] = label

    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        global_id=torch.tensor(global_id_list, dtype=torch.long),
    )


def build_windowed_dataset(path, node_vocab, feature_map, label_map, window_size):
    """Build the full ordered sequence of per-window Data objects for one file.

    Parameters mirror build_window_data(); *path* is windowed via
    windowing.generate_windows() first.

    Returns
    -------
    list of torch_geometric.data.Data, in chronological order.
    """
    windows = generate_windows(path, window_size)
    dataset = []
    for i, window_rows in enumerate(windows):
        data = build_window_data(window_rows, node_vocab, feature_map, label_map)
        show(f'Window {i}: {data.num_nodes:,} nodes, '
             f'{data.edge_index.size(1):,} edges (incl. self-loops)')
        dataset.append(data)
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Build a time-ordered sequence of per-window Data '
                     'objects for one scene/split, using the Stage 1 '
                     'global node vocabulary.'
    )
    parser.add_argument('--scene', type=str, required=True,
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'test'])
    parser.add_argument('--window-size', type=int, default=50000,
                        help='Rows per window (default: 50000).')
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Path to the Stage 1 vocab file. Defaults to '
                             '../models/<scene>_node_vocab.txt')
    parser.add_argument('--save', type=str, default=None,
                        help='Optional path to torch.save() the resulting '
                             'list of Data objects to, for inspection.')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    split_path = base + args.scene + '_' + args.split + '.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'

    for p in (train_path, split_path, node_vocab_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')

    node_vocab = load_node_vocab(node_vocab_path)
    # Type vocab always comes from the training file, matching the
    # existing baseline convention (feature.txt / label.txt are built
    # once from the training file and reused for test).
    feature_map, label_map = build_type_vocab(train_path)

    dataset = build_windowed_dataset(
        split_path, node_vocab, feature_map, label_map, args.window_size
    )

    show(f'Built {len(dataset)} windows for {args.scene}/{args.split}')
    if args.save:
        torch.save(dataset, args.save)
        show(f'Saved windowed dataset to {args.save}')


if __name__ == '__main__':
    main()
