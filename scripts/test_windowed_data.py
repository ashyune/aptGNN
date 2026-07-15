"""Correctness checks for windowed_data.py.

Run directly: python test_windowed_data.py

Requires torch and torch_geometric (the project's existing `threatrace`
conda environment already has both). This file could not be executed in
the sandbox this was written in -- that environment only offers a
CUDA-only torch build with no way to satisfy its runtime library
dependencies, and installing the full dependency chain risked filling the
sandbox's disk for no benefit, since it's unrelated to your actual
environment. Run this here before trusting windowed_data.py.
"""

import os
import tempfile

import torch
from torch_geometric.utils import add_self_loops

from windowing import build_type_vocab, generate_windows
from windowed_data import build_window_data


def _write(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


LINES = [
    "uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t100\n",
    "uuid-B\tFile\tuuid-C\tFile\tEVENT_WRITE\t200\n",
    "uuid-A\tProcess\tuuid-C\tFile\tEVENT_WRITE\t300\n",
    "uuid-C\tFile\tuuid-D\tNetFlow\tEVENT_CONNECT\t400\n",
]


def test_single_window_matches_mydataset_construction():
    """The core regression check: build_window_data()'s x / y / edge_index
    construction must match data_process_train.py's MyDataset() exactly
    (mod the local<->global id bookkeeping, which MyDataset doesn't have).
    Timestamps in the fixture are already strictly increasing so sorting
    cannot reorder anything, isolating this check from windowing's own
    sort behaviour (covered separately in test_windowing.py).
    """
    node_vocab = {'uuid-A': 10, 'uuid-B': 11, 'uuid-C': 12, 'uuid-D': 13}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, LINES)

        feature_map, label_map = build_type_vocab(path)
        windows = generate_windows(path, window_size=100)  # one window
        assert len(windows) == 1

        data = build_window_data(windows[0], node_vocab, feature_map, label_map)

        # Manual mirror of MyDataset's x / y / edge_index construction
        # (data_process_train.py), applied to the same rows in the same
        # order, using local ids assigned the same way.
        feature_num = len(feature_map)
        local_id = {}
        edge_s, edge_e = [], []
        provenance = []
        for src_uuid, src_type, dst_uuid, dst_type, edge_type, _ts in windows[0]:
            if src_type not in label_map or dst_type not in label_map \
                    or edge_type not in feature_map:
                continue
            if src_uuid not in local_id:
                local_id[src_uuid] = len(local_id)
            if dst_uuid not in local_id:
                local_id[dst_uuid] = len(local_id)
            edge_s.append(local_id[src_uuid])
            edge_e.append(local_id[dst_uuid])
            provenance.append((local_id[src_uuid], label_map[src_type],
                                local_id[dst_uuid], label_map[dst_type],
                                feature_map[edge_type]))

        num_nodes = len(local_id)
        expected_x = torch.zeros((num_nodes, feature_num * 2), dtype=torch.float)
        expected_y = torch.zeros(num_nodes, dtype=torch.long)
        for s, s_type, e, e_type, feat in provenance:
            expected_x[s, feat] += 1
            expected_y[s] = s_type
            expected_x[e, feat + feature_num] += 1
            expected_y[e] = e_type

        expected_edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
        expected_edge_index, _ = add_self_loops(expected_edge_index, num_nodes=num_nodes)

        assert torch.equal(data.x, expected_x)
        assert torch.equal(data.y, expected_y)
        assert torch.equal(data.edge_index, expected_edge_index)
        assert data.global_id.tolist() == [
            node_vocab[u] for u in ['uuid-A', 'uuid-B', 'uuid-C', 'uuid-D']
        ]
    print('PASS: single_window_matches_mydataset_construction')


def test_global_id_consistent_across_windows():
    """A node appearing in two different windows must resolve to the same
    global_id in both, even though its LOCAL index differs between them.
    This is the property Stage 3's memory manager depends on.
    """
    node_vocab = {'uuid-A': 7, 'uuid-B': 8, 'uuid-C': 9, 'uuid-E': 20}
    feature_map = {'EVENT_READ': 0, 'EVENT_WRITE': 1}
    label_map = {'Process': 0, 'File': 1}

    window1_rows = [
        ('uuid-B', 'File', 'uuid-A', 'Process', 'EVENT_READ', 100),
    ]
    window2_rows = [
        ('uuid-A', 'Process', 'uuid-C', 'File', 'EVENT_WRITE', 200),
        ('uuid-E', 'Process', 'uuid-A', 'Process', 'EVENT_READ', 250),
    ]

    data1 = build_window_data(window1_rows, node_vocab, feature_map, label_map)
    data2 = build_window_data(window2_rows, node_vocab, feature_map, label_map)

    # uuid-A is local index 1 in window 1 (B=0 seen first, A=1) and local
    # index 0 in window 2 (A=0 seen first, C=1) -- different local slots,
    # same global id.
    a_local_in_1 = 1
    a_local_in_2 = 0
    assert data1.global_id[a_local_in_1].item() == node_vocab['uuid-A']
    assert data2.global_id[a_local_in_2].item() == node_vocab['uuid-A']
    assert (data1.global_id[a_local_in_1].item()
            == data2.global_id[a_local_in_2].item())
    print('PASS: global_id_consistent_across_windows')


def test_type_filtering_matches_existing_behavior():
    """Rows with an edge type or node type not in the vocab must be
    dropped entirely, matching MyDatasetA's existing `continue` behaviour
    in data_process_test.py.
    """
    node_vocab = {'uuid-A': 1, 'uuid-B': 2, 'uuid-C': 3}
    feature_map = {'EVENT_READ': 0}          # EVENT_WRITE deliberately absent
    label_map = {'Process': 0, 'File': 1}    # NetFlow deliberately absent

    rows = [
        ('uuid-A', 'Process', 'uuid-B', 'File', 'EVENT_READ', 100),   # kept
        ('uuid-B', 'File', 'uuid-C', 'NetFlow', 'EVENT_READ', 200),   # dropped: dst type unknown
        ('uuid-A', 'Process', 'uuid-C', 'File', 'EVENT_WRITE', 300),  # dropped: edge type unknown
    ]

    data = build_window_data(rows, node_vocab, feature_map, label_map)

    assert data.num_nodes == 2
    assert data.global_id.tolist() == [node_vocab['uuid-A'], node_vocab['uuid-B']]
    print('PASS: type_filtering_matches_existing_behavior')


def test_self_loops_added_per_window():
    node_vocab = {'uuid-A': 1, 'uuid-B': 2}
    feature_map = {'EVENT_READ': 0}
    label_map = {'Process': 0, 'File': 1}
    rows = [('uuid-A', 'Process', 'uuid-B', 'File', 'EVENT_READ', 100)]

    data = build_window_data(rows, node_vocab, feature_map, label_map)
    # 1 real edge + 2 self-loops (one per node) = 3 total.
    assert data.edge_index.size(1) == 3
    src, dst = data.edge_index
    for local_idx in range(data.num_nodes):
        assert ((src == local_idx) & (dst == local_idx)).any().item(), (
            f'node {local_idx} is missing its self-loop'
        )
    print('PASS: self_loops_added_per_window')


def test_edge_type_aligned_with_edge_index():
    """edge_type[i] must be the relation id of edge_index[:, i]: real
    edges first, in row order, with their feature_map ids; then the
    appended self-loops with the dedicated id len(feature_map). Guards
    the add_self_loops append-at-end behaviour Extension 2's relational
    routing relies on, in case a PyG upgrade ever changes it.
    """
    node_vocab = {'uuid-A': 1, 'uuid-B': 2, 'uuid-C': 3}
    feature_map = {'EVENT_READ': 0, 'EVENT_WRITE': 1}
    label_map = {'Process': 0, 'File': 1}
    rows = [
        ('uuid-A', 'Process', 'uuid-B', 'File', 'EVENT_READ', 100),
        ('uuid-B', 'File', 'uuid-C', 'File', 'EVENT_WRITE', 200),
        ('uuid-A', 'Process', 'uuid-C', 'File', 'EVENT_WRITE', 300),
    ]

    data = build_window_data(rows, node_vocab, feature_map, label_map)

    assert data.edge_type.size(0) == data.edge_index.size(1)
    # Real edges keep row order and their feature_map relation ids...
    assert data.edge_type[:3].tolist() == [0, 1, 1]
    src, dst = data.edge_index
    assert src[:3].tolist() == [0, 1, 0] and dst[:3].tolist() == [1, 2, 2]
    # ...then exactly one self-loop per node with the dedicated id.
    loop_id = len(feature_map)
    assert data.edge_type[3:].tolist() == [loop_id] * data.num_nodes
    assert torch.equal(src[3:], dst[3:])
    print('PASS: edge_type_aligned_with_edge_index')


if __name__ == '__main__':
    test_single_window_matches_mydataset_construction()
    test_global_id_consistent_across_windows()
    test_type_filtering_matches_existing_behavior()
    test_self_loops_added_per_window()
    test_edge_type_aligned_with_edge_index()
    print('All windowed_data checks passed.')
