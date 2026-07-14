"""
Verify edge_type/num_relations construction against real DARPA TC CADETS
data. Run from your src/ directory (same relative-path assumptions as
train_darpatc.py: ../models/, ../graphchi-cpp-master/graph_data/darpatc/).

Usage:
    python3 verify_cadets_graph_construction.py

Only checks graph construction -- does not train anything, does not
import SAGEMemNet. Safe to run before you've touched train_darpatc.py
at all.
"""
import time
import torch
from data_process_train import MyDataset
from data_process_test import MyDatasetA

TRAIN_PATH = '../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt'
TEST_PATH = '../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt'
NUM_WINDOWS = 3  # match train_darpatc.py's NUM_WINDOWS


def _relation_names():
    """feature.txt maps name -> id; invert it for readable printouts.
    Only exists after MyDataset has run at least once (it's written as
    a side effect of _load_provenance)."""
    names = {}
    try:
        with open('../models/feature.txt', 'r') as f:
            for line in f:
                name, idx = line.strip('\n').split('\t')
                names[int(idx)] = name
    except FileNotFoundError:
        pass
    return names


def check_train_side():
    print('=' * 70)
    print('TRAIN-SIDE: data_process_train.py :: MyDataset')
    print('=' * 70)

    t0 = time.time()
    windows, feature_num, label_num, num_relations = MyDataset(
        TRAIN_PATH, num_windows=NUM_WINDOWS
    )
    elapsed = time.time() - t0
    print(f'MyDataset() took {elapsed:.1f}s')
    print(f'feature_num={feature_num}  label_num={label_num}  '
          f'num_relations={num_relations}')
    print(f'windows returned: {len(windows)} (requested {NUM_WINDOWS})')

    names = _relation_names()

    prev_active = None
    prev_mass = None
    total_edges = 0

    for w_idx, data in enumerate(windows):
        n_edges = data.edge_index.shape[1]
        total_edges += n_edges
        active = int(data.active_mask.sum())
        print(f'\n--- window {w_idx} ---')
        print(f'  total nodes so far (x.shape[0]) : {data.x.shape[0]}')
        print(f'  edges in this window            : {n_edges}')
        print(f'  active nodes (cumulative)       : {active}')

        # --- correctness checks ---
        assert data.edge_type.shape[0] == n_edges, (
            f'window {w_idx}: edge_type length {data.edge_type.shape[0]} '
            f'!= edge count {n_edges}'
        )
        assert int(data.edge_type.min()) >= 0
        assert int(data.edge_type.max()) < num_relations, (
            f'window {w_idx}: relation id {int(data.edge_type.max())} '
            f'>= num_relations ({num_relations})'
        )

        if prev_active is not None:
            assert torch.all(data.active_mask | ~prev_active), (
                f'active_mask shrank between window {w_idx - 1} and {w_idx}'
            )
        prev_active = data.active_mask

        mass = data.x.sum().item()
        if prev_mass is not None:
            assert mass >= prev_mass, (
                f'x mass decreased window {w_idx - 1} -> {w_idx}: '
                f'{prev_mass} -> {mass}'
            )
        prev_mass = mass

        # relation-id histogram, named where possible
        counts = torch.bincount(data.edge_type, minlength=num_relations)
        top = torch.argsort(counts, descending=True)[:10]
        print('  top relation types this window:')
        for rid in top.tolist():
            c = int(counts[rid])
            if c == 0:
                continue
            label = names.get(rid, f'id_{rid}')
            print(f'    {label:30s} {c:>10d}')

    print(f'\nTotal edges across all windows: {total_edges}')
    print('TRAIN-SIDE: all checks passed.')
    return num_relations


def check_test_side(expected_num_relations):
    print('\n' + '=' * 70)
    print('TEST-SIDE: data_process_test.py :: MyDatasetA')
    print('=' * 70)

    t0 = time.time()
    (data, feature_num, label_num, adj, adj2,
     nodeA, _nodeA, _neighbour, num_relations) = MyDatasetA(TEST_PATH, 0)
    elapsed = time.time() - t0
    print(f'MyDatasetA() took {elapsed:.1f}s')
    print(f'feature_num={feature_num}  label_num={label_num}  '
          f'num_relations={num_relations}')
    print(f'nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}  '
          f'ground-truth nodes (nodeA)={len(nodeA)}')

    assert data.edge_type.shape[0] == data.edge_index.shape[1], (
        'edge_type length does not match edge count'
    )
    assert int(data.edge_type.min()) >= 0
    assert int(data.edge_type.max()) < num_relations

    # This is the check that actually matters most: train and test must
    # share ONE relation vocabulary, or a relation-aware conv will treat
    # the same id as two different things depending on which graph it's
    # looking at.
    assert num_relations == expected_num_relations, (
        f'VOCABULARY MISMATCH: train-side num_relations='
        f'{expected_num_relations} but test-side num_relations='
        f'{num_relations}. This should be impossible since both read '
        f'../models/feature.txt -- if you see this, feature.txt was '
        f'probably overwritten between the two runs.'
    )
    print('TEST-SIDE: all checks passed, vocabulary matches train side.')


if __name__ == '__main__':
    num_relations = check_train_side()
    check_test_side(num_relations)
    print('\nAll graph-construction checks passed on real CADETS data.')