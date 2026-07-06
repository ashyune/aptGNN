"""Correctness checks for node_vocab.py.

Run directly: python test_node_vocab.py

Plain assertions rather than a test framework, to match this project's
existing style -- no test runner (pytest, unittest, etc.) is used anywhere
else in the codebase.
"""

import os
import tempfile

from node_vocab import (
    build_node_vocab,
    save_node_vocab,
    load_node_vocab,
)

TRAIN_LINES = [
    "uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t100\n",
    "uuid-B\tFile\tuuid-C\tFile\tEVENT_WRITE\t200\n",
    "uuid-A\tProcess\tuuid-C\tFile\tEVENT_WRITE\t300\n",
]

TEST_LINES = [
    "uuid-A\tProcess\tuuid-D\tNetFlow\tEVENT_CONNECT\t400\n",
    "uuid-E\tProcess\tuuid-D\tNetFlow\tEVENT_CONNECT\t500\n",
]


def _write(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


def test_first_appearance_order():
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, 'train.txt')
        _write(train_path, TRAIN_LINES)
        vocab = build_node_vocab([train_path])
        # Line 1 introduces A then B; line 2 introduces C.
        assert vocab['uuid-A'] == 0
        assert vocab['uuid-B'] == 1
        assert vocab['uuid-C'] == 2
        assert len(vocab) == 3
    print('PASS: first_appearance_order')


def test_determinism():
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, 'train.txt')
        _write(train_path, TRAIN_LINES)
        vocab1 = build_node_vocab([train_path])
        vocab2 = build_node_vocab([train_path])
        assert vocab1 == vocab2
    print('PASS: determinism')


def test_stable_across_train_and_test():
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, 'train.txt')
        test_path = os.path.join(d, 'test.txt')
        _write(train_path, TRAIN_LINES)
        _write(test_path, TEST_LINES)

        train_only_vocab = build_node_vocab([train_path])
        combined_vocab = build_node_vocab([train_path, test_path])

        # Every node seen during training must keep the exact same id once
        # the test file is added -- this is the property cross-window
        # memory depends on.
        for uuid, node_id in train_only_vocab.items():
            assert combined_vocab[uuid] == node_id, (
                f'{uuid} changed id from {node_id} to {combined_vocab[uuid]}'
            )

        max_train_id = max(train_only_vocab.values())
        new_uuids = set(combined_vocab) - set(train_only_vocab)
        assert new_uuids == {'uuid-D', 'uuid-E'}
        for uuid in new_uuids:
            assert combined_vocab[uuid] > max_train_id
    print('PASS: stable_across_train_and_test')


def test_save_and_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, 'train.txt')
        vocab_path = os.path.join(d, 'vocab.txt')
        _write(train_path, TRAIN_LINES)

        vocab = build_node_vocab([train_path])
        save_node_vocab(vocab, vocab_path)
        reloaded = load_node_vocab(vocab_path)

        assert reloaded == vocab
    print('PASS: save_and_load_round_trip')


def test_matches_existing_mydataset_ordering():
    """Reproduces the id-assignment fragment of MyDataset()
    (data_process_train.py) on the same fixture and checks it agrees with
    build_node_vocab() exactly. This is the regression check: Stage 1 must
    not change how ids would have been assigned in a train-only run.
    """
    with tempfile.TemporaryDirectory() as d:
        train_path = os.path.join(d, 'train.txt')
        _write(train_path, TRAIN_LINES)

        # Inline re-implementation of the relevant fragment of
        # data_process_train.py's MyDataset(), kept minimal on purpose.
        node_cnt = 0
        nodeId_map = {}
        with open(train_path, 'r') as f:
            for line in f:
                temp = line.strip('\n').split('\t')
                if temp[0] not in nodeId_map:
                    nodeId_map[temp[0]] = node_cnt
                    node_cnt += 1
                if temp[2] not in nodeId_map:
                    nodeId_map[temp[2]] = node_cnt
                    node_cnt += 1

        vocab = build_node_vocab([train_path])
        assert vocab == nodeId_map
    print('PASS: matches_existing_mydataset_ordering')


if __name__ == '__main__':
    test_first_appearance_order()
    test_determinism()
    test_stable_across_train_and_test()
    test_save_and_load_round_trip()
    test_matches_existing_mydataset_ordering()
    print('All node_vocab checks passed.')