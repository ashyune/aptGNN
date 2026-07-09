"""Correctness checks for windowing.py.

Run directly: python test_windowing.py

Plain assertions rather than a test framework, matching this project's
existing style. This file has no PyTorch dependency and can be run
anywhere, including environments without the GNN stack installed.
"""

import os
import tempfile
from collections import Counter

from windowing import generate_windows, build_type_vocab, _parse_line

OUT_OF_ORDER_LINES = [
    "uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t300\n",
    "uuid-B\tFile\tuuid-C\tFile\tEVENT_WRITE\t100\n",
    "uuid-A\tProcess\tuuid-C\tFile\tEVENT_WRITE\t200\n",
]


def _write(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


def test_windows_cover_all_rows_exactly_once():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, OUT_OF_ORDER_LINES)

        windows = generate_windows(path, window_size=2)
        all_rows = [row for window in windows for row in window]

        with open(path, 'r') as f:
            input_rows = [_parse_line(line) for line in f]

        assert Counter(all_rows) == Counter(input_rows), (
            'windowing must not drop or duplicate rows'
        )
    print('PASS: windows_cover_all_rows_exactly_once')


def test_chronological_ordering():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, OUT_OF_ORDER_LINES)

        windows = generate_windows(path, window_size=2)
        for i in range(len(windows) - 1):
            max_this = max(r[5] for r in windows[i])
            min_next = min(r[5] for r in windows[i + 1])
            assert max_this <= min_next, (
                f'window {i} (max={max_this}) overlaps window {i + 1} '
                f'(min={min_next}) out of time order'
            )
    print('PASS: chronological_ordering')


def test_out_of_order_input_gets_sorted():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, OUT_OF_ORDER_LINES)

        windows = generate_windows(path, window_size=100)  # one big window
        assert len(windows) == 1
        timestamps = [r[5] for r in windows[0]]
        assert timestamps == sorted(timestamps)
        assert timestamps == [100, 200, 300], (
            'file order (300, 100, 200) must not leak into window order'
        )
    print('PASS: out_of_order_input_gets_sorted')


def test_window_chunking_sizes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, OUT_OF_ORDER_LINES)  # 3 rows

        windows = generate_windows(path, window_size=2)
        assert [len(w) for w in windows] == [2, 1]
    print('PASS: window_chunking_sizes')


def test_build_type_vocab_matches_mydataset():
    """Reproduces the vocabulary-building fragment of MyDataset()
    (data_process_train.py) on the same fixture and checks it agrees with
    build_type_vocab() exactly, including on out-of-order-timestamp input
    -- type vocab must use plain file order, never time-sorted order.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, OUT_OF_ORDER_LINES)

        feature_map, label_map = build_type_vocab(path)

        # Inline re-implementation of the relevant fragment of
        # data_process_train.py's MyDataset(), kept minimal on purpose.
        nodeType_map, edgeType_map = {}, {}
        nodeType_cnt = edgeType_cnt = 0
        with open(path, 'r') as f:
            for line in f:
                temp = line.strip('\n').split('\t')
                if temp[1] not in nodeType_map:
                    nodeType_map[temp[1]] = nodeType_cnt
                    nodeType_cnt += 1
                if temp[3] not in nodeType_map:
                    nodeType_map[temp[3]] = nodeType_cnt
                    nodeType_cnt += 1
                if temp[4] not in edgeType_map:
                    edgeType_map[temp[4]] = edgeType_cnt
                    edgeType_cnt += 1

        assert label_map == nodeType_map
        assert feature_map == edgeType_map
    print('PASS: build_type_vocab_matches_mydataset')


if __name__ == '__main__':
    test_windows_cover_all_rows_exactly_once()
    test_chronological_ordering()
    test_out_of_order_input_gets_sorted()
    test_window_chunking_sizes()
    test_build_type_vocab_matches_mydataset()
    print('All windowing checks passed.')
