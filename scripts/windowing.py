"""Temporal window generation and type-vocabulary construction for
cross-window memory.

ThreaTrace's existing pipeline treats a whole provenance file (the entire
train file, or the entire test file) as a single static graph -- there is
no notion of a sequence of time-ordered sub-graphs anywhere in the current
code. This module adds that: it groups a file's rows into an ordered
sequence of fixed-size, time-sorted windows. windowed_data.py then turns
each window into a per-window Data object.

This module has no PyTorch / PyTorch Geometric dependency -- it only
parses text and sorts/chunks plain Python tuples. Tensor construction
lives in windowed_data.py. Keeping the two separate means the highest-risk
logic here (correct time ordering, complete row coverage, and matching the
existing type-vocabulary construction exactly) can be tested in complete
isolation from the GNN framework.
"""

import time


def show(str_msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {str_msg}')


def _parse_line(line):
    """Split one provenance tsv line into its typed fields.

    Column layout matches parse_darpatc.py's output and the existing
    data_process_train.py / data_process_test.py convention:
        [0] source UUID        [1] source type
        [2] destination UUID   [3] destination type
        [4] edge type          [5] timestamp (nanoseconds)
    """
    temp = line.strip('\n').split('\t')
    return (temp[0], temp[1], temp[2], temp[3], temp[4], int(temp[5]))


def generate_windows(path, window_size):
    """Split one provenance file into an ordered sequence of time windows.

    Parameters
    ----------
    path : str
        A provenance tsv file (e.g. a `*_train.txt` or `*_test.txt` file
        produced by parse_darpatc.py).
    window_size : int
        Number of rows per window. The final window may be smaller if the
        file's row count isn't an exact multiple of window_size.

    Returns
    -------
    list of list of tuple
        windows[i] is the i-th window, in chronological order, as a list
        of (src_uuid, src_type, dst_uuid, dst_type, edge_type, timestamp)
        tuples, itself sorted by timestamp.

    Notes
    -----
    Rows are sorted by timestamp before chunking rather than trusting file
    order -- see the module docstring. This uses Python's stable sort, so
    rows sharing an identical timestamp keep their original relative file
    order rather than being shuffled, which keeps window contents
    deterministic given the same input file.
    """
    show(f'Reading rows for windowing: {path}')
    rows = []
    with open(path, 'r') as f:
        for line in f:
            rows.append(_parse_line(line))
    show(f'{path}: {len(rows):,} rows read')

    rows.sort(key=lambda r: r[5])

    windows = [
        rows[i:i + window_size]
        for i in range(0, len(rows), window_size)
    ]
    show(f'{path}: split into {len(windows):,} windows of up to '
         f'{window_size:,} rows each')
    return windows


def build_type_vocab(path):
    """Build edge-type and node-type vocabularies from a provenance file.

    Mirrors the vocabulary-building fragment of MyDataset() in
    data_process_train.py exactly -- same columns, same first-appearance
    order -- so calling this on the training file produces the identical
    feature_map / label_map that MyDataset would write to feature.txt /
    label.txt today. It is duplicated here rather than imported from
    data_process_train.py so Stage 2 does not modify or depend on the
    training pipeline's internals.

    Unlike generate_windows, rows here are read in plain FILE order, not
    time-sorted order. This is deliberate: MyDataset never sorts by time
    either, so matching its exact integer assignments (which any existing
    feature.txt / label.txt, or a model already trained against them,
    depend on) requires reading in the same order it does.

    Returns
    -------
    (feature_map, label_map) : (dict[str, int], dict[str, int])
        feature_map maps edge-type strings to indices (mirrors
        feature.txt). label_map maps node-type strings to indices
        (mirrors label.txt).
    """
    show(f'Building type vocabulary: {path}')
    feature_map = {}
    label_map = {}
    feature_cnt = 0
    label_cnt = 0

    with open(path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')

            if temp[1] not in label_map:
                label_map[temp[1]] = label_cnt
                label_cnt += 1

            if temp[3] not in label_map:
                label_map[temp[3]] = label_cnt
                label_cnt += 1

            if temp[4] not in feature_map:
                feature_map[temp[4]] = feature_cnt
                feature_cnt += 1

    show(f'{path}: {len(label_map)} node types, {len(feature_map)} edge types')
    return feature_map, label_map
