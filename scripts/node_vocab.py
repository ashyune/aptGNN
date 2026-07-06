import os.path as osp
import time


def show(str_msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {str_msg}')


def build_node_vocab(paths):
    """Assign a stable global integer id to every node UUID.

    Parameters
    ----------
    paths : list of str
        Provenance tsv files, scanned in the given order. Ids are assigned
        in first-appearance order across all files combined, so the order
        of *paths* determines which file "claims" a node that appears in
        more than one. Pass the train file(s) before the test file(s) so a
        node's id matches exactly what a train-only run assigns it today.

    Returns
    -------
    dict[str, int]
        Mapping from DARPA CDM UUID to global node id.

    Notes
    -----
    Column layout matches parse_darpatc.py's output and the existing
    data_process_train.py / data_process_test.py convention:
        temp[0] = source UUID       temp[1] = source type
        temp[2] = destination UUID  temp[3] = destination type
        temp[4] = edge type         temp[5] = timestamp (unused here)
    Only temp[0] and temp[2] (the UUIDs) are used -- this function builds
    identity, not features or labels, and deliberately mirrors the
    src-before-dst check order used throughout the rest of the codebase.
    """
    vocab = {}
    node_cnt = 0

    for path in paths:
        show(f'Scanning for node identities: {path}')
        line_cnt = 0
        with open(path, 'r') as f:
            for line in f:
                line_cnt += 1
                temp = line.strip('\n').split('\t')

                src_uuid = temp[0]
                if src_uuid not in vocab:
                    vocab[src_uuid] = node_cnt
                    node_cnt += 1

                dst_uuid = temp[2]
                if dst_uuid not in vocab:
                    vocab[dst_uuid] = node_cnt
                    node_cnt += 1

        show(f'{path}: {line_cnt:,} lines scanned, {node_cnt:,} unique nodes so far')

    return vocab


def save_node_vocab(vocab, path):
    """Persist a node vocab as uuid<TAB>id, one entry per line.

    Same tab-separated convention as feature.txt / label.txt, so it can be
    read back with the identical parsing pattern already used throughout
    the project, and can be inspected by hand.
    """
    with open(path, 'w') as fw:
        for uuid, node_id in vocab.items():
            fw.write(f'{uuid}\t{node_id}\n')


def load_node_vocab(path):
    """Load a node vocab previously written by save_node_vocab()."""
    vocab = {}
    with open(path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            vocab[temp[0]] = int(temp[1])
    return vocab


def build_and_save_node_vocab(paths, output_path):
    """Convenience wrapper: build the vocab and persist it in one call."""
    vocab = build_node_vocab(paths)
    save_node_vocab(vocab, output_path)
    show(f'Wrote {len(vocab):,} node identities to {output_path}')
    return vocab


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Build a stable global node-identity vocabulary '
                     "spanning a scene's train and test provenance files."
    )
    parser.add_argument('--scene', type=str, required=True,
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output path. Defaults to '
                             '../models/<scene>_node_vocab.txt')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    output_path = args.output or f'../models/{args.scene}_node_vocab.txt'

    for p in (train_path, test_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected provenance file not found: {p}')

    build_and_save_node_vocab([train_path, test_path], output_path)


if __name__ == '__main__':
    main()