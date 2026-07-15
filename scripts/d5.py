"""
Diagnostic 5 -- identity and structural context of nodes 605775 and 608335.

Diagnostic 4 found their stored 2-hop neighbourhoods reach 12,836 and
12,833 of 12,852 ground-truth nodes respectively, with the second
adding ZERO net coverage beyond the first -- suggesting one shared
structural cause rather than two independent ones.

Single question answered:
    What CDM object type do these two resolve to, how many raw edges
    (and how diverse in edge-type / neighbour-type / timestamp) does
    the test file record for them, are they themselves ground truth,
    and do their stored neighbourhoods overlap almost completely or
    is one directly connected to the other?

No torch import: the overlap check reads alarm_windowed.txt's already-
stored neighbour lists rather than recomputing two_hop_neighbors();
identity/type/edge-profile is a plain-text scan of the test tsv plus
the vocab file.

Note: the edge-profile count below includes every raw edge referencing
the uuid regardless of whether the OTHER endpoint is itself in
node_vocab, so it may run slightly higher than build_whole_file_adjacency's
own degree (which requires both endpoints in vocab).
"""

import argparse

WATCH_IDS = [605775, 608335]


def resolve_uuids(vocab_path, global_ids):
    reverse = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            uuid, gid = temp[0], int(temp[1])
            if gid in global_ids:
                reverse[gid] = uuid
    return reverse


def edge_profile(test_path, uuid_set):
    profile = {u: {'type': None, 'edge_types': set(), 'neighbours': set(),
                    'neighbour_types': set(), 'timestamps': [],
                    'edge_count': 0} for u in uuid_set}

    with open(test_path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            src, src_type, dst, dst_type, edge_type, ts = temp[:6]

            if src in profile:
                p = profile[src]
                p['type'] = src_type
                p['edge_types'].add(edge_type)
                p['neighbours'].add(dst)
                p['neighbour_types'].add(dst_type)
                p['timestamps'].append(ts)
                p['edge_count'] += 1

            if dst in profile:
                p = profile[dst]
                p['type'] = dst_type
                p['edge_types'].add(edge_type)
                p['neighbours'].add(src)
                p['neighbour_types'].add(src_type)
                p['timestamps'].append(ts)
                p['edge_count'] += 1

    return profile


def alarm_neighbour_sets(alarm_path, global_ids):
    """Read each watched node's ALREADY-STORED 2-hop list straight out
    of alarm_windowed.txt -- no recomputation."""
    sets = {}
    with open(alarm_path, 'r') as f:
        for line in f:
            if line == '\n' or ':' not in line:
                continue
            line = line.strip('\n')
            gid_str, neighbours_str = line.split(':')
            gid = int(gid_str)
            if gid in global_ids:
                sets[gid] = {int(x) for x in neighbours_str.strip(' ').split(' ') if x != ''}
    return sets


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 5: identity and structural context of '
                    'the extreme-reach alarm nodes.')
    parser.add_argument('--scene', type=str, default='cadets')
    parser.add_argument('--node-vocab', type=str, default=None)
    parser.add_argument('--out-dir', type=str,
                        default='../models/windowed_cadets/eval')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    test_path = base + args.scene + '_test.txt'
    vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    alarm_path = f'{args.out_dir}/alarm_windowed.txt'
    gt_path = f'{args.out_dir}/groundtruth_global_id.txt'

    watch = set(WATCH_IDS)

    gt = set()
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line:
                gt.add(int(line))
    for gid in WATCH_IDS:
        print(f"Node {gid}: {'IS' if gid in gt else 'is NOT'} itself a ground-truth node")

    reverse = resolve_uuids(vocab_path, watch)
    uuid_set = set(reverse.values())
    profiles = edge_profile(test_path, uuid_set)

    print()
    for gid in WATCH_IDS:
        uuid = reverse.get(gid)
        if uuid is None:
            print(f"Node {gid}: NOT FOUND in vocab file -- stop, investigate this first")
            continue
        p = profiles[uuid]
        ts_sorted = sorted(p['timestamps'])
        span = (ts_sorted[0], ts_sorted[-1]) if ts_sorted else (None, None)
        print(f"Node {gid} -> uuid {uuid}")
        print(f"  CDM type: {p['type']}")
        print(f"  Raw edges in test file: {p['edge_count']:,}")
        print(f"  Distinct edge types: {sorted(p['edge_types'])}")
        print(f"  Distinct raw neighbours: {len(p['neighbours']):,}")
        print(f"  Distinct neighbour CDM types: {sorted(p['neighbour_types'])}")
        print(f"  Timestamp span: {span[0]} .. {span[1]}")
        print()

    stored = alarm_neighbour_sets(alarm_path, watch)
    a, b = WATCH_IDS
    if a in stored and b in stored:
        set_a, set_b = stored[a], stored[b]
        overlap = set_a & set_b
        print(f"Node {a}'s stored 2-hop set: {len(set_a):,} nodes")
        print(f"Node {b}'s stored 2-hop set: {len(set_b):,} nodes")
        print(f"Overlap: {len(overlap):,} ({100*len(overlap)/min(len(set_a), len(set_b)):.2f}% of the smaller set)")
        print(f"{a} in {b}'s 2-hop set: {a in set_b} | {b} in {a}'s 2-hop set: {b in set_a}")


if __name__ == '__main__':
    main()