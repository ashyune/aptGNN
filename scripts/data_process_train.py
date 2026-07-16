import torch
from torch_geometric.data import HeteroData


def _load_provenance(path):
    """Single pass over the file: build the global node/type vocabularies
    ONCE so ids stay consistent across every time window of this scene
    (a node seen in window 3 must map to the same row as if it were seen
    in window 0). Also writes feature.txt / label.txt exactly as before.

    Each raw line has 6 tab-separated fields:
        src_uuid  src_type  dst_uuid  dst_type  edge_type  timestamp_ns
    timestamp_ns is a real Unix nanosecond timestamp. We keep it so
    provenance can be sorted into genuine chronological order before
    windowing.

    Returns edge_type_names as well: a list where edge_type_names[idx] is
    the string name of edge-type idx, built from the SAME edgeType_map
    written to feature.txt, in idx order (not dict insertion order, so
    it's robust regardless of how the dict was populated). This is the
    canonical relation-name ordering used to build HeteroConv's relation
    set in train_darpatc.py -- MyDatasetA derives an equivalent list from
    the same feature.txt file at test time, so train/test relation sets
    always agree.
    """
    node_cnt = 0
    nodeType_cnt = 0
    edgeType_cnt = 0
    provenance = []          # each entry: [srcId, srcType, dstId, dstType, edge, ts_ns]
    nodeType_map = {}
    edgeType_map = {}
    nodeId_map = {}

    with open(path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')
            ts_ns = int(temp[5])

            if temp[0] not in nodeId_map:
                nodeId_map[temp[0]] = node_cnt
                node_cnt += 1
            temp[0] = nodeId_map[temp[0]]

            if temp[2] not in nodeId_map:
                nodeId_map[temp[2]] = node_cnt
                node_cnt += 1
            temp[2] = nodeId_map[temp[2]]

            if temp[1] not in nodeType_map:
                nodeType_map[temp[1]] = nodeType_cnt
                nodeType_cnt += 1
            temp[1] = nodeType_map[temp[1]]

            if temp[3] not in nodeType_map:
                nodeType_map[temp[3]] = nodeType_cnt
                nodeType_cnt += 1
            temp[3] = nodeType_map[temp[3]]

            if temp[4] not in edgeType_map:
                edgeType_map[temp[4]] = edgeType_cnt
                edgeType_cnt += 1
            temp[4] = edgeType_map[temp[4]]

            provenance.append([temp[0], temp[1], temp[2], temp[3], temp[4], ts_ns])

    provenance.sort(key=lambda e: e[5])

    with open('../models/feature.txt', 'w') as f_feature:
        for name, idx in edgeType_map.items():
            f_feature.write(f'{name}\t{idx}\n')
    with open('../models/label.txt', 'w') as f_label:
        for name, idx in nodeType_map.items():
            f_label.write(f'{name}\t{idx}\n')

    edge_type_names = [None] * edgeType_cnt
    for name, idx in edgeType_map.items():
        edge_type_names[idx] = name

    return provenance, node_cnt, edgeType_cnt, nodeType_cnt, edge_type_names


def _equal_count_chunks(provenance, num_windows, min_edges=1):
    """Equal-sized chunks of the (now timestamp-sorted) edge list."""
    n = len(provenance)
    if min_edges > 1 and num_windows > 1:
        max_windows_allowed = max(1, n // min_edges)
        if num_windows > max_windows_allowed:
            print(f'[data_process_train] requested {num_windows} windows would '
                  f'average {n // num_windows} edges/window (< min_edges='
                  f'{min_edges}); reducing to {max_windows_allowed} windows.')
            num_windows = max_windows_allowed

    num_windows = max(1, min(num_windows, n))
    size = max(1, n // num_windows)
    chunks = [provenance[i:i + size] for i in range(0, n, size)]

    if len(chunks) > num_windows:
        surplus = chunks[num_windows:]
        del chunks[num_windows:]
        for extra in surplus:
            chunks[-1].extend(extra)
    return chunks


def MyDataset(path, num_windows=1, min_edges_per_window=1):
    """Returns (windows, feature_num, label_num, edge_type_names).

    windows: list of HeteroData objects, one per chronological window.
    Single node type 'node'. One relation ('node', edge_type_name, 'node')
    per distinct edge type -- only relations with at least one edge in a
    given window are populated for that window's HeteroData object
    (HeteroConv simply skips relations absent from edge_index_dict, so a
    window missing a rare edge type is handled correctly, not an error).

    data['node'].x is CUMULATIVE across windows (unchanged from before);
    per-relation edge_index is window-scoped.

    edge_type_names: the FULL global list of edge type names for this
    scene (in canonical idx order), independent of which relations any
    single window happens to populate. Pass this to the model constructor
    so HeteroConv's relation set is fixed for the whole run, not
    window-dependent -- this is what lets weights for a relation that
    first appears in window 2 still exist (freshly initialized, untouched)
    from window 0 onward.
    """
    provenance, node_cnt, edgeType_cnt, nodeType_cnt, edge_type_names = \
        _load_provenance(path)
    feature_num, label_num = edgeType_cnt, nodeType_cnt

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)  # persists across windows
    y = torch.zeros(node_cnt, dtype=torch.long)
    seen = torch.zeros(node_cnt, dtype=torch.bool)

    windows = []
    for chunk in _equal_count_chunks(provenance, num_windows, min_edges=min_edges_per_window):
        # Group this window's edges by type so each relation gets its own
        # edge_index tensor.
        per_type_edges = {}   # edge_type_idx -> ([src...], [dst...])

        for temp in chunk:
            srcId, srcType, dstId, dstType, edge, _ts_ns = temp
            x[srcId, edge] += 1
            y[srcId] = srcType
            x[dstId, edge + feature_num] += 1
            y[dstId] = dstType
            seen[srcId] = True
            seen[dstId] = True

            s, d = per_type_edges.setdefault(edge, ([], []))
            s.append(srcId)
            d.append(dstId)

        active_mask = seen.clone()

        data = HeteroData()
        data['node'].x = x.clone()
        data['node'].y = y.clone()
        data['node'].train_mask = active_mask.clone()
        data['node'].test_mask = active_mask.clone()
        data['node'].active_mask = active_mask.clone()

        for edge_type_idx, (s, d) in per_type_edges.items():
            rel_name = edge_type_names[edge_type_idx]
            data['node', rel_name, 'node'].edge_index = torch.tensor([s, d], dtype=torch.long)

        windows.append(data)

    feature_num *= 2
    return windows, feature_num, label_num, edge_type_names
