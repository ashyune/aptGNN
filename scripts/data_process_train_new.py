import torch
from torch_geometric.data import Data


def _load_provenance(path):
    """Single pass over the file: build the global node/type vocabularies
    ONCE so ids stay consistent across every time window of this scene
    (a node seen in window 3 must map to the same row as if it were seen
    in window 0). Also writes feature.txt / label.txt exactly as before.

    Each raw line has 6 tab-separated fields:
        src_uuid  src_type  dst_uuid  dst_type  edge_type  timestamp_ns
    timestamp_ns is a real Unix nanosecond timestamp (confirmed against
    cadets_train.txt: 1522828474810632615 -> 2018-04-04, matches the
    DARPA TC3 collection window). We keep it so provenance can be sorted
    into GENUINE chronological order before windowing, instead of relying
    on file line order as only an approximation of time order.

    NOTE (edge-type awareness): temp[4] (the "edge" field) already IS a
    dense integer id for the syscall/edge type (READ, WRITE, EXECUTE, ...)
    via edgeType_map. Historically this file only ever used that id to
    build a one-hot histogram feature (x). It is now ALSO carried through
    unchanged as the relation id fed to the edge-type-aware convolution
    (edge_type). No new vocabulary, no new file format -- we are reusing
    a value that was already being computed.
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

    # Sort into genuine chronological order. Stable sort preserves original
    # file order among any edges that share the exact same nanosecond
    # timestamp (common here -- audit logs batch-write bursts of events
    # under one timestamp), rather than shuffling ties arbitrarily.
    provenance.sort(key=lambda e: e[5])

    with open('../models/feature.txt', 'w') as f_feature:
        for name, idx in edgeType_map.items():
            f_feature.write(f'{name}\t{idx}\n')
    with open('../models/label.txt', 'w') as f_label:
        for name, idx in nodeType_map.items():
            f_label.write(f'{name}\t{idx}\n')

    return provenance, node_cnt, edgeType_cnt, nodeType_cnt


def _equal_count_chunks(provenance, num_windows, min_edges=1):
    """Equal-sized chunks of the (now timestamp-sorted) edge list. We chunk
    by count rather than by equal wall-clock duration: this dataset has
    bursty, batched timestamps (many edges sharing the exact same
    nanosecond), so fixed-duration bins risk producing empty or wildly
    uneven windows. Equal-count chunking over truly sorted data still
    gives genuine chronological windows without that risk.

    min_edges: if the requested num_windows would make each window thinner
    than this, num_windows is silently reduced until windows meet the
    floor. This exists because a window's edge_index is what the
    relation-aware convolution actually message-passes over -- a window
    with too few edges gives the model too little neighbourhood structure
    to learn from, independent of how many total nodes are "active"
    (cumulative) in that window.

    Merges a too-small tail chunk into the previous one so the last
    window isn't a near-empty sliver.
    """
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

    # range(0, n, size) yields ceil(n / size) chunks, which is num_windows +
    # ceil((n % num_windows) / size) -- i.e. it can overshoot num_windows by
    # one or more. Fold every surplus chunk into the last kept one. When the
    # surplus is a single small tail (the only case that arises under
    # MIN_WINDOW_EDGES) this is identical to merging that tail into
    # chunks[-2], so window boundaries are unchanged for the shipped config.
    if len(chunks) > num_windows:
        surplus = chunks[num_windows:]
        del chunks[num_windows:]
        for extra in surplus:
            chunks[-1].extend(extra)
    return chunks


def MyDataset(path, num_windows=1, min_edges_per_window=1):
    """Returns (windows, feature_num, label_num, num_relations).

    windows: list of Data objects, one per chronological window, sharing
    one global node vocabulary. x is CUMULATIVE (a node's edge-type
    histogram keeps growing window over window); edge_index, edge_type
    and the masks are window-scoped (only edges/activity that happened
    in that window).

    data.active_mask: nodes seen in this window OR any earlier window.
    Use this as train_mask/test_mask so we never score a node before it
    has appeared at least once.

    EDGE-TYPE AWARENESS (new): every window's Data object now also carries
    `edge_type`, a LongTensor of shape [num_edges_in_window] aligned
    1:1 with the columns of edge_index, giving the relation id (READ,
    WRITE, EXECUTE, ...) of each edge. This is the only structural
    addition -- chronological chunking, the cumulative x histogram, and
    all three masks (train_mask/test_mask/active_mask) are computed
    exactly as before.

    num_relations (new): the total number of distinct edge types seen in
    this scene (== edgeType_cnt from _load_provenance, i.e. feature_num
    BEFORE it gets doubled below). The caller needs this to size the
    relation-aware convolution's per-relation weight basis. It is a
    property of the *scene*, not of any individual window, so a single
    value is returned once rather than per-window.

    num_windows=1 reproduces the old single-snapshot behaviour exactly,
    modulo the return signature (list-of-one instead of a bare Data, and
    the new trailing num_relations value).
    """
    provenance, node_cnt, edgeType_cnt, nodeType_cnt = _load_provenance(path)
    feature_num, label_num = edgeType_cnt, nodeType_cnt
    num_relations = edgeType_cnt   # captured BEFORE feature_num is doubled

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)  # persists across windows
    y = torch.zeros(node_cnt, dtype=torch.long)
    seen = torch.zeros(node_cnt, dtype=torch.bool)

    windows = []
    for chunk in _equal_count_chunks(provenance, num_windows, min_edges=min_edges_per_window):
        edge_s, edge_e, edge_t = [], [], []
        for temp in chunk:
            srcId, srcType, dstId, dstType, edge, _ts_ns = temp
            x[srcId, edge] += 1
            y[srcId] = srcType
            x[dstId, edge + feature_num] += 1
            y[dstId] = dstType
            edge_s.append(srcId)
            edge_e.append(dstId)
            edge_t.append(edge)          # relation id, aligned with edge_s/edge_e
            seen[srcId] = True
            seen[dstId] = True

        edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
        edge_type = torch.tensor(edge_t, dtype=torch.long)
        active_mask = seen.clone()

        windows.append(Data(
            x=x.clone(), y=y.clone(), edge_index=edge_index,
            edge_type=edge_type,
            train_mask=active_mask.clone(),
            test_mask=active_mask.clone(),
            active_mask=active_mask.clone(),
        ))

    feature_num *= 2
    return windows, feature_num, label_num, num_relations
