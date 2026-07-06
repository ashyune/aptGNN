import torch
from torch_geometric.data import Data


def _load_provenance(path):
    """Single pass over the file: build the global node/type vocabularies
    ONCE so ids stay consistent across every time window of this scene
    (a node seen in window 3 must map to the same row as if it were seen
    in window 0). Also writes feature.txt / label.txt exactly as before.
    """
    node_cnt = 0
    nodeType_cnt = 0
    edgeType_cnt = 0
    provenance = []          # kept in FILE ORDER == our proxy for time order
    nodeType_map = {}
    edgeType_map = {}
    nodeId_map = {}

    with open(path, 'r') as f:
        for line in f:
            temp = line.strip('\n').split('\t')

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

            # Only keep the 5 fields we actually use — some DARPA TC lines
            # carry extra trailing columns (e.g. a timestamp, or a stray
            # trailing tab from the graphchi export) that would otherwise
            # break the 5-value unpack below.
            provenance.append(temp[:5])

    with open('../models/feature.txt', 'w') as f_feature:
        for name, idx in edgeType_map.items():
            f_feature.write(f'{name}\t{idx}\n')
    with open('../models/label.txt', 'w') as f_label:
        for name, idx in nodeType_map.items():
            f_label.write(f'{name}\t{idx}\n')

    return provenance, node_cnt, edgeType_cnt, nodeType_cnt


def _chronological_chunks(provenance, num_windows):
    """Equal-sized, order-preserving chunks. Merge a too-small tail chunk
    into the previous one so the last window isn't a near-empty sliver."""
    n = len(provenance)
    num_windows = max(1, min(num_windows, n))
    size = max(1, n // num_windows)
    chunks = [provenance[i:i + size] for i in range(0, n, size)]
    if len(chunks) > num_windows and len(chunks[-1]) < size // 2:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def MyDataset(path, num_windows=1):
    """Returns (windows, feature_num, label_num).

    windows: list of Data objects, one per chronological window, sharing
    one global node vocabulary. x is CUMULATIVE (a node's edge-type
    histogram keeps growing window over window); edge_index and the masks
    are window-scoped (only edges/activity that happened in that window).

    data.active_mask: nodes seen in this window OR any earlier window.
    Use this as train_mask/test_mask so we never score a node before it
    has appeared at least once.

    num_windows=1 reproduces the old single-snapshot behaviour exactly,
    modulo the return signature (list-of-one instead of a bare Data).
    """
    provenance, node_cnt, edgeType_cnt, nodeType_cnt = _load_provenance(path)
    feature_num, label_num = edgeType_cnt, nodeType_cnt

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)  # persists across windows
    y = torch.zeros(node_cnt, dtype=torch.long)
    seen = torch.zeros(node_cnt, dtype=torch.bool)

    windows = []
    for chunk in _chronological_chunks(provenance, num_windows):
        edge_s, edge_e = [], []
        for temp in chunk:
            srcId, srcType, dstId, dstType, edge = temp
            x[srcId, edge] += 1
            y[srcId] = srcType
            x[dstId, edge + feature_num] += 1
            y[dstId] = dstType
            edge_s.append(srcId)
            edge_e.append(dstId)
            seen[srcId] = True
            seen[dstId] = True

        edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
        active_mask = seen.clone()

        windows.append(Data(
            x=x.clone(), y=y.clone(), edge_index=edge_index,
            train_mask=active_mask.clone(),
            test_mask=active_mask.clone(),
            active_mask=active_mask.clone(),
        ))

    feature_num *= 2
    return windows, feature_num, label_num