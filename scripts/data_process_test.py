import time
import torch
from torch_geometric.data import HeteroData


def show(s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {s}')


def MyDatasetA(path, model):
    """Returns (data, feature_num, label_num, adj, adj2, nodeA, _nodeA,
    _neighbour, edge_type_names).

    data is now a HeteroData object (single node type 'node', one relation
    per edge type) mirroring MyDataset's structure, so the SAME model
    class (built with the same edge_type_names) can be evaluated on it.

    adj / adj2 / _neighbour remain type-agnostic flat dicts keyed by plain
    node id -- these exist purely for the 2-hop ground-truth scoring used
    by validate() and the alarm.txt writer in test_darpatc.py, which never
    needed to distinguish edge types; only the message-passing structure
    (edge_index) needed that distinction.
    """
    feature_num = 0
    label_num = 0

    feature_map = {}
    with open('../models/feature.txt', 'r') as f_feature:
        for line in f_feature:
            temp = line.strip('\n').split('\t')
            feature_map[temp[0]] = int(temp[1])
            feature_num += 1

    label_map = {}
    with open('../models/label.txt', 'r') as f_label:
        for line in f_label:
            temp = line.strip('\n').split('\t')
            label_map[temp[0]] = int(temp[1])
            label_num += 1

    # Canonical idx-ordered edge type name list, built from the SAME
    # feature.txt that data_process_train.py wrote -- guarantees this
    # matches MyDataset's edge_type_names exactly, so the HeteroConv
    # relation set constructed at test time lines up with the trained
    # checkpoint's relation set.
    edge_type_names = [None] * feature_num
    for name, idx in feature_map.items():
        edge_type_names[idx] = name

    ground_truth = {}
    with open('groundtruth_uuid.txt', 'r') as f_gt:
        for line in f_gt:
            ground_truth[line.strip('\n')] = 1

    node_cnt = 0
    provenance = []
    adj = {}
    adj2 = {}
    nodeId_map = {}
    nodeA = []

    show(f'Loading: {path}')

    with open('groundtruth_nodeId.txt', 'w') as fw, \
         open('id_to_uuid.txt', 'w') as fw2, \
         open(path, 'r') as f:

        for line in f:
            temp = line.strip('\n').split('\t')

            if temp[1] not in label_map:
                continue
            if temp[3] not in label_map:
                continue
            if temp[4] not in feature_map:
                continue

            if temp[0] not in nodeId_map:
                nodeId_map[temp[0]] = node_cnt
                fw2.write(f'{node_cnt} {temp[0]}\n')
                if temp[0] in ground_truth:
                    fw.write(f'{node_cnt} {temp[1]} {temp[0]}\n')
                    nodeA.append(node_cnt)
                node_cnt += 1
            temp[0] = nodeId_map[temp[0]]

            if temp[2] not in nodeId_map:
                nodeId_map[temp[2]] = node_cnt
                fw2.write(f'{node_cnt} {temp[2]}\n')
                if temp[2] in ground_truth:
                    fw.write(f'{node_cnt} {temp[3]} {temp[2]}\n')
                    nodeA.append(node_cnt)
                node_cnt += 1
            temp[2] = nodeId_map[temp[2]]

            temp[1] = label_map[temp[1]]
            temp[3] = label_map[temp[3]]
            temp[4] = feature_map[temp[4]]

            adj.setdefault(temp[2], []).append(temp[0])   # incoming (backward)
            adj2.setdefault(temp[0], []).append(temp[2])  # outgoing (forward)

            provenance.append(temp)

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)
    y = torch.zeros(node_cnt, dtype=torch.long)
    per_type_edges = {}   # edge_type_idx -> ([src...], [dst...])

    for temp in provenance:
        srcId   = temp[0]
        srcType = temp[1]
        dstId   = temp[2]
        dstType = temp[3]
        edge    = temp[4]
        x[srcId, edge] += 1
        y[srcId] = srcType
        x[dstId, edge + feature_num] += 1
        y[dstId] = dstType

        s, d = per_type_edges.setdefault(edge, ([], []))
        s.append(srcId)
        d.append(dstId)

    test_mask = torch.ones(node_cnt, dtype=torch.bool)

    data = HeteroData()
    data['node'].x = x
    data['node'].y = y
    data['node'].test_mask = test_mask

    for edge_type_idx, (s, d) in per_type_edges.items():
        rel_name = edge_type_names[edge_type_idx]
        data['node', rel_name, 'node'].edge_index = torch.tensor([s, d], dtype=torch.long)

    feature_num *= 2

    # --- 2-hop neighbourhood expansion around ground-truth nodes ---
    # (type-agnostic -- see module docstring)
    neighbour = set()
    _neighbour = {}

    for i in nodeA:
        neighbour.add(i)
        _neighbour.setdefault(i, set()).add(i)

        if i in adj:
            for j in adj[i]:
                neighbour.add(j)
                _neighbour.setdefault(j, set()).add(i)

                if j in adj:
                    for k in adj[j]:
                        neighbour.add(k)
                        _neighbour.setdefault(k, set()).add(i)

        if i in adj2:
            for j in adj2[i]:
                neighbour.add(j)
                _neighbour.setdefault(j, set()).add(i)

                if j in adj2:
                    for k in adj2[j]:
                        neighbour.add(k)
                        _neighbour.setdefault(k, set()).add(i)

    _nodeA = list(neighbour)
    _neighbour = {node: list(anchors) for node, anchors in _neighbour.items()}

    return data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour, edge_type_names
