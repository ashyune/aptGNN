import time
import torch
from torch_geometric.data import Data


def show(s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {s}')

def MyDatasetA(path, model):
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

    ground_truth = {}
    with open('groundtruth_uuid.txt', 'r') as f_gt:
        for line in f_gt:
            ground_truth[line.strip('\n')] = 1

    node_cnt = 0
    provenance = []
    edge_s = []
    edge_e = []
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

            edge_s.append(temp[0])
            edge_e.append(temp[2])

            adj.setdefault(temp[2], []).append(temp[0])
            adj2.setdefault(temp[0], []).append(temp[2])

            provenance.append(temp)

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)
    y = torch.zeros(node_cnt, dtype=torch.long)

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

    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)

    # Only test_mask is used downstream; train_mask is omitted intentionally.
    test_mask = torch.ones(node_cnt, dtype=torch.bool)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        test_mask=test_mask,
    )

    feature_num *= 2

    # --- 2-hop neighbourhood expansion around ground-truth nodes ---
    # neighbour  : flat set of all nodes within 2 hops (either direction) of
    #              any nodeA member. Used by validate() as the FP rule:
    #              an alarm outside this set is a false positive.
    # _neighbour : maps each such node to the list of nodeA anchors it is
    #              within 2 hops of. Used by validate() as the TP rule:
    #              an alarm credits every anchor in its list as detected.
    #              Both directions must be recorded here, otherwise the TP
    #              rule and the FP rule disagree about what "within 2 hops"
    #              means, and validate() stops matching evaluate_darpatc.py.
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

    return data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour