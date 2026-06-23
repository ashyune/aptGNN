import torch
from torch_geometric.data import Data


def MyDataset(path, model):
    node_cnt = 0
    nodeType_cnt = 0
    edgeType_cnt = 0
    provenance = []
    nodeType_map = {}
    edgeType_map = {}
    edge_s = []
    edge_e = []
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

            edge_s.append(temp[0])
            edge_e.append(temp[2])
            provenance.append(temp)

    with open('../models/feature.txt', 'w') as f_feature:
        for name, idx in edgeType_map.items():
            f_feature.write(f'{name}\t{idx}\n')

    with open('../models/label.txt', 'w') as f_label:
        for name, idx in nodeType_map.items():
            f_label.write(f'{name}\t{idx}\n')

    feature_num = edgeType_cnt
    label_num = nodeType_cnt

    x = torch.zeros((node_cnt, feature_num * 2), dtype=torch.float)
    y = torch.zeros(node_cnt, dtype=torch.long)
    train_mask = torch.ones(node_cnt, dtype=torch.bool)

    for temp in provenance:
        srcId  = temp[0]
        srcType = temp[1]
        dstId  = temp[2]
        dstType = temp[3]
        edge   = temp[4]
        x[srcId, edge] += 1
        y[srcId] = srcType
        x[dstId, edge + feature_num] += 1
        y[dstId] = dstType

    # test_mask is an independent clone so in-place mutations to one
    # do not silently corrupt the other.
    test_mask = train_mask.clone()

    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        test_mask=test_mask,
    )

	# adj, adj2 placeholders kept for compatibility
	# The last two zeros are ugly, but if train_darpatc.py expects five return values, changing it will break the caller.
    feature_num *= 2
    return data, feature_num, label_num