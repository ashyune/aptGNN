import os
import os.path as osp
import argparse
import re
import glob
import shutil
import torch
import time
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from data_process_test import MyDatasetA
from model import SAGEMemNet  # single source of truth, shared with train_darpatc.py


def show(str_msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {str_msg}')


def _predict_batch(model, batch, device, thre, state):
    """Run forward pass and apply confidence-ratio threshold for one batch.

    Returns
    -------
    pred   : [batch_size] — predicted class (100 = uncertain / below threshold)
    y_true : [batch_size] — ground-truth labels for seed nodes only
    n_ids  : [batch_size] — global node ids for seed nodes only
    """
    batch = batch.to(device)
    out, _ = model(batch.x, batch.edge_index, state)

    # Slice to seed nodes — remaining rows are sampled neighbours used
    # only for message-passing context.
    out    = out[:batch.batch_size]
    y_true = batch.y[:batch.batch_size]
    n_ids  = batch.n_id[:batch.batch_size]

    pred = out.max(1)[1].clone()
    pro  = F.softmax(out, dim=1)
    pro1 = pro.max(1)

    for i in range(batch.batch_size):
        pro[i][pro1[1][i]] = -1
    pro2 = pro.max(1)

    for i in range(batch.batch_size):
        if pro2[0][i] <= 0 or pro1[0][i] / pro2[0][i] < thre:
            pred[i] = 100

    return pred, y_true, n_ids


def make_loader(data, mask, b_size):
    return NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        input_nodes=mask,
        batch_size=b_size,
        shuffle=False,
    )


def run_test(model, data, b_size, device, thre, state):
    """Evaluate model over data.test_mask, using a FIXED memory state for
    every batch (same convention as train_darpatc.py's test()/final_test():
    no intra-eval drift, order-independent result).

    Rebuilds the loader from the current mask state on every call so that
    mask mutations between iterations are correctly reflected.

    Returns
    -------
    fp : list of global node ids classified incorrectly
    tn : list of global node ids classified correctly
    acc: float accuracy over active test nodes
    """
    loader = make_loader(data, data.test_mask, b_size)
    model.eval()
    fp, tn = [], []
    correct = 0

    with torch.no_grad():
        for batch in loader:
            pred, y_true, n_ids = _predict_batch(model, batch, device, thre, state)
            for i in range(batch.batch_size):
                nid = int(n_ids[i].item())
                if y_true[i] != pred[i]:
                    fp.append(nid)
                else:
                    tn.append(nid)
            correct += pred.eq(y_true).sum().item()

    denom = data.test_mask.sum().item()
    acc = correct / denom if denom > 0 else 0.0
    return fp, tn, acc


def _find_model_indices(model_dir='../models'):
    """Return a sorted list of integer indices for files named model_N.

    Uses a strict regex so auxiliary files such as
    fp_feature_label_0_1.txt are never mistakenly matched.
    """
    pattern = re.compile(r'^model_(\d+)$')
    indices = []
    for path in glob.glob(osp.join(model_dir, 'model_*')):
        m = pattern.match(osp.basename(path))
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='0')
    parser.add_argument('--scene', type=str, default='')
    args = parser.parse_args()
    assert args.scene in ['cadets', 'trace', 'theia', 'fivedirections'], (
        f"Unknown scene '{args.scene}'. "
        "Choose from: cadets, trace, theia, fivedirections"
    )

    thre_map = {
        "cadets": 1.5, "trace": 1.0,
        "theia": 1.5,  "fivedirections": 1.0,
    }
    b_size = 5000
    path   = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_test.txt'
    )
    graphId = 1
    show(f'Start testing graph {graphId} in model {args.model}')

    # --- Data loading ---
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour = \
        MyDatasetA(path, args.model)

    device = torch.device('cpu')
    model  = SAGEMemNet(feature_num, label_num).to(device)
    thre   = thre_map[args.scene]

    # --- Discover saved models ---
    model_indices = _find_model_indices()
    if not model_indices:
        show('No saved models found in ../models/ — aborting.')
        return

    # --- Iterative evaluation loop ---
    # For each saved model (in index order):
    #   1. Load weights + the memory snapshot saved alongside that checkpoint.
    #   2. Evaluate over currently-active test nodes.
    #   3. Remove correctly-classified nodes from test_mask.
    #   4. Stop early if accuracy reaches 1.0 (no remaining errors).
    for loop_num in model_indices:
        model_path = f'../models/model_{loop_num}'

        # Defensive check — file could have been deleted between discovery
        # and this point (e.g. concurrent run).
        if not osp.exists(model_path):
            show(f'WARNING: model_{loop_num} disappeared — skipping.')
            continue

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        mem_path = f'../models/memory_{loop_num}.pt'
        if osp.exists(mem_path):
            state = torch.load(mem_path, map_location=device)['state']
        else:
            show(f'WARNING: memory_{loop_num}.pt not found — using zero state.')
            state = model.init_state(device)

        fp, tn, acc = run_test(model, data, b_size, device, thre, state)

        print(f'[Model {loop_num}] Acc: {acc:.4f} | FP: {len(fp)} | TN: {len(tn)}')

        # Remove correctly-classified nodes so subsequent models see only
        # the still-uncertain ones.
        for i in tn:
            data.test_mask[i] = False

        if acc == 1.0:
            break

    # --- Write alarm file ---
    # Format expected by evaluate_darpatc.py:
    #   Line 1 : total number of nodes in the graph
    #   Per remaining flagged node:
    #     blank line
    #     "<node_id>: <neighbour_id> <neighbour_id> ..."
    #       where neighbours are the 2-hop subgraph around that node
    total_nodes = data.test_mask.size(0)
    with open('alarm.txt', 'w') as fw:
        fw.write(f'{total_nodes}\n')
        for i in range(total_nodes):
            if not data.test_mask[i].item():
                continue
            fw.write('\n')
            fw.write(f'{i}:')
            neighbours = set()

            # Incoming edges (adj): 2-hop backwards
            for j in adj.get(i, []):
                neighbours.add(j)
                for k in adj.get(j, []):
                    neighbours.add(k)

            # Outgoing edges (adj2): 2-hop forwards
            for j in adj2.get(i, []):
                neighbours.add(j)
                for k in adj2.get(j, []):
                    neighbours.add(k)

            for j in neighbours:
                fw.write(f' {j}')

    show(f'Finish testing graph {graphId} in model {args.model}')


if __name__ == '__main__':
    main()