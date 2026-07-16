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
from torch_geometric.nn import SAGEConv, HeteroConv
from data_process_test import MyDatasetA

MEM_DIM = 64  # must match train_darpatc.py


def show(str_msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{ts}] {str_msg}')


class SAGEMemNet(torch.nn.Module):
    """Same architecture as train_darpatc.py's SAGEMemNet -- duplicated here
    because this script is standalone and doesn't import from
    train_darpatc.py. If you change one, change the other.
    """
    def __init__(self, in_channels, out_channels, edge_types, mem_dim=MEM_DIM):
        super().__init__()
        self.mem_dim = mem_dim
        self.edge_types = list(edge_types)

        self.conv1 = HeteroConv({
            ('node', et, 'node'): SAGEConv(in_channels, 32, normalize=False)
            for et in self.edge_types
        }, aggr='sum')

        self.readout = torch.nn.Linear(32, mem_dim)
        self.readout_norm = torch.nn.LayerNorm(mem_dim)
        self.gru = torch.nn.GRUCell(mem_dim, mem_dim)
        self.ctx_proj = torch.nn.Linear(mem_dim, 32)
        self.ctx_gate = torch.nn.Parameter(torch.tensor(-2.0))

        self.conv2 = HeteroConv({
            ('node', et, 'node'): SAGEConv(32, out_channels, normalize=False)
            for et in self.edge_types
        }, aggr='sum')

    def forward(self, x, edge_index_dict, prev_state):
        h_dict = self.conv1({'node': x}, edge_index_dict)
        h = F.relu(h_dict['node'])

        g = self.readout_norm(self.readout(h).mean(dim=0, keepdim=True))
        new_state = torch.tanh(self.gru(g, prev_state))

        ctx = self.ctx_proj(new_state).expand(h.size(0), -1)
        gate = torch.sigmoid(self.ctx_gate)
        h = F.dropout(h + gate * ctx, p=0.5, training=self.training)

        out_dict = self.conv2({'node': h}, edge_index_dict)
        out = out_dict['node']
        return F.log_softmax(out, dim=1), new_state

    def init_state(self, device):
        return torch.zeros(1, self.mem_dim, device=device)


def _predict_batch(model, batch, device, thre, state):
    """Run forward pass and apply confidence-ratio threshold for one batch.

    Returns
    -------
    pred   : [batch_size] — predicted class (100 = uncertain / below threshold)
    y_true : [batch_size] — ground-truth labels for seed nodes only
    n_ids  : [batch_size] — global node ids for seed nodes only
    """
    batch = batch.to(device)
    node_store = batch['node']
    out, _ = model(node_store.x, batch.edge_index_dict, state)

    bs = node_store.batch_size
    out    = out[:bs]
    y_true = node_store.y[:bs]
    n_ids  = node_store.n_id[:bs]

    pred = out.max(1)[1].clone()
    pro  = F.softmax(out, dim=1)
    pro1 = pro.max(1)

    for i in range(bs):
        pro[i][pro1[1][i]] = -1
    pro2 = pro.max(1)

    for i in range(bs):
        if pro2[0][i] <= 0 or pro1[0][i] / pro2[0][i] < thre:
            pred[i] = 100

    return pred, y_true, n_ids


def make_loader(data, mask, b_size):
    num_neighbors = {et: [-1, -1] for et in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('node', mask),
        batch_size=b_size,
        shuffle=False,
    )


def run_test(model, data, b_size, device, thre, state):
    """Evaluate model over data['node'].test_mask, using a FIXED memory
    state for every batch. Rebuilds the loader from the current mask state
    on every call so mask mutations between iterations are reflected.
    """
    loader = make_loader(data, data['node'].test_mask, b_size)
    model.eval()
    fp, tn = [], []
    correct = 0

    with torch.no_grad():
        for batch in loader:
            pred, y_true, n_ids = _predict_batch(model, batch, device, thre, state)
            for i in range(len(n_ids)):
                nid = int(n_ids[i].item())
                if y_true[i] != pred[i]:
                    fp.append(nid)
                else:
                    tn.append(nid)
            correct += pred.eq(y_true).sum().item()

    denom = data['node'].test_mask.sum().item()
    acc = correct / denom if denom > 0 else 0.0
    return fp, tn, acc


def _find_model_indices(model_dir='../models'):
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

    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour, edge_type_names = \
        MyDatasetA(path, args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Using device: {device}')
    model  = SAGEMemNet(feature_num, label_num, edge_type_names).to(device)
    thre   = thre_map[args.scene]

    model_indices = _find_model_indices()
    if not model_indices:
        show('No saved models found in ../models/ — aborting.')
        return

    for loop_num in model_indices:
        model_path = f'../models/model_{loop_num}'

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

        for i in tn:
            data['node'].test_mask[i] = False

        if acc == 1.0:
            break

    # --- Write alarm file (unchanged: adj/adj2 are type-agnostic) ---
    total_nodes = data['node'].test_mask.size(0)
    with open('alarm.txt', 'w') as fw:
        fw.write(f'{total_nodes}\n')
        for i in range(total_nodes):
            if not data['node'].test_mask[i].item():
                continue
            fw.write('\n')
            fw.write(f'{i}:')
            neighbours = set()

            for j in adj.get(i, []):
                neighbours.add(j)
                for k in adj.get(j, []):
                    neighbours.add(k)

            for j in adj2.get(i, []):
                neighbours.add(j)
                for k in adj2.get(j, []):
                    neighbours.add(k)

            for j in neighbours:
                fw.write(f' {j}')

    show(f'Finish testing graph {graphId} in model {args.model}')


if __name__ == '__main__':
    main()
