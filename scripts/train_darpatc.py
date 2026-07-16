import os.path as osp
import os
import shutil
import glob
import argparse
import torch
import time
import torch.nn.functional as F
import numpy as np
import subprocess
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv
from data_process_train import MyDataset
from data_process_test import MyDatasetA

thre_map = {"cadets": 1.5, "trace": 1.0, "theia": 1.5, "fivedirections": 1.0}
NUM_WINDOWS = 3     # DEFAULT chronological chunks per scene; overridable via
                    # --num_windows.
MIN_WINDOW_EDGES = 5000    # floor: a window's own edge_index needs enough
                            # edges for SAGEConv to have real neighbourhoods
                            # to aggregate over.
MEM_DIM = 64        # size of the global cross-window memory vector


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAGEMemNet(torch.nn.Module):
    """Edge-type-aware GraphSAGE classifier + a persistent, GRU-updated
    GLOBAL memory vector carried across time windows.

    Edge-type awareness: conv1/conv2 are now HeteroConv layers with ONE
    SAGEConv per edge type ('node', edge_type_name, 'node'), instead of a
    single SAGEConv that aggregates every neighbour identically regardless
    of what kind of event connected them. HeteroConv is built from
    edge_types -- the FULL global relation set for this scene -- so every
    relation has its own weights from the start of training, even if a
    given window's edges don't touch every relation (HeteroConv silently
    skips relations absent from a particular forward call's
    edge_index_dict; it does not error, and does not need every relation
    present every window).

    The memory mechanism itself (readout -> GRU -> gated broadcast) is
    unchanged from the homogeneous version -- it still operates on the
    single-node-type embedding `h` after each HeteroConv layer collapses
    back down to a plain per-node tensor.
    """
    def __init__(self, in_channels, out_channels, edge_types, mem_dim=MEM_DIM):
        super().__init__()
        self.mem_dim = mem_dim
        self.edge_types = list(edge_types)

        self.conv1 = HeteroConv({
            ('node', et, 'node'): SAGEConv(in_channels, 32, normalize=False)
            for et in self.edge_types
        }, aggr='mean')   # mean, not sum: with 20+ relations now contributing
                          # per node, summing let per-node activation scale
                          # grow with how many relation types touched that
                          # node, which reintroduced the kind of unbounded-
                          # magnitude instability the LayerNorm/gate fix on
                          # the memory pathway was built to prevent in the
                          # first place. mean keeps a node's activation scale
                          # roughly independent of how many edge types it
                          # happens to participate in.

        self.readout = torch.nn.Linear(32, mem_dim)
        self.readout_norm = torch.nn.LayerNorm(mem_dim)
        self.gru = torch.nn.GRUCell(mem_dim, mem_dim)
        self.ctx_proj = torch.nn.Linear(mem_dim, 32)
        self.ctx_gate = torch.nn.Parameter(torch.tensor(-2.0))

        self.conv2 = HeteroConv({
            ('node', et, 'node'): SAGEConv(32, out_channels, normalize=False)
            for et in self.edge_types
        }, aggr='mean')

    def forward(self, x, edge_index_dict, prev_state):
        """x: plain [N, in_channels] tensor for the single 'node' type.
        edge_index_dict: {('node', edge_type, 'node'): edge_index, ...} --
        only relations present in the current batch need be included.
        """
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


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_loader(data, mask, b_size, shuffle=False):
    """Build a NeighborLoader scoped to the nodes indicated by *mask*, for
    a HeteroData object. num_neighbors is rebuilt from data.edge_types
    every call, since different windows can have different relations
    present (a window with zero edges of some rare type simply won't have
    that relation in data.edge_types, and num_neighbors must match)."""
    num_neighbors = {et: [-1, -1] for et in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('node', mask),
        batch_size=b_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Shared prediction helper
# ---------------------------------------------------------------------------

def _predict_batch(model, batch, device, thre, state):
    """Run forward pass and apply confidence-ratio threshold for one batch.

    Returns
    -------
    pred      : [batch_size] — predicted class (100 = below-threshold / uncertain)
    y_true    : [batch_size] — ground-truth labels for seed nodes
    n_ids     : [batch_size] — global node ids for seed nodes
    new_state : [1, mem_dim] — memory state after seeing this batch
    """
    batch = batch.to(device)
    node_store = batch['node']
    out, new_state = model(node_store.x, batch.edge_index_dict, state)

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

    return pred, y_true, n_ids, new_state


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, device, data, thre, state):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        node_store = batch['node']
        out, new_state = model(node_store.x, batch.edge_index_dict, state)
        bs = node_store.batch_size
        loss = F.nll_loss(out[:bs], node_store.y[:bs])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs
        state = new_state.detach()
    train_node_count = data['node'].train_mask.sum().item()
    avg_loss = total_loss / train_node_count if train_node_count > 0 else 0.0
    return avg_loss, state


def test(model, loader, device, thre, mask, state):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, _, _ = _predict_batch(model, batch, device, thre, state)
            correct += pred.eq(y_true).sum().item()
    denom = mask.sum().item()
    return correct / denom if denom > 0 else 0.0


def final_test(model, loader, device, thre, mask, fp, tn, state):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, n_ids, _ = _predict_batch(model, batch, device, thre, state)
            for i in range(len(n_ids)):
                nid = int(n_ids[i].item())
                if y_true[i] != pred[i]:
                    fp.append(nid)
                else:
                    tn.append(nid)
            correct += pred.eq(y_true).sum().item()
    denom = mask.sum().item()
    return correct / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature-log writer
# ---------------------------------------------------------------------------

def _save_feature_log(data, nodes, graphId, loop_num, tag):
    path = f'../models/{tag}_feature_label_{graphId}_{loop_num}.txt'
    x_list = data['node'].x[nodes]
    y_list = data['node'].y[nodes]
    n = len(x_list)
    if n > 1:
        order   = np.argsort(y_list.numpy(), axis=0)
        x_list  = x_list.numpy()[order]
        y_list  = y_list.numpy()[order]
    else:
        x_list = x_list.numpy()
        y_list = y_list.numpy()
    with open(path, 'w') as fw:
        for i in range(n):
            fw.write(str(y_list[i]) + ':')
            for v in x_list[i]:
                fw.write(' ' + str(v))
            fw.write('\n')


# ---------------------------------------------------------------------------
# Model-cleanup helpers
# ---------------------------------------------------------------------------

def _delete_model_files(graphId, from_loop):
    loop = from_loop
    while True:
        mp = f'../models/model_{loop}'
        if not osp.exists(mp):
            break
        os.remove(mp)
        mem_p = f'../models/memory_{loop}.pt'
        if osp.exists(mem_p):
            os.remove(mem_p)
        for tag in ('fp', 'tn'):
            p = f'../models/{tag}_feature_label_{graphId}_{loop}.txt'
            if osp.exists(p):
                os.remove(p)
        loop += 1


def _delete_all_model_files():
    for pattern in (
        '../models/model_*',
        '../models/memory_*.pt',
        '../models/tn_feature_label_*',
        '../models/fp_feature_label_*',
    ):
        for p in glob.glob(pattern):
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def validate(args, b_size, thre, graphId, device):
    """Evaluate saved models (and their matching memory snapshots) against
    the test snapshot. Returns 1 if precision/recall thresholds are met,
    else 0.
    """
    show('Start validating')
    path = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_test.txt'
    )
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour, edge_type_names = \
        MyDatasetA(path, 0)

    if len(nodeA) == 0:
        show('WARNING: no ground-truth nodes found — skipping validation')
        return 0

    print(data)
    model = SAGEMemNet(feature_num, label_num, edge_type_names).to(device)
    eps = 1e-10

    out_loop = -1
    while True:
        out_loop += 1
        model_path = f'../models/model_{out_loop}'
        if not osp.exists(model_path):
            break

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        mem_path = f'../models/memory_{out_loop}.pt'
        if osp.exists(mem_path):
            state = torch.load(mem_path, map_location=device)['state']
        else:
            state = model.init_state(device)

        loader = make_loader(data, data['node'].test_mask, b_size)

        fp, tn = [], []
        final_test(model, loader, device, thre, data['node'].test_mask, fp, tn, state)

        _fp = 0
        _tp = 0
        tempNodeA = {i: 1 for i in nodeA}

        for i in fp:
            if i not in _nodeA:
                _fp += 1
            for j in _neighbour.get(i, []):
                if j in tempNodeA:
                    tempNodeA[j] = 0

        for v in tempNodeA.values():
            if v == 0:
                _tp += 1

        precision = _tp / (_tp + _fp + eps)
        recall    = _tp / len(nodeA)
        print(
            f'[Model {out_loop}] Precision: {precision:.4f} | Recall: {recall:.4f}'
            f' | FP: {_fp} | TN: {len(tn)}'
        )

        if recall > 0.8 and precision > 0.7:
            _delete_model_files(graphId, out_loop + 1)
            return 1

        if recall <= 0.8:
            return 0

        for j in tn:
            data['node'].test_mask[j] = False

    return 0


# ---------------------------------------------------------------------------
# train_pro()
# ---------------------------------------------------------------------------

def train_pro(args, b_size, thre, num_windows=NUM_WINDOWS):
    subprocess.run(['python', 'setup.py'], check=True)

    path = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_train.txt'
    )
    graphId = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Using device: {device}')

    windows, feature_num, label_num, edge_type_names = MyDataset(
        path, num_windows, min_edges_per_window=MIN_WINDOW_EDGES
    )
    show(f'feature {feature_num}; label {label_num}; {len(windows)} windows '
         f'(requested {num_windows}); {len(edge_type_names)} edge types')

    model = SAGEMemNet(feature_num, label_num, edge_type_names).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)

    state = model.init_state(device)
    global_loop = 0
    max_thre = 3

    for w_idx, data in enumerate(windows):
        show(f'--- window {w_idx}/{len(windows) - 1} '
             f'({int(data["node"].active_mask.sum().item())} active nodes) ---')

        train_loader = make_loader(data, data['node'].train_mask, b_size)
        test_loader  = make_loader(data, data['node'].test_mask,  b_size)

        for epoch in range(1, 30):
            loss, state = train(model, train_loader, optimizer, device, data, thre, state)
            auc = test(model, test_loader, device, thre, data['node'].test_mask, state)
            ts = time.strftime("%H:%M:%S", time.localtime())
            print(f'[{ts}] window {w_idx} epoch {epoch} | Loss: {loss:.4f} | Acc: {auc:.4f}')

        bad_cnt = 0
        while True:
            fp, tn = [], []
            test_loader = make_loader(data, data['node'].test_mask, b_size)
            final_test(model, test_loader, device, thre, data['node'].test_mask, fp, tn, state)

            bad_cnt = bad_cnt + 1 if len(tn) == 0 else 0
            if bad_cnt >= max_thre:
                break

            if len(tn) > 0:
                for i in tn:
                    data['node'].train_mask[i] = False
                    data['node'].test_mask[i]  = False

                _save_feature_log(data, fp, graphId, global_loop, 'fp')
                _save_feature_log(data, tn, graphId, global_loop, 'tn')
                torch.save(model.state_dict(), f'../models/model_{global_loop}')
                torch.save({'state': state.clone()}, f'../models/memory_{global_loop}.pt')
                global_loop += 1

                if len(fp) == 0:
                    break

            train_loader = make_loader(data, data['node'].train_mask, b_size)
            test_loader  = make_loader(data, data['node'].test_mask,  b_size)

            last_loss = None
            for epoch in range(1, 150):
                loss, state = train(model, train_loader, optimizer, device, data, thre, state)
                auc = test(model, test_loader, device, thre, data['node'].test_mask, state)
                ts = time.strftime("%H:%M:%S", time.localtime())
                print(f'[{ts}] window {w_idx} refine epoch {epoch} | Loss: {loss:.4f} | Acc: {auc:.4f}')
                last_loss = loss
                if loss < 1:
                    break

            if last_loss is not None and last_loss < 1:
                for extra_epoch in range(2, 6):
                    loss, state = train(model, train_loader, optimizer, device, data, thre, state)
                    auc = test(model, test_loader, device, thre, data['node'].test_mask, state)
                    ts = time.strftime("%H:%M:%S", time.localtime())
                    print(f'[{ts}] window {w_idx} refine (extra) epoch {extra_epoch} | Loss: {loss:.4f} | Acc: {auc:.4f}')

    show(f'Finish training graph {graphId} across {len(windows)} windows')
    return graphId, device


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SAGE')
    parser.add_argument('--scene', type=str, default='theia')
    parser.add_argument('--num_windows', type=int, default=NUM_WINDOWS,
                         help='Requested chronological window count. May be '
                              'reduced automatically if it would make windows '
                              f'thinner than MIN_WINDOW_EDGES ({MIN_WINDOW_EDGES} '
                              'edges).')
    parser.add_argument('--seed', type=int, default=None,
                         help='Fix the random seed for reproducibility.')
    args = parser.parse_args()
    assert args.model in ['SAGE']
    assert args.scene in ['cadets', 'trace', 'theia', 'fivedirections']

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        show(f'Fixed random seed: {args.seed}')

    b_size = 5000
    thre   = thre_map[args.scene]

    src = f'../groundtruth/{args.scene}.txt'
    if not osp.exists(src):
        raise FileNotFoundError(f'Ground-truth file not found: {src}')
    shutil.copy2(src, 'groundtruth_uuid.txt')

    while True:
        graphId, device = train_pro(args, b_size, thre, num_windows=args.num_windows)
        flag = validate(args, b_size, thre, graphId, device)
        if flag == 1:
            break
        show('Validation failed — resetting models and retrying.')
        _delete_all_model_files()


if __name__ == '__main__':
    graphchi_root = osp.abspath(
        osp.join(os.getcwd(), '../graphchi-cpp-master')
    )
    os.environ['GRAPHCHI_ROOT'] = graphchi_root
    main()