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
from torch_geometric.nn import SAGEConv
from data_process_train import MyDataset
from data_process_test import MyDatasetA

thre_map = {"cadets": 1.5, "trace": 1.0, "theia": 1.5, "fivedirections": 1.0}
NUM_WINDOWS = 3     # DEFAULT chronological chunks per scene; overridable via
                    # --num_windows. 5 was too fine-grained for cadets at
                    # the OLD (file-order) chunking -- see MIN_WINDOW_EDGES
                    # below for the safeguard that now applies regardless
                    # of how high --num_windows is set.
MIN_WINDOW_EDGES = 5000    # floor: a window's own edge_index needs enough
                            # edges for SAGEConv to have real neighbourhoods
                            # to aggregate over. If --num_windows would make
                            # windows thinner than this, we silently reduce
                            # the effective window count instead of creating
                            # message-passing-starved windows. Lowered from
                            # 20000 -> 5000 to allow much finer windows (was
                            # capping --num_windows well below what's needed
                            # to approach a teammate's ~87-window setup);
                            # 5000 matches the smaller end of their tested
                            # edges-per-window range as a reference floor,
                            # not a tuned value -- revisit if windows this
                            # thin turn out unstable.
MEM_DIM = 64        # size of the global cross-window memory vector


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAGEMemNet(torch.nn.Module):
    """Two-layer GraphSAGE classifier + a persistent, GRU-updated GLOBAL
    memory vector carried across time windows.

    Why global rather than per-node: this vector is meant to transfer
    between the training scene graph and the separate test scene graph
    (validate()), which don't share a node-id space. A per-node memory
    bank couldn't cross that boundary; a single "what has the whole
    system looked like so far" vector can.

    The memory update (readout + GRU) happens INSIDE forward(), before
    broadcasting back into node features for classification -- so the
    ordinary classification loss is what trains the GRU/readout weights.
    Only the carried-over `state` tensor gets detached by the caller
    between calls (truncated BPTT of depth 1), which is what keeps the
    backward graph bounded no matter how many windows/batches a full
    training run covers.

    forward() accepts a NeighborLoader batch's x/edge_index plus the
    incoming memory state, and returns (log_probs, updated_state). The
    caller slices out[:batch.batch_size] before loss/accuracy/threshold
    logic, same convention as before.
    """
    def __init__(self, in_channels, out_channels, mem_dim=MEM_DIM):
        super().__init__()
        self.mem_dim = mem_dim
        self.conv1 = SAGEConv(in_channels, 32, normalize=False)
        self.readout = torch.nn.Linear(32, mem_dim)
        self.readout_norm = torch.nn.LayerNorm(mem_dim)   # bounds the GRU's input
        self.gru = torch.nn.GRUCell(mem_dim, mem_dim)
        self.ctx_proj = torch.nn.Linear(mem_dim, 32)
        # Learnable scalar gate (init near 0) so the context term starts as
        # a no-op and the model only leans on memory once it's earned a
        # role in the loss, instead of immediately swamping h with an
        # untrained, potentially large ctx vector from step 1.
        self.ctx_gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.conv2 = SAGEConv(32, out_channels, normalize=False)

    def forward(self, x, edge_index, prev_state):
        h = F.relu(self.conv1(x, edge_index))                       # [N, 32]

        g = self.readout_norm(self.readout(h).mean(dim=0, keepdim=True))  # [1, mem_dim], bounded
        new_state = torch.tanh(self.gru(g, prev_state))                  # keep state in [-1, 1]

        ctx = self.ctx_proj(new_state).expand(h.size(0), -1)              # broadcast to every node
        gate = torch.sigmoid(self.ctx_gate)                               # starts ~0.12, learns upward
        h = F.dropout(h + gate * ctx, p=0.5, training=self.training)

        out = self.conv2(h, edge_index)
        return F.log_softmax(out, dim=1), new_state

    def init_state(self, device):
        return torch.zeros(1, self.mem_dim, device=device)


# ---------------------------------------------------------------------------
# Loader factory — always rebuilds so mask mutations are picked up
# ---------------------------------------------------------------------------

def make_loader(data, mask, b_size, shuffle=False):
    """Build a NeighborLoader scoped to the nodes indicated by *mask*.

    shuffle stays False by default: batch order is our only proxy for
    intra-window chronology, and the memory update relies on batches
    being consumed in a stable order.
    """
    return NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        input_nodes=mask,
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
    out, new_state = model(batch.x, batch.edge_index, state)

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

    return pred, y_true, n_ids, new_state


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, device, data, thre, state):
    """Returns (avg_loss, new_state). `state` updates batch-by-batch as the
    loader streams through this window's nodes; the caller carries the
    final detached value into the next epoch / window."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, new_state = model(batch.x, batch.edge_index, state)
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        state = new_state.detach()       # truncate BPTT at the batch boundary
    train_node_count = data.train_mask.sum().item()
    avg_loss = total_loss / train_node_count if train_node_count > 0 else 0.0
    return avg_loss, state


def test(model, loader, device, thre, mask, state):
    """Accuracy over the nodes covered by *loader*, using a FIXED memory
    state for every batch (no intra-eval drift, order-independent result).
    Does not mutate the caller's persistent state."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, _, _ = _predict_batch(model, batch, device, thre, state)
            correct += pred.eq(y_true).sum().item()
    denom = mask.sum().item()
    return correct / denom if denom > 0 else 0.0


def final_test(model, loader, device, thre, mask, fp, tn, state):
    """Like test() but also populates *fp* and *tn* with global node ids."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, n_ids, _ = _predict_batch(model, batch, device, thre, state)
            for i in range(batch.batch_size):
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
    x_list = data.x[nodes]
    y_list = data.y[nodes]
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
# Model-cleanup helpers used by validate() and main()
# ---------------------------------------------------------------------------

def _delete_model_files(graphId, from_loop):
    """Remove model_N, memory_N, fp_feature_label_*, tn_feature_label_* from
    *from_loop* upward."""
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
    """Wipe all model artefacts between outer training attempts."""
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
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour = \
        MyDatasetA(path, 0)

    if len(nodeA) == 0:
        show('WARNING: no ground-truth nodes found — skipping validation')
        return 0

    print(data)
    model = SAGEMemNet(feature_num, label_num).to(device)
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

        # Load the memory snapshot that matches this checkpoint. Falls back
        # to a zero vector if it's missing (e.g. old checkpoints from before
        # this feature existed).
        mem_path = f'../models/memory_{out_loop}.pt'
        if osp.exists(mem_path):
            state = torch.load(mem_path, map_location=device)['state']
        else:
            state = model.init_state(device)

        # Rebuild loader each iteration so mask mutations are honoured.
        loader = make_loader(data, data.test_mask, b_size)

        fp, tn = [], []
        final_test(model, loader, device, thre, data.test_mask, fp, tn, state)

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
            data.test_mask[j] = False

    return 0


# ---------------------------------------------------------------------------
# train_pro()
# ---------------------------------------------------------------------------

def train_pro(args, b_size, thre, num_windows=NUM_WINDOWS):
    """Train sequentially over `num_windows` chronological windows of the
    training snapshot, carrying both model weights AND the global memory
    state forward from window to window. Returns graphId for validate().
    """
    subprocess.run(['python', 'setup.py'], check=True)

    path = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_train.txt'
    )
    graphId = 0
    device = torch.device('cpu')

    windows, feature_num, label_num = MyDataset(path, num_windows, min_edges_per_window=MIN_WINDOW_EDGES)
    show(f'feature {feature_num}; label {label_num}; {len(windows)} windows (requested {num_windows})')

    model = SAGEMemNet(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)

    state = model.init_state(device)   # persists across ALL windows below
    global_loop = 0
    max_thre = 3

    for w_idx, data in enumerate(windows):
        show(f'--- window {w_idx}/{len(windows) - 1} '
             f'({int(data.active_mask.sum().item())} active nodes) ---')

        # Initial warm-up: 30 epochs over this window's active nodes.
        train_loader = make_loader(data, data.train_mask, b_size)
        test_loader  = make_loader(data, data.test_mask,  b_size)

        for epoch in range(1, 30):
            loss, state = train(model, train_loader, optimizer, device, data, thre, state)
            auc = test(model, test_loader, device, thre, data.test_mask, state)
            ts = time.strftime("%H:%M:%S", time.localtime())
            print(f'[{ts}] window {w_idx} epoch {epoch} | Loss: {loss:.4f} | Acc: {auc:.4f}')

        bad_cnt = 0
        while True:
            fp, tn = [], []
            test_loader = make_loader(data, data.test_mask, b_size)
            final_test(model, test_loader, device, thre, data.test_mask, fp, tn, state)

            bad_cnt = bad_cnt + 1 if len(tn) == 0 else 0
            if bad_cnt >= max_thre:
                break

            if len(tn) > 0:
                for i in tn:
                    data.train_mask[i] = False
                    data.test_mask[i]  = False

                _save_feature_log(data, fp, graphId, global_loop, 'fp')
                _save_feature_log(data, tn, graphId, global_loop, 'tn')
                torch.save(model.state_dict(), f'../models/model_{global_loop}')
                torch.save({'state': state.clone()}, f'../models/memory_{global_loop}.pt')
                global_loop += 1

                if len(fp) == 0:
                    break

            train_loader = make_loader(data, data.train_mask, b_size)
            test_loader  = make_loader(data, data.test_mask,  b_size)

            last_loss = None
            for epoch in range(1, 150):
                loss, state = train(model, train_loader, optimizer, device, data, thre, state)
                auc = test(model, test_loader, device, thre, data.test_mask, state)
                ts = time.strftime("%H:%M:%S", time.localtime())
                print(f'[{ts}] window {w_idx} refine epoch {epoch} | Loss: {loss:.4f} | Acc: {auc:.4f}')
                last_loss = loss
                if loss < 1:
                    break

            # If we broke out on epoch 1 because the carried-over weights
            # were already converged, the model hasn't actually adapted to
            # the newly-pruned mask at all -- that's what was causing dozens
            # of near-identical, zero-progress prune/retrain cycles in a
            # row. Force a couple more epochs so each cycle does real work.
            if last_loss is not None and last_loss < 1:
                for extra_epoch in range(2, 6):
                    loss, state = train(model, train_loader, optimizer, device, data, thre, state)
                    auc = test(model, test_loader, device, thre, data.test_mask, state)
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
                              'edges) -- see the printed message at startup.')
    args = parser.parse_args()
    assert args.model in ['SAGE']
    assert args.scene in ['cadets', 'trace', 'theia', 'fivedirections']

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