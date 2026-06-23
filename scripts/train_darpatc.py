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


def show(*s):
    for i in range(len(s)):
        print(str(s[i]) + ' ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAGENet(torch.nn.Module):
    """Two-layer GraphSAGE classifier.

    forward() accepts a NeighborLoader batch object.  The caller is
    responsible for slicing out[:batch.batch_size] before computing loss,
    accuracy, or threshold logic — this keeps the module itself unaware of
    the seed-node convention so it can also be used for full-graph inference.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 32, normalize=False)
        self.conv2 = SAGEConv(32, out_channels, normalize=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Loader factory — always rebuilds so mask mutations are picked up
# ---------------------------------------------------------------------------

def make_loader(data, mask, b_size, shuffle=False):
    """Build a NeighborLoader scoped to the nodes indicated by *mask*.

    Called after every mask mutation so the sampler sees the current set of
    active nodes rather than the stale set from construction time.
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

def _predict_batch(model, batch, device, thre):
    """Run forward pass and apply confidence-ratio threshold for one batch.

    Returns
    -------
    pred   : [batch_size] — predicted class (100 = below-threshold / uncertain)
    y_true : [batch_size] — ground-truth labels for seed nodes
    n_ids  : [batch_size] — global node ids for seed nodes
    """
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index)

    # Slice to seed nodes only — the remaining rows are sampled neighbours
    # included only for message-passing context.
    out    = out[:batch.batch_size]
    y_true = batch.y[:batch.batch_size]
    n_ids  = batch.n_id[:batch.batch_size]

    pred = out.max(1)[1].clone()
    pro  = F.softmax(out, dim=1)
    pro1 = pro.max(1)

    # Zero out the top class so we can find the second-best confidence.
    for i in range(batch.batch_size):
        pro[i][pro1[1][i]] = -1
    pro2 = pro.max(1)

    for i in range(batch.batch_size):
        if pro2[0][i] <= 0 or pro1[0][i] / pro2[0][i] < thre:
            pred[i] = 100

    return pred, y_true, n_ids


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, device, data, thre):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[:batch.batch_size],
                          batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
    train_node_count = data.train_mask.sum().item()
    return total_loss / train_node_count if train_node_count > 0 else 0.0


def test(model, loader, device, thre, mask):
    """Return accuracy over the nodes covered by *loader*.

    *mask* is used only to compute the denominator — the loader itself
    already restricts which nodes are evaluated.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, _ = _predict_batch(model, batch, device, thre)
            correct += pred.eq(y_true).sum().item()
    denom = mask.sum().item()
    return correct / denom if denom > 0 else 0.0


def final_test(model, loader, device, thre, mask, fp, tn):
    """Like test() but also populates *fp* and *tn* with global node ids.

    *fp* and *tn* are mutated in-place so the caller can accumulate across
    multiple calls if needed.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pred, y_true, n_ids = _predict_batch(model, batch, device, thre)
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
# Model-cleanup helper used by validate() and main()
# ---------------------------------------------------------------------------

def _delete_model_files(graphId, from_loop):
    """Remove model_N, fp_feature_label_*, tn_feature_label_* from *from_loop* upward."""
    loop = from_loop
    while True:
        mp = f'../models/model_{loop}'
        if not osp.exists(mp):
            break
        os.remove(mp)
        for tag in ('fp', 'tn'):
            p = f'../models/{tag}_feature_label_{graphId}_{loop}.txt'
            if osp.exists(p):
                os.remove(p)
        loop += 1


def _delete_all_model_files():
    """Wipe all model artefacts between outer training attempts."""
    for pattern in (
        '../models/model_*',
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
    """Evaluate saved models against the test snapshot.

    Returns 1 if precision/recall thresholds are met, else 0.
    graphId is passed explicitly (not a global) so validate() is self-contained.
    """
    show('Start validating')
    path = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_test.txt'
    )
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour = MyDatasetA(path, 0)

    if len(nodeA) == 0:
        show('WARNING: no ground-truth nodes found — skipping validation')
        return 0

    print(data)
    model = SAGENet(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    eps = 1e-10

    out_loop = -1
    while True:
        out_loop += 1
        model_path = f'../models/model_{out_loop}'
        if not osp.exists(model_path):
            break

        print(f'validating in model {out_loop}')
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        # Rebuild loader each iteration so mask mutations are honoured.
        loader = make_loader(data, data.test_mask, b_size)

        fp, tn = [], []
        final_test(model, loader, device, thre, data.test_mask, fp, tn)
        print(f'fp and fn: {len(fp)} {len(tn)}')

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
        print(f'Precision: {precision:.4f}')
        print(f'Recall:    {recall:.4f}')

        if recall > 0.8 and precision > 0.7:
            # Passed — delete any surplus models that were never needed.
            _delete_model_files(graphId, out_loop + 1)
            return 1

        if recall <= 0.8:
            return 0

        # Remove correctly-classified nodes from the test mask and continue
        # to the next saved model.
        for j in tn:
            data.test_mask[j] = False

    return 0


# ---------------------------------------------------------------------------
# train_pro()
# ---------------------------------------------------------------------------

def train_pro(args, b_size, thre):
    """Train on the training snapshot.  Returns graphId for use by validate()."""
    subprocess.run(['python', 'setup.py'], check=True)

    path = (
        '../graphchi-cpp-master/graph_data/darpatc/'
        + args.scene + '_train.txt'
    )
    graphId = 0
    show(f'Start training graph {graphId}')

    data, feature_num, label_num = MyDataset(path, 0)
    print(data)
    print(f'feature {feature_num}; label {label_num}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SAGENet(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)

    # Initial warm-up: 30 epochs over the full training set.
    train_loader = make_loader(data, data.train_mask, b_size)
    test_loader  = make_loader(data, data.test_mask,  b_size)

    for epoch in range(1, 30):
        loss = train(model, train_loader, optimizer, device, data, thre)
        auc  = test(model, test_loader, device, thre, data.test_mask)
        show(epoch, loss, auc)

    loop_num  = 0
    max_thre  = 3
    bad_cnt   = 0

    while True:
        fp, tn = [], []

        # Rebuild loaders so the current mask state is reflected.
        test_loader = make_loader(data, data.test_mask, b_size)

        final_test(model, test_loader, device, thre, data.test_mask, fp, tn)

        if len(tn) == 0:
            bad_cnt += 1
        else:
            bad_cnt = 0

        if bad_cnt >= max_thre:
            break

        if len(tn) > 0:
            # Remove correctly-classified nodes from both masks.
            for i in tn:
                data.train_mask[i] = False
                data.test_mask[i]  = False

            _save_feature_log(data, fp, graphId, loop_num, 'fp')
            _save_feature_log(data, tn, graphId, loop_num, 'tn')
            torch.save(model.state_dict(), f'../models/model_{loop_num}')
            loop_num += 1

            if len(fp) == 0:
                break

        # Re-train on the pruned mask.
        auc = 0.0
        train_loader = make_loader(data, data.train_mask, b_size)
        test_loader  = make_loader(data, data.test_mask,  b_size)

        for epoch in range(1, 150):
            loss = train(model, train_loader, optimizer, device, data, thre)
            auc  = test(model, test_loader, device, thre, data.test_mask)
            show(epoch, loss, auc)
            if loss < 1:
                break

    show(f'Finish training graph {graphId}')
    return graphId, device


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SAGE')
    parser.add_argument('--scene', type=str, default='theia')
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
        graphId, device = train_pro(args, b_size, thre)
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