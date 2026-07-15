"""
Sweep the confidence-ratio threshold (thre) against ALREADY-TRAINED
checkpoints in ../models/, without retraining.

Why this exists: validate()'s precision/recall depends heavily on thre
(the pro1/pro2 confidence-ratio cutoff in _predict_batch), which was
tuned for the old, poorly-calibrated SAGEConv-era confidence distribution.
Now that the model is properly calibrated (log1p + grad clipping fixed
the numerical instability), thre=1.5 is apparently far too lenient --
almost nothing gets rejected, so almost nothing gets flagged, so recall
collapses. Retraining from scratch to test a different thre is pure
waste: the MODEL doesn't need to change, only the post-hoc threshold
applied to its (already fine) softmax outputs.

This script runs ONE forward pass per checkpoint (the expensive part --
same cost as one validate() iteration), caches each node's raw argmax
prediction + top-2 softmax probabilities, then re-scores against many
thre values instantly using the exact same precision/recall logic as
validate() in train_darpatc.py.

Usage:
    python3 sweep_threshold.py --scene cadets
    python3 sweep_threshold.py --scene cadets --model 0       # one checkpoint
    python3 sweep_threshold.py --scene cadets --thresholds 1,2,5,10,20,50
"""
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from data_process_test import MyDatasetA
from model import SAGEMemNet


def make_loader(data, mask, b_size):
    return NeighborLoader(
        data, num_neighbors=[-1, -1], input_nodes=mask,
        batch_size=b_size, shuffle=False,
    )


def collect_raw_predictions(model, loader, device, state):
    """One forward pass. Returns per-node global id, true label, raw
    argmax prediction, and the top-2 softmax probabilities -- everything
    needed to re-derive the thresholded prediction for ANY thre value
    without touching the model again."""
    model.eval()
    n_ids, y_trues, preds, pro1s, pro2s = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index, batch.edge_type, state)
            out = out[:batch.batch_size]
            y_true = batch.y[:batch.batch_size]
            n_id = batch.n_id[:batch.batch_size]

            pred = out.max(1)[1].clone()
            pro = F.softmax(out, dim=1)
            pro1 = pro.max(1)
            for i in range(batch.batch_size):
                pro[i][pro1[1][i]] = -1
            pro2 = pro.max(1)

            n_ids.append(n_id)
            y_trues.append(y_true)
            preds.append(pred)
            pro1s.append(pro1[0])
            pro2s.append(pro2[0])

    return (torch.cat(n_ids), torch.cat(y_trues), torch.cat(preds),
            torch.cat(pro1s), torch.cat(pro2s))


def score_at_threshold(n_ids, y_true, pred, pro1, pro2, thre,
                        nodeA, _nodeA, _neighbour):
    """Exact same scoring as validate() in train_darpatc.py: a node's
    misclassification (post-threshold) only counts as a genuine false
    positive if it's outside every ground-truth node's 2-hop neighbourhood;
    a ground-truth node counts as caught (_tp) if ANY misclassified node
    lands in its 2-hop neighbourhood."""
    eps = 1e-10
    reject = (pro2 <= 0) | (pro1 / (pro2 + 1e-30) < thre)
    thresholded_pred = pred.clone()
    thresholded_pred[reject] = 100

    fp_nodes = n_ids[thresholded_pred != y_true].tolist()

    _fp = 0
    tempNodeA = {i: 1 for i in nodeA}
    for i in fp_nodes:
        if i not in _nodeA:
            _fp += 1
        for j in _neighbour.get(i, []):
            if j in tempNodeA:
                tempNodeA[j] = 0

    _tp = sum(1 for v in tempNodeA.values() if v == 0)
    precision = _tp / (_tp + _fp + eps)
    recall = _tp / len(nodeA)
    fscore = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, fscore, _fp, len(nodeA) - _tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--model', type=int, default=None,
                         help='Sweep only this checkpoint index. '
                              'Default: sweep every saved checkpoint.')
    parser.add_argument(
        '--thresholds', type=str,
        default='1.0,1.5,2,3,5,8,12,20,30,50,80,120,200,500,1000,5000',
        help='Comma-separated thre values to try. Wide range on purpose: '
             'a well-calibrated model can need a MUCH higher ratio than '
             'the old thre=1.5 to force rejection on borderline nodes.'
    )
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(',')]

    path = f'../graphchi-cpp-master/graph_data/darpatc/{args.scene}_test.txt'
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour, num_relations = \
        MyDatasetA(path, 0)

    if len(nodeA) == 0:
        print('No ground-truth nodes found -- aborting.')
        return

    device = torch.device('cpu')
    model = SAGEMemNet(feature_num, label_num, num_relations).to(device)

    if args.model is not None:
        checkpoints = [args.model]
    else:
        checkpoints = []
        i = 0
        while osp.exists(f'../models/model_{i}'):
            checkpoints.append(i)
            i += 1

    if not checkpoints:
        print('No checkpoints found in ../models/. Run train_darpatc.py first.')
        return

    print(f'Sweeping {len(thresholds)} threshold values against '
          f'{len(checkpoints)} checkpoint(s). {len(nodeA)} ground-truth '
          f'nodes in this test set.\n')

    for ckpt in checkpoints:
        model.load_state_dict(
            torch.load(f'../models/model_{ckpt}', map_location=device,
                       weights_only=True)
        )
        mem_path = f'../models/memory_{ckpt}.pt'
        state = (torch.load(mem_path, map_location=device)['state']
                  if osp.exists(mem_path) else model.init_state(device))

        loader = make_loader(data, data.test_mask, 5000)
        n_ids, y_true, pred, pro1, pro2 = collect_raw_predictions(
            model, loader, device, state
        )

        raw_acc = (pred == y_true).float().mean().item()
        print(f'=== Model {ckpt} (raw node-type accuracy, no threshold: '
              f'{raw_acc:.4f}) ===')
        print(f'{"thre":>8} | {"Precision":>9} | {"Recall":>7} | '
              f'{"F-Score":>7} | {"FP":>7} | {"FN":>7}')
        for thre in thresholds:
            precision, recall, fscore, fp_count, fn_count = score_at_threshold(
                n_ids, y_true, pred, pro1, pro2, thre, nodeA, _nodeA, _neighbour
            )
            print(f'{thre:>8.2f} | {precision:>9.4f} | {recall:>7.4f} | '
                  f'{fscore:>7.4f} | {fp_count:>7} | {fn_count:>7}')
        print()


if __name__ == '__main__':
    main()