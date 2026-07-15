"""
Diagnose the step-function (cliff) pattern seen in sweep_threshold.py's
output: recall jumps from near-0 to near-1 across a narrow thre range,
for a small change in FP -- inconsistent with individually-varying node
confidences, consistent with a large cohort of nodes sharing near-
identical pro1/pro2 ratios that all flip at once as thre crosses that
shared value.

Leading hypothesis: nodes with near-all-zero cumulative x (low-degree /
rarely-touched nodes -- x is a histogram of edge-type counts, so a node
barely involved in any edges yet has a near-empty feature vector) get
near-identical RGCNConv output, hence near-identical pro1/pro2, hence a
mass flip at one threshold instead of individually-timed flips.

This script:
  1. Runs one forward pass on a given checkpoint (same as sweep_threshold.py)
  2. Buckets nodes by "feature magnitude" (sum of x before log1p) into
     zero / low / high groups
  3. Reports the pro1/pro2 ratio DISTRIBUTION within each bucket --
     if the zero/low-feature bucket shows a tight cluster of near-
     identical ratios (vs. a spread in the high-feature bucket), that
     confirms the hypothesis.
  4. Prints a coarse histogram of ratio values across ALL test nodes so
     you can see the cluster directly.

Usage:
    python3 diagnose_confidence_distribution.py --scene cadets --model 3
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


def collect(model, loader, device, state, x_full):
    model.eval()
    n_ids, pro1s, pro2s, feat_mags = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index, batch.edge_type, state)
            out = out[:batch.batch_size]
            n_id = batch.n_id[:batch.batch_size]

            pro = F.softmax(out, dim=1)
            pro1 = pro.max(1)
            pro_copy = pro.clone()
            for i in range(batch.batch_size):
                pro_copy[i][pro1[1][i]] = -1
            pro2 = pro_copy.max(1)

            n_ids.append(n_id)
            pro1s.append(pro1[0])
            pro2s.append(pro2[0])
            # raw (pre-log1p) feature magnitude, straight from the full x --
            # this is what tells us if a node is "quiet" (near-zero
            # cumulative edge-type histogram) vs. a busy hub node.
            feat_mags.append(x_full[n_id].sum(dim=1))

    return (torch.cat(n_ids), torch.cat(pro1s), torch.cat(pro2s),
            torch.cat(feat_mags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--model', type=int, required=True)
    args = parser.parse_args()

    path = f'../graphchi-cpp-master/graph_data/darpatc/{args.scene}_test.txt'
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neighbour, num_relations = \
        MyDatasetA(path, 0)

    device = torch.device('cpu')
    model = SAGEMemNet(feature_num, label_num, num_relations).to(device)
    model.load_state_dict(
        torch.load(f'../models/model_{args.model}', map_location=device,
                   weights_only=True)
    )
    mem_path = f'../models/memory_{args.model}.pt'
    state = (torch.load(mem_path, map_location=device)['state']
              if osp.exists(mem_path) else model.init_state(device))

    loader = make_loader(data, data.test_mask, 5000)
    n_ids, pro1, pro2, feat_mag = collect(model, loader, device, state, data.x)

    ratio = pro1 / (pro2 + 1e-30)
    ratio_clamped = torch.clamp(ratio, max=1e6)  # for readable histogram bins

    # --- Bucket by feature magnitude ---
    zero_mask = feat_mag == 0
    low_mask = (feat_mag > 0) & (feat_mag <= 5)
    high_mask = feat_mag > 5

    print(f'Total test nodes evaluated: {len(n_ids)}')
    print(f'  zero-feature nodes (sum(x)==0):  {int(zero_mask.sum())}  '
          f'({100*zero_mask.float().mean():.1f}%)')
    print(f'  low-feature nodes  (0<sum(x)<=5): {int(low_mask.sum())}  '
          f'({100*low_mask.float().mean():.1f}%)')
    print(f'  high-feature nodes (sum(x)>5):    {int(high_mask.sum())}  '
          f'({100*high_mask.float().mean():.1f}%)')

    def describe_ratio(mask, label):
        if mask.sum() == 0:
            print(f'\n{label}: no nodes in this bucket')
            return
        r = ratio_clamped[mask]
        # how concentrated is this bucket's ratio distribution? if a huge
        # fraction sits within a tiny range, that's the "cohort" signature.
        median = r.median().item()
        p10 = r.kthvalue(max(1, int(0.10 * len(r)))).values.item()
        p90 = r.kthvalue(max(1, int(0.90 * len(r)))).values.item()
        # fraction within +-20% of the median -- a tight cluster around
        # its own median indicates near-identical outputs within the bucket
        near_median = ((r > 0.8 * median) & (r < 1.2 * median)).float().mean().item()
        print(f'\n{label} (n={int(mask.sum())}):')
        print(f'  median ratio: {median:.4f}   10th pct: {p10:.4f}   90th pct: {p90:.4f}')
        print(f'  fraction within +-20% of median: {100*near_median:.1f}%  '
              f'(high % here = tight cluster = cohort behavior)')

    describe_ratio(zero_mask, 'ZERO-feature bucket')
    describe_ratio(low_mask, 'LOW-feature bucket')
    describe_ratio(high_mask, 'HIGH-feature bucket')

    # --- Coarse histogram across ALL nodes ---
    print('\nOverall ratio histogram (log-spaced bins):')
    bins = [0, 1, 1.5, 2, 3, 5, 8, 12, 20, 50, 100, 500, 1000, 5000, 1e6]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = int(((ratio_clamped >= lo) & (ratio_clamped < hi)).sum())
        pct = 100 * count / len(ratio_clamped)
        bar = '#' * int(pct / 2)
        print(f'  [{lo:>8g}, {hi:>8g}): {count:>7} ({pct:5.1f}%) {bar}')

    # --- Ground-truth-specific check ---
    gt_mask = torch.zeros(len(n_ids), dtype=torch.bool)
    nodeA_set = set(nodeA)
    for i, nid in enumerate(n_ids.tolist()):
        if nid in nodeA_set:
            gt_mask[i] = True
    if gt_mask.sum() > 0:
        print(f'\nGround-truth nodes found in this evaluation: {int(gt_mask.sum())}')
        describe_ratio(gt_mask, 'GROUND-TRUTH nodes specifically')
        describe_ratio(gt_mask & zero_mask, 'GROUND-TRUTH nodes with zero features')


if __name__ == '__main__':
    main()