"""Extension 3: behavior training enriched with cheap window statistics.

E3 = the D1-revised task (train_behavior.py) with three extra scored
observables per node per window -- binned n_distinct, rep_ratio and
span_frac (window_features.py; measured motivation in
reports/2026-07-17_feature_enrichment_survey.md). E1/E2 files are
imported, never modified; their checkpoints remain the frozen comparison
arms (shape-incompatible with this model by design).

How the features enter (and how they may NOT)
-----------------------------------------------
The score must remain surprise-given-history, so current-window
statistics of a node's own events are TARGETS, never forward() inputs --
exactly like the edge-type profile:

  target side : each feature is a categorical over its preregistered
                bins, predicted by its own softmax head off the shared
                hidden state. Per-node score/loss =
                    profile_nll + mean_k(feature-bin NLL_k)
                The equal 1/NUM_FEATURES weighting is LOCKED in
                feature_nll() below -- there is deliberately no weight
                argument, no CLI flag, and nothing for a driver to
                sweep. Changing it is a new experiment.
  memory side : the same bins are appended one-hot to the parameter-free
                writeback [h.detach() ; normalized profile ; bin
                one-hots], so next-window predictions are conditioned on
                them.

Everything else -- three read channels (neighbor memories via SAGE or
the E2 relational conv, own memory, static type one-hot), zero-memory
collapse to a per-type prior, chronological window loop, per-epoch
memory reset, checkpoint = best val NLL on the chronological train tail
(the SAME rule as E1/E2, applied to the combined NLL) -- mirrors
train_behavior.py.
"""

import argparse
import os
import os.path as osp
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from node_vocab import load_node_vocab
from node_memory import NodeMemory
from windowing import build_type_vocab
from window_features import FEATURE_BIN_SIZES, NUM_FEATURES, \
    build_enriched_dataset
from train_behavior import RelationalMemoryConv, profile_nll, type_onehot


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


class EnrichedBehaviorNet(torch.nn.Module):
    """BehaviorNet + one softmax head per binned window feature.

    memory_dim = hidden + profile_dim + sum(FEATURE_BIN_SIZES): stored
    memory is [past hidden ; last normalized profile ; last feature-bin
    one-hots]. The read channels are identical to BehaviorNet; only the
    output side gains the per-feature heads, all fed from the same
    dropped hidden state as lin_out. forward() takes neither data.x nor
    data.feat_bins -- both are targets.
    """

    def __init__(self, num_types, profile_dim, hidden_channels=32,
                 num_relations=None, num_bases=8):
        super().__init__()
        memory_dim = hidden_channels + profile_dim + sum(FEATURE_BIN_SIZES)
        self.memory_dim = memory_dim
        self.relational = num_relations is not None
        if self.relational:
            self.conv_mem = RelationalMemoryConv(memory_dim, hidden_channels,
                                                 num_relations, num_bases)
        else:
            self.conv_mem = SAGEConv(memory_dim, hidden_channels,
                                     normalize=False, root_weight=False)
        self.lin_self = torch.nn.Linear(memory_dim, hidden_channels,
                                        bias=False)
        self.lin_type = torch.nn.Linear(num_types, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, profile_dim)
        self.feat_heads = torch.nn.ModuleList(
            torch.nn.Linear(hidden_channels, k) for k in FEATURE_BIN_SIZES)

    def forward(self, type_onehot, memory, edge_index, edge_type=None):
        if self.relational:
            agg = self.conv_mem(memory, edge_index, edge_type)
        else:
            agg = self.conv_mem(memory, edge_index)
        h = F.relu(agg
                   + self.lin_self(memory)
                   + self.lin_type(type_onehot))
        h_dropped = F.dropout(h, p=0.5, training=self.training)
        profile_log_probs = F.log_softmax(self.lin_out(h_dropped), dim=1)
        feat_log_probs = [F.log_softmax(head(h_dropped), dim=1)
                          for head in self.feat_heads]
        return profile_log_probs, feat_log_probs, h


def feature_nll(feat_log_probs, feat_bins):
    """Per-node mean NLL over the feature heads [N].

    Each head observes exactly one bin per node per window, so its NLL
    is a single gather. The combination is the plain unweighted mean --
    LOCKED, no weight parameter exists on purpose (see module docstring).
    """
    nlls = [-lp.gather(1, feat_bins[:, k:k + 1]).squeeze(1)
            for k, lp in enumerate(feat_log_probs)]
    return torch.stack(nlls, dim=0).mean(dim=0)


def combined_nll(profile_log_probs, x, feat_log_probs, feat_bins):
    """The E3 per-node score/loss [N]: profile surprise + feature surprise."""
    return profile_nll(profile_log_probs, x) \
        + feature_nll(feat_log_probs, feat_bins)


def make_writeback_e3(h, x, feat_bins):
    """Parameter-free write vector [N, H + profile_dim + sum(bins)].

    [h.detach() ; per-event-normalized profile ; feature-bin one-hots].
    Same rationale as train_behavior.make_writeback: stored memory is
    detached, so nothing learned may sit on the write path.
    """
    prof = x / x.sum(dim=1, keepdim=True).clamp(min=1.0)
    onehots = [F.one_hot(feat_bins[:, k], num_classes=n).to(x.dtype)
               for k, n in enumerate(FEATURE_BIN_SIZES)]
    return torch.cat([h.detach(), prof] + onehots, dim=1)


def train_one_epoch(model, dataset, node_memory, optimizer, device,
                    num_types):
    model.train()
    node_memory.reset()
    total_loss = 0.0
    total_nodes = 0

    for window_idx, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        data = data.to(device)
        memory = node_memory.get_decayed(data.global_id, window_idx)

        optimizer.zero_grad()
        out, feat_out, hidden = model(type_onehot(data.y, num_types), memory,
                                      data.edge_index, data.edge_type)
        loss = combined_nll(out, data.x, feat_out, data.feat_bins).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
        node_memory.update(data.global_id,
                           make_writeback_e3(hidden, data.x, data.feat_bins),
                           window_idx)

    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate_loss(model, dataset, node_memory, device, num_types,
                  timestep_offset=0, reset=True):
    """Mean combined NLL -- the checkpoint-selection criterion (same
    best-val-NLL rule as E1/E2, on the same chronological val tail; also
    logs the profile/feature split for interpretation only -- the split
    plays no role in selection)."""
    model.eval()
    if reset:
        node_memory.reset()
    total = 0.0
    total_prof = 0.0
    total_nodes = 0

    for i, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        window_idx = timestep_offset + i
        data = data.to(device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, feat_out, hidden = model(type_onehot(data.y, num_types), memory,
                                      data.edge_index, data.edge_type)
        prof = profile_nll(out, data.x)
        comb = prof + feature_nll(feat_out, data.feat_bins)
        total += comb.mean().item() * data.num_nodes
        total_prof += prof.mean().item() * data.num_nodes
        total_nodes += data.num_nodes
        node_memory.update(data.global_id,
                           make_writeback_e3(hidden, data.x, data.feat_bins),
                           window_idx)

    if total_nodes == 0:
        return 0.0, 0.0
    return total / total_nodes, total_prof / total_nodes


def main():
    parser = argparse.ArgumentParser(
        description='Extension 3: history-conditioned behavior training '
                    'with binned window-statistic targets '
                    '(n_distinct / rep_ratio / span_frac).'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--window-size', type=int, default=5000)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--decay-rate', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Defaults to ../models/<scene>_node_vocab.txt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to ../models/behavior_<scene>_e3/')
    parser.add_argument('--max-norm', type=float, default=100.0,
                        help='NodeMemory clamp over the whole write vector; '
                             'must match test_behavior_e3.py. 0 = memory-off '
                             'ablation (collapses to the static type prior '
                             'on every head).')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Chronological TAIL of the training windows '
                             'held out for checkpoint selection; the test '
                             'file is never read by this script.')
    parser.add_argument('--relational', action='store_true',
                        help='E2-style relation-aware neighbor-memory '
                             'routing under the E3 task. Must match '
                             'test_behavior_e3.py.')
    parser.add_argument('--rel-bases', type=int, default=8)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    out_dir = args.out_dir or f'../models/behavior_{args.scene}_e3'

    for p in (train_path, node_vocab_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')
    os.makedirs(out_dir, exist_ok=True)

    show(f'Loading node vocab: {node_vocab_path}')
    node_vocab = load_node_vocab(node_vocab_path)
    show(f'{len(node_vocab):,} global node identities')

    feature_map, label_map = build_type_vocab(train_path)
    profile_dim = len(feature_map) * 2
    num_types = len(label_map)
    show(f'{num_types} node types | profile dim {profile_dim} | '
         f'feature bins {FEATURE_BIN_SIZES} ({NUM_FEATURES} heads, '
         f'equal-weight NLL, locked)')

    show(f'Building enriched windowed train dataset (window={args.window_size})')
    dataset = build_enriched_dataset(
        train_path, node_vocab, feature_map, label_map, args.window_size)
    n_val = int(len(dataset) * args.val_fraction)
    fit_windows = dataset[:len(dataset) - n_val] if n_val else dataset
    val_windows = dataset[len(dataset) - n_val:] if n_val else []
    show(f'{len(dataset)} windows: {len(fit_windows)} fit / '
         f'{len(val_windows)} validation tail')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Device: {device}')

    num_relations = len(feature_map) + 1 if args.relational else None
    if args.relational:
        show(f'Relational routing ON: {num_relations} relations, '
             f'{args.rel_bases} bases')
    model = EnrichedBehaviorNet(num_types, profile_dim, args.hidden_dim,
                                num_relations=num_relations,
                                num_bases=args.rel_bases).to(device)
    node_memory = NodeMemory(len(node_vocab), model.memory_dim,
                             args.decay_rate,
                             device=device, max_norm=args.max_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_criterion = float('inf')
    best_path = osp.join(out_dir, 'best_model.pt')
    latest_path = osp.join(out_dir, 'latest_model.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, fit_windows, node_memory,
                                     optimizer, device, num_types)
        if val_windows:
            criterion, val_prof = evaluate_loss(
                model, val_windows, node_memory, device, num_types,
                timestep_offset=len(fit_windows), reset=False)
            label = 'Val loss'
            split = (f' (profile {val_prof:.4f} / features '
                     f'{criterion - val_prof:.4f})')
        else:
            criterion = train_loss
            label = 'Train loss (no val tail)'
            split = ''
        show(f'Epoch {epoch} | Train loss: {train_loss:.4f} | '
             f'{label}: {criterion:.4f}{split} | {time.time() - t0:.1f}s')

        torch.save(model.state_dict(), latest_path)
        if criterion < best_criterion:
            best_criterion = criterion
            torch.save(model.state_dict(), best_path)

    show(f'Done. Best {label.lower()}: {best_criterion:.4f}')
    show(f'Checkpoints: {best_path} (best) / {latest_path} (latest)')


if __name__ == '__main__':
    main()
