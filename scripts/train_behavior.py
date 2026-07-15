"""Stage 4-B (D1-revised): history-conditioned behavior training.

New entry point implementing the revised D1 score design
(reports/2026-07-14_ext1_sweep_report_and_d1_design.md §5). The existing
pipeline (train_windowed.py / test_windowed.py) stays untouched as the
Extension-1-v1 reference; this file exists alongside it, sharing Stages
1-3 (node_vocab, windowing, windowed_data, node_memory) as imports.

Task inversion
---------------
The v1 task predicts a node's CDM *type* (data.y) from its per-window
edge-type histogram (data.x) -- a mapping so nearly deterministic that
81%+ of test scores were exactly 0.0 and NodeMemory had nothing left to
explain. This file inverts it: predict the node's *behavior this window*
(the 56-dim edge-type participation profile -- literally data.x) from
what history says about the node, and score by how surprised the model
was.

Memory as the only node-specific channel
------------------------------------------
forward() receives exactly three things about a node:
  (a) its own decayed memory,
  (b) a SAGE aggregation of its current neighbors' decayed memories
      (the channel that matters for CADETS attack entities: they are
      one-shot (0.18% recurrence), but their neighbors -- nginx,
      sendmail, sshd -- are persistent hubs whose memories encode what
      their typical partners do), and
  (c) its static CDM type one-hot, kept deliberately so the memory-off
      ablation degrades to a *per-type behavior prior* (a meaningful
      classical baseline) rather than a single global distribution.
data.x NEVER enters forward() -- it is the target and, on the write
side, part of the next memory. With --max-norm 0, (a) and (b) are
identically zero and the model provably collapses to the type prior:
the ablation delta is exactly "what history buys".

Why the memory write path is parameter-free
---------------------------------------------
Stored memory is detached (see node_memory.py) -- gradients never flow
across windows -- so any *learned* transform used only on the write path
would receive no gradient at all. The write vector is therefore a
parameter-free concatenation:
    [ h.detach() (H dims) ; this window's normalized profile (56 dims) ]
h is on the gradient path of its own window's loss (it produces the
prediction), so the layers that shape it do learn; the profile half
carries the raw observed behavior forward so that memory contains what a
node actually *did*, not merely a function of older memories and types.
The read side (lin_self / conv_mem below) is fully learned -- learning
happens where gradients exist.

Everything else -- chronological full-batch window loop, per-epoch
memory reset, chronological validation tail for checkpoint selection
(--val-fraction), never reading the test file -- mirrors
train_windowed.py exactly.
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
from windowing import build_type_vocab
from windowed_data import build_windowed_dataset
from node_memory import NodeMemory


def show(*s):
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = ' '.join(str(x) for x in s)
    print(f'[{ts}] {msg}')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BehaviorNet(torch.nn.Module):
    """Predict a node's current-window edge-type profile from history.

    memory_dim = hidden_channels + profile_dim: each stored memory row is
    [past hidden state ; last observed normalized profile] (see module
    docstring). Three read channels sum into the hidden state:

      conv_mem : SAGEConv over the *memories* along this window's edges
                 (root_weight=False -- the root's contribution comes from
                 lin_self, keeping the own-history and neighbor-history
                 channels separable for analysis; note build_window_data
                 adds self-loops, so a node's own memory also appears in
                 its neighbor aggregation -- harmless, own memory is a
                 legitimate input channel, unlike the v1 task where the
                 self-loop round-tripped the answer).
      lin_self : the node's own decayed memory.
      lin_type : static CDM type one-hot; its bias doubles as the global
                 behavior prior every node shares when memory is zeroed.

    forward() deliberately has no `x` argument: the observed profile is
    the target, not an input.
    """

    def __init__(self, num_types, profile_dim, hidden_channels=32):
        super().__init__()
        memory_dim = hidden_channels + profile_dim
        self.memory_dim = memory_dim
        self.conv_mem = SAGEConv(memory_dim, hidden_channels,
                                 normalize=False, root_weight=False)
        self.lin_self = torch.nn.Linear(memory_dim, hidden_channels, bias=False)
        self.lin_type = torch.nn.Linear(num_types, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, profile_dim)

    def forward(self, type_onehot, memory, edge_index):
        h = F.relu(self.conv_mem(memory, edge_index)
                   + self.lin_self(memory)
                   + self.lin_type(type_onehot))
        h_dropped = F.dropout(h, p=0.5, training=self.training)
        out = self.lin_out(h_dropped)
        return F.log_softmax(out, dim=1), h


def profile_nll(log_probs, x):
    """Per-node mean-per-event NLL of the observed profile [N].

    -(x . log p) / sum(x): the average surprise per observed event, so a
    node's score does not grow with its degree. Continuous by
    construction -- exact zero requires the model to put probability 1 on
    every event the node emitted, so there is no saturation floor like
    the v1 type-NLL's 81%-at-0.0. Every node in a window participates in
    at least one real edge (nodes exist only via edges; self-loops are
    added after x is built), but the count is still clamped defensively.
    """
    events = x.sum(dim=1).clamp(min=1.0)
    return -(x * log_probs).sum(dim=1) / events


def make_writeback(h, x):
    """Parameter-free memory write vector [N, H + profile_dim].

    [h.detach() ; per-event-normalized observed profile]. See the module
    docstring for why nothing learned is allowed on the write path.
    """
    prof = x / x.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.cat([h.detach(), prof], dim=1)


def type_onehot(y, num_types):
    return F.one_hot(y, num_classes=num_types).to(torch.float32)


# ---------------------------------------------------------------------------
# Train / eval loops -- same window-sequence threading as train_windowed.py
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataset, node_memory, optimizer, device, num_types):
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
        out, hidden = model(type_onehot(data.y, num_types), memory,
                            data.edge_index)
        loss = profile_nll(out, data.x).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
        node_memory.update(data.global_id, make_writeback(hidden, data.x),
                           window_idx)

    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate_loss(model, dataset, node_memory, device, num_types,
                  timestep_offset=0, reset=True):
    """Mean profile NLL -- the checkpoint-selection criterion.

    Same continuation semantics as train_windowed.evaluate_loss: the
    validation tail is evaluated with reset=False and a timestep offset
    so its window clock and memory continue from where the epoch's
    training pass ended (same file, genuinely contiguous time).
    """
    model.eval()
    if reset:
        node_memory.reset()
    total_loss = 0.0
    total_nodes = 0

    for i, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        window_idx = timestep_offset + i
        data = data.to(device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, hidden = model(type_onehot(data.y, num_types), memory,
                            data.edge_index)
        total_loss += profile_nll(out, data.x).mean().item() * data.num_nodes
        total_nodes += data.num_nodes
        node_memory.update(data.global_id, make_writeback(hidden, data.x),
                           window_idx)

    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate_loss_zero_memory(model, dataset, device, num_types, memory_dim):
    """Mean profile NLL with memory zeroed -- the type-prior forward.

    With a zero memory matrix, conv_mem and lin_self contribute nothing
    and the prediction collapses to the static type prior (same semantics
    as --max-norm 0). Reads and writes no NodeMemory state, so it cannot
    perturb the epoch's live memory. Paired with the mem-on val loss to
    log the per-epoch memory-utility gap
        delta = NLL(zero memory) - NLL(memory on),
    the preregistered checkpoint-selection proxy for the 2026-07-15
    epoch sweep (only computed under --save-every-epoch).
    """
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    for data in dataset:
        if data.num_nodes == 0:
            continue
        data = data.to(device)
        zero_mem = torch.zeros(data.num_nodes, memory_dim, device=device)
        out, _ = model(type_onehot(data.y, num_types), zero_mem,
                       data.edge_index)
        total_loss += profile_nll(out, data.x).mean().item() * data.num_nodes
        total_nodes += data.num_nodes
    return total_loss / total_nodes if total_nodes > 0 else 0.0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 4-B (D1-revised): history-conditioned behavior '
                    'training with cross-window NodeMemory.'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--window-size', type=int, default=5000,
                        help='Default 5000 per the 2026-07-14 D4 sweep '
                             'decision (best raw-GT ranking, lowest '
                             'score saturation, highest recurrence).')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Untuned starting point -- sweep AFTER the '
                             'memory ablation shows a nonzero delta.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=None,
                        help='torch/python RNG seed. Unset = OS entropy '
                             '(the 2026-07-14 seed check is why this '
                             'flag exists in the new files).')
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Defaults to ../models/<scene>_node_vocab.txt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to ../models/behavior_<scene>/')
    parser.add_argument('--max-norm', type=float, default=100.0,
                        help='NodeMemory max_norm clamp over the whole '
                             '[hidden ; profile] write vector. Must match '
                             'test_behavior.py. 0 zeroes all stored '
                             'memory: the model provably collapses to '
                             'the static type prior (the honest '
                             'memory-off ablation arm).')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Chronological TAIL of the training windows '
                             'held out for checkpoint selection; the '
                             'test file is never read by this script.')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='Additionally save model_epoch<NNN>.pt each '
                             'epoch and log the per-epoch memory-utility '
                             'gap delta = val NLL(zero memory) - val '
                             'NLL(memory on) to the console and to '
                             '<out-dir>/epoch_metrics.tsv. Off by '
                             'default; the training path is unchanged '
                             'when unset.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    out_dir = args.out_dir or f'../models/behavior_{args.scene}'

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
    show(f'{num_types} node types | profile dim {profile_dim}')

    show(f'Building windowed train dataset (window={args.window_size})')
    dataset = build_windowed_dataset(
        train_path, node_vocab, feature_map, label_map, args.window_size)
    n_val = int(len(dataset) * args.val_fraction)
    fit_windows = dataset[:len(dataset) - n_val] if n_val else dataset
    val_windows = dataset[len(dataset) - n_val:] if n_val else []
    show(f'{len(dataset)} windows: {len(fit_windows)} fit / '
         f'{len(val_windows)} validation tail')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Device: {device}')

    model = BehaviorNet(num_types, profile_dim, args.hidden_dim).to(device)
    node_memory = NodeMemory(len(node_vocab),
                             args.hidden_dim + profile_dim,
                             args.decay_rate,
                             device=device, max_norm=args.max_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_criterion = float('inf')
    best_path = osp.join(out_dir, 'best_model.pt')
    latest_path = osp.join(out_dir, 'latest_model.pt')

    if args.save_every_epoch:
        metrics_path = osp.join(out_dir, 'epoch_metrics.tsv')
        with open(metrics_path, 'w') as f:
            f.write('epoch\ttrain_loss\tval_nll_mem\tval_nll_zeromem\t'
                    'delta_mem\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, fit_windows, node_memory,
                                     optimizer, device, num_types)
        if val_windows:
            criterion = evaluate_loss(model, val_windows, node_memory,
                                      device, num_types,
                                      timestep_offset=len(fit_windows),
                                      reset=False)
            label = 'Val loss'
        else:
            criterion = train_loss
            label = 'Train loss (no val tail)'
        extra = ''
        if args.save_every_epoch:
            torch.save(model.state_dict(),
                       osp.join(out_dir, f'model_epoch{epoch:03d}.pt'))
            if val_windows:
                nll_zeromem = evaluate_loss_zero_memory(
                    model, val_windows, device, num_types,
                    args.hidden_dim + profile_dim)
                delta = nll_zeromem - criterion
                extra = (f' | Val zero-mem: {nll_zeromem:.4f}'
                         f' | delta_mem: {delta:.4f}')
                with open(metrics_path, 'a') as f:
                    f.write(f'{epoch}\t{train_loss:.6f}\t{criterion:.6f}\t'
                            f'{nll_zeromem:.6f}\t{delta:.6f}\n')
        show(f'Epoch {epoch} | Train loss: {train_loss:.4f} | '
             f'{label}: {criterion:.4f}{extra} | {time.time() - t0:.1f}s')

        torch.save(model.state_dict(), latest_path)
        if criterion < best_criterion:
            best_criterion = criterion
            torch.save(model.state_dict(), best_path)

    show(f'Done. Best {label.lower()}: {best_criterion:.4f}')
    show(f'Checkpoints: {best_path} (best) / {latest_path} (latest)')


if __name__ == '__main__':
    main()
