"""Stage 4: cross-window memory training integration.

Wires Stage 3's NodeMemory into an actual training loop over the Stage 2
windowed dataset, replacing data_process_train.py's single static graph
with the chronological window sequence windowed_data.py already builds.

Relationship to the baseline (train_darpatc.py)
-------------------------------------------------
This is a NEW, separate entry point rather than a modification of
train_darpatc.py, on purpose:

- train_darpatc.py must stay runnable unmodified as the baseline you're
  comparing Extension 1+2 against (and against MAGIC). Editing its
  SAGENet or training loop in place would put that comparison at risk.
- Windowed cross-window-memory training is a fundamentally different
  temporal loop (a chronological pass over ~87 small per-window graphs
  with state threaded between them) from the baseline's single-big-graph
  neighbor-sampled mini-batch loop. Bolting one onto the other in place
  would obscure both.

What this does NOT (yet) reproduce from the baseline, deliberately
--------------------------------------------------------------------
- The iterative prune-cascade scheme in train_darpatc.py's train_pro() /
  validate() (train, remove confidently-correct nodes, retrain on the
  remainder, save a cascade of models) is not ported here. This loop is a
  single, straightforward supervised pass per epoch over the window
  sequence. Folding the cascade scheme on top of windowed memory is a
  real, separate design question (does "confidently correct" get tracked
  per global_id across the whole sequence or per-window? does a new
  cascade round reset NodeMemory?) that deserves its own discussion
  rather than being decided silently here.
- The confidence-ratio thresholding in the baseline's _predict_batch()
  ("pred=100 if too uncertain") and evaluate_darpatc.py's full
  precision/recall/alarm-file pipeline aren't reproduced. evaluate()
  below reports plain classification accuracy only -- enough to confirm
  the mechanism trains and to compare decay-rate settings, not a
  replacement for the full anomaly-detection evaluation pipeline.

Why each window is processed full-batch (no NeighborLoader)
--------------------------------------------------------------
windowing.generate_windows() already caps each window at window_size
*events*, so a window has at most 2 * window_size nodes (50,000 by
default -> at most 100,000, and typically far fewer given repeated hub
nodes). That's small enough to run conv1/conv2 over the whole window in
one forward/backward pass. Doing this avoids re-deriving NeighborLoader's
seed/neighbor bookkeeping for a fundamentally different loop shape, and
it sidesteps the baseline's actual scaling problem in the first place:
full-neighborhood sampling (num_neighbors=[-1,-1]) is expensive
specifically because high-degree nodes get re-expanded across many
overlapping seed batches drawn from one huge graph. A window is already
small; there's nothing to sample.

If a real window ever turns out too large for full-batch on your GPU,
NeighborLoader could be reintroduced for that window specifically,
following make_loader()'s pattern in train_darpatc.py -- confirm this is
actually necessary first (see the validation instructions) before adding
that complexity back.

Train/test memory continuity is an open question, not a silent decision
--------------------------------------------------------------------------
cadets_train.txt and cadets_test.txt come from separate source tars
(ta1-cadets-e3-official vs ta1-cadets-e3-official-2 per parse_darpatc.py)
-- whether the test window sequence is genuinely the chronological
continuation of the train sequence isn't something this code can verify.
By default, NodeMemory resets between the train and test passes (treating
them as two independent evaluations). --continue-memory-into-test
switches to carrying the training memory's final state (and window-index
clock) forward into testing instead, which is more faithful to a
"deployed detector" scenario IF the timing is truly contiguous.
"""

import argparse
import os
import os.path as osp
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

class SAGENetWithMemory(torch.nn.Module):
    """train_darpatc.py's SAGENet, instrumented for cross-window memory.

    Identical two-layer GraphSAGE computation when memory=None (same
    layer widths, same normalize=False/root_weight=False conv config,
    same dropout, same log_softmax) -- this is not a redesign of the
    baseline classifier, just an instrumented version of it. Two
    differences:

    (a) forward() accepts an optional `memory` tensor of shape
        [num_nodes, hidden_channels], added to conv1's output BEFORE the
        ReLU -- a temporal residual connection (see NodeMemory). Adding
        before the nonlinearity keeps the same "always non-negative
        post-activation" shape as the memoryless path and follows the
        usual residual convention (add, then nonlinearity).
    (b) forward() returns (log_probs, hidden) instead of just log_probs.
        Callers need the post-ReLU, PRE-dropout hidden state to write
        back into NodeMemory afterwards -- pre-dropout deliberately,
        since dropout is stochastic and train-only; storing a dropped-out
        vector would inject random noise into future windows and would
        make memory differ between train/eval mode for reasons that have
        nothing to do with elapsed time.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=False, root_weight=False)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=False, root_weight=False)

    def forward(self, x, edge_index, memory=None):
        h = self.conv1(x, edge_index)
        if memory is not None:
            h = h + memory
        h = F.relu(h)
        h_dropped = F.dropout(h, p=0.5, training=self.training)
        out = self.conv2(h_dropped, edge_index)
        return F.log_softmax(out, dim=1), h


# ---------------------------------------------------------------------------
# Train / eval loops -- sequential over the window list, threading memory
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataset, node_memory, optimizer, device):
    """One full supervised pass over *dataset*, in chronological order.

    node_memory is reset at the start -- see NodeMemory.reset() for why
    epoch boundaries must not leak state into each other. Each window is
    a single full-batch forward/backward pass (see module docstring).
    """
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
        out, hidden = model(data.x, data.edge_index, memory)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
        node_memory.update(data.global_id, hidden.detach(), window_idx)

    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate(model, dataset, node_memory, device, timestep_offset=0, reset=True):
    """Plain classification accuracy over *dataset*.

    timestep_offset/reset exist for --continue-memory-into-test: when
    continuing a NodeMemory instance from training straight into testing,
    the caller passes reset=False and timestep_offset=len(train_dataset)
    so the test windows' timesteps keep counting forward from where
    training left off instead of restarting at 0. Does not (yet)
    reproduce the baseline's confidence-ratio thresholding or
    precision/recall pipeline -- see module docstring.
    """
    model.eval()
    if reset:
        node_memory.reset()
    correct = 0
    total = 0

    for i, data in enumerate(dataset):
        if data.num_nodes == 0:
            continue
        window_idx = timestep_offset + i
        data = data.to(device)
        memory = node_memory.get_decayed(data.global_id, window_idx)
        out, hidden = model(data.x, data.edge_index, memory)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_nodes
        node_memory.update(data.global_id, hidden.detach(), window_idx)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 4: sequential windowed training with '
                     'cross-window NodeMemory.'
    )
    parser.add_argument('--scene', type=str, default='cadets',
                        choices=['cadets', 'trace', 'theia', 'fivedirections'])
    parser.add_argument('--window-size', type=int, default=50000)
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='SAGEConv hidden width AND NodeMemory '
                             'embedding width -- the two are the same '
                             'tensor by design (see module docstring).')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Untuned starting point -- sweep this.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--node-vocab', type=str, default=None,
                        help='Defaults to ../models/<scene>_node_vocab.txt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to ../models/windowed_<scene>/')
    parser.add_argument('--train-cache', type=str, default=None,
                        help='Optional pre-built windowed training set '
                             '(torch.save()-d list of Data, e.g. from '
                             'windowed_data.py --save) used instead of '
                             'rebuilding from the raw provenance file.')
    parser.add_argument('--test-cache', type=str, default=None)
    parser.add_argument('--continue-memory-into-test', action='store_true',
                        help='Reuse the training NodeMemory for '
                             'evaluation (continuing its window-index '
                             'clock forward) instead of resetting. Only '
                             'sensible if the test file is genuinely the '
                             'chronological continuation of the training '
                             'file -- see module docstring. Default: off.')
    args = parser.parse_args()

    base = '../graphchi-cpp-master/graph_data/darpatc/'
    train_path = base + args.scene + '_train.txt'
    test_path = base + args.scene + '_test.txt'
    node_vocab_path = args.node_vocab or f'../models/{args.scene}_node_vocab.txt'
    out_dir = args.out_dir or f'../models/windowed_{args.scene}'

    for p in (train_path, test_path, node_vocab_path):
        if not osp.exists(p):
            raise FileNotFoundError(f'Expected file not found: {p}')
    os.makedirs(out_dir, exist_ok=True)

    show(f'Loading node vocab: {node_vocab_path}')
    node_vocab = load_node_vocab(node_vocab_path)
    show(f'{len(node_vocab):,} global node identities')

    # Type vocab always comes from the training file (matching the
    # baseline convention and windowed_data.py's own main()), so train
    # and test windows share identical feature/label integer assignments.
    feature_map, label_map = build_type_vocab(train_path)
    in_channels = len(feature_map) * 2
    out_channels = len(label_map)
    show(f'in_channels={in_channels} ({len(feature_map)} edge types), '
         f'out_channels={out_channels} ({len(label_map)} node types)')

    if args.train_cache:
        show(f'Loading cached train windows: {args.train_cache}')
        train_dataset = torch.load(args.train_cache, weights_only=False)
    else:
        show('Building train windows')
        train_dataset = build_windowed_dataset(
            train_path, node_vocab, feature_map, label_map, args.window_size)

    if args.test_cache:
        show(f'Loading cached test windows: {args.test_cache}')
        test_dataset = torch.load(args.test_cache, weights_only=False)
    else:
        show('Building test windows')
        test_dataset = build_windowed_dataset(
            test_path, node_vocab, feature_map, label_map, args.window_size)

    show(f'{len(train_dataset)} train windows, {len(test_dataset)} test windows')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    show(f'Device: {device}')

    model = SAGENetWithMemory(in_channels, out_channels, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_memory = NodeMemory(len(node_vocab), args.hidden_dim, args.decay_rate, device=device, max_norm=100.0)
    if args.continue_memory_into_test:
        test_memory = train_memory
        test_offset = len(train_dataset)
        test_reset = False
        show('Evaluation will continue the training NodeMemory forward '
             '(--continue-memory-into-test)')
    else:
        test_memory = NodeMemory(len(node_vocab), args.hidden_dim, args.decay_rate, device=device, max_norm=100.0)
        test_offset = 0
        test_reset = True

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_dataset, train_memory, optimizer, device)
        acc = evaluate(model, test_dataset, test_memory, device,
                       timestep_offset=test_offset, reset=test_reset)
        show(f'Epoch {epoch} | Loss: {loss:.4f} | Test Acc: {acc:.4f} '
             f'| {time.time() - t0:.1f}s')

        torch.save(model.state_dict(), osp.join(out_dir, 'latest_model.pt'))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), osp.join(out_dir, 'best_model.pt'))

    show(f'Finished. Best test accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
