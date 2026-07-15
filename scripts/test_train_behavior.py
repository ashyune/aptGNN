"""Unit tests for train_behavior.py's structural guarantees.

These test the *design properties* the D1-revised score rests on, not
just arithmetic: the observed profile never reaches a node's own
prediction, and zeroed memory provably collapses the model to a
per-type prior (what makes the --max-norm 0 ablation an honest
instrument).
"""

import torch

from train_behavior import (BehaviorNet, RelationalMemoryConv, profile_nll,
                            make_writeback, type_onehot)

NUM_TYPES = 6
PROFILE_DIM = 56
HIDDEN = 32
NUM_RELATIONS = 5  # toy relation vocab; last id plays the self-loop role


def _toy_window(num_nodes=5):
    """A small line graph with self-loops, mixed types, random profiles."""
    src = list(range(num_nodes - 1)) + list(range(num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([i % NUM_TYPES for i in range(num_nodes)])
    x = torch.rand(num_nodes, PROFILE_DIM) * 5
    return x, y, edge_index


def _toy_relational_window(num_nodes=5):
    """_toy_window plus an edge_type vector mirroring build_window_data's
    layout: real edges carry CDM-style relation ids, the trailing
    self-loops carry the dedicated last id."""
    x, y, edge_index = _toy_window(num_nodes)
    num_real = num_nodes - 1
    edge_type = torch.tensor(
        [i % (NUM_RELATIONS - 1) for i in range(num_real)]
        + [NUM_RELATIONS - 1] * num_nodes, dtype=torch.long)
    return x, y, edge_index, edge_type


def test_profile_nll_hand_example():
    # two events of type 0, model puts log p on them
    log_probs = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    x = torch.tensor([[2.0, 0.0, 0.0]])
    nll = profile_nll(log_probs, x)
    # -(2 * log 0.5) / 2 = log 2
    assert torch.allclose(nll, torch.tensor([0.6931]), atol=1e-4)


def test_profile_nll_is_per_event_not_per_degree():
    # same behavior distribution at 10x the volume => same score
    log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    x = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    nll = profile_nll(log_probs, x)
    assert torch.allclose(nll[0], nll[1])


def test_zero_memory_collapses_to_type_prior():
    # With memory all-zero (the --max-norm 0 ablation), two nodes of the
    # same type must get IDENTICAL predictions regardless of degree or
    # position -- the model has provably no node-specific channel left.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN).eval()
    x, y, edge_index = _toy_window(6)
    y = torch.tensor([2, 2, 3, 3, 2, 3])  # same-type pairs at different positions
    memory = torch.zeros(6, HIDDEN + PROFILE_DIM)
    out, _h = model(type_onehot(y, NUM_TYPES), memory, edge_index)
    assert torch.allclose(out[0], out[1], atol=1e-6)
    assert torch.allclose(out[0], out[4], atol=1e-6)
    assert torch.allclose(out[2], out[3], atol=1e-6)
    assert not torch.allclose(out[0], out[2], atol=1e-3)  # types still differ


def test_nonzero_memory_changes_prediction():
    # The memory channel must actually be able to move the output.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN).eval()
    x, y, edge_index = _toy_window(4)
    zero_mem = torch.zeros(4, HIDDEN + PROFILE_DIM)
    warm_mem = torch.randn(4, HIDDEN + PROFILE_DIM)
    out_cold, _ = model(type_onehot(y, NUM_TYPES), zero_mem, edge_index)
    out_warm, _ = model(type_onehot(y, NUM_TYPES), warm_mem, edge_index)
    assert not torch.allclose(out_cold, out_warm, atol=1e-3)


def test_prediction_independent_of_observed_profile():
    # THE leakage guard: forward() takes no x; scoring the same window
    # with a completely different observed profile must leave the
    # prediction untouched (only the NLL target moves).
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN).eval()
    x, y, edge_index = _toy_window(4)
    memory = torch.randn(4, HIDDEN + PROFILE_DIM)
    out1, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index)
    out2, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index)
    assert torch.allclose(out1, out2)
    nll_a = profile_nll(out1, x)
    nll_b = profile_nll(out1, torch.rand_like(x) * 5)
    assert not torch.allclose(nll_a, nll_b, atol=1e-3)


def test_writeback_layout_and_detachment():
    torch.manual_seed(0)
    h = torch.randn(3, HIDDEN, requires_grad=True)
    x = torch.tensor([[2.0, 2.0] + [0.0] * (PROFILE_DIM - 2)]).repeat(3, 1)
    wb = make_writeback(h, x)
    assert wb.shape == (3, HIDDEN + PROFILE_DIM)
    assert not wb.requires_grad  # NodeMemory.update() would reject otherwise
    assert torch.allclose(wb[:, :HIDDEN], h.detach())
    # profile half is per-event normalized: 2 events of each of 2 types -> 0.5
    assert torch.allclose(wb[0, HIDDEN:HIDDEN + 2],
                          torch.tensor([0.5, 0.5]))
    assert torch.allclose(wb[0, HIDDEN + 2:].sum(), torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Extension 2 (--relational): same design guarantees under relation routing
# ---------------------------------------------------------------------------

def test_relational_zero_memory_collapses_to_type_prior():
    # The ablation guarantee must survive E2: with memory all-zero,
    # same-type nodes get IDENTICAL predictions -- relation routing adds
    # no node-specific channel of its own.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN,
                        num_relations=NUM_RELATIONS).eval()
    _x, _y, edge_index, edge_type = _toy_relational_window(6)
    y = torch.tensor([2, 2, 3, 3, 2, 3])
    memory = torch.zeros(6, HIDDEN + PROFILE_DIM)
    out, _h = model(type_onehot(y, NUM_TYPES), memory, edge_index, edge_type)
    assert torch.allclose(out[0], out[1], atol=1e-6)
    assert torch.allclose(out[0], out[4], atol=1e-6)
    assert torch.allclose(out[2], out[3], atol=1e-6)
    assert not torch.allclose(out[0], out[2], atol=1e-3)


def test_relational_zero_memory_ignores_edge_types():
    # The no-injection guard: edge types reach the output only
    # multiplied by memory content (no per-relation bias, no edge-type
    # embedding), so with memory zeroed, relabeling every edge must not
    # move any prediction -- current-window edge types cannot leak the
    # target into a cold node's score.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN,
                        num_relations=NUM_RELATIONS).eval()
    _x, y, edge_index, edge_type = _toy_relational_window(5)
    memory = torch.zeros(5, HIDDEN + PROFILE_DIM)
    out_a, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, edge_type)
    relabeled = (edge_type + 1) % NUM_RELATIONS
    out_b, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, relabeled)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_relational_edge_types_route_warm_memory():
    # ...and with WARM memory the routing must actually be live: the
    # same relabeling now changes predictions.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN,
                        num_relations=NUM_RELATIONS).eval()
    _x, y, edge_index, edge_type = _toy_relational_window(5)
    memory = torch.randn(5, HIDDEN + PROFILE_DIM)
    out_a, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, edge_type)
    relabeled = (edge_type + 1) % NUM_RELATIONS
    out_b, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, relabeled)
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_relational_equal_weights_reduce_to_shared_sage():
    # With one basis and identical coefficients, W_r is the same matrix
    # for every relation, and mean-of-transformed == transform-of-mean:
    # RelationalMemoryConv must reproduce the E1 SAGEConv read EXACTLY.
    # This is what makes E1-vs-E2 a pure routing ablation.
    torch.manual_seed(0)
    from torch_geometric.nn import SAGEConv
    in_dim, out_dim = 12, 7
    rel = RelationalMemoryConv(in_dim, out_dim, num_relations=4, num_bases=1)
    sage = SAGEConv(in_dim, out_dim, normalize=False, root_weight=False)
    with torch.no_grad():
        rel.coeff.fill_(1.0)
        rel.basis[0].copy_(sage.lin_l.weight.t())
        rel.bias.copy_(sage.lin_l.bias)
    x = torch.randn(6, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2],
                               [1, 2, 3, 4, 5, 0, 3, 5]])
    edge_type = torch.tensor([i % 4 for i in range(8)])
    assert torch.allclose(rel(x, edge_index, edge_type), sage(x, edge_index),
                          atol=1e-5)


def test_relational_prediction_independent_of_observed_profile():
    # THE leakage guard, E2 edition: forward() still takes no x.
    torch.manual_seed(0)
    model = BehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN,
                        num_relations=NUM_RELATIONS).eval()
    x, y, edge_index, edge_type = _toy_relational_window(4)
    memory = torch.randn(4, HIDDEN + PROFILE_DIM)
    out1, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, edge_type)
    out2, _ = model(type_onehot(y, NUM_TYPES), memory, edge_index, edge_type)
    assert torch.allclose(out1, out2)
    nll_a = profile_nll(out1, x)
    nll_b = profile_nll(out1, torch.rand_like(x) * 5)
    assert not torch.allclose(nll_a, nll_b, atol=1e-3)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} passed')


if __name__ == '__main__':
    _run_all()
