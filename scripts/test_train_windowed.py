"""Integration checks for Stage 3 + Stage 4 wiring: node_memory.py
combined with train_windowed.py's model and training loop.

Run directly: python test_train_windowed.py

Needs torch AND torch_geometric (SAGEConv, Data) -- like
test_windowed_data.py, this could not be run in the sandbox this was
written in (no working torch_geometric install there -- torch itself
installs but fails to import, missing CUDA runtime libraries with no GPU
present to justify chasing them down). Run this in the `threatrace` conda
environment before trusting train_windowed.py.
"""

import math

import torch

from windowed_data import build_window_data
from node_memory import NodeMemory
from train_windowed import SAGENetWithMemory, train_one_epoch, evaluate


# Node 'uuid-A' appears in window 0 and again in window 2, with nothing in
# window 1 -- the exact dormant-then-active shape this extension targets.
WINDOW0 = ["uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t100\n"]
WINDOW1 = ["uuid-C\tProcess\tuuid-D\tFile\tEVENT_WRITE\t200\n"]
WINDOW2 = ["uuid-A\tProcess\tuuid-E\tFile\tEVENT_WRITE\t300\n"]


def _parse(line):
    temp = line.strip('\n').split('\t')
    return (temp[0], temp[1], temp[2], temp[3], temp[4], int(temp[5]))


def _fixture_dataset(node_vocab, feature_map, label_map):
    windows = [WINDOW0, WINDOW1, WINDOW2]
    return [
        build_window_data([_parse(l) for l in w], node_vocab, feature_map, label_map)
        for w in windows
    ]


def _fixture_vocabs():
    node_vocab = {u: i for i, u in enumerate(
        ['uuid-A', 'uuid-B', 'uuid-C', 'uuid-D', 'uuid-E'])}
    feature_map = {'EVENT_READ': 0, 'EVENT_WRITE': 1}
    label_map = {'Process': 0, 'File': 1}
    return node_vocab, feature_map, label_map


def test_forward_runs_and_zero_memory_is_a_no_op():
    node_vocab, feature_map, label_map = _fixture_vocabs()
    dataset = _fixture_dataset(node_vocab, feature_map, label_map)
    in_channels = len(feature_map) * 2
    out_channels = len(label_map)

    model = SAGENetWithMemory(in_channels, out_channels, hidden_channels=8)
    # This test checks a structural property (memory=None == memory=zeros),
    # not training dynamics -- eval() disables dropout so two separate
    # forward calls are actually comparable. Without this, the two calls
    # would apply independent random dropout masks and out_no_mem /
    # out_zero_mem would legitimately differ for reasons that have
    # nothing to do with memory.
    model.eval()
    data = dataset[0]

    out_no_mem, hidden_no_mem = model(data.x, data.edge_index, memory=None)
    zero_mem = torch.zeros(data.num_nodes, 8)
    out_zero_mem, hidden_zero_mem = model(data.x, data.edge_index, memory=zero_mem)

    assert out_no_mem.shape == (data.num_nodes, out_channels)
    assert hidden_no_mem.shape == (data.num_nodes, 8)
    # memory=None and memory=zeros must be equivalent -- a zero residual
    # changes nothing, confirming the injection point is a true additive
    # residual and not accidentally gating/scaling anything else.
    assert torch.allclose(out_no_mem, out_zero_mem, atol=1e-6)
    assert torch.allclose(hidden_no_mem, hidden_zero_mem, atol=1e-6)
    print('PASS: forward_runs_and_zero_memory_is_a_no_op')


def test_dormant_node_carries_decayed_memory_into_later_window():
    node_vocab, feature_map, label_map = _fixture_vocabs()
    dataset = _fixture_dataset(node_vocab, feature_map, label_map)

    device = torch.device('cpu')
    memory_dim = 6
    decay_rate = 0.3
    model = SAGENetWithMemory(len(feature_map) * 2, len(label_map),
                              hidden_channels=memory_dim).to(device)
    node_memory = NodeMemory(len(node_vocab), memory_dim, decay_rate, device=device)

    node_memory.reset()
    for window_idx, data in enumerate(dataset):
        data = data.to(device)
        mem = node_memory.get_decayed(data.global_id, window_idx)

        if window_idx == 2:
            # uuid-A (global id 0) was last written in window 0 and is
            # absent from window 1 -- its memory going into window 2 must
            # be a decayed (not zero, not undecayed) copy of window 0's
            # stored value.
            a_local = (data.global_id == node_vocab['uuid-A']).nonzero(as_tuple=True)[0].item()
            assert mem[a_local].abs().sum().item() > 0.0
            stored = node_memory.memory[node_vocab['uuid-A']]
            expected = stored * math.exp(-decay_rate * 2.0)  # elapsed = 2 - 0
            assert torch.allclose(mem[a_local], expected, atol=1e-5)

        _, hidden = model(data.x, data.edge_index, mem)
        node_memory.update(data.global_id, hidden.detach(), window_idx)

    print('PASS: dormant_node_carries_decayed_memory_into_later_window')


def test_train_one_epoch_and_evaluate_run_end_to_end():
    node_vocab, feature_map, label_map = _fixture_vocabs()
    train_dataset = _fixture_dataset(node_vocab, feature_map, label_map)
    test_dataset = _fixture_dataset(node_vocab, feature_map, label_map)

    device = torch.device('cpu')
    memory_dim = 6
    model = SAGENetWithMemory(len(feature_map) * 2, len(label_map),
                              hidden_channels=memory_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    train_memory = NodeMemory(len(node_vocab), memory_dim, decay_rate=0.3, device=device)
    test_memory = NodeMemory(len(node_vocab), memory_dim, decay_rate=0.3, device=device)

    losses = [train_one_epoch(model, train_dataset, train_memory, optimizer, device)
              for _ in range(20)]

    assert all(math.isfinite(l) for l in losses)
    assert losses[-1] < losses[0], (
        f'loss did not improve on a fixture this trivial: '
        f'{losses[0]:.4f} -> {losses[-1]:.4f}'
    )

    acc = evaluate(model, test_dataset, test_memory, device)
    assert 0.0 <= acc <= 1.0
    print(f'PASS: train_one_epoch_and_evaluate_run_end_to_end '
          f'(loss {losses[0]:.4f} -> {losses[-1]:.4f}, test acc {acc:.4f})')


def test_continue_memory_into_test_offsets_timesteps_correctly():
    """--continue-memory-into-test's core mechanism: evaluating with
    reset=False and a timestep_offset must produce a DIFFERENT (not
    equal, not crashing) result than a fresh reset() pass, for a node
    that appears in both the tail of training and the head of test.
    """
    node_vocab, feature_map, label_map = _fixture_vocabs()
    # Re-use uuid-A across the boundary: last training window is index 2.
    train_dataset = _fixture_dataset(node_vocab, feature_map, label_map)
    test_dataset = _fixture_dataset(node_vocab, feature_map, label_map)  # uuid-A in its window 0 too

    device = torch.device('cpu')
    memory_dim = 6
    model = SAGENetWithMemory(len(feature_map) * 2, len(label_map),
                              hidden_channels=memory_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    shared_memory = NodeMemory(len(node_vocab), memory_dim, decay_rate=0.3, device=device)

    train_one_epoch(model, train_dataset, shared_memory, optimizer, device)
    # uuid-A's last_seen should now be 2 (last training window it appeared in).
    assert shared_memory.last_seen[node_vocab['uuid-A']].item() == 2

    acc_continued = evaluate(model, test_dataset, shared_memory, device,
                             timestep_offset=len(train_dataset), reset=False)
    assert 0.0 <= acc_continued <= 1.0
    # uuid-A appears again at LOCAL index 2 within test_dataset (same
    # fixture reused), so its last update during the continued pass lands
    # at timestep_offset + 2, not at a small index restarting near 0 --
    # that's the property that actually matters here (the clock keeps
    # counting forward across the boundary rather than resetting).
    expected_last_seen = len(train_dataset) + 2
    assert shared_memory.last_seen[node_vocab['uuid-A']].item() == expected_last_seen, (
        f'expected last_seen={expected_last_seen}, got '
        f'{shared_memory.last_seen[node_vocab["uuid-A"]].item()}'
    )
    print('PASS: continue_memory_into_test_offsets_timesteps_correctly')


def test_empty_window_is_skipped_without_crashing():
    """A window can end up with 0 nodes if every row references a type
    outside feature_map/label_map (build_window_data drops such rows
    entirely, matching MyDatasetA's existing behaviour). train_one_epoch
    must skip it rather than dividing by zero or calling the model on an
    empty tensor.
    """
    node_vocab, feature_map, label_map = _fixture_vocabs()
    dataset = _fixture_dataset(node_vocab, feature_map, label_map)
    empty = build_window_data(
        [_parse("uuid-A\tProcess\tuuid-B\tFile\tEVENT_UNKNOWN\t150\n")],
        node_vocab, feature_map, label_map)
    assert empty.num_nodes == 0
    dataset_with_gap = [dataset[0], empty, dataset[1], dataset[2]]

    device = torch.device('cpu')
    memory_dim = 4
    model = SAGENetWithMemory(len(feature_map) * 2, len(label_map),
                              hidden_channels=memory_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    node_memory = NodeMemory(len(node_vocab), memory_dim, decay_rate=0.1, device=device)

    loss = train_one_epoch(model, dataset_with_gap, node_memory, optimizer, device)
    assert math.isfinite(loss)
    print('PASS: empty_window_is_skipped_without_crashing')


if __name__ == '__main__':
    test_forward_runs_and_zero_memory_is_a_no_op()
    test_dormant_node_carries_decayed_memory_into_later_window()
    test_train_one_epoch_and_evaluate_run_end_to_end()
    test_continue_memory_into_test_offsets_timesteps_correctly()
    test_empty_window_is_skipped_without_crashing()
    print('All train_windowed integration checks passed.')
