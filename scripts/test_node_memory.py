"""Correctness checks for node_memory.py.

Run directly: python test_node_memory.py

Unlike the Stage 2 test files, this one has no torch_geometric
dependency at all -- NodeMemory only manipulates plain tensors -- so it
should run in any environment with bare torch installed.
"""

import math

import torch

from node_memory import NodeMemory


def test_never_seen_returns_zero():
    mem = NodeMemory(num_nodes=5, embedding_dim=4, decay_rate=0.1)
    out = mem.get_decayed(torch.tensor([0, 1, 2]), current_timestep=10)
    assert torch.equal(out, torch.zeros(3, 4))
    print('PASS: never_seen_returns_zero')


def test_zero_elapsed_is_undecayed():
    mem = NodeMemory(num_nodes=5, embedding_dim=3, decay_rate=0.5)
    emb = torch.tensor([[1.0, 2.0, 3.0]])
    mem.update(torch.tensor([2]), emb, current_timestep=7)
    out = mem.get_decayed(torch.tensor([2]), current_timestep=7)
    assert torch.allclose(out, emb)
    print('PASS: zero_elapsed_is_undecayed')


def test_decay_matches_closed_form():
    decay_rate = 0.2
    mem = NodeMemory(num_nodes=5, embedding_dim=2, decay_rate=decay_rate)
    emb = torch.tensor([[10.0, -4.0]])
    mem.update(torch.tensor([1]), emb, current_timestep=3)

    out = mem.get_decayed(torch.tensor([1]), current_timestep=8)  # elapsed = 5
    expected = emb * math.exp(-decay_rate * 5)
    assert torch.allclose(out, expected, atol=1e-6)
    print('PASS: decay_matches_closed_form')


def test_decay_rate_zero_never_decays():
    mem = NodeMemory(num_nodes=3, embedding_dim=2, decay_rate=0.0)
    emb = torch.tensor([[5.0, 5.0]])
    mem.update(torch.tensor([0]), emb, current_timestep=0)
    out = mem.get_decayed(torch.tensor([0]), current_timestep=10_000)
    assert torch.allclose(out, emb)
    print('PASS: decay_rate_zero_never_decays')


def test_update_overwrites_value_and_timestamp():
    mem = NodeMemory(num_nodes=3, embedding_dim=2, decay_rate=1.0)
    mem.update(torch.tensor([0]), torch.tensor([[1.0, 1.0]]), current_timestep=0)

    # Heavy decay by timestep 5 if it were still the old value.
    decayed_old = mem.get_decayed(torch.tensor([0]), current_timestep=5)
    assert decayed_old.abs().sum().item() < 1.0 - 1e-3  # meaningfully decayed

    # Overwrite at timestep 5 -- a fresh lookup AT timestep 5 must be
    # undecayed again (elapsed resets to 0 from the new write).
    mem.update(torch.tensor([0]), torch.tensor([[9.0, 9.0]]), current_timestep=5)
    fresh = mem.get_decayed(torch.tensor([0]), current_timestep=5)
    assert torch.allclose(fresh, torch.tensor([[9.0, 9.0]]))
    print('PASS: update_overwrites_value_and_timestamp')


def test_negative_elapsed_is_clamped_not_amplified():
    mem = NodeMemory(num_nodes=3, embedding_dim=2, decay_rate=0.5)
    emb = torch.tensor([[2.0, 2.0]])
    mem.update(torch.tensor([0]), emb, current_timestep=10)
    # Looked up "before" its own last_seen (e.g. caller misuse / out-of-
    # order calls) -- must clamp to elapsed=0, never go negative and
    # amplify the stored value above what was actually stored.
    out = mem.get_decayed(torch.tensor([0]), current_timestep=3)
    assert torch.allclose(out, emb)
    print('PASS: negative_elapsed_is_clamped_not_amplified')


def test_reset_clears_everything():
    mem = NodeMemory(num_nodes=4, embedding_dim=2, decay_rate=0.1)
    mem.update(torch.tensor([0, 1]),
              torch.tensor([[1.0, 1.0], [2.0, 2.0]]), current_timestep=0)
    assert len(mem) == 2
    mem.reset()
    assert len(mem) == 0
    out = mem.get_decayed(torch.tensor([0, 1]), current_timestep=0)
    assert torch.equal(out, torch.zeros(2, 2))
    print('PASS: reset_clears_everything')


def test_update_rejects_grad_tensor():
    mem = NodeMemory(num_nodes=3, embedding_dim=2, decay_rate=0.1)
    emb = torch.tensor([[1.0, 1.0]], requires_grad=True)
    try:
        mem.update(torch.tensor([0]), emb, current_timestep=0)
        assert False, 'expected ValueError for a grad-tracked tensor'
    except ValueError:
        pass
    print('PASS: update_rejects_grad_tensor')


def test_update_rejects_shape_mismatch():
    mem = NodeMemory(num_nodes=3, embedding_dim=4, decay_rate=0.1)
    try:
        mem.update(torch.tensor([0, 1]), torch.zeros(2, 3), current_timestep=0)
        assert False, 'expected ValueError for embedding_dim mismatch'
    except ValueError:
        pass
    try:
        mem.update(torch.tensor([0, 1]), torch.zeros(1, 4), current_timestep=0)
        assert False, 'expected ValueError for row-count mismatch'
    except ValueError:
        pass
    print('PASS: update_rejects_shape_mismatch')


def test_batch_independence():
    """Looking up one node's memory must not disturb another's, and
    lookup order in the batch must not matter."""
    mem = NodeMemory(num_nodes=5, embedding_dim=2, decay_rate=0.3)
    mem.update(torch.tensor([0, 3]),
              torch.tensor([[1.0, 0.0], [0.0, 1.0]]), current_timestep=0)
    out = mem.get_decayed(torch.tensor([3, 1, 0]), current_timestep=0)
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0]))  # node 3
    assert torch.allclose(out[1], torch.zeros(2))             # node 1, unseen
    assert torch.allclose(out[2], torch.tensor([1.0, 0.0]))  # node 0
    print('PASS: batch_independence')


def test_accepts_plain_list_ids():
    """get_decayed/update should also accept plain Python lists, not just
    tensors -- convenient for quick scripts/tests."""
    mem = NodeMemory(num_nodes=3, embedding_dim=2, decay_rate=0.1)
    mem.update([0, 1], torch.tensor([[1.0, 2.0], [3.0, 4.0]]), current_timestep=0)
    out = mem.get_decayed([1, 0], current_timestep=0)
    assert torch.allclose(out[0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(out[1], torch.tensor([1.0, 2.0]))
    print('PASS: accepts_plain_list_ids')


def test_constructor_validation():
    for kwargs in [dict(num_nodes=0, embedding_dim=2, decay_rate=0.1),
                   dict(num_nodes=5, embedding_dim=0, decay_rate=0.1),
                   dict(num_nodes=5, embedding_dim=2, decay_rate=-0.1)]:
        try:
            NodeMemory(**kwargs)
            assert False, f'expected ValueError for {kwargs}'
        except ValueError:
            pass
    print('PASS: constructor_validation')


if __name__ == '__main__':
    test_never_seen_returns_zero()
    test_zero_elapsed_is_undecayed()
    test_decay_matches_closed_form()
    test_decay_rate_zero_never_decays()
    test_update_overwrites_value_and_timestamp()
    test_negative_elapsed_is_clamped_not_amplified()
    test_reset_clears_everything()
    test_update_rejects_grad_tensor()
    test_update_rejects_shape_mismatch()
    test_batch_independence()
    test_accepts_plain_list_ids()
    test_constructor_validation()
    print('All node_memory checks passed.')
