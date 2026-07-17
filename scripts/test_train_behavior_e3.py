"""Unit tests for Extension 3's structural guarantees.

Mirrors test_train_behavior.py's pattern (plain python, no pytest):
these test the design properties the E3 score rests on -- the binned
window features are targets and memory content only, never inputs to a
node's own prediction; zeroed memory still collapses every head to a
per-type prior; the profile/feature score combination is the locked
unweighted mean; and the feature computation + Data alignment match
brute force under the exact build_window_data row filter.
"""

import math
import os
import tempfile

import torch

from windowing import build_type_vocab
from windowed_data import build_window_data
from window_features import (FEATURE_BIN_SIZES, NUM_FEATURES,
                             bin_n_distinct, bin_rep_ratio, bin_span_frac,
                             compute_window_feature_bins,
                             feature_bins_tensor, build_enriched_dataset)
from train_behavior import profile_nll, type_onehot
from train_behavior_e3 import (EnrichedBehaviorNet, feature_nll,
                               combined_nll, make_writeback_e3)

NUM_TYPES = 6
PROFILE_DIM = 56
HIDDEN = 32


def _toy_window(num_nodes=5):
    src = list(range(num_nodes - 1)) + list(range(num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([i % NUM_TYPES for i in range(num_nodes)])
    x = torch.rand(num_nodes, PROFILE_DIM) * 5
    feat_bins = torch.stack([
        torch.randint(0, k, (num_nodes,)) for k in FEATURE_BIN_SIZES
    ], dim=1)
    return x, y, edge_index, feat_bins


def test_bin_edges():
    # n_distinct / rep_ratio: log2 buckets with a top cap
    assert bin_n_distinct(1) == 0
    assert bin_n_distinct(2) == 1
    assert bin_n_distinct(3) == 1
    assert bin_n_distinct(4) == 2
    assert bin_n_distinct(4096) == FEATURE_BIN_SIZES[0] - 1
    assert bin_n_distinct(10 ** 6) == FEATURE_BIN_SIZES[0] - 1
    assert bin_rep_ratio(1.0) == 0
    assert bin_rep_ratio(1.99) == 0
    assert bin_rep_ratio(2.0) == 1
    assert bin_rep_ratio(1666.67) == 10
    assert bin_rep_ratio(10 ** 9) == FEATURE_BIN_SIZES[1] - 1
    # span_frac: 10 linear bins + dedicated ==1.0 bin
    assert bin_span_frac(0.0) == 0
    assert bin_span_frac(0.05) == 0
    assert bin_span_frac(0.1) == 1
    assert bin_span_frac(0.95) == 9
    assert bin_span_frac(0.9999999) == 9  # never leaks into the 1.0 bin
    assert bin_span_frac(1.0) == FEATURE_BIN_SIZES[2] - 1


def test_window_feature_bins_brute_force():
    """Hand-checkable window; also exercises the type filter."""
    label_map = {'P': 0, 'F': 1}
    feature_map = {'READ': 0, 'WRITE': 1}
    rows = [
        # a (P) touches b twice and c once; b (F) sees only a; c (F) only a
        ('a', 'P', 'b', 'F', 'READ', 100),
        ('a', 'P', 'b', 'F', 'WRITE', 150),
        ('x', 'UNKNOWN_TYPE', 'b', 'F', 'READ', 160),   # filtered out
        ('a', 'P', 'c', 'F', 'BOGUS_EDGE', 170),        # filtered out
        ('a', 'P', 'c', 'F', 'READ', 200),
        ('d', 'P', 'e', 'F', 'READ', 200),
    ]
    bins = compute_window_feature_bins(rows, feature_map, label_map)
    assert set(bins) == {'a', 'b', 'c', 'd', 'e'}
    # window span over FILTERED rows: 100..200
    # a: 3 events, 2 distinct, span (200-100)/100 = 1.0
    assert bins['a'] == (bin_n_distinct(2), bin_rep_ratio(3 / 2),
                         bin_span_frac(1.0))
    # b: 2 events, 1 distinct, span (150-100)/100 = 0.5
    assert bins['b'] == (bin_n_distinct(1), bin_rep_ratio(2.0),
                         bin_span_frac(0.5))
    # c: 1 event at t=200, span 0
    assert bins['c'] == (0, 0, bin_span_frac(0.0))
    # d, e: single shared event, span 0
    assert bins['d'] == (0, 0, 0)
    # the filtered rows contributed nothing anywhere
    assert 'x' not in bins


def test_feature_tensor_alignment_via_global_id():
    """feat_bins row i must describe the node at data.global_id[i],
    regardless of local index order."""
    label_map = {'P': 0, 'F': 1}
    feature_map = {'READ': 0}
    rows = [
        ('n1', 'P', 'n2', 'F', 'READ', 10),
        ('n3', 'P', 'n1', 'P', 'READ', 20),
        ('n3', 'P', 'n4', 'F', 'READ', 30),
    ]
    node_vocab = {'n1': 40, 'n2': 41, 'n3': 42, 'n4': 43}
    gid_to_uuid = {g: u for u, g in node_vocab.items()}
    data = build_window_data(rows, node_vocab, feature_map, label_map)
    bins = compute_window_feature_bins(rows, feature_map, label_map)
    t = feature_bins_tensor(data, bins, gid_to_uuid)
    assert t.shape == (data.num_nodes, NUM_FEATURES)
    for i, gid in enumerate(data.global_id.tolist()):
        assert tuple(t[i].tolist()) == bins[gid_to_uuid[gid]]
    # n3: 2 events, 2 distinct partners, span (30-20)/(30-10) = 0.5
    i3 = data.global_id.tolist().index(42)
    assert tuple(t[i3].tolist()) == (1, 0, bin_span_frac(0.5))


def test_build_enriched_dataset_end_to_end():
    """The enriched builder attaches an aligned feat_bins to every
    window while leaving the underlying Data construction identical."""
    lines = [
        'u1\tP\tu2\tF\tREAD\t5',
        'u1\tP\tu3\tF\tREAD\t1',
        'u4\tP\tu2\tF\tWRITE\t3',
        'u1\tP\tu2\tF\tREAD\t8',
    ]
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(lines) + '\n')
        path = f.name
    try:
        feature_map, label_map = build_type_vocab(path)
        node_vocab = {'u1': 0, 'u2': 1, 'u3': 2, 'u4': 3}
        dataset = build_enriched_dataset(path, node_vocab, feature_map,
                                         label_map, window_size=2)
        assert len(dataset) == 2  # 4 time-sorted rows, 2 per window
        for data in dataset:
            assert data.feat_bins.shape == (data.num_nodes, NUM_FEATURES)
            assert data.feat_bins.dtype == torch.long
        # window 0 = ts {1, 3}: u1 has 1 event/1 partner, span 0
        d0 = dataset[0]
        i_u1 = d0.global_id.tolist().index(0)
        assert tuple(d0.feat_bins[i_u1].tolist()) == (0, 0, 0)
        # window 1 = ts {5, 8}: u1->u2 twice; u1: 2 events, 1 distinct,
        # span (8-5)/(8-5) = 1.0 -> dedicated top bin
        d1 = dataset[1]
        i_u1 = d1.global_id.tolist().index(0)
        assert tuple(d1.feat_bins[i_u1].tolist()) == (
            0, bin_rep_ratio(2.0), FEATURE_BIN_SIZES[2] - 1)
    finally:
        os.unlink(path)


def test_feature_nll_locked_unweighted_mean():
    """Score combination is exactly profile_nll + (1/3) * sum of head
    NLLs -- hand-computed, no tolerance for a hidden weight."""
    torch.manual_seed(0)
    n = 4
    feat_log_probs = [torch.log_softmax(torch.randn(n, k), dim=1)
                      for k in FEATURE_BIN_SIZES]
    feat_bins = torch.stack([
        torch.randint(0, k, (n,)) for k in FEATURE_BIN_SIZES], dim=1)
    got = feature_nll(feat_log_probs, feat_bins)
    for i in range(n):
        expect = -sum(feat_log_probs[k][i, feat_bins[i, k]].item()
                      for k in range(NUM_FEATURES)) / NUM_FEATURES
        assert abs(got[i].item() - expect) < 1e-6
    # and combined_nll is the plain sum with the profile term
    x = torch.rand(n, PROFILE_DIM) * 5
    profile_log_probs = torch.log_softmax(torch.randn(n, PROFILE_DIM), dim=1)
    comb = combined_nll(profile_log_probs, x, feat_log_probs, feat_bins)
    assert torch.allclose(
        comb, profile_nll(profile_log_probs, x) + got, atol=1e-6)


def test_zero_memory_collapses_all_heads_to_type_prior():
    """With zeroed memory, every head's prediction depends only on the
    node's static type -- the ablation stays honest for E3."""
    torch.manual_seed(1)
    model = EnrichedBehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN)
    model.eval()
    x, y, edge_index, feat_bins = _toy_window(6)
    zero_mem = torch.zeros(6, model.memory_dim)
    out, feat_out, _ = model(type_onehot(y, NUM_TYPES), zero_mem, edge_index)
    for i in range(6):
        for j in range(6):
            if y[i] == y[j]:
                assert torch.allclose(out[i], out[j], atol=1e-6)
                for lp in feat_out:
                    assert torch.allclose(lp[i], lp[j], atol=1e-6)


def test_prediction_independent_of_observed_profile_and_features():
    """Neither data.x nor data.feat_bins reaches forward(): predictions
    are a function of (type, memory, edges) only. Structurally, forward
    has no argument for them; behaviorally, the score of a window is
    computed from targets the prediction provably never saw."""
    torch.manual_seed(2)
    model = EnrichedBehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN)
    model.eval()
    x, y, edge_index, feat_bins = _toy_window(5)
    memory = torch.randn(5, model.memory_dim)
    out1, feat1, h1 = model(type_onehot(y, NUM_TYPES), memory, edge_index)
    # change the current window's observations completely
    x2 = torch.rand_like(x) * 9
    feat_bins2 = torch.stack([
        torch.randint(0, k, (5,)) for k in FEATURE_BIN_SIZES], dim=1)
    out2, feat2, h2 = model(type_onehot(y, NUM_TYPES), memory, edge_index)
    assert torch.equal(out1, out2)
    assert all(torch.equal(a, b) for a, b in zip(feat1, feat2))
    assert torch.equal(h1, h2)
    # while the scores of course differ
    s1 = combined_nll(out1, x, feat1, feat_bins)
    s2 = combined_nll(out2, x2, feat2, feat_bins2)
    assert not torch.allclose(s1, s2)


def test_writeback_layout_and_detachment():
    """[h.detach() ; normalized profile ; one-hot bins], parameter-free."""
    torch.manual_seed(3)
    x, y, edge_index, feat_bins = _toy_window(5)
    h = torch.randn(5, HIDDEN, requires_grad=True)
    wb = make_writeback_e3(h, x, feat_bins)
    assert wb.shape == (5, HIDDEN + PROFILE_DIM + sum(FEATURE_BIN_SIZES))
    assert not wb.requires_grad
    assert torch.allclose(wb[:, :HIDDEN], h.detach())
    prof = wb[:, HIDDEN:HIDDEN + PROFILE_DIM]
    assert torch.allclose(prof.sum(dim=1), torch.ones(5), atol=1e-5)
    offset = HIDDEN + PROFILE_DIM
    for k, size in enumerate(FEATURE_BIN_SIZES):
        block = wb[:, offset:offset + size]
        assert torch.allclose(block.sum(dim=1), torch.ones(5))
        assert torch.equal(block.argmax(dim=1), feat_bins[:, k])
        offset += size


def test_memory_dim_consistency():
    """Model memory_dim matches the writeback width -- the contract the
    NodeMemory allocation in train/test relies on."""
    model = EnrichedBehaviorNet(NUM_TYPES, PROFILE_DIM, HIDDEN)
    x, y, edge_index, feat_bins = _toy_window(5)
    h = torch.randn(5, HIDDEN)
    wb = make_writeback_e3(h, x, feat_bins)
    assert wb.shape[1] == model.memory_dim


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} passed')


if __name__ == '__main__':
    _run_all()
