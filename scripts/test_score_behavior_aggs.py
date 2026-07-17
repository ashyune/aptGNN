"""Structural tests for score_behavior_aggs.per_event_nll_aggs.

Run with plain python (pytest not installed in the threatrace env):
    python test_score_behavior_aggs.py
"""

import math

import torch

from score_behavior_aggs import per_event_nll_aggs
from train_behavior import profile_nll


def brute_force(log_probs_row, x_row, q=0.95):
    """Expand the event multiset explicitly and aggregate."""
    events = []
    for k, cnt in enumerate(x_row.tolist()):
        events.extend([-log_probs_row[k].item()] * int(cnt))
    if not events:
        return 0.0, 0.0, 0.0
    events_sorted = sorted(events)
    rank = max(1, math.ceil(q * len(events)))
    return (sum(events) / len(events), max(events),
            events_sorted[rank - 1])


def random_case(n, d, seed, max_count=6, sparsity=0.7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, d, generator=g)
    log_probs = torch.log_softmax(logits, dim=1)
    x = torch.randint(0, max_count + 1, (n, d), generator=g).float()
    x = x * (torch.rand(n, d, generator=g) > sparsity).float()
    # Guarantee at least one event per node except node 0 (zero-profile
    # defensive path).
    for i in range(1, n):
        if x[i].sum() == 0:
            x[i, i % d] = 1.0
    x[0] = 0.0
    return log_probs, x


def test_matches_brute_force():
    log_probs, x = random_case(64, 56, seed=7)
    aggs = per_event_nll_aggs(log_probs, x)
    for i in range(x.size(0)):
        bf_mean, bf_max, bf_p95 = brute_force(log_probs[i], x[i])
        assert abs(aggs['mean'][i].item() - bf_mean) < 1e-5, i
        assert abs(aggs['max'][i].item() - bf_max) < 1e-6, i
        assert abs(aggs['p95'][i].item() - bf_p95) < 1e-6, i
    print('test_matches_brute_force OK (64 nodes x 56 categories)')


def test_mean_is_exactly_profile_nll():
    log_probs, x = random_case(128, 56, seed=11)
    aggs = per_event_nll_aggs(log_probs, x)
    ref = profile_nll(log_probs, x)
    assert torch.equal(aggs['mean'], ref)
    print('test_mean_is_exactly_profile_nll OK (bitwise)')


def test_single_event_all_aggs_equal():
    g = torch.Generator().manual_seed(3)
    log_probs = torch.log_softmax(torch.randn(10, 56, generator=g), dim=1)
    x = torch.zeros(10, 56)
    for i in range(10):
        x[i, (3 * i) % 56] = 1.0
    aggs = per_event_nll_aggs(log_probs, x)
    assert torch.allclose(aggs['mean'], aggs['max'])
    assert torch.allclose(aggs['p95'], aggs['max'])
    print('test_single_event_all_aggs_equal OK')


def test_zero_profile_scores_zero():
    log_probs = torch.log_softmax(torch.randn(1, 56), dim=1)
    x = torch.zeros(1, 56)
    aggs = per_event_nll_aggs(log_probs, x)
    for agg, v in aggs.items():
        assert v.item() == 0.0, agg
    print('test_zero_profile_scores_zero OK')


def test_max_dominates_p95_dominates_mean_ordering():
    # max >= p95 always; p95 >= per-node min, and mean <= max.
    log_probs, x = random_case(200, 56, seed=23)
    aggs = per_event_nll_aggs(log_probs, x)
    assert (aggs['max'] >= aggs['p95'] - 1e-6).all()
    assert (aggs['max'] >= aggs['mean'] - 1e-6).all()
    print('test_max_dominates_p95_dominates_mean_ordering OK')


if __name__ == '__main__':
    test_matches_brute_force()
    test_mean_is_exactly_profile_nll()
    test_single_event_all_aggs_equal()
    test_zero_profile_scores_zero()
    test_max_dominates_p95_dominates_mean_ordering()
    print('ALL TESTS PASSED')
