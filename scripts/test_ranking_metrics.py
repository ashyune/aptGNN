"""Tests for ranking_metrics.py.

Every expected value below is computed BY HAND in the comment above the
assertion, so a failure points at a specific arithmetic claim rather than
at a vague "number changed". Runs with plain asserts and no test framework
(pure standard library), so it executes in any environment.
"""

from ranking_metrics import auroc, auprc, precision_at_k

TOL = 1e-9


def approx(a, b, tol=TOL):
    return a is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_perfect_separation():
    # positives hold the two highest scores -> AUROC = 1.0
    assert approx(auroc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]), 1.0)


def test_auroc_perfect_inversion():
    # positives hold the two lowest scores -> AUROC = 0.0
    assert approx(auroc([0.9, 0.8, 0.3, 0.1], [0, 0, 1, 1]), 0.0)


def test_auroc_all_ties():
    # all scores equal -> every pair is a tie counted 0.5 -> AUROC = 0.5
    assert approx(auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)


def test_auroc_mixed_with_boundary_tie():
    # scores 0.9,0.6,0.6,0.2  labels 1,0,1,0
    # pairs (pos,neg): (0.9,0.6)=1 (0.9,0.2)=1 (0.6,0.6)=0.5 (0.6,0.2)=1
    # sum=3.5 over 2*2=4 pairs -> 0.875
    assert approx(auroc([0.9, 0.6, 0.6, 0.2], [1, 0, 1, 0]), 0.875)


def test_auroc_undefined_no_negatives():
    assert auroc([0.9, 0.8], [1, 1]) is None


def test_auroc_undefined_no_positives():
    assert auroc([0.9, 0.8], [0, 0]) is None


# ---------------------------------------------------------------------------
# AUPRC (average precision)
# ---------------------------------------------------------------------------

def test_auprc_perfect_separation():
    # thresholds: @0.9 P=1,R=0.5 ; @0.8 P=1,R=1.0
    # AP = (0.5-0)*1 + (1.0-0.5)*1 = 1.0
    assert approx(auprc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]), 1.0)


def test_auprc_perfect_inversion():
    # @0.9 P=0,R=0 ; @0.8 P=0,R=0 ; @0.3 P=1/3,R=0.5 ; @0.1 P=0.5,R=1
    # AP = 0 + 0 + (0.5)*(1/3) + (0.5)*(0.5) = 1/6 + 1/4 = 0.4166666667
    assert approx(auprc([0.9, 0.8, 0.3, 0.1], [0, 0, 1, 1]), 1.0 / 6 + 0.25)


def test_auprc_boundary_tie():
    # scores 0.9,0.6,0.6,0.2  labels 1,0,1,0
    # @0.9      P=1,   R=0.5
    # @0.6 tie  tp=2,fp=1 -> P=2/3, R=1.0
    # AP = (0.5-0)*1 + (1.0-0.5)*(2/3) = 0.5 + 1/3 = 0.8333333333
    assert approx(auprc([0.9, 0.6, 0.6, 0.2], [1, 0, 1, 0]), 0.5 + 1.0 / 3)


def test_auprc_all_positives():
    # precision is 1 at every step, recall climbs 0->1 -> AP = 1.0
    assert approx(auprc([0.9, 0.8, 0.7], [1, 1, 1]), 1.0)


def test_auprc_undefined_no_positives():
    assert auprc([0.9, 0.8], [0, 0]) is None


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k_basic():
    # top 2 by score are 0.9(label 1) and 0.8(label 0) -> 1/2 = 0.5
    assert approx(precision_at_k([0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0], 2), 0.5)


def test_precision_at_k_full():
    # k = all 4 -> 2 positives / 4 = 0.5
    assert approx(precision_at_k([0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0], 4), 0.5)


def test_precision_at_k_out_of_range():
    assert precision_at_k([0.9, 0.8], [1, 0], 0) is None
    assert precision_at_k([0.9, 0.8], [1, 0], 3) is None


# ---------------------------------------------------------------------------
# shape validation
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    for fn in (auroc, auprc):
        try:
            fn([0.1, 0.2, 0.3], [1, 0])
            assert False, 'expected ValueError'
        except ValueError:
            pass


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'PASS  {t.__name__}')
    print(f'\n{len(tests)} tests passed')


if __name__ == '__main__':
    _run_all()
