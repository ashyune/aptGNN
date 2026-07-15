"""Correctness checks for evaluate_windowed.py's node-level scorer.

Run directly: python test_evaluate_windowed.py

Zero dependencies beyond the standard library and ranking_metrics.py, so
this runs anywhere -- including a torch-free sandbox. Every expected value
is derived by hand in the comment above the assertion.

Scores-file format under test: one line per node, '<gid> <score> <flag>'.
"""

import os
import tempfile

from evaluate_windowed import score_node_file, score_alarm_file

TOL = 1e-6


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)


def _run(scores_text, gt_text, **kw):
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, 'scores.txt')
        gp = os.path.join(d, 'gt.txt')
        _write(sp, scores_text)
        _write(gp, gt_text)
        return score_node_file(sp, gp, **kw)


# ---------------------------------------------------------------------------
# Operating point + ranking, all GT scored
# ---------------------------------------------------------------------------

def test_mixed_case_operating_point_and_ranking():
    # gid score flag  | label  outcome
    #  1  0.9   1     |  GT    TP
    #  2  0.7   1     |  --    FP
    #  3  0.5   0     |  GT    FN
    #  4  0.3   0     |  --    TN
    #  5  0.1   0     |  --    TN
    scores = '1 0.9 1\n2 0.7 1\n3 0.5 0\n4 0.3 0\n5 0.1 0\n'
    gt = '1\n3\n'
    r = _run(scores, gt)

    # operating point
    assert r['tp'] == 1 and r['fp'] == 1 and r['fn'] == 1 and r['tn'] == 2, r
    assert abs(r['precision'] - 0.5) < TOL, r      # 1/(1+1)
    assert abs(r['recall'] - 0.5) < TOL, r         # 1/(1+1)
    assert abs(r['f1'] - 0.5) < TOL, r
    assert abs(r['fpr'] - 1.0 / 3) < TOL, r        # 1/(1+2)

    # ranking: scores [0.9,0.7,0.5,0.3,0.1] labels [1,0,1,0,0]
    # AUROC pair count = 5/6 ; AUPRC = 0.5 + 1/3 ; P@2 = 1/2
    assert abs(r['auroc'] - 5.0 / 6) < TOL, r
    assert abs(r['auprc'] - (0.5 + 1.0 / 3)) < TOL, r
    assert abs(r['precision_at_k'] - 0.5) < TOL, r
    assert r['n_gt_unscored'] == 0, r
    print('PASS: mixed_case_operating_point_and_ranking')


def test_perfect_detection():
    # positives hold the two highest scores and are the only flags
    scores = '1 0.9 1\n2 0.8 1\n3 0.2 0\n4 0.1 0\n'
    gt = '1\n2\n'
    r = _run(scores, gt)
    assert r['tp'] == 2 and r['fp'] == 0 and r['fn'] == 0 and r['tn'] == 2, r
    assert abs(r['precision'] - 1.0) < TOL and abs(r['recall'] - 1.0) < TOL, r
    assert abs(r['fpr'] - 0.0) < TOL, r
    assert abs(r['auroc'] - 1.0) < TOL, r
    assert abs(r['auprc'] - 1.0) < TOL, r
    assert abs(r['precision_at_k'] - 1.0) < TOL, r
    print('PASS: perfect_detection')


# ---------------------------------------------------------------------------
# Unscored ground truth (coverage)
# ---------------------------------------------------------------------------

def test_unscored_gt_counted_as_missed_by_default():
    # GT = {1,2,3}. Only gid 1 appears (flagged). gid 4 is a benign flag.
    # gids 2 and 3 never appear -> unscored GT.
    scores = '1 0.9 1\n4 0.5 1\n5 0.1 0\n'
    gt = '1\n2\n3\n'
    r = _run(scores, gt)

    # operating point (default includes unscored GT as FN)
    assert r['n_gt_unscored'] == 2, r
    assert r['tp'] == 1, r          # gid 1
    assert r['fp'] == 1, r          # gid 4
    assert r['tn'] == 1, r          # gid 5
    assert r['fn'] == 2, r          # gids 2,3 never seen
    assert abs(r['precision'] - 0.5) < TOL, r     # 1/(1+1)
    assert abs(r['recall'] - 1.0 / 3) < TOL, r    # 1/(1+2)
    print('PASS: unscored_gt_counted_as_missed_by_default')


def test_exclude_unscored_gt_toggle():
    # Same input; excluding unscored GT should NOT count 2,3 as FN.
    scores = '1 0.9 1\n4 0.5 1\n5 0.1 0\n'
    gt = '1\n2\n3\n'
    r = _run(scores, gt, include_unscored_gt=False)
    assert r['fn'] == 0, r
    assert abs(r['recall'] - 1.0) < TOL, r        # 1/(1+0)
    assert r['n_gt_unscored'] == 2, r             # still reported
    print('PASS: exclude_unscored_gt_toggle')


# ---------------------------------------------------------------------------
# Degenerate / guard cases
# ---------------------------------------------------------------------------

def test_no_flags_at_all():
    # nothing flagged: every GT node is FN, formulas stay finite
    scores = '1 0.9 0\n2 0.5 0\n3 0.1 0\n'
    gt = '1\n'
    r = _run(scores, gt)
    assert r['tp'] == 0 and r['fp'] == 0, r
    assert r['fn'] == 1, r
    assert r['recall'] < TOL, r
    assert r['precision'] < TOL, r
    print('PASS: no_flags_at_all')


def test_zero_score_node_is_not_dropped():
    # a node whose only score is 0.0 must still be in the population
    scores = '1 0.0 0\n2 0.9 1\n'
    gt = '2\n'
    r = _run(scores, gt)
    assert r['n_scored'] == 2, r
    assert r['tp'] == 1 and r['tn'] == 1, r
    print('PASS: zero_score_node_is_not_dropped')


# ---------------------------------------------------------------------------
# Legacy scorer still importable and functional (used by old diagnostics)
# ---------------------------------------------------------------------------

def test_legacy_score_alarm_file_still_works():
    with tempfile.TemporaryDirectory() as d:
        ap = os.path.join(d, 'alarm.txt')
        gp = os.path.join(d, 'gt.txt')
        _write(gp, '1\n2\n')
        _write(ap, '5\n\n1:\n\n2:\n')
        res = score_alarm_file(ap, gp)
        assert res['tp'] == 2 and res['fp'] == 0 and res['fn'] == 0, res
    print('PASS: legacy_score_alarm_file_still_works')


if __name__ == '__main__':
    test_mixed_case_operating_point_and_ranking()
    test_perfect_detection()
    test_unscored_gt_counted_as_missed_by_default()
    test_exclude_unscored_gt_toggle()
    test_no_flags_at_all()
    test_zero_score_node_is_not_dropped()
    test_legacy_score_alarm_file_still_works()
    print('All evaluate_windowed checks passed.')