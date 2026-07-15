"""Unit tests for calibrate_within_type.py (pure-python, no torch)."""

from calibrate_within_type import load_scores, within_type_quantiles


def test_midrank_quantile_within_single_type():
    rows = [(1, 0.0, '0'), (2, 0.0, '0'), (3, 1.0, '0'), (4, 2.0, '1')]
    gid_type = {1: 'A', 2: 'A', 3: 'A', 4: 'A'}
    cal = within_type_quantiles(rows, gid_type)
    # ties share a mid-rank; strictly higher scores get strictly higher quantiles
    assert cal[1] == cal[2] == (0 + 0.5 * 2) / 4
    assert cal[3] == (2 + 0.5) / 4
    assert cal[4] == (3 + 0.5) / 4


def test_groups_are_independent():
    # identical raw scores land at different quantiles in different groups
    rows = [(1, 5.0, '0'), (2, 5.0, '0'), (3, 1.0, '0')]
    gid_type = {1: 'A', 2: 'B', 3: 'B'}
    cal = within_type_quantiles(rows, gid_type)
    assert cal[1] == 0.5 / 1          # alone in A
    assert cal[2] == (1 + 0.5) / 2    # top of B
    assert cal[3] == 0.5 / 2


def test_unknown_gids_form_their_own_group():
    rows = [(1, 3.0, '0'), (2, 4.0, '0')]
    cal = within_type_quantiles(rows, {1: 'A'})  # gid 2 unresolved
    assert cal[1] == 0.5 / 1
    assert cal[2] == 0.5 / 1  # alone in UNKNOWN, not mixed into A


def test_load_scores_preserves_order_and_flag(tmp_path):
    p = tmp_path / 'scores_windowed.txt'
    p.write_text('7 1.5 0\n9 0.0 1\n')
    rows = load_scores(str(p))
    assert rows == [(7, 1.5, '0'), (9, 0.0, '1')]
