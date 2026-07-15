"""Correctness checks for the pure-Python helpers in test_windowed.py.

Covers:
- build_whole_file_adjacency / two_hop_neighbors (kept for the diagnostic
  scripts; still exercised here so a future edit can't silently break them);
- score_nodes (Stage 2: negative-log-prob anomaly score);
- update_max_scores (Stage 2: cross-window MAX aggregation);
- write_scores_file (Stage 2: the scores_windowed.txt line format).

Run directly: python test_test_windowed.py

Needs torch AND torch_geometric to import at all -- test_windowed.py
imports SAGENetWithMemory from train_windowed.py at module level. Run it in
the `threatrace` conda environment; it cannot run in a torch-free sandbox.
The score_nodes test needs torch specifically; the other Stage 2 helpers
are pure Python and their logic was additionally verified standalone.
"""

import math
import os
import tempfile

import torch
import torch.nn.functional as F

from test_windowed import (
    build_whole_file_adjacency,
    two_hop_neighbors,
    score_nodes,
    update_max_scores,
    write_scores_file,
)


def _write(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Adjacency / 2-hop (unchanged behaviour, still guarded)
# ---------------------------------------------------------------------------

def test_adjacency_keyed_by_global_id():
    node_vocab = {'uuid-A': 10, 'uuid-B': 11, 'uuid-C': 12, 'uuid-D': 13}
    lines = [
        "uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t100\n",
        "uuid-B\tFile\tuuid-C\tFile\tEVENT_WRITE\t200\n",
        "uuid-C\tFile\tuuid-D\tNetFlow\tEVENT_CONNECT\t300\n",
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, lines)
        adj, adj2 = build_whole_file_adjacency(path, node_vocab)
        assert adj[11] == [10]
        assert adj[12] == [11]
        assert adj[13] == [12]
        assert adj2[10] == [11]
        assert adj2[11] == [12]
        assert adj2[12] == [13]
    print('PASS: adjacency_keyed_by_global_id')


def test_unresolvable_uuid_rows_are_skipped():
    node_vocab = {'uuid-A': 1, 'uuid-B': 2}
    lines = [
        "uuid-A\tProcess\tuuid-B\tFile\tEVENT_READ\t100\n",
        "uuid-A\tProcess\tuuid-UNKNOWN\tFile\tEVENT_WRITE\t200\n",
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fixture.txt')
        _write(path, lines)
        adj, adj2 = build_whole_file_adjacency(path, node_vocab)
        assert adj2[1] == [2]
        assert 1 not in adj
        assert len(adj2[1]) == 1
    print('PASS: unresolvable_uuid_rows_are_skipped')


def test_two_hop_neighbors_is_symmetric_both_directions():
    adj = {1: [6], 6: [5]}
    adj2 = {1: [2], 2: [3]}
    result = two_hop_neighbors(1, adj, adj2)
    assert result == {6, 5, 2, 3}, result
    print('PASS: two_hop_neighbors_is_symmetric_both_directions')


def test_two_hop_neighbors_empty_for_isolated_node():
    adj, adj2 = {}, {}
    assert two_hop_neighbors(42, adj, adj2) == set()
    print('PASS: two_hop_neighbors_empty_for_isolated_node')


# ---------------------------------------------------------------------------
# Stage 2: continuous score
# ---------------------------------------------------------------------------

def test_score_nodes_negative_log_prob():
    """Score = -log p(true type). Direction and the exact uniform case.

    Node 0: confident in the true class -> surprise near 0.
    Node 1: uniform logits -> p(true)=1/3 -> surprise = log 3.
    Node 2: confident in the WRONG class -> large surprise.
    """
    logits = torch.tensor([[10.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0]])
    out = F.log_softmax(logits, dim=1)
    y = torch.tensor([0, 0, 0])  # true type = class 0 for all three
    scores = score_nodes(out, y)

    assert scores[0].item() < scores[1].item() < scores[2].item()
    assert abs(scores[1].item() - math.log(3.0)) < 1e-5
    assert scores[0].item() < 1e-3
    print('PASS: score_nodes_negative_log_prob')


def test_score_nodes_length_matches_input():
    out = F.log_softmax(torch.randn(7, 4), dim=1)
    y = torch.randint(0, 4, (7,))
    assert score_nodes(out, y).shape[0] == 7
    print('PASS: score_nodes_length_matches_input')


# ---------------------------------------------------------------------------
# Stage 2: cross-window MAX aggregation
# ---------------------------------------------------------------------------

def test_update_max_scores_keeps_maximum():
    sbg = {}
    update_max_scores(sbg, [1, 2, 3], [0.5, 0.9, 0.1])
    assert sbg == {1: 0.5, 2: 0.9, 3: 0.1}
    # node 1 reappears higher -> updated; node 2 lower -> kept; node 4 new
    update_max_scores(sbg, [1, 2, 4], [0.8, 0.4, 0.2])
    assert sbg == {1: 0.8, 2: 0.9, 3: 0.1, 4: 0.2}
    print('PASS: update_max_scores_keeps_maximum')


def test_update_max_scores_first_write_wins_when_alone():
    sbg = {}
    update_max_scores(sbg, [99], [0.0])
    assert sbg == {99: 0.0}  # a zero score must still register, not be skipped
    print('PASS: update_max_scores_first_write_wins_when_alone')


# ---------------------------------------------------------------------------
# Stage 2: scores file format
# ---------------------------------------------------------------------------

def test_write_scores_file_format_and_flags():
    sbg = {10: 0.5, 11: 1.25, 12: 0.0}
    flagged = {11}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'scores.txt')
        write_scores_file(path, sbg, flagged)
        lines = open(path).read().splitlines()
    parsed = {}
    for ln in lines:
        gid, score, flag = ln.split(' ')
        parsed[int(gid)] = (float(score), int(flag))
    assert parsed[10] == (0.5, 0)
    assert parsed[11] == (1.25, 1)   # only node 11 was flagged
    assert parsed[12] == (0.0, 0)
    assert len(parsed) == 3          # every node written, flagged or not
    print('PASS: write_scores_file_format_and_flags')


if __name__ == '__main__':
    test_adjacency_keyed_by_global_id()
    test_unresolvable_uuid_rows_are_skipped()
    test_two_hop_neighbors_is_symmetric_both_directions()
    test_two_hop_neighbors_empty_for_isolated_node()
    test_score_nodes_negative_log_prob()
    test_score_nodes_length_matches_input()
    test_update_max_scores_keeps_maximum()
    test_update_max_scores_first_write_wins_when_alone()
    test_write_scores_file_format_and_flags()
    print('All test_windowed helper checks passed.')