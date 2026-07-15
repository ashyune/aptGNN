"""Threshold-free ranking metrics for node-level anomaly scoring.

Pure standard-library Python -- no torch, no numpy, no sklearn -- matching
the zero-dependency convention of evaluate_darpatc.py / evaluate_windowed.py,
so scoring can run anywhere, including outside the GNN environment.

All three metrics take two equal-length sequences:
    scores : per-node continuous anomaly score (higher = more anomalous)
    labels : per-node ground-truth label (1 = malicious, 0 = benign)

Convention: a HIGHER score means "more anomalous / more likely positive".
The node scores produced by test_windowed.py (Stage 2) are negative
log-probabilities of each node's true CDM type -- higher means the model
was more surprised by that node, i.e. more anomalous.

Tie handling is explicit in every function. The real node scores will
contain many ties (e.g. large numbers of never-surprising nodes sharing a
near-identical low score), and a naive implementation would produce
order-dependent, irreproducible numbers. Each function documents exactly
how ties are resolved.
"""


def _check(scores, labels):
    """Validate shapes; return (n_pos, n_neg)."""
    if len(scores) != len(labels):
        raise ValueError(
            f'scores has {len(scores)} entries but labels has {len(labels)}'
        )
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    return n_pos, n_neg


def auroc(scores, labels):
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    Equals P(score(random positive) > score(random negative)), counting
    ties as 0.5. Computed from average ranks rather than by sweeping
    thresholds, so ties are handled exactly and there is no
    curve-resolution parameter.

    Returns None when undefined (no positives, or no negatives).
    """
    n_pos, n_neg = _check(scores, labels)
    if n_pos == 0 or n_neg == 0:
        return None

    # Ascending sort; assign average (1-indexed) ranks within tied groups.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while (j + 1 < len(order)
               and scores[order[j + 1]] == scores[order[i]]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # mean of the 1-indexed positions
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[idx] for idx in range(len(labels))
                       if labels[idx] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auprc(scores, labels):
    """Average precision (area under the precision-recall curve).

    Uses the step-wise average-precision definition
        AP = sum_t (recall_t - recall_{t-1}) * precision_t
    with thresholds placed at each distinct score and evaluated high to
    low. This is the standard AP (it matches sklearn's
    average_precision_score) and is preferred over trapezoidal PR
    integration, which is optimistically biased. Ties are handled by
    consuming all examples at the same score into one threshold step
    before reading precision/recall.

    Returns None when undefined (no positives).
    """
    n_pos, _ = _check(scores, labels)
    if n_pos == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    ap = 0.0
    prev_recall = 0.0
    tp = 0
    fp = 0
    i = 0
    while i < len(order):
        j = i
        while (j + 1 < len(order)
               and scores[order[j + 1]] == scores[order[i]]):
            j += 1
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1

    return ap


def precision_at_k(scores, labels, k):
    """Fraction of the top-k highest-scoring nodes that are positive.

    With k = |ground truth| this is an interpretable operating point:
    "if I raise exactly |GT| alarms, ranked by score, how many are right?"

    Ties straddling the k-th position are resolved by the sort order and
    so are mildly order-dependent; negligible for large k, but it is why
    this is a reported convenience rather than a primary metric.

    Returns None when k <= 0 or k > len(scores).
    """
    n = len(scores)
    _check(scores, labels)
    if k <= 0 or k > n:
        return None
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    hits = sum(1 for idx in order[:k] if labels[idx] == 1)
    return hits / k
