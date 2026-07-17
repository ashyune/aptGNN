"""Extension 3: cheap per-node per-window statistical features.

Computes, for every node in every window, the three features the
2026-07-17 enrichment survey (reports/2026-07-17_feature_enrichment_survey
.md, probe: diagnose_feature_separation.py) measured as carrying marginal
signal for the raw CADETS ground truth:

  n_distinct : distinct counterpart uuids this window (either role)
  rep_ratio  : total events / n_distinct (partner re-use)
  span_frac  : (node last ts - node first ts) / window span

Each feature is discretized into a small categorical bin so it can be
(a) predicted by its own softmax head and scored by NLL exactly like the
edge-type profile, and (b) appended one-hot to the parameter-free memory
writeback. The bin edges below are PREREGISTERED constants of the E3
experiment -- they are deliberately not exposed as CLI flags anywhere;
changing them is a new experiment, not a knob (project rule: no
hyperparameter tuning against the 46-label GT).

This module is additive: windowed_data.py is untouched. Features are
computed from the same window_rows that build_window_data consumes, under
the SAME row filter (drop rows whose src/dst/edge type is missing from
the train vocab), keyed by uuid, and then aligned onto the window's Data
object through data.global_id -- so alignment holds by construction and a
missing node fails loudly instead of silently misaligning.
"""

import math

import torch

from windowing import generate_windows, show
from windowed_data import build_window_data

# Preregistered bin layout (2026-07-17). Sizes are consumed by the E3
# model (head widths + writeback one-hot dims) and by feature_nll.
N_DISTINCT_BINS = 13   # log2 bins: 1, 2-3, 4-7, ..., >=4096
REP_RATIO_BINS = 13    # log2 bins, same edges (ratio >= 1 always)
SPAN_FRAC_BINS = 11    # 10 linear bins over [0, 1) + dedicated ==1.0 bin
FEATURE_BIN_SIZES = (N_DISTINCT_BINS, REP_RATIO_BINS, SPAN_FRAC_BINS)
NUM_FEATURES = len(FEATURE_BIN_SIZES)


def bin_n_distinct(v):
    """v >= 1 (a node exists only via at least one edge)."""
    return min(int(math.log2(v)), N_DISTINCT_BINS - 1)


def bin_rep_ratio(v):
    """v >= 1.0 (events >= distinct partners by definition)."""
    return min(int(math.log2(v)), REP_RATIO_BINS - 1)


def bin_span_frac(v):
    """v in [0, 1]; the ==1.0 bin is dedicated (survey: only 78 test
    nodes ever span a full window, 7 of them GT)."""
    if v >= 1.0:
        return SPAN_FRAC_BINS - 1
    return min(int(v * (SPAN_FRAC_BINS - 1)), SPAN_FRAC_BINS - 2)


def compute_window_feature_bins(window_rows, feature_map, label_map):
    """Per-uuid feature bins for one window.

    Applies exactly the row filter build_window_data applies (skip rows
    whose src type, dst type, or edge type is absent from the train
    vocab), so the set of uuids returned equals the set of nodes in the
    window's Data object. The window span is taken over the FILTERED
    rows, matching the survey probe.

    Returns dict[str, tuple(int, int, int)]:
        uuid -> (n_distinct bin, rep_ratio bin, span_frac bin)
    """
    events = {}
    partners = {}
    first_ts = {}
    last_ts = {}
    w_first = None
    w_last = None
    for src, src_type, dst, dst_type, edge_type, ts in window_rows:
        if src_type not in label_map:
            continue
        if dst_type not in label_map:
            continue
        if edge_type not in feature_map:
            continue
        if w_first is None:
            w_first = ts
        w_last = ts
        for node, other in ((src, dst), (dst, src)):
            events[node] = events.get(node, 0) + 1
            partners.setdefault(node, set()).add(other)
            if node not in first_ts:
                first_ts[node] = ts
            last_ts[node] = ts
    wspan = (w_last - w_first) if (w_first is not None
                                   and w_last > w_first) else 0
    bins = {}
    for node, cnt in events.items():
        nd = len(partners[node])
        span = ((last_ts[node] - first_ts[node]) / wspan) if wspan > 0 else 0.0
        bins[node] = (bin_n_distinct(nd),
                      bin_rep_ratio(cnt / nd),
                      bin_span_frac(span))
    return bins


def feature_bins_tensor(data, uuid_bins, gid_to_uuid):
    """[num_local_nodes, NUM_FEATURES] LongTensor aligned with data.x rows.

    Row i holds the bins of the node at local index i, resolved through
    data.global_id -- never through iteration order. Every local node
    must have bins (same rows, same filter); a miss raises KeyError.
    """
    out = torch.zeros((data.num_nodes, NUM_FEATURES), dtype=torch.long)
    for i, gid in enumerate(data.global_id.tolist()):
        out[i] = torch.tensor(uuid_bins[gid_to_uuid[gid]], dtype=torch.long)
    return out


def build_enriched_dataset(path, node_vocab, feature_map, label_map,
                           window_size):
    """build_windowed_dataset + a data.feat_bins tensor per window.

    Returns the same chronological list of Data objects the E1/E2 stack
    consumes, each carrying an extra feat_bins field. build_window_data
    is imported, not modified.
    """
    gid_to_uuid = {gid: uuid for uuid, gid in node_vocab.items()}
    windows = generate_windows(path, window_size)
    dataset = []
    for i, window_rows in enumerate(windows):
        data = build_window_data(window_rows, node_vocab, feature_map,
                                 label_map)
        uuid_bins = compute_window_feature_bins(window_rows, feature_map,
                                                label_map)
        data.feat_bins = feature_bins_tensor(data, uuid_bins, gid_to_uuid)
        show(f'Window {i}: {data.num_nodes:,} nodes, '
             f'{data.edge_index.size(1):,} edges (incl. self-loops), '
             f'feature bins attached')
        dataset.append(data)
    return dataset
