# Operating-point metrics (precision/recall/F1/MCC) vs Orthrus & Velox — 2026-07-16

Purpose: our headline metric is AUPRC (area under the full precision–recall curve), but
Orthrus and Velox (Bilot et al.) report single-threshold node-level precision on raw
entity GT for E3-CADETS. This report converts our existing E1/E2 seedrun outputs into the
same unit — TP/FP/FN counts, precision, recall, F1, MCC at a concrete operating threshold —
with no new training and no code changes. Inputs: the uncalibrated headline score files
(`scores_windowed.txt`) from `models/behavior_cadets/seedrun/` (E1 mem, SAGE) and
`models/behavior_cadets/e2seedrun/` (E2 relational), seeds {101, 202, 303}, scored against
the raw 46-node GT (`models/windowed_cadets/groundtruth_raw_global_id.txt`). 357,174 nodes
scored per arm; all 46 GT nodes scored in every arm.

## Threshold selection — why neither operating point is label-fitting

Two operating points, both fixed without ever looking at test labels:

- **A. Train-quantile flag (the existing flag column).** `test_behavior.py` sets the
  threshold at the 99.9th nearest-rank percentile of the **training-window** score
  distribution (`--flag-quantile 0.999`) and flags every test node at or above it. The
  threshold is a function of benign training scores only — it is the "calibrate on
  held-out non-test data" option. This is the model's own deployed alarm rule.
- **B. Top-K alarm budget, K = |raw GT| = 46.** Rank all 357,174 test nodes by score and
  alarm on the top 46. Uses only the *count* of GT entities (a budget assumption also made
  in P@|GT|-style reporting), never their identities. Equivalent to the P@46 already in the
  headline tables, here expanded into a full confusion matrix.

Explicitly **not** used: any threshold tuned to maximize precision/F1/MCC on the 46 test
labels. That would be label-fitting and would overstate every number below.

Tie handling for top-K: scores sorted descending with deterministic id tie-break; two arms
had score ties straddling rank 46 (E1 s101: 33 tied nodes; E2 s303: 2), but **no GT node
is in either tied group**, so TP is tie-invariant in all six arms.

Calibrated (within-type) scores are not re-reported here: calibration hurt both arms on
every seed (2026-07-16 E2 report §3) and the headline is uncalibrated.

## Operating point A — train 99.9th-percentile flag threshold

| seed | arm | alarms | TP | FP | FN | precision | recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 101 | E1 mem | 495 | 7 | 488 | 39 | 0.0141 | 0.152 | 0.026 | 0.046 |
| 101 | E2 rel | 498 | 6 | 492 | 40 | 0.0120 | 0.130 | 0.022 | 0.039 |
| 202 | E1 mem | 517 | 10 | 507 | 36 | 0.0193 | 0.217 | 0.036 | 0.065 |
| 202 | E2 rel | 493 | 7 | 486 | 39 | 0.0142 | 0.152 | 0.026 | 0.046 |
| 303 | E1 mem | 518 | 8 | 510 | 38 | 0.0154 | 0.174 | 0.028 | 0.051 |
| 303 | E2 rel | 527 | 6 | 521 | 40 | 0.0114 | 0.130 | 0.021 | 0.038 |

TP counts match the "flag TP/46" column of the 2026-07-16 E2 report exactly (cross-check
passed). TN = 357,174 − 46 − FP in every row.

## Operating point B — top-46 alarm budget (K = |raw GT|)

| seed | arm | rank-46 score | TP | FP | FN | precision | recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 101 | E1 mem | 4.904 | 3 | 43 | 43 | 0.0652 | 0.0652 | 0.0652 | 0.0651 |
| 101 | E2 rel | 4.636 | 3 | 43 | 43 | 0.0652 | 0.0652 | 0.0652 | 0.0651 |
| 202 | E1 mem | 4.827 | 3 | 43 | 43 | 0.0652 | 0.0652 | 0.0652 | 0.0651 |
| 202 | E2 rel | 4.522 | 3 | 43 | 43 | 0.0652 | 0.0652 | 0.0652 | 0.0651 |
| 303 | E1 mem | 4.817 | 2 | 44 | 44 | 0.0435 | 0.0435 | 0.0435 | 0.0434 |
| 303 | E2 rel | 4.501 | 3 | 43 | 43 | 0.0652 | 0.0652 | 0.0652 | 0.0651 |

With K = |GT|, precision = recall = F1 by construction, and precision equals the P@46
already reported. This is our best label-free operating point: it dominates the flag rule
on precision, F1, and MCC in every arm.

## Side by side with Orthrus and Velox (E3-CADETS, node level, raw entity GT)

| system | operating point | TP | FP | FN | precision | recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| Orthrus (paper) | their detection threshold | 25 | 23 | 43 | 0.52 | 0.368 | 0.43 | 0.44 |
| Velox (paper) | best case | — | — | — | 0.85 | — | — | — (ADP 0.94) |
| E1 mem, best seed (202) | top-46 budget | 3 | 43 | 43 | 0.065 | 0.065 | 0.065 | 0.065 |
| E1 mem, worst seed (303) | top-46 budget | 2 | 44 | 44 | 0.043 | 0.043 | 0.043 | 0.043 |
| E2 rel, all seeds | top-46 budget | 3 | 43 | 43 | 0.065 | 0.065 | 0.065 | 0.065 |
| E1 mem, best seed (202) | train-quantile flag | 10 | 507 | 36 | 0.019 | 0.217 | 0.036 | 0.065 |
| E2 rel, best seed (202) | train-quantile flag | 7 | 486 | 39 | 0.014 | 0.152 | 0.026 | 0.046 |

(Orthrus recall/F1 derived from their published counts: R = 25/68, F1 = 2PR/(P+R). Velox
reports best-case precision and Attack Detection Precision without a comparable count
table.)

**Verdict: at any honestly-chosen threshold we are roughly an order of magnitude below
Orthrus on precision (0.04–0.07 vs 0.52) and MCC (0.04–0.07 vs 0.44), on both arms and
all seeds.** The AUPRC headline (~0.012–0.021 vs random 0.0001, AUROC ~0.93–0.95) shows
the score carries real signal, but the ranking concentrates far too few of the 46
entities at the very top to be competitive at a deployable operating point. Recall at
matched recall tells the same story: to reach Orthrus's recall (0.37 → 17/46 here), the
best arm (E1 s101) needs the top 2,849 alarms (precision 0.0060); the other five arms need
4,295–5,075 (precision 0.0033–0.0040).

## Footing caveats (read before quoting)

1. **GT denominators differ.** Both use the KAIROS/Orthrus entity label set, but
   Orthrus's E3-CADETS arithmetic implies 68 attack nodes (TP 25 + FN 43); our capture
   resolves 46 of the 72 unique UUIDs, because our test shard covers 2018-04-11 → 04-13
   02:24 (the 04-06 attack falls between our train/test captures and 04-13 mostly after —
   see `groundtruth/cadets_raw_SOURCE.md`). Same label family, not identical node sets.
2. **MCC is population-dependent.** Ours is computed over 357,174 scored test nodes;
   Orthrus's graph/candidate population differs. At these effect sizes the ~order-of-
   magnitude gap is robust to that, but the third decimal is not meaningful.
3. **Threshold provenance differs.** Orthrus tunes its detection threshold per its own
   validation protocol; Velox's 0.85 is explicitly best-case. Our two operating points are
   conservative (never touch test labels). This asymmetry favors them, but no label-free
   threshold on our scores comes close to closing a 10× gap — max precision anywhere on
   our PR curves is bounded by the AUPRC evidence already reported.

## Bottom line for the tuning-sweep decision

The apples-to-apples numbers confirm the gap AUPRC implied: E1/E2 are ~0.04–0.07
precision/MCC at label-free operating points vs Orthrus 0.52/0.44 and Velox 0.85
(best-case). No seed and no arm is an outlier — the gap is structural (ranking quality in
the extreme top of the score distribution), not a threshold-choice artifact.
