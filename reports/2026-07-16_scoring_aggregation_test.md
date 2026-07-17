# Scoring-aggregation test: window MAX / p95 per-event NLL vs window MEAN — 2026-07-16

Hypothesis under test (from the operating-point comparison of the same date): the
per-window MEAN per-event NLL dilutes a few anomalous events inside a mostly-normal
window, and replacing it with a tail statistic (per-window MAX, or nearest-rank 95th
percentile, of the per-event NLL multiset) should recover the GT entities' ranks.
Success criterion implied by the diagnosis: the 24th-ranked GT node (rank 11,308–17,758
under mean) must move toward rank ≤ 46 for Orthrus-level precision at a 46-alarm budget.

No retraining: all three statistics are functions of the same frozen-checkpoint forward
pass (a node's window events are category draws from its 56-dim edge-type profile), so
this was a pure re-inference pass over all 6 seedrun arms (E1 mem SAGE + E2 relational,
seeds {101, 202, 303}). New additive scorer `scripts/score_behavior_aggs.py` (+ 5
structural tests in `scripts/test_score_behavior_aggs.py`, incl. brute-force multiset
verification; all pass); pipeline files untouched. Faithfulness check: the recomputed
mean score matched each arm's existing `scores_windowed.txt` bit-exactly (max |diff| = 0
over 357,174 nodes, all 6 arms). Per-aggregation flag thresholds recomputed from that
aggregation's train-window 99.9th-percentile score (same protocol as the headline). Raw
46-node GT, uncalibrated scores. Driver + log + outputs + analysis script:
`models/behavior_cadets/aggscoring/`.

## Verdict up front

**Null-to-negative. Tail aggregation does not close the rank gap and degrades every
operating-point metric.** Median GT rank stays ~10k–16k under all three aggregations
(best single improvement: E1 s202 p95, 12,828 → 9,942); the 24th GT node never ranks
better than 10,532 (needed: ≤ 46). AUPRC moves in opposite directions on the two arms
(p95 > mean on all 3 E1 seeds; p95 < mean on all 3 E2 seeds), so by the project standard
(reproducible across seeds *and* arms) the AUPRC effect is not adoptable. Operating-point
precision/MCC and P@46 get uniformly worse under max and worse-or-equal under p95.

## 1. Ranking metrics (raw GT, uncalibrated)

| seed | arm | agg | AUPRC | AUROC | P@46 |
|---|---|---|---|---|---|
| 101 | E1 | mean | 0.0161 | 0.9502 | 0.0652 (3) |
| 101 | E1 | max | 0.0144 | 0.9324 | 0.0435 (2) |
| 101 | E1 | p95 | 0.0193 | 0.9341 | 0.0652 (3) |
| 202 | E1 | mean | 0.0211 | 0.9509 | 0.0652 (3) |
| 202 | E1 | max | 0.0232 | 0.9320 | 0.0217 (1) |
| 202 | E1 | p95 | 0.0242 | 0.9317 | 0.0217 (1) |
| 303 | E1 | mean | 0.0129 | 0.9324 | 0.0435 (2) |
| 303 | E1 | max | 0.0234 | 0.9204 | 0.0217 (1) |
| 303 | E1 | p95 | 0.0251 | 0.9209 | 0.0217 (1) |
| 101 | E2 | mean | 0.0133 | 0.9296 | 0.0652 (3) |
| 101 | E2 | max | 0.0041 | 0.9243 | 0.0217 (1) |
| 101 | E2 | p95 | 0.0075 | 0.9213 | 0.0217 (1) |
| 202 | E2 | mean | 0.0123 | 0.9315 | 0.0652 (3) |
| 202 | E2 | max | 0.0050 | 0.9279 | 0.0217 (1) |
| 202 | E2 | p95 | 0.0064 | 0.9338 | 0.0217 (1) |
| 303 | E2 | mean | 0.0186 | 0.9293 | 0.0652 (3) |
| 303 | E2 | max | 0.0087 | 0.9221 | 0.0217 (1) |
| 303 | E2 | p95 | 0.0140 | 0.9251 | 0.0217 (1) |

Paired per-seed AUPRC delta (p95 − mean): E1 +0.0032 / +0.0031 / +0.0122;
E2 −0.0058 / −0.0059 / −0.0046. Same-signed within each arm, opposite-signed across
arms — an arm-dependent artifact, not a scoring-rule improvement.

## 2. Operating points (raw GT; flag = per-agg train-0.999-quantile threshold)

| seed | arm | agg | flag TP/FP | flag P | flag MCC | top-46 TP | top-46 P | top-46 MCC |
|---|---|---|---|---|---|---|---|---|
| 101 | E1 | mean | 7/488 | 0.0141 | 0.046 | 3 | 0.065 | 0.065 |
| 101 | E1 | max | 4/443 | 0.0089 | 0.028 | 2 | 0.043 | 0.043 |
| 101 | E1 | p95 | 5/418 | 0.0118 | 0.036 | 3 | 0.065 | 0.065 |
| 202 | E1 | mean | 10/507 | 0.0193 | 0.065 | 3 | 0.065 | 0.065 |
| 202 | E1 | max | 1/438 | 0.0023 | 0.007 | 1 | 0.022 | 0.022 |
| 202 | E1 | p95 | 2/410 | 0.0049 | 0.014 | 1 | 0.022 | 0.022 |
| 303 | E1 | mean | 8/510 | 0.0154 | 0.051 | 2 | 0.043 | 0.043 |
| 303 | E1 | max | 1/414 | 0.0024 | 0.007 | 1 | 0.022 | 0.022 |
| 303 | E1 | p95 | 4/628 | 0.0063 | 0.023 | 1 | 0.022 | 0.022 |
| 101 | E2 | mean | 6/492 | 0.0120 | 0.039 | 3 | 0.065 | 0.065 |
| 101 | E2 | max | 1/505 | 0.0020 | 0.006 | 1 | 0.022 | 0.022 |
| 101 | E2 | p95 | 4/556 | 0.0071 | 0.025 | 1 | 0.022 | 0.022 |
| 202 | E2 | mean | 7/486 | 0.0142 | 0.046 | 3 | 0.065 | 0.065 |
| 202 | E2 | max | 1/529 | 0.0019 | 0.006 | 1 | 0.022 | 0.022 |
| 202 | E2 | p95 | 3/459 | 0.0065 | 0.020 | 1 | 0.022 | 0.022 |
| 303 | E2 | mean | 6/521 | 0.0114 | 0.038 | 3 | 0.065 | 0.065 |
| 303 | E2 | max | 1/466 | 0.0021 | 0.006 | 1 | 0.022 | 0.022 |
| 303 | E2 | p95 | 3/439 | 0.0068 | 0.021 | 1 | 0.022 | 0.022 |

Mean wins or ties every row of this table. Under max, five of six arms keep exactly one
GT node in the top 46 (a single entity rockets to rank 1–8, see below) and lose the rest.

## 3. GT rank distribution — the quantity the hypothesis had to move

| seed | arm | agg | median GT rank | 24th GT rank (needs ≤46) | best GT rank |
|---|---|---|---|---|---|
| 101 | E1 | mean | 11,306 | 11,308 | 5 |
| 101 | E1 | max | 11,624 | 11,935 | 2 |
| 101 | E1 | p95 | 10,580 | 10,676 | 2 |
| 202 | E1 | mean | 12,828 | 12,872 | 3 |
| 202 | E1 | max | 15,252 | 16,399 | 1 |
| 202 | E1 | p95 | 9,942 | 10,667 | 1 |
| 303 | E1 | mean | 14,515 | 14,588 | 8 |
| 303 | E1 | max | 15,024 | 15,332 | 1 |
| 303 | E1 | p95 | 16,477 | 20,101 | 1 |
| 101 | E2 | mean | 13,020 | 14,549 | 4 |
| 101 | E2 | max | 13,182 | 13,267 | 8 |
| 101 | E2 | p95 | 12,505 | 14,189 | 5 |
| 202 | E2 | mean | 11,955 | 11,960 | 15 |
| 202 | E2 | max | 12,300 | 12,324 | 6 |
| 202 | E2 | p95 | 10,153 | 10,532 | 6 |
| 303 | E2 | mean | 15,286 | 17,758 | 5 |
| 303 | E2 | max | 16,310 | 18,486 | 3 |
| 303 | E2 | p95 | 13,949 | 16,651 | 2 |

The needed improvement was ~250–435×; the best observed is ~1.3× (E1 s202 p95), and the
sign is not even consistent across seeds within an arm.

## 4. What this says about the dilution hypothesis

If GT entities emitted a few high-surprise events that the window mean averages away,
per-window MAX would vault them up the ranking. It does the opposite for all but one or
two entities per arm: one GT node reaches rank 1–8 (this is what inflates E1 max/p95
AUPRC — an early-precision spike), while the other ~45 sink or stay put, because
**thousands of benign nodes also have at least one rare event per window** — the tail
statistic promotes them just as readily. The bulk of the GT set has neither individually
surprising events (max doesn't find them) nor broadly deviant profiles (mean ranks them
~10k-15k). The bottleneck is therefore not the aggregation rule inside the window: the
56-dim edge-type participation profile itself does not carry enough information to
distinguish most attack entities' activity from benign activity, no matter which
statistic summarizes it. This strengthens the architectural/representation reading of
the 10× gap to Orthrus (event-level temporal features), and it retires the last cheap
"scoring rule" lever short of changing the representation.

## 5. Status

- Window-tail aggregation (max, p95): **rejected** — rank gap unmoved, operating points
  uniformly worse, AUPRC direction arm-inconsistent. Headline stays window-mean.
- Faithfulness verified: recomputed mean = existing headline scores, bit-exact, 6/6 arms.
- Artifacts: `models/behavior_cadets/aggscoring/` (driver, log, per-arm score files for
  all three aggregations, analysis script); scorer + tests in `scripts/`.
- Next per the endgame plan: paper draft. This result supplies the "we tested the obvious
  scoring-rule fix and the gap is representational" paragraph.
