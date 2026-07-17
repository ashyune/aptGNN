# E3 feature-enrichment results: binned n_distinct / rep_ratio / span_frac as scored targets — 2026-07-17

Experiment (user-approved, design per §3 of the same-day survey): E3 = the D1-revised
task with three extra scored observables per node per window — log2-binned n_distinct
(13 bins), log2-binned rep_ratio (13 bins), and span_frac (10 linear bins + a dedicated
==1.0 bin). Features enter (a) as their own softmax heads over the shared hidden state,
node score = profile NLL + **unweighted mean** of the three feature-bin NLLs, and (b)
one-hot into the parameter-free memory writeback. They never enter `forward()` — the
leakage guard and the zero-memory type-prior collapse are preserved and structurally
tested. Bin edges and the equal weighting are preregistered constants locked in
`window_features.py` / `train_behavior_e3.py` (no CLI knob exists). Checkpoint rule =
best val NLL on the chronological train tail (same as E1/E2, applied to the combined
NLL). New additive files only (`window_features.py`, `train_behavior_e3.py`,
`test_behavior_e3.py`, `test_train_behavior_e3.py` — 9/9 structural tests pass);
E1/E2 stack untouched. 3 seeds {101, 202, 303}, w5000, 30 epochs, max-norm 100,
same scoring protocol (per-node MAX over windows, train-0.999-quantile flag).
Driver + log + outputs + analysis: `models/behavior_cadets/e3seedrun/`. All 3 arms
clean (exit 0, 357,174 nodes scored each).

Survey prediction being tested: P@46 moves 3 → ~6–9; median GT rank does not
approach 46.

## Verdict up front

**Mixed, with both directions reproducible across all 3 seeds — and the survey's P@46
prediction was wrong.** The features carry real signal: median GT rank roughly HALVES
on every seed, AUROC rises to ~0.97 on every seed (and becomes far more seed-stable),
and the flag operating point improves on every seed (precision ~1.5–2×, MCC up, with
FEWER false positives). This is the first reproducible cross-seed detection improvement
of the project. But the extreme head of the ranking gets WORSE: P@46 drops to 2/1/1
(vs E1's 3/3/2), no GT node survives above rank ~19, and AUPRC dips slightly on every
seed. The 46-alarm operating point — the Orthrus-comparison number — degrades; the
crossover where E3 beats E1 is at a ~200–400-alarm budget.

## 1. Ranking metrics (raw GT, uncalibrated; E2 shown for completeness)

| seed | arm | AUPRC | AUROC | P@46 | best val NLL |
|---|---|---|---|---|---|
| 101 | E1 | 0.0161 | 0.9502 | 0.0652 (3) | 1.0155 |
| 101 | E2 | 0.0133 | 0.9296 | 0.0652 (3) | 0.9768 |
| 101 | E3 | 0.0119 | **0.9708** | 0.0435 (2) | 1.4183 combined |
| 202 | E1 | 0.0211 | 0.9509 | 0.0652 (3) | 1.0255 |
| 202 | E2 | 0.0123 | 0.9315 | 0.0652 (3) | 0.9678 |
| 202 | E3 | 0.0157 | **0.9696** | 0.0217 (1) | 1.4265 combined |
| 303 | E1 | 0.0129 | 0.9324 | 0.0435 (2) | 1.0152 |
| 303 | E2 | 0.0186 | 0.9293 | 0.0652 (3) | 0.9694 |
| 303 | E3 | 0.0127 | **0.9707** | 0.0217 (1) | 1.4296 combined |

E3 AUROC spread across seeds: 0.0012 (E1: 0.019). E3's combined val NLL is not
comparable to E1/E2 (different objective), but its logged epoch-30 profile/feature
split (profile 1.003 / 1.011 / 1.004 across seeds) shows the profile head is not
degraded by the added heads — slightly better than E1's val NLL (1.015–1.026) — so
the trade below is a scoring-composition effect, not a modeling regression.

## 2. Operating points (raw GT)

| seed | arm | flag TP/FP | flag P | flag recall | flag MCC | top-46 TP | top-46 P=MCC |
|---|---|---|---|---|---|---|---|
| 101 | E1 | 7/488 | 0.0141 | 0.152 | 0.046 | 3 | 0.065 |
| 101 | E3 | 8/385 | **0.0204** | 0.174 | **0.059** | 2 | 0.043 |
| 202 | E1 | 10/507 | 0.0193 | 0.217 | 0.065 | 3 | 0.065 |
| 202 | E3 | 11/374 | **0.0286** | 0.239 | **0.082** | 1 | 0.022 |
| 303 | E1 | 8/510 | 0.0154 | 0.174 | 0.051 | 2 | 0.043 |
| 303 | E3 | 12/383 | **0.0304** | 0.261 | **0.089** | 1 | 0.022 |

Flag point: E3 wins every seed on precision, recall, MCC — more TPs from fewer total
alarms (385–395 vs 486–517). Top-46: E3 loses every seed. Still 17–25× below Orthrus
(P 0.52 / MCC 0.44) at the flag point.

## 3. GT rank distribution

| seed | arm | median GT rank | 24th GT rank (needs ≤46) | best GT rank |
|---|---|---|---|---|
| 101 | E1 | 11,306 | 11,308 | 5 |
| 101 | E3 | **6,841** | **6,858** | 20 |
| 202 | E1 | 12,828 | 12,872 | 3 |
| 202 | E3 | **5,596** | **5,694** | 22 |
| 303 | E1 | 14,515 | 14,588 | 8 |
| 303 | E3 | **5,602** | **5,633** | 28 |

Median and 24th GT rank roughly halve on every seed — the largest reproducible bulk
improvement observed in the project — but 5,6k–6,9k is still ~120–150× short of 46,
as the survey's ceiling analysis said it would be.

## 4. TP at alarm budgets — where the crossover sits

| seed | arm | @46 | @100 | @200 | @400 | @1000 | @5000 |
|---|---|---|---|---|---|---|---|
| 101 | E1 | 3 | 5 | 7 | 7 | 10 | 19 |
| 101 | E3 | 2 | 2 | 7 | **8** | **16** | 19 |
| 202 | E1 | 3 | 5 | 9 | 10 | 13 | 18 |
| 202 | E3 | 1 | **6** | 9 | **12** | **15** | **19** |
| 303 | E1 | 2 | 5 | 5 | 8 | 12 | 18 |
| 303 | E3 | 1 | 2 | **6** | **12** | **15** | **22** |

E3 loses inside ~100 alarms, ties by ~200, and wins at 400 and 1000 on every seed
(at 1000: recall 0.33–0.35 vs 0.22–0.28).

## 5. Node-level mechanism (tracked GT uuids, E1 rank → E3 rank)

The feature-extreme GT nodes from the survey are recovered exactly as the marginal
analysis predicted — by one to three orders of magnitude:

| node (type, why tracked) | s101 | s202 | s303 |
|---|---|---|---|
| 7CED5514 (proc, span==1.0 + rep) | 837 → **20** | 20,277 → **99** | 71,737 → **28** |
| 51110606 (file, fan_in #1) | 53,923 → **182** | 59,308 → **177** | 70,256 → **305** |
| 416AB837 (file, burst #1 / events #3) | 18,457 → **163** | 18,585 → **128** | 17,778 → **232** |
| 47E61FFC (proc, fan_out #1) | 9,847 → **487** | 10,117 → **285** | 9,928 → **338** |
| 1131CD42 (proc, fan_out #5) | 1,711 → **162** | 1,651 → **94** | 2,663 → **200** |
| 7CF2DDA4 (proc, rep #37) | 45,309 → 5,561 | 43,077 → 5,694 | 57,941 → 4,938 |

…but E1's profile-surprise hits are simultaneously DILUTED by the added feature term
(their feature bins are ordinary, so the additive combination drags them down while
benign feature-extreme nodes jump ahead):

| node | s101 | s202 | s303 |
|---|---|---|---|
| 1902FFA3 (file, 1 event; E1 top hit) | 5 → 36 | 20 → 22 | 15 → 66 |
| 1633500B (file; E1 top hit) | 13 → 719 | 3 → 70 | 8 → 162 |
| 5453C813 (file, 4 events) | 93 → 10,354 | 42 → 6,130 | 1,075 → 6,128 |
| 48289024 (proc; E1/E2 top hit) | 40 → 172 | 90 → 89 | 87 → 156 |

The top ~20 of the E3 ranking is occupied by benign nodes with rare feature bins —
the survey's marginal tables already showed the raw features' extreme tails are mostly
benign (e.g. 71 of the 78 full-span nodes), and history/type conditioning does not
neutralize them because one-shot benign nodes have no history either. Net effect: the
head is contested, the bulk is compressed upward.

## 6. Standard checks (same pattern as E1/E2)

- Within-type calibration HURTS E3 on every seed (raw AUPRC 0.0119→0.0042,
  0.0157→0.0055, 0.0127→0.0042; AUROC 0.97→0.82–0.84). Headline stays uncalibrated.
- Expanded GT is anti-signal again (uncal AUROC 0.41/0.64/0.41; calibrated 0.15–0.17).
  Both GT variants reported per protocol; raw GT remains the design target.
- Flag threshold recomputed per arm from the train-window 99.9th percentile of the
  combined score (same label-free protocol).

## 7. What this changes

1. **The representational diagnosis stands, refined**: enriching the scored
   representation moved the bulk of the GT distribution reproducibly — the first
   intervention in this project to do so — confirming the bottleneck was information
   content, not capacity (E2) or scoring rule (aggregation test). But the enrichment
   is equally available to benign one-shots, so the 46-alarm head does not improve.
2. **The survey's P@46 prediction (3 → 6–9) is refuted**: P@46 went DOWN (2/1/1).
   What improved instead: flag-point precision/MCC (every seed), mid-budget recall
   (every seed, ≥200–400 alarms), median GT rank (halved, every seed), AUROC
   (+0.02–0.04, every seed, with 16× less seed spread).
3. **Paper framing**: E3 is an honest mixed result — "cheap statistical enrichment of
   the scored profile reproducibly improves bulk ranking and realistic-budget operating
   points, but the top-of-ranking precision that Orthrus-style comparisons report is
   dominated by benign feature-extremes and does not improve." The ~10× gap at a
   46-alarm budget stays (now 17–25× at the flag point in precision terms vs Orthrus's
   0.52, unchanged in order).
4. Any recombination of the two score components other than the preregistered
   unweighted mean (max-of-ranks, per-head quantile fusion, learned weights…) is a NEW
   experiment with its own preregistration — explicitly not tried here, per the locked
   no-mixing-weight rule.

## 8. Status

- Artifacts: `models/behavior_cadets/e3seedrun/` (driver, run.log, 3 seed dirs with
  checkpoints + eval outputs incl. calibrated scores, analyze_e3.py);
  `scripts/{window_features,train_behavior_e3,test_behavior_e3,test_train_behavior_e3}.py`.
  All uncommitted, alongside the 07-16 reports and the survey.
- Headline arm for the paper remains E1 window-mean at 46-alarm budgets; E3 is the
  documented mixed-result extension (report both, per-seed).
