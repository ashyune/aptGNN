# Extension 2 (relation-aware neighbor-memory routing) seeded run — results (2026-07-16)

Run: 3 seeds {101, 202, 303} × relational mem-on (`--relational --max-norm 100`, `--rel-bases 8`
default) at w5000, 30 epochs, `scripts/train_behavior.py` / `test_behavior.py`, best checkpoint
by validation per-event NLL on the chronological train tail — the same rule as the E1 headline
(the Δ-max rule failed its preregistered evaluation on 2026-07-15 and is not used). Evaluation:
uncalibrated and within-type-calibrated scores against both GT variants (raw 46-node /
expanded 12,852-node). E1 comparison arms are the existing D1-revised seedrun
(`models/behavior_cadets/seedrun/`, report `2026-07-14_d1_revised_seedrun_results.md`) — same
seeds, window size, epochs, and checkpoint rule, differing only in the convolution.
Driver + full log: `models/behavior_cadets/e2seedrun/` (`run_e2_seedrun.sh`, `run.log`).
All 3 arms trained and tested with zero failures; runner exited cleanly (~70 min total,
~12 s/epoch relational vs ~2.5 s E1 SAGE).

Architecture under test: `RelationalMemoryConv` — per-edge-type read of neighbor memories
(29 relations = 28 edge types + self-loop id), RGCN basis decomposition (8 bases), no
per-relation bias or edge-type embedding, so edge types act only multiplicatively on memory
content and the zero-memory ⇒ type-prior collapse is preserved (structurally tested; shared
basis reduces exactly to the E1 SAGEConv, parity-tested).

## 1. Headline — raw GT, uncalibrated (E2 vs E1 side by side)

| seed | arm | AUPRC | AUROC | P@46 | flag TP/46 | flag recall | flag FPR |
|---|---|---|---|---|---|---|---|
| 101 | E1 mem | 0.0161 | 0.9502 | 0.0652 (3) | 7 | 0.152 | 0.0014 |
| 101 | E2 rel | 0.0133 | 0.9296 | 0.0652 (3) | 6 | 0.130 | 0.0014 |
| 202 | E1 mem | 0.0211 | 0.9509 | 0.0652 (3) | 10 | 0.217 | 0.0014 |
| 202 | E2 rel | 0.0123 | 0.9315 | 0.0652 (3) | 7 | 0.152 | 0.0014 |
| 303 | E1 mem | 0.0129 | 0.9324 | 0.0435 (2) | 8 | 0.174 | 0.0014 |
| 303 | E2 rel | 0.0186 | 0.9293 | 0.0652 (3) | 6 | 0.130 | 0.0015 |

Random AUPRC baseline ≈ 0.0001. Nodes scored: 357,174; GT never scored: 0 in all arms.
E1 type-prior ablation reference (all seeds): AUPRC 0.0018–0.0019, 0 flag TP, P@46 = 0.

**Relation-aware routing does not reproducibly improve detection.** E1 wins AUPRC on seeds
101 and 202, E2 wins on 303; the ranges overlap almost completely (E2 0.0123–0.0186 vs E1
0.0129–0.0211) and the per-seed winner flips — by the same standard applied to every prior
claim (non-overlapping across all seeds), this is a null result. E2 preserves the
memory-is-the-signal property (~7–10× the type-prior ablation on every seed, 6–7/46 flag TPs
where the ablation gets 0), it just does not add to it.

Two secondary observations, descriptive only (single 3-seed run, not adoption criteria):

- **E2 is more seed-stable.** AUROC spread across seeds is 0.0022 (0.9293–0.9315) vs E1's
  0.0185 (0.9324–0.9509); AUPRC spread 0.0063 vs 0.0082; P@46 is 3/46 on all three seeds
  where E1 dropped to 2 on s303; flag TP spread 6–7 vs 7–10. Consistent with per-relation
  weights regularizing the read path — and with the epoch-sweep finding that per-checkpoint
  detection variance is the dominant noise source, so lower variance at equal mean is worth
  reporting but not headline-grade.
- **E2's AUROC is uniformly ~0.02 below E1's.** Small, but the same sign on all seeds.

## 2. Validation NLL — E2 predicts behavior better, detects no better

| seed | E1 mem best val NLL | E2 rel best val NLL | E2 best epoch |
|---|---|---|---|
| 101 | 1.0155 | 0.9768 | 19 |
| 202 | 1.0255 | 0.9678 | 30 |
| 303 | 1.0152 | 0.9694 | 18 |

E2 improves validation per-event NLL by 0.04–0.06 nats on every seed (type-prior ablation
reference ≈ 1.50) — routing neighbor memories by edge type genuinely helps predict benign
behavior profiles. That this does not translate into detection is the same dissociation the
epoch sweep established between val NLL and anomaly contrast: better modeling of the benign
distribution is not the binding constraint on separating the 46 attack entities.

## 3. Calibration (within-type) — raw GT

| seed | arm | AUPRC cal | AUROC cal | P@46 cal |
|---|---|---|---|---|
| 101 | E1 mem | 0.0097 | 0.7206 | 0.0435 |
| 101 | E2 rel | 0.0077 | 0.6718 | 0.0435 |
| 202 | E1 mem | 0.0102 | 0.7134 | 0.0217 |
| 202 | E2 rel | 0.0116 | 0.6665 | 0.0435 |
| 303 | E1 mem | 0.0082 | 0.7005 | 0.0217 |
| 303 | E2 rel | 0.0118 | 0.6784 | 0.0435 |

Within-type calibration hurts E2 on every seed, as it did E1 (AUPRC roughly halved on 2/3
seeds, AUROC 0.93→0.67). Headline stays uncalibrated. E2-vs-E1 ordering under calibration is
again mixed (E1 wins s101, E2 wins s202/s303) — no view of the data makes either arm a
reproducible winner.

## 4. Expanded GT (reported per protocol; random AUPRC baseline ≈ 0.0360)

| seed | arm | AUPRC uncal | AUROC uncal | AUPRC cal | AUROC cal |
|---|---|---|---|---|---|
| 101 | E1 mem | 0.0249 | 0.0790 | 0.0266 | 0.1209 |
| 101 | E2 rel | 0.0249 | 0.0776 | 0.0263 | 0.1206 |
| 202 | E1 mem | 0.0353 | 0.3080 | 0.0263 | 0.1100 |
| 202 | E2 rel | 0.0247 | 0.0718 | 0.0246 | 0.0610 |
| 303 | E1 mem | 0.0356 | 0.3109 | 0.0266 | 0.1195 |
| 303 | E2 rel | 0.0248 | 0.0743 | 0.0259 | 0.1059 |

Same anti-signal pattern as D4 and the E1 run: a behavior-deviation score ranks the
expanded set's benign-behaving neighbor nodes low, so mem-on sits far below random AUROC.
E2 is even more uniformly anti-correlated (0.072–0.078 vs E1's 0.079–0.311) — one more data
point that expanded GT rewards type-prior-like scoring and is not the design target.

## 5. Status

- E2 (relation-aware routing of neighbor memories) is a **null result for detection**:
  per-seed winner vs E1 flips, ranges overlap, on both reporting variants and both GTs.
- E2 is a **positive result for behavior modeling**: val NLL better on every seed
  (0.968–0.977 vs 1.015–1.026), and detection metrics are visibly more seed-stable.
- Honest paper framing: edge-type structure helps the model predict benign behavior but the
  extra capacity does not sharpen anomaly contrast on the 46 raw-GT entities; combined with
  the epoch-sweep results this locates the bottleneck in score/checkpoint variance and the
  scoring rule, not in encoder expressiveness.
- Checkpoints + per-seed eval outputs: `models/behavior_cadets/e2seedrun/s{101,202,303}/`.
- Next per the endgame plan: paper draft (E1 + E2 + negative-result framing); no further
  hyperparameter exploration (agreed cut).
