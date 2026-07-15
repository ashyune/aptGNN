# D1-revised full seeded run — results (2026-07-14 night)

Run: 3 seeds {101, 202, 303} × {mem-on (--max-norm 100), memory-off ablation (--max-norm 0)}
at w5000, 30 epochs, `scripts/train_behavior.py` / `test_behavior.py`, best checkpoint by
validation per-event NLL on the chronological train tail. Evaluation: uncalibrated and
within-type-calibrated scores against both GT variants (raw 46-node / expanded 12,852-node).
Driver + full log: `models/behavior_cadets/seedrun/` (`run_behavior_seedrun.sh`, `run.log`).
All 6 arms trained and tested with zero failures; runner exited cleanly.

## 1. Headline — raw GT, uncalibrated

| seed | arm | AUPRC | AUROC | P@46 | flag TP/46 | flag recall | flag FPR |
|---|---|---|---|---|---|---|---|
| 101 | mem | 0.0161 | 0.9502 | 0.0652 (3) | 7 | 0.152 | 0.0014 |
| 202 | mem | 0.0211 | 0.9509 | 0.0652 (3) | 10 | 0.217 | 0.0014 |
| 303 | mem | 0.0129 | 0.9324 | 0.0435 (2) | 8 | 0.174 | 0.0014 |
| 101 | nomem | 0.0019 | 0.9089 | 0.0000 (0) | 0 | 0.000 | 0.0012 |
| 202 | nomem | 0.0019 | 0.9044 | 0.0000 (0) | 0 | 0.000 | 0.0012 |
| 303 | nomem | 0.0018 | 0.8815 | 0.0000 (0) | 0 | 0.000 | 0.0012 |

Random AUPRC baseline ≈ 0.0001. Nodes scored: 357,174; GT never scored: 0 in all arms.

**The memory ablation now separates cleanly and stably.** Mem-on beats the type-prior
ablation on raw AUPRC by ~9× (0.013–0.021 vs 0.0018–0.0019), non-overlapping across all
three seeds, and the ablation never places a single true node in the top-46 or in the
flagged set while mem-on catches 7–10 of 46. This is the designed behavior: with
--max-norm 0 the model provably collapses to a type-prior, and that prior carries no
node-level detection signal. In v1 (score = NLL of own type) memory was null-to-negative;
in D1-revised (history-conditioned behavior profile) memory is the entire signal.

## 2. Calibration (within-type) — raw GT

| seed | arm | AUPRC cal | AUROC cal | P@46 cal |
|---|---|---|---|---|
| 101 | mem | 0.0097 | 0.7206 | 0.0435 |
| 202 | mem | 0.0102 | 0.7134 | 0.0217 |
| 303 | mem | 0.0082 | 0.7005 | 0.0217 |
| 101 | nomem | 0.0017 | 0.6452 | 0.0000 |
| 202 | nomem | 0.0017 | 0.6437 | 0.0000 |
| 303 | nomem | 0.0017 | 0.6455 | 0.0000 |

Within-type calibration **hurts** D1-revised on every seed (AUPRC roughly halved, AUROC
0.95→0.71). Unlike v1, the behavior score's cross-type scale differences are apparently
informative, not artifact. Mem > nomem ordering unchanged. Headline stays uncalibrated.

## 3. Expanded GT (reported per protocol; random AUPRC baseline ≈ 0.0360)

| seed | arm | AUPRC uncal | AUROC uncal | AUPRC cal | AUROC cal |
|---|---|---|---|---|---|
| 101 | mem | 0.0249 | 0.0790 | 0.0266 | 0.1209 |
| 202 | mem | 0.0353 | 0.3080 | 0.0263 | 0.1100 |
| 303 | mem | 0.0356 | 0.3109 | 0.0266 | 0.1195 |
| 101 | nomem | 0.0902 | 0.6389 | 0.0364 | 0.0317 |
| 202 | nomem | 0.0902 | 0.6387 | 0.0364 | 0.0305 |
| 303 | nomem | 0.0908 | 0.6414 | 0.0364 | 0.0304 |

Mem-on sits **at or below random** on expanded GT (AUROC 0.08–0.31), while the type-prior
ablation looks "good" (0.64) — the expanded set is dominated by benign-behaving neighbor
nodes of common types, so a type-prior ranks them high and a behavior-deviation score
ranks them low. This is the same anti-signal pattern D4 established; the raw/expanded
divergence is expected and reinforces raw GT as the design target.

## 4. Caveat — 30-epoch training underperforms the 2-epoch smoke

The pre-run smoke test (2 epochs, unseeded, throwaway checkpoint) scored raw-GT AUPRC
0.0399 / P@46 0.152 / AUROC 0.920 — roughly 2× the best full-run arm here (0.0211).
Longer training (best-val-NLL checkpointing over 30 epochs) appears to *reduce* detection:
as the model gets better at predicting behavior profiles generally, contrast on anomalous
nodes shrinks. The val-NLL model-selection objective is not aligned with detection.
Also for reference: v1's best arm (trainnomem uncalibrated) reached 0.033–0.035 AUPRC,
above D1-revised mem-on at 30 epochs but below the D1-revised 2-epoch smoke.

**Candidate follow-ups (not started, need approval):** epoch/checkpoint sweep on the
existing D1-revised code (e.g. score checkpoints at 1/2/3/5/10/30 epochs across the 3
seeds — new driver script only, no pipeline changes); alternatively investigate a
detection-aligned early-stop proxy.

## 5. Status

- v1 negative-result framing (memory adds variance without reproducible benefit) stands.
- D1-revised structural claim now seed-verified: forcing scoring through memory makes the
  memory ablation honest and cleanly positive (9×, all seeds, non-overlapping).
- Absolute performance at the current training schedule is not yet at its apparent
  ceiling (smoke evidence); training-length sensitivity is the obvious next experiment.
