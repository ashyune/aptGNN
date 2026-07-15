# Epoch/checkpoint sweep — preregistration (2026-07-15, committed BEFORE compute)

Purpose of this sweep: **evaluate a preregistered checkpoint-selection rule**, not
to pick the best epoch from the raw-GT curve. Written and user-approved before any
sweep training or scoring ran.

## Motivation

The D1-revised seedrun (reports/2026-07-14_d1_revised_seedrun_results.md §4) showed
the 2-epoch smoke checkpoint scoring ~2x the best 30-epoch best-val-NLL arm on
raw-GT AUPRC: val-NLL checkpoint selection appears misaligned with detection.
Selecting the epoch that maximizes raw-GT AUPRC directly would fit a hyperparameter
to the 46-node test label set, so the selection rule must be label-free and stated
in advance.

## Primary rule (label-free)

Select the checkpoint maximizing the **memory-utility gap** on the existing
chronological train-tail validation windows:

    delta(e) = val per-event NLL with memory zeroed (type-prior forward)
             - val per-event NLL with memory on

computed at each epoch e during training (train_behavior.py --save-every-epoch,
logged to epoch_metrics.tsv). Pick argmax_e delta(e). Never touches the test file
or any ground truth.

Rationale: D1-revised's detection signal is deviation-from-history; contrast on
anomalous nodes requires the model to actually lean on memory. Hypothesis: late in
training the model keeps shaving val NLL by sharpening the type prior / smoothing
(hedging), which flattens the deviation signal; delta should peak roughly where
detection peaks.

## Fallback rule (also preregistered; used only if delta is monotone or flat)

Val-score tail contrast: ratio of the 99.9th-percentile per-node score to the
median on the validation windows, maximized over checkpoints. No third proxy may
be introduced after seeing curves.

Rejected rule: fixed 2-3 epoch budget — contaminated, since the smoke checkpoint's
raw-GT score is already known.

## Adoption criteria (comparative, no tunable parameters)

The rule REPLACES the incumbent (best-val-NLL checkpoint selection) only if, on
raw-GT uncalibrated AUPRC:

1. **Beat the incumbent:** the delta-max pick >= the best-val-NLL pick, on all 3
   seeds; AND
2. **Beat blind selection:** the delta-max pick >= the median checkpoint of the
   scored grid, on all 3 seeds.

Magnitude sanity check: the improvement over the incumbent should be visible above
seed noise, i.e. exceed the seedrun's across-seed spread of the incumbent arm
(mem-on AUPRC 0.0129-0.0211, spread 0.0082). Oracle-recovery (rule pick vs
label-optimal pick) is reported **descriptively only** — no threshold, because the
oracle is a max over noisy evaluations and is upward-biased with grid density.

## Reporting rules

- Headline numbers come only from a rule-selected checkpoint, never from the
  curve's argmax. The raw-GT-vs-epoch curve appears as sensitivity analysis,
  clearly labeled.
- Per seed: rule-picked epoch vs oracle epoch, regret, and Spearman correlation
  between delta(e) and raw-GT AUPRC across the scored grid.
- Per-seed numbers side by side in all tables (user standing rule); both GT
  variants reported; expanded GT is anti-signal (D4) and is NOT a holdout.
- Attack-day split (~40 Apr-12 / ~6 Apr-13 entities) reported at the rule-picked
  checkpoint as a generalization red-flag check; never selected on.
- If the rule fails the adoption criteria, the outcome is reported as
  "detection-aligned checkpoint selection remains open"; headline stays at
  best-val-NLL with the curve as sensitivity. The oracle epoch is NOT promoted.

## Sweep design

3 seeds {101, 202, 303}, mem-on (--max-norm 100) only, w5000, 30 epochs with
--save-every-epoch (checkpoints + per-epoch delta at all 30 epochs); full test
pass (test_behavior.py, protocol unchanged) at grid epochs {1,2,3,5,10,15,20,30};
uncalibrated + within-type-calibrated evaluation against both GT variants per
checkpoint. Driver: run_epochsweep.sh. Pipeline files untouched except the
user-approved additive --save-every-epoch flag in train_behavior.py.
