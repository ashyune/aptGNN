# Epoch/checkpoint sweep — preregistered rule evaluation (2026-07-15)

**Preregistration:** `models/behavior_cadets/epochsweep/PREREGISTRATION.md`
(committed and user-approved before compute). **Driver + logs:**
`models/behavior_cadets/epochsweep/{run_epochsweep.sh, run.log, run_picks.log}`.
**Design:** 3 seeds {101, 202, 303}, mem-on (`--max-norm 100`), w5000, 30 epochs
with per-epoch checkpoints and Δ logging (`train_behavior.py --save-every-epoch`);
full test passes at grid epochs {1, 2, 3, 5, 10, 15, 20, 30} plus the rule-picked
and incumbent-picked checkpoints. All runs clean; incumbent picks reproduce the
2026-07-14 seedrun numbers exactly (seeded training is deterministic).

## 1. Verdict: the Δ-max rule FAILS both preregistered adoption criteria

Rule = argmax over epochs of Δ(e) = val NLL(zero memory) − val NLL(memory on).
Δ-max picked **epoch 22 on all three seeds** (the rule itself is highly seed-stable).
Raw-GT **uncalibrated AUPRC**, per seed:

| seed | rule pick (e22) | incumbent (best-val-NLL) | grid median | oracle (grid) |
|---|---|---|---|---|
| 101 | 0.0121 | **0.0161** (e27) | 0.0129 | 0.0196 (e30) |
| 202 | 0.0211 | 0.0211 (e22 — same checkpoint) | 0.0113 | 0.0352 (e5) |
| 303 | 0.0128 | 0.0129 (e21) | 0.0188 | 0.0418 (e3) |

- **Criterion 1 (≥ incumbent on all 3 seeds): FAIL** — loses on s101 (0.0121 vs
  0.0161) and marginally on s303; ties on s202 only because both rules select the
  same epoch.
- **Criterion 2 (≥ grid median on all 3 seeds): FAIL** — below median on s101 and s303.
- Magnitude check moot. Oracle recovery (descriptive only): 62% / 60% / 31%.
- **Fallback rule (val-score tail contrast): NOT evaluated.** Its preregistered
  trigger — Δ monotone or flat — did not occur (Δ has a stable argmax). Per the
  preregistration, no further proxies may be tried after seeing these curves.

**Outcome (as preregistered for this case): detection-aligned checkpoint selection
remains open. The headline remains best-val-NLL checkpoint selection — i.e. the
2026-07-14 seedrun numbers stand unchanged.**

## 2. The premise of the sweep was itself a single-run mirage

The sweep was motivated by the 2-epoch unseeded smoke checkpoint scoring 0.0399
(≈2× the best 30-epoch arm), suggesting a systematic early-training detection peak.
The seeded grid refutes this. Raw-GT uncalibrated AUPRC across epochs:

| epoch | s101 | s202 | s303 |
|---|---|---|---|
| 1  | 0.0056 | 0.0105 | 0.0149 |
| 2  | 0.0075 | 0.0102 | 0.0252 |
| 3  | 0.0135 | 0.0112 | **0.0418** |
| 5  | 0.0128 | **0.0352** | 0.0196 |
| 10 | 0.0131 | 0.0317 | 0.0310 |
| 15 | 0.0160 | 0.0087 | 0.0117 |
| 20 | 0.0123 | 0.0229 | 0.0113 |
| 30 | **0.0196** | 0.0115 | 0.0180 |

- **No consistent detection peak exists**: oracle epochs are 30 / 5 / 3 across
  seeds, and every seeded epoch-2 checkpoint (0.0075–0.0252) is well below the
  0.0399 smoke — the smoke was a favorable OS-entropy draw, the same failure mode
  the v1 seed check exposed. (F5)
- **Detection-vs-epoch is dominated by run noise, not by a training-length trend**:
  swings of 2–4× between adjacent grid epochs within a seed (s202: 0.0352 → 0.0087
  between e5 and e15; s303 e10 has AUPRC 0.0310 with AUROC 0.5375 — a top-heavy
  ranking over a near-random bulk). (F6)
- Spearman(Δ, AUPRC) across the grid: +0.50 / +0.76 / −0.48 — sign flips on s303;
  the Δ proxy does not reliably track detection. (F7)

## 3. Δ itself: memory reliance grows with training (hypothesis refuted, cleanly)

The rule's motivating hypothesis — that late training erodes memory reliance via
prior-sharpening — is wrong in its simple form. Δ *rises* from ~0.6–0.7 (epochs
1–3) to a noisy plateau ≥1.0 nats/event from mid-training on every seed
(per-epoch values in `s*/epoch_metrics.tsv`). The model leans on memory *more*
with training, as measured on benign validation behavior, while detection wanders
without trend. "Memory usefulness on benign data" and "anomaly contrast" are
simply different quantities. Δ remains a useful health metric (it is the live,
label-free version of the ablation delta), just not a checkpoint selector.

## 4. Protocol completions at the pick checkpoints

**Attack-day split (red-flag check, preregistered at rule pick; incumbent shown
descriptively).** Both days carry ranking signal at every pick — no
generalization red flag. Day-13 resolves only 6 entities; its P@6 = 0 everywhere,
so top-of-list hits are all day-12 entities:

| seed·epoch | day-12 AUPRC / AUROC (43 gids) | day-13 AUPRC / AUROC (6 gids) |
|---|---|---|
| s101 e22 | 0.0120 / 0.9366 | 0.0054 / 0.9219 |
| s101 e27 (inc.) | 0.0156 / 0.9501 | 0.0068 / 0.9341 |
| s202 e22 (=inc.) | 0.0207 / 0.9509 | 0.0044 / 0.9378 |
| s303 e22 | 0.0126 / 0.9518 | 0.0033 / 0.9439 |
| s303 e21 (inc.) | 0.0126 / 0.9321 | 0.0035 / 0.9203 |

(Day-13 random baseline ≈ 0.000017, so 0.0033–0.0068 is ~200–400× random.
Day subsets derived from the pinned ProvenanceAnalytics commit's per-day CSVs;
43 + 6 = 49 > 46 because 3 entities persist across both days.)

**Both GT variants / both reporting variants at the picks** (per protocol;
consistent with the seedrun — calibration hurts D1-revised, expanded GT stays
at/below random for mem-on):

| seed·epoch | raw uncal AUPRC/AUROC | raw cal AUPRC/AUROC | exp uncal AUPRC/AUROC | exp cal AUPRC/AUROC |
|---|---|---|---|---|
| s101 e22 | 0.0121 / 0.9383 | 0.0062 / 0.6899 | 0.0359 / 0.3136 | 0.0267 / 0.1221 |
| s101 e27 (inc.) | 0.0161 / 0.9502 | 0.0097 / 0.7206 | 0.0249 / 0.0790 | 0.0266 / 0.1209 |
| s202 e22 (=inc.) | 0.0211 / 0.9509 | 0.0102 / 0.7134 | 0.0353 / 0.3080 | 0.0263 / 0.1100 |
| s303 e22 | 0.0128 / 0.9520 | 0.0097 / 0.7302 | 0.0249 / 0.0812 | 0.0267 / 0.1231 |
| s303 e21 (inc.) | 0.0129 / 0.9324 | 0.0082 / 0.7005 | 0.0356 / 0.3109 | 0.0266 / 0.1195 |

## 5. Status & what this changes

1. **Headline unchanged:** D1-revised results remain the 2026-07-14 seedrun
   numbers (best-val-NLL checkpoints): raw-GT uncal AUPRC 0.0161 / 0.0211 / 0.0129,
   memory ~9× over the type-prior ablation on every seed.
2. **Retired:** the "val-NLL checkpointing leaves 2× on the table" caveat from the
   seedrun report §4. The correct caveat is: *per-checkpoint detection metrics
   carry large run-to-run variance around the level the seedrun already reports;
   no label-free selection rule evaluated so far beats best-val-NLL.*
3. **Open question reframed:** from "which checkpoint" to "why does the detection
   metric swing 2–4× between adjacent checkpoints while val NLL and Δ are stable."
   Candidate follow-ups (NOT started, need approval): (a) score-averaging across
   the last k checkpoints (snapshot ensembling of scores, not weights) to damp
   checkpoint noise; (b) rank-stability analysis of which GT entities move between
   adjacent checkpoints (are the swings a few high-rank entities flickering?).
4. The preregistration discipline worked as intended: without it, the natural move
   would have been to report s303's e3 = 0.0418 — a number this sweep shows is
   irreproducible.
