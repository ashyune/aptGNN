# Extension 1 — D4 window-size sweep results & revised D1 score design

**Date:** 2026-07-14 · **Pipeline:** fixed post-D2 (`train_windowed.py` never reads test file; val-tail checkpoint selection; train-quantile flag calibration at q=0.999) · **Scene:** cadets E3 · **Eval:** `evaluate_windowed.py` node-level protocol (no neighborhood credit), both GT variants reported side by side per protocol.

## 1. Run provenance

The D4 sweep ran on 2026-07-14 (morning session, interrupted by laptop shutdown ~11:25). Post-shutdown audit:

- **Completed and intact** (verified: 357,174 lines each, well-formed): all six score sets — `sweep/{w50000,w10000,w5000}/{eval,eval_nomem}/scores_windowed.txt` plus checkpoints. `eval` = memory active at test time; `eval_nomem` = same mem-trained checkpoint scored with `--max-norm 0` (test-time ablation).
- **Lost to the shutdown:** the `w5000_trainnomem` arm (train-time ablation: trained from scratch with `--max-norm 0`) — directory was created at 11:25 but empty. Rerun from scratch in the evening session (log: `sweep/w5000_trainnomem/run.log`).
- Metric printouts from the morning session existed only in the (wiped) session scratchpad; all metrics below are recomputed from the intact score files — deterministic, no loss.

## 2. Sweep results

Ranking metrics (threshold-free). Expanded GT = `groundtruth/cadets.txt`, 12,852 resolved (2-hop-expanded, ThreaTrace/MAGIC-comparable; prevalence 0.0360). Raw GT = `groundtruth/cadets_raw_uuid.txt`, 46 resolved of 72 (Orthrus-style entity labels; prevalence 0.00013).

| Arm | Test mem | Expanded AUPRC | Expanded AUROC | Raw AUPRC | Raw AUROC | Raw P@46 | Zero-score frac | Mean score |
|---|---|---|---|---|---|---|---|---|
| w50000 | on  | 0.0372 | 0.4103 | 0.0076 | 0.9433 | 0.0217 | 82.2% | 0.6231 |
| w50000 | off | 0.0372 | 0.4051 | 0.0078 | 0.9509 | 0.0217 | 81.1% | 0.6134 |
| w10000 | on  | 0.0372 | 0.0951 | 0.0085 | 0.9322 | 0.0435 | 21.1% | 0.2501 |
| w10000 | off | 0.0372 | 0.0922 | 0.0086 | 0.9490 | 0.0435 | 20.5% | 0.2704 |
| w5000  | on  | 0.0353 | 0.0375 | 0.0185 | 0.9404 | 0.0435 | 9.0%  | 0.2219 |
| w5000  | off | 0.0353 | 0.0335 | 0.0189 | 0.9518 | 0.0652 | 8.2%  | 0.2514 |
| w5000 trained-nomem | off | 0.0383 | 0.0757 | **0.0336** | 0.9440 | 0.0435 | 17.3% | 0.2485 |

Operating point (q=0.999 train-quantile flag) stays in the FPR≈0.002–0.004 band across arms; raw-GT recall at that point peaks at w5000 mem-on (12/46 = 0.26 at precision 0.0095). Operating-point numbers are secondary per protocol.

### 2.1 Train-time ablation arm (w5000_trainnomem)

Trained from scratch with `--max-norm 0` (memory zeroed during training as well as test), w5000, otherwise identical config. Result: **raw-GT AUPRC 0.0336 — the best of all seven arms**, ~1.8× the memory-trained w5000 arms (0.0185/0.0189), with comparable raw AUROC (0.9440) and the best operating-point F1 on raw GT (0.0306 at FPR 0.0015). Caveat: single seed per arm, no variance estimate — the 1.8× gap is larger than the on/off test-time deltas but could still partly be run-to-run training noise; a seed-repeat would settle it if the point ever becomes load-bearing. Directionally it is consistent with, and stronger than, F3: under the current architecture, memory during training does not merely fail to help — the arm that never had memory ranks attack entities better.

## 3. Findings

**F1 — Window size is a real lever on the raw-GT signal.** Raw-GT AUPRC rises monotonically as windows shrink: 0.0076 → 0.0085 → 0.0185 (50k → 10k → 5k), i.e. ~59× → ~145× random. Mechanism is consistent with the saturation diagnosis: smaller windows mean less context per window, a harder type-prediction task, and far fewer saturated ties (exact-zero scores drop 82% → 21% → 9%). The score becomes more informative as the task stops being trivially solvable.

**F2 — The expanded GT is now demonstrably anti-signal, not just diluted.** At w50000, 99.5% of expanded GT sat inside the zero-score tie and expanded AUROC ≈ 0.41 was tie-handling arithmetic. At w5000 the ties are gone (9% zeros) and expanded AUROC *collapses to 0.037* — the 12,852 expanded-GT nodes score systematically *below* random negatives, while the 46 raw entities hold AUROC ≈ 0.94. The 2-hop hull around the attack is dominated by high-frequency benign-behaving nodes the model finds maximally unsurprising. Implication: expanded-GT metrics should be reported for comparability only, never optimized against; raw GT is the design target.

**F3 — Memory ablation is null-to-negative everywhere, now confirmed across all window sizes and both ablation types.** Test-time memory on/off differs by ≤0.005 AUPRC / ≤0.02 AUROC in every arm, memory-off is *slightly better* on raw-GT AUROC in all three sizes (e.g. 0.9518 vs 0.9404 at w5000), and the train-time ablation (§2.1) outperforms both w5000 memory arms outright on raw AUPRC (0.0336 vs 0.0185). Even at w5000 — where warm events are 20.8% and the score is no longer saturated — memory contributes nothing (or slightly negative noise). This eliminates "score saturation was masking the memory effect" as the remaining explanation: the score is no longer saturated at w5000, and the ablation is still flat. The cause is now cleanly the architectural one: memory is a small additive residual (norm ≤100 vs conv1 activations ~2,100) on a task that neighborhood types already solve, and GT entities are one-shot (0.18% recurrence) so their own memory is always cold. §5's design responds to exactly this.

**F4 — The detector, as-is, is a rarity ranker.** Raw AUROC ≈ 0.94 with raw AUPRC ≈ 0.02 means attack entities are reliably in the top tail but buried under ~50–100× as many equally-surprising benign rarities. Within-type calibration (D3 follow-up: 0.0068 → 0.027 on the old checkpoint) and the w5000 window size stack; whether they stack together is an open measurement, not assumed.

## 4. What this means for Extension 1's thesis claim

The honest current statement: *windowing + smaller windows improves node-level attack-entity ranking (F1), but cross-window memory — the actual contribution of Extension 1 — has no measurable effect under the current score (F3).* Extension 1 cannot be defended on these numbers. The D1 redesign below is therefore not polish; it is the load-bearing change that gives the memory mechanism a channel through which it *can* matter, so that the ablation becomes a meaningful test of the thesis rather than a foregone null.

## 5. Revised D1 design — history-conditioned behavior scoring (memory structurally load-bearing)

### 5.1 Why the current score cannot exercise memory

Current: `h = ReLU(conv1(x) + memory); score = NLL(conv2-softmax, true type)`, max over windows.

1. **Target leakage:** x is the node's per-window edge-type histogram (56-dim, `feature_num*2` — src-role and dst-role counts per CDM edge type), inherited from the baseline's featurization; y is the node type. A node's type is a near-deterministic function of its own interaction histogram (only processes EXECUTE, only netflows CONNECT…), and with `root_weight=False` that histogram still round-trips to the node's own prediction through its neighbors (own x → neighbor h in conv1 → back via conv2), while neighbor histograms alone are nearly sufficient anyway → 82% exact-zero NLL at w50000. Memory competes against an input channel that already contains the answer.
2. **Scale mismatch:** memory is norm-clamped ≤100, added to conv1 outputs with median norm ~2,100 — a ≤5% perturbation.
3. **Cold-start × max-aggregation:** GT entities recur at 0.18%; their first (usually only) appearance has zero memory, and max-aggregation typically selects that cold moment. The nodes we care about are structurally excluded from ever being scored *with* memory.

The fix cannot be rescaling (F3 shows even the unsaturated w5000 score gains nothing). Memory must move from *additive nudge on a self-sufficient pathway* to *the only node-specific conditioning channel of the prediction task*.

### 5.2 Proposed task: predict this window's behavior from history

**Target.** For node v in window w: the observed **edge-type participation profile** — a 56-dim count vector (28 CDM edge types × {v as src, v as dst}) accumulated over w's events. This is *literally the existing `data.x`* that `build_window_data()` already produces — D1-revised is an exact inversion of the current task (predict x from history + type, instead of y from x), so no new dataset builder is required and the windowing/feature code stays untouched. Loss/score = multinomial NLL of the observed profile under the model's predicted distribution (per-event mean NLL, so score doesn't scale with degree). Continuous target ⇒ no exact-zero saturation floor by construction.

**Inputs (per node v, window w):**
- **(a) v's own decayed memory** — the *only* v-specific historical channel. Zero for cold nodes, and that is informative ("never-before-seen entity"), not an artifact.
- **(b) Neighbor-memory aggregation:** SAGE-style aggregation of the decayed memories of v's *current-window* neighbors. This is the channel the recurrence analysis demands: attack entities are one-shot, but their neighbors (nginx, sendmail, sshd) are persistent hubs whose memories encode what their typical interaction partners do. A fresh process spawned by nginx is scored against nginx's history.
- **(c) v's static node type** (6-dim one-hot) as a coarse class prior — *kept deliberately*, so the memory-off ablation degrades to a per-type behavior prior (a meaningful classical baseline) rather than to a single global distribution. **Nothing else from the current window enters v's prediction** — in particular the current window's own edge-type observations must not reach v's output (they are the target). Implementation constraint: the encoder over inputs (a)+(b) runs on the window's *node set and memory vectors*, using edge_index only as the aggregation structure, never edge-type features of w.

**Memory write-back:** unchanged mechanics (post-aggregation hidden state, detached, decayed, max-norm clamp) — but the stored vector is now trained to be *predictive of future behavior*, because it is the only channel through which window w's information can reduce window w+1's loss. This is the structural load-bearing property: the training objective itself is unsatisfiable-beyond-prior without memory, so gradient pressure must put behavioral history into it. The `--max-norm 0` ablation now removes inputs (a) and (b) entirely and provably reduces the model to the type-prior baseline — the ablation delta becomes exactly "what history buys."

**Scoring & protocol:** per-window per-node score = mean per-event NLL; aggregation across windows stays **max**; evaluation unchanged through `evaluate_windowed.py`, both GT variants, no neighborhood credit, threshold from train quantile. Nothing in the eval protocol moves — only the score-producing model.

**Why this is not MAGIC-shaped:** no masking, no reconstruction of input features or structure, no embedding-space distance scoring. It is a forward *predictive* (history → future observation) likelihood, closer to the TGN/temporal-point-process family than to masked-autoencoder detectors, and the thing being predicted is raw observable behavior (event-type counts), not latent codes.

**Known risks, stated up front:** (i) per-type behavior priors may already be sharp (UNIX sockets do little else than send/recv) — mitigation is that the score is *relative to* that prior, so only deviation ranks high; (ii) hub memories may average over so many partners that they carry little discriminative signal — measurable directly via the ablation delta, which is now an honest instrument; (iii) two hyperparameters (decay rate, max-norm) remain untuned — sweep after the mechanism shows any nonzero delta, not before.

### 5.3 Extension 2 interactions (explicit)

E2 = edge-type-aware convolution (HeteroSAGEConv-style per-relation weights). Under this D1 design:

1. **Natural slot:** E2 replaces the *neighbor-memory aggregation* (input b) — per-edge-type weight matrices decide how a neighbor's memory is read depending on the connecting relation ("memory of my parent process" ≠ "memory of a file I read"). Same module boundary, no change to task, target, or score semantics ⇒ E1-vs-E1+E2 comparison stays clean.
2. **Shared vocabulary:** E2's relation set and D1's 56-dim target both derive from the same 28-edge-type vocabulary (`models/feature.txt` / `build_type_vocab`), so E2 does not introduce a second label space.
3. **Leakage guard carries over:** E2's per-relation weights are conditioned on edge *types of past windows' aggregation into memory* and the current window's *graph structure*; the constraint that current-window edge-type observations of v never reach v's own prediction applies identically — E2 must use relation-aware weights only for routing neighbor memories, not for injecting the target.
4. **Ordering:** implement and ablate D1 first with plain mean aggregation; E2 then has a well-posed measurable question ("does relation-aware memory routing improve history-conditioned prediction?") instead of compounding two untested changes.

## 6. Within-type quantile calibration (folded into reporting, 2026-07-14 eve)

Approved as a reporting-only step (decision 4). Now reproducible via `scripts/calibrate_within_type.py` (+ unit tests): replaces each raw max-NLL score with its mid-rank quantile within its CDM node-type group; flag column passes through; output evaluated by `evaluate_windowed.py` unchanged. All 357,174 scored gids resolve to a type (0 conflicts, 0 unknown).

| Arm | Raw AUPRC (uncal → cal) | Raw AUROC (uncal → cal) | Raw P@46 (uncal → cal) |
|---|---|---|---|
| w50000 mem-on | 0.0076 → **0.0287** | 0.943 → 0.924 | 0.022 → 0.065 |
| w5000 mem-on | 0.0185 → **0.0273** | 0.940 → 0.830 | 0.043 → 0.065 |
| w5000 mem-off (test) | 0.0189 → 0.0217 | 0.952 → 0.842 | 0.065 → 0.065 |
| w5000 trained-nomem | 0.0336 → 0.0152 | 0.944 → 0.875 | 0.043 → 0.065 |

Two things to note. (1) The w50000 lift reproduces the D3 old-checkpoint finding (0.0068 → 0.027) on the refreshed pipeline. (2) **Calibration flips F3's sharpest claim:** the trainnomem arm's uncalibrated AUPRC advantage (0.0336) collapses to 0.0152 under calibration — much of it was cross-type score-scale artifact, not better within-type ranking — while the memory-trained arm improves to 0.0273. Under calibrated reporting, memory-trained ≥ trainnomem. The honest summary of F3 is therefore: *memory's effect is metric-fragile and everywhere small; nothing about it is stable enough to build on* — which still motivates the D1 redesign, but retires "memory actively hurts" as a headline. The seed check (§8) reports both calibrated and uncalibrated numbers per seed.

## 7. Decision list

1. **D1-revised (§5.2):** approve the history-conditioned edge-type-profile score for implementation as new files (`train_windowed2.py`-style separate entry points; current pipeline untouched as the E1-v1 reference)? Naming/placement to your preference.
2. **Input (c):** keep static type one-hot as the memory-off fallback prior (recommended), or go fully history-only?
3. **Sweep default going forward:** adopt w5000 as the default window size for D1-revised experiments (best raw-GT numbers, lowest saturation, highest recurrence), keeping w50000 for baseline comparability?
4. **Within-type calibration:** fold the D3 within-type quantile calibration into the reported scoring path for the current detector (as a documented post-hoc step), or hold it until D1-revised lands?
5. **trainnomem arms for w50000/w10000:** the train-time ablation currently exists only at w5000 — sufficient (recommended, given F3's uniformity), or complete the matrix?

## 8. Addendum (2026-07-14 evening): provisional decisions & seed check

User decisions (provisional, pending seed check): (1) D1-revised approved in principle — start as new files, existing pipeline untouched; (2) keep static type one-hot as memory-off fallback; (3) w5000 default window size going forward; (4) within-type calibration folded into reporting now (done, §6); (5) no trainnomem arms at w50000/w10000.

**Seed-stability check (requested before final D1 sign-off):** seeds {101, 202, 303} × {mem-trained, trainnomem} at w5000, mem arm also scored with test-time memory off (9 score sets). Driver + seed wrapper: `models/windowed_cadets/sweep/seedcheck/{run_seedcheck.sh, seed_train.py}` (seeds set externally; `train_windowed.py` itself sets no seed, so all earlier arms were OS-entropy seeded). Results:

Raw-GT AUPRC per seed (uncal = raw max-NLL score, cal = within-type quantile calibrated):

| Arm | s101 | s202 | s303 | mean | original (OS-seeded) |
|---|---|---|---|---|---|
| mem-trained, test mem-on (uncal) | 0.0112 | 0.0138 | 0.0114 | 0.0121 | 0.0185 |
| mem-trained, test mem-off (uncal) | 0.0106 | 0.0127 | 0.0113 | 0.0115 | 0.0189 |
| trainnomem (uncal) | 0.0354 | 0.0326 | 0.0331 | **0.0337** | 0.0336 |
| mem-trained, test mem-on (cal) | 0.0123 | 0.0169 | 0.0240 | 0.0177 | 0.0273 |
| mem-trained, test mem-off (cal) | 0.0090 | 0.0090 | 0.0167 | 0.0116 | 0.0217 |
| trainnomem (cal) | 0.0154 | 0.0163 | 0.0163 | 0.0160 | 0.0152 |

Calibrated AUROC is likewise consistently higher for trainnomem (0.866–0.878) than mem-trained (0.817–0.845); calibrated P@46 is 0.0652 for every arm at every seed.

**Verdict:**
1. **Uncalibrated, "memory hurts" is a stable finding, not noise:** trainnomem beats mem-trained at every seed by ~2.8× (0.033–0.035 vs 0.011–0.014, non-overlapping ranges). The original mem arm's 0.0185 was a favorable draw.
2. **Calibrated, the §6 flip does NOT survive seeds:** mem-trained calibrated AUPRC is high-variance (0.0123–0.0240; the original 0.0273 was another favorable draw), trainnomem is remarkably stable (0.0154–0.0163); means overlap (0.0177 vs 0.0160) with the winner changing per seed. No stable ordering either way.
3. **Test-time memory off is never distinguishable from on** (≤0.003 uncal), consistent with all earlier arms.
4. The across-the-board pattern: the memoryless arm is *stable* across seeds in every metric; the memory-trained arm adds seed variance without adding any stable ranking benefit. The settled summary for the report and thesis: **under v1, cross-window memory contributes no reproducible detection value on any reporting variant, and destabilizes training** — which is precisely the situation the D1-revised redesign addresses. The seed check does not change the story; it sharpens it.

**D1-revised implementation (started per decision 1, new files only):**
- `scripts/train_behavior.py` — `BehaviorNet` + windowed training loop. Memory row = `[hidden(32) ; last normalized profile(56)]`; read side learned (SAGE aggregation of neighbor memories with `root_weight=False`, plus own-memory linear and type-prior linear), write side parameter-free (detach-compatible — a learned write transform would receive no gradient). Has `--seed`. Default `--window-size 5000` per decision 3.
- `scripts/test_behavior.py` — identical protocol to `test_windowed.py` (max aggregation, cold start, train-quantile flag); imports the shared protocol helpers from `test_windowed.py` so the two scorers cannot drift. Output format unchanged ⇒ `evaluate_windowed.py` and `calibrate_within_type.py` apply as-is.
- `scripts/test_train_behavior.py` — 6 structural unit tests, all passing, including the two load-bearing guarantees: zero memory ⇒ same-type nodes get *identical* predictions (provable collapse to type prior), and predictions are independent of the observed profile (no target leakage path exists).
- Smoke run (2 epochs, w5000): trains cleanly, val per-event NLL 1.80 → 1.11 (uniform floor log 56 ≈ 4.03), no zero-saturation by construction. Scoring smoke on the throwaway 2-epoch checkpoint: 357,174 nodes scored, 0.00% exact-zero scores, and — treat as preliminary, single OS-seeded run, untuned — **raw-GT AUPRC 0.0399, P@46 0.152 (7/46), AUROC 0.920 uncalibrated**, i.e. already above every v1 arm's best number under either reporting (v1 best: 0.0336 uncal / 0.0287 cal, P@46 0.065). Not a substitute for the approved full experiment; recorded here only as an early signal that the history-conditioned target carries real information.

Full training/evaluation of D1-revised is HELD until final user go-ahead post-seed-check.
