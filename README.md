# aptGNN — Setup & Run Guide

Branch: `cwm-2`

This README documents how to set up a fresh machine and reproduce every result
(baseline, E1, E2, E3). See `reports/` for full write-ups of each experiment.

## 0. Setup (once per machine)

```bash
git clone <repo-url>
cd aptGNN
git checkout cwm-2
conda activate threatrace
```

Confirm you're on the right commit:

```bash
git log --oneline -3
```

## 1. Data prep (one-time, slow)

**Skip this entire step if `cadets_train.txt` / `cadets_test.txt` already exist**
(e.g. copy them over from another machine instead of re-parsing ~15GB of raw
tarballs — much faster).

If parsing from scratch is required:

```bash
cd scripts
python parse_darpatc.py      # raw CDM JSON -> 6-column TSVs (slow)
python node_vocab.py --scene cadets    # builds the global UUID -> node-id vocab
```

## 2. Run the experiments

Use the driver scripts — they already have the correct seeds/flags baked in.
Do not call `train_behavior.py` manually per-seed; use these instead.

```bash
# E1 — headline result: cross-window memory (SAGE read), + type-prior ablation
bash ../models/behavior_cadets/seedrun/run_behavior_seedrun.sh

# E2 — relational/heterogeneous extension (edge-type-aware neighbor-memory routing)
bash ../models/behavior_cadets/e2seedrun/run_e2_seedrun.sh

# E3 — feature enrichment (n_distinct / rep_ratio / span_frac added to the profile)
bash ../models/behavior_cadets/e3seedrun/run_e3_seedrun.sh
```

Each script runs all 3 seeds {101, 202, 303} internally and writes scores +
logs to its own directory (`seedrun/`, `e2seedrun/`, `e3seedrun/`).

**Optional — diagnostic side-experiments, not needed for core paper numbers:**

```bash
bash ../models/behavior_cadets/epochsweep/run_epochsweep.sh   # Δ-max checkpoint rule (rejected)
bash ../models/behavior_cadets/aggscoring/run_aggscoring.sh    # window max/p95 aggregation (rejected)
```

## 3. Evaluate / verify numbers

```bash
python evaluate_windowed.py \
  --scores-file <path_to_scores> \
  --groundtruth-file ../models/windowed_cadets/groundtruth_raw_global_id.txt
```

Should reproduce the AUPRC / AUROC / P@46 numbers already in
`reports/2026-07-14_*`, `2026-07-16_*`, `2026-07-17_*`.

Optional within-type calibrated variant (reported for protocol completeness;
known to hurt all arms):

```bash
python calibrate_within_type.py --scores-file <scores>
```

## 4. Sanity-check the code itself (optional but recommended on a new machine)

```bash
python test_train_behavior.py       # 16 structural tests — E1/E2
python test_train_behavior_e3.py    # 9 structural tests — E3
```

All should pass with no errors. These check leakage guards, ablation collapse
to the type prior, etc. — not the experiment results themselves.

## GPU note

**Every existing number in every report is CPU-only, by design — not an
oversight.** `torch` in this env is CPU-only (`cuda.is_available() == False`).
The model is tiny (~8–23k parameters, ~3s/epoch), so CPU is not a bottleneck.

If running on a GPU machine or a teammate's laptop with a GPU:
- PyTorch will auto-select CUDA if available.
- **This may change results slightly** — seeded bit-exact reproducibility was
  only verified on CPU; some GPU ops (e.g. `index_add_`/scatter) are not
  deterministic the same way CPU ops are.
- To keep results comparable to existing reports, prefer forcing CPU explicitly
  if the script exposes a `--device` flag (confirm before running on GPU).
- The paper should state: **CPU, single machine, seeds {101, 202, 303}.**

## File map (`scripts/`)

### Current — reproduces paper numbers

| File | Role |
|---|---|
| `parse_darpatc.py` | Stage 0: raw CDM JSON → TSVs |
| `node_vocab.py` | Stage 1: global UUID → node-id vocab |
| `windowing.py` | Library: time-sorts + chunks into windows (imported only) |
| `windowed_data.py` | Library: builds per-window PyG `Data` objects (imported only) |
| `node_memory.py` | Library: cross-window per-node memory w/ decay (imported only) |
| `train_behavior.py` | **Headline model** (E1 / E2 via `--relational` flag) |
| `test_behavior.py` | Scores test set under a frozen checkpoint |
| `window_features.py` | Library: E3's binned features (imported only) |
| `train_behavior_e3.py` / `test_behavior_e3.py` | E3 trainer/scorer |
| `evaluate_windowed.py` | Computes AUPRC/AUROC/P@\|GT\| vs a GT file |
| `calibrate_within_type.py` | Within-type calibrated reporting variant |
| `ranking_metrics.py` | Library: verified metric implementations (imported only) |
| `train_windowed.py` / `test_windowed.py` | v1 Extension-1 stack — the paper's v1 negative result; still current, shares protocol helpers |
| `score_behavior_aggs.py` | Aggregation experiment scorer (rejected, kept for reproducibility) |
| `test_*.py` (unit tests) | Structural guarantees — run directly with `python <file>` |

**Note:** files named `test_windowed.py`, `test_behavior.py`, `test_behavior_e3.py`,
`test_darpatc.py` are **scoring scripts** ("test" = test-set inference), NOT unit
tests. The actual unit tests are `test_train_*.py` / `test_node_*.py` /
`test_ranking_metrics.py` / etc.

### Diagnostics — historical record, not needed for runs

`diagnose_ground_truth_footprint.py`, `diagnostic2_alarm_footprint.py`,
`diagnostic3_random_baseline.py`, `d3-1.py`, `d4.py`, `d5.py`, `d6.py`,
`diagnose_feature_separation.py` — back specific report sections; kept for
provenance.

### Deprecated / frozen — never modify, not used by extension work

Original ThreaTrace baseline: `data_process_train.py`, `data_process_test.py`,
`train_darpatc.py`, `test_darpatc.py`, `evaluate_darpatc.py`, `setup.py`,
`moniter.py`, plus streamspot/unicornsc parsers (datasets not on disk).

Stray baseline **artifacts** (outputs, not code): `alarm.txt`,
`groundtruth_nodeId.txt`, `groundtruth_uuid.txt`, `id_to_uuid.txt`,
`training_log.txt`.

## Ground truth

Raw GT (`models/windowed_cadets/groundtruth_raw_global_id.txt`) derives from
`groundtruth/cadets_raw_uuid.txt` (72 UUIDs; provenance documented in
`groundtruth/cadets_raw_SOURCE.md`), resolved through the Stage-1 vocab.
46 of 72 resolve into the dataset. **This is the raw/conservative protocol**,
not the 2-hop expanded ground truth — see `reports/` for why that distinction
matters.

## Results summary

| Experiment | Result |
|---|---|
| E1 (cross-window memory) | **Positive** — ~9× the type-prior ablation, reproducible across 3 seeds |
| E2 (relational/heterogeneous routing) | **Null** — no reproducible detection improvement over E1, despite better validation-NLL |
| E3 (feature enrichment) | **Mixed** — improves broad-rank detection (top ~200–1000), does not improve narrow top-46 precision |

Full per-seed tables and methodology in `reports/2026-07-14` through
`2026-07-17`.
