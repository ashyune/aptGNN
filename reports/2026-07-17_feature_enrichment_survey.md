# Cheap-feature enrichment survey: what's already parsed, and would it separate the raw GT? — 2026-07-17

Question posed: the 2026-07-16 aggregation test concluded the 10× operating-point gap
to Orthrus is representational — the 56-dim edge-type participation profile carries too
little information — so, before building anything: what per-node, per-window information
is ALREADY in the parsed data but unused, how cheap is each candidate, and is it
plausible each would separate the 46 raw-GT nodes? Report only; no pipeline changes.
Plausibility here is **measured, not guessed**: a new read-only diagnostic
(`scripts/diagnose_feature_separation.py`, exact pipeline windowing and row filter,
per-node MAX over windows, tie-aware ranks) computed every candidate for all 357,174
test nodes and ranked the 46 GT nodes against the population on each raw feature alone.

## 1. Inventory: what the parsed data contains

The parsed TSVs (`cadets_{train,test}.txt`) carry exactly six fields per event row:
`src_uuid, src_type, dst_uuid, dst_type, edge_type, timestamp(ns)`
(`parse_darpatc.py`). Of these, the current model consumes:

- **edge_type × role** → the 56-dim profile `x` (target + memory writeback),
- **node type** → `y` / type one-hot,
- **uuid pairs** → `edge_index` for message passing (identities are *routed over*,
  never *counted*),
- **timestamp** → used ONLY to sort rows into windows, then discarded.

So two whole information channels sit unused in data already on disk: **partner-identity
statistics** (degree/fan-out/re-use — how many distinct things a node touches) and
**within-window timing** (burstiness, active span, ordering). Everything richer —
file paths, process names/cmdlines, ports/IPs, event sizes/return values — was dropped
at parse time; the raw CDM JSON tarballs still exist
(`graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official*.json.tar.gz`, ~0.9 GB),
so those are recoverable but need a new parse pass + vocab design (hours-scale; this is
the semantic-attribute axis Orthrus-class systems actually use).

## 2. Candidates: cost, additivity, and MEASURED separation

GT composition (all 46 present in test windows): 26 SUBJECT_PROCESS,
14 FILE_OBJECT_FILE, 6 NetFlowObject.

Global tie-aware ranking of GT nodes by each raw feature (per-node max over windows;
optimistic rank = 1 + #strictly-greater / pessimistic = #greater-or-equal):

| feature | GT in top-46 (opt/pes) | GT median rank | best GT ranks (opt) |
|---|---|---|---|
| n_events | 2 / 2 | 10,847 | 3, 5, 64, 111, 154 |
| n_distinct | 2 / 2 | 25,232 | 1, 5, 208, 1885 |
| rep_ratio | 3 / 3 | 37,064 | 17, 26, 37, 259 |
| fan_out | 2 / 2 | 25,118 | 1, 5, 1844 |
| fan_in | 2 / 2 | 2,293 | 1, 9, 132, 383 |
| n_etypes | 0 / 0 | 25,644 | 4,144 |
| burst_cv | 2 / 2 | 24,901 | 1, 7, 121 |
| span_frac | 7 / 0 | 25,856 | 78-node tie block at 1.0 contains 7 GT |

Within-type median GT percentile (the right proxy for a type-prior/history-conditioned
model; higher = more extreme):

| type | n_events | n_distinct | rep_ratio | fan_out | fan_in | n_etypes | burst_cv | span_frac |
|---|---|---|---|---|---|---|---|---|
| SUBJECT_PROCESS (26 GT / 26,297) | 61.3 | 59.4 | 61.4 | 59.4 | 3.7 | 10.1 | 16.1 | 50.2 |
| FILE_OBJECT_FILE (14 GT / 89,376) | 76.3 | 99.1 | 60.8 | 0.0 | 99.1 | 96.7 | 86.9 | 99.4 |
| NetFlowObject (6 GT / 7,076) | 99.4 | 0.0 | 99.9 | 0.0 | 0.0 | 9.3 | 99.1 | 99.2 |

Headline findings:

- **span_frac is the single strongest signal found in this project so far.** Only 78
  nodes in the entire test set are ever active from the first to the last event of a
  5000-row window; **7 of them are GT** (3 files + 4 processes). That tie block alone is
  9.0% precision at a 78-alarm budget — better than any model operating point we have
  (flag 1.1–1.9%, top-46 6.5%).
- **GT files deviate on fan-in**: touched by unusually many distinct processes
  (99.1th within-type percentile; two at global rank 1 and 9) over unusually long spans
  (span_frac 99.4th).
- **GT netflows deviate on volume/re-use**: 99.4th percentile events, 99.9th
  rep_ratio, 99.1th burstiness.
- **GT processes — 26 of 46, the majority — are statistically ordinary** at window
  granularity: 50–61st within-type percentile on every activity feature, and *below*
  median on fan-in, n_etypes, burst_cv. Only 4–5 individual processes (e.g.
  47E61FFC…, 1131CD42…, 7CED5514…, 7CF2DDA4…) are extreme on anything.
- **n_etypes carries nothing** (0 in top-46; ≤10th percentile for processes/netflows) —
  unsurprising, it's a coarsening of the profile the model already scores.

Per-candidate cost/additivity summary:

| candidate | raw data present? | additive? | measured verdict |
|---|---|---|---|
| n_distinct / fan_in / fan_out | yes (uuid pairs) | yes — new columns from same window rows | **signal** (files) |
| rep_ratio (events per distinct partner) | yes (derived) | yes | **signal** (netflows; 3 GT top-46 alone) |
| span_frac (active-span fraction) | yes (timestamps) | yes | **signal** (7 GT in 78-node block) |
| burst_cv (inter-event gap CV) | yes (timestamps) | yes | signal, but correlated with span_frac |
| n_events | yes | yes | redundant: = rep_ratio × n_distinct |
| n_etypes | yes | yes | **no signal** |
| edge-type sequence bigrams | yes (ordering) | no — 28²-dim target restructuring; ns-timestamp ties make order partly arbitrary | not measured; not first-pass |
| cross-window partner novelty | yes | needs cross-window state | confounded: GT is one-shot (0.18% recurrence), so novelty is trivially maximal for GT *and* every cold-start benign node |
| paths / names / ports / sizes | **no — needs re-parse of raw JSON** (tarballs on disk) | no — new parser, vocab, embedding | the real representational axis; separate decision |

## 3. How new features must enter the model (leakage constraint)

Current-window statistics of a node's own events must **not** be inputs to `forward()`
— the score is surprise-given-history, and `data.x` (with anything correlated with it)
is the target. New features therefore enter exactly the way the profile itself does:

1. **Target side (the part that changes the score):** each feature discretized into a
   small bin histogram (e.g. log2 buckets), predicted by its own small softmax head;
   node score = profile NLL + unweighted mean of the per-feature bin NLLs (fixed
   weighting, preregistered — no mixing-weight tuning).
2. **Memory side:** the same binned observations appended to the parameter-free
   writeback `[h ; profile ; feature bins]`, so next-window predictions are conditioned
   on them (NodeMemory is dimension-generic).

Consequence: **retraining is required** — checkpoint shapes change. Measured cost is
small: E1 ≈ 3 s/epoch, E2 ≈ 12 s/epoch × 30 epochs, so a full 3-seed enriched arm is
~15–30 min end-to-end including windowing and eval. Code cost: a window-feature builder
(wrapper over `generate_windows`, `windowed_data.py` untouched) + an enriched
train/test pair (new files or flags alongside `train_behavior.py` / `test_behavior.py`);
E1/E2 checkpoints stay frozen as comparison arms.

## 4. Recommended first pass (3 features) and honest ceiling

Recommended minimal set — one per measured signal direction:

1. **n_distinct** (log2-binned; optionally split fan_in/fan_out) — GT files.
2. **rep_ratio** (log2-binned) — GT netflows.
3. **span_frac** (binned, with a dedicated ==1.0 bin) — GT files + the only processes
   that light up on anything.

Excluded from the first pass: n_events (deterministic function of 1×2), burst_cv
(correlated with span_frac; add later only if the first pass moves), n_etypes (no
signal), bigrams/novelty/raw-JSON semantics (not cheap or confounded, above).

**Ceiling, stated plainly:** these features can plausibly help the 14 files and 6
netflows, plus ~4–5 unusual processes — and they are largely *complementary* to what
the model already finds: the current top-46 GT hits (union over arms: 1902FFA3…,
5453C813… — near-silent files with 1–4 events found by profile surprise — plus
1633500B…, 48289024…, 11C64B2C…) overlap the feature-extreme set only at 1633500B/
11C64B2C. Realistic best case is therefore P@46 moving from 3 to ~6–9 (precision
0.065 → ~0.13–0.20), i.e. **a plausible doubling, not gap closure**: 26 of 46 GT
nodes are statistically ordinary processes that no cheap window statistic separates,
so the median GT rank will not approach 46 and Orthrus-level 0.52 precision is out of
reach on this axis. Worth building only with that expectation set.

## 5. Status

- Read-only diagnostic vendored at `scripts/diagnose_feature_separation.py` (re-run
  reproduces every number above; ~25 s). No pipeline files touched; no training run.
- Decision pending: whether to build the 3-feature enriched arm (E3) under the design
  in §3, 3 seeds, same protocol/metrics as E1/E2.
