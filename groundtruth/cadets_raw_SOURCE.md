# cadets_raw_uuid.txt — provenance and citation

Raw (un-expanded) malicious **entity** ground truth for DARPA TC Engagement 3,
CADETS host. One CDM UUID per line, 72 unique entities. This is the
entity-level label set, in contrast to `cadets.txt` (12,858 lines), which is
ThreaTrace's published label set produced by 2-hop neighborhood expansion
around the attack entities and therefore measures a structural footprint, not
attack membership.

## Source

Files retrieved 2026-07-14 from the ProvenanceAnalytics ground-truth
repository (the KAIROS authors' curated DARPA TC labels), which the ORTHRUS
artifact (USENIX Security 2025) vendors verbatim as its `Ground_Truth/darpa`
submodule:

- Repository: https://github.com/ProvenanceAnalytics/ground-truth
- Commit: `59d2d1a99ca8a6f36a893d3795d4760e6da79150` (branch `main`, HEAD at
  retrieval time)
- Files (union, first-appearance order, deduplicated across files):
  - `darpa/E3-CADETS/node_Nginx_Backdoor_06.csv` — 8 entities (2018-04-06 attack)
  - `darpa/E3-CADETS/node_Nginx_Backdoor_12.csv` — 43 entities (2018-04-12 attack)
  - `darpa/E3-CADETS/node_Nginx_Backdoor_13.csv` — 24 entities (2018-04-13 attack)
  - 3 UUIDs appear in more than one file (entities persisting across attack
    days), hence 75 rows → 72 unique UUIDs.
- Referenced via: https://github.com/ubc-provenance/orthrus (submodule pin
  observed at orthrus HEAD `e7f25dfee1ddd182a955b88f8a90a8cbd4a8e543`).

Source CSV row format: `UUID,{'kind': 'description'},node_id` — only the UUID
column is used here; the description (process name / file path / 5-tuple) and
the source repo's internal node id are dropped.

## Papers to cite

- KAIROS: "KAIROS: Practical Intrusion Detection and Investigation using
  Whole-system Provenance", IEEE S&P 2024 — origin of the curated label set.
- ORTHRUS: "ORTHRUS: Achieving High Quality of Attribution in
  Provenance-based Intrusion Detection Systems", USENIX Security 2025 —
  adopts the same entity-level label set for DARPA TC evaluation.

## Coverage caveat for this repo's cadets split

This repo's `cadets_train.txt` covers 2018-04-04 07:54 → 2018-04-05 17:35 UTC
and `cadets_test.txt` covers 2018-04-11 20:36 → 2018-04-13 02:24 UTC (see
parse_darpatc.py's shard selection). Consequently the 2018-04-06 attack (8
entities) falls entirely between the two captures, and the 2018-04-13 attack
(24 entities) falls mostly after the test capture ends. Expect only the
2018-04-12 attack's entities (plus any cross-day persistent entities) to
resolve against the node vocabulary; always report the resolved count next to
any metric computed from this file.
