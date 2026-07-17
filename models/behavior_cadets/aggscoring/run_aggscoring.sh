#!/bin/bash
# Aggregation-variant scoring experiment (2026-07-16): does replacing
# the per-window MEAN per-event NLL with MAX / p95 (computed from the
# SAME frozen checkpoints, no retraining) fix the rank dilution the
# operating-point comparison diagnosed? All 6 seedrun arms (E1 mem +
# E2 relational, seeds 101/202/303), one inference pass per arm; the
# recomputed mean must match the arm's existing scores_windowed.txt
# (faithfulness regression check inside score_behavior_aggs.py).
set -u
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
OUT=../models/behavior_cadets/aggscoring
for s in 101 202 303; do
  echo "===== E1 mem s$s ====="
  python score_behavior_aggs.py \
    --checkpoint ../models/behavior_cadets/seedrun/s$s/mem/best_model.pt \
    --window-size 5000 --max-norm 100.0 \
    --ref-scores ../models/behavior_cadets/seedrun/s$s/mem/eval/scores_windowed.txt \
    --out-dir $OUT/e1_s$s || echo "FAILED e1_s$s"
done
for s in 101 202 303; do
  echo "===== E2 rel s$s ====="
  python score_behavior_aggs.py --relational \
    --checkpoint ../models/behavior_cadets/e2seedrun/s$s/best_model.pt \
    --window-size 5000 --max-norm 100.0 \
    --ref-scores ../models/behavior_cadets/e2seedrun/s$s/eval/scores_windowed.txt \
    --out-dir $OUT/e2_s$s || echo "FAILED e2_s$s"
done
echo AGGSCORING_DONE
