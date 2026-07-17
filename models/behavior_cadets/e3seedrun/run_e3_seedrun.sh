#!/bin/bash
# Extension 3 seeded experiment (2026-07-17, user-approved):
# 3 seeds x enriched mem-on (binned n_distinct / rep_ratio / span_frac
# as extra scored-target heads + memory writeback dims; bins and the
# equal-weight NLL combination are preregistered constants LOCKED in
# window_features.py / train_behavior_e3.py -- this driver has no knob
# for them by design). w5000, 30 epochs, checkpoint = best val NLL on
# the chronological train tail (same rule as E1/E2, applied to the
# combined NLL). Uncalibrated + within-type-calibrated evaluation
# against both GT variants. Comparison arms: E1 =
# models/behavior_cadets/seedrun/, E2 = models/behavior_cadets/e2seedrun/.
set -u
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
ES=../models/behavior_cadets/e3seedrun
RAWGT=../models/windowed_cadets/groundtruth_raw_global_id.txt
for s in 101 202 303; do
  d=$ES/s$s
  mkdir -p $d
  echo "===== seed $s E3 train (enriched, max-norm 100) ====="
  python train_behavior_e3.py --seed $s --window-size 5000 \
    --max-norm 100.0 --out-dir $d || { echo "TRAIN_FAILED s$s"; continue; }
  echo "===== seed $s E3 test ====="
  python test_behavior_e3.py --window-size 5000 --max-norm 100.0 \
    --checkpoint $d/best_model.pt --out-dir $d/eval || echo "TEST_FAILED s$s"
done
echo "===== metrics ====="
for f in $(find $ES -path "*/s[0-9]*/eval/scores_windowed.txt" | sort); do
  d=$(dirname $f)
  echo "--- $d uncalibrated raw GT ---"
  python evaluate_windowed.py --scores-file $f --groundtruth-file $RAWGT
  echo "--- $d uncalibrated expanded GT ---"
  python evaluate_windowed.py --scores-file $f \
    --groundtruth-file $d/groundtruth_global_id.txt
  python calibrate_within_type.py --scores-file $f
  echo "--- $d calibrated raw GT ---"
  python evaluate_windowed.py --scores-file $d/scores_windowed_caltype.txt \
    --groundtruth-file $RAWGT
  echo "--- $d calibrated expanded GT ---"
  python evaluate_windowed.py --scores-file $d/scores_windowed_caltype.txt \
    --groundtruth-file $d/groundtruth_global_id.txt
done
echo E3_SEEDRUN_DONE
