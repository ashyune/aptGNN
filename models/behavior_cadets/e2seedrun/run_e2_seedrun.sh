#!/bin/bash
# Extension 2 seeded experiment (2026-07-15, user-approved):
# 3 seeds x relational mem-on (--relational, --max-norm 100) at w5000,
# 30 epochs, checkpoint = best val NLL (same rule as the E1 headline;
# the delta-max rule failed its preregistered evaluation and is NOT
# used). Uncalibrated + within-type-calibrated evaluation against both
# GT variants. E1 comparison arms = models/behavior_cadets/seedrun/.
set -u
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
ES=../models/behavior_cadets/e2seedrun
RAWGT=../models/windowed_cadets/groundtruth_raw_global_id.txt
for s in 101 202 303; do
  d=$ES/s$s
  mkdir -p $d
  echo "===== seed $s E2 train (relational, max-norm 100) ====="
  python train_behavior.py --seed $s --relational --window-size 5000 \
    --max-norm 100.0 --out-dir $d || { echo "TRAIN_FAILED s$s"; continue; }
  echo "===== seed $s E2 test ====="
  python test_behavior.py --relational --window-size 5000 --max-norm 100.0 \
    --checkpoint $d/best_model.pt --out-dir $d/eval || echo "TEST_FAILED s$s"
done
echo "===== metrics ====="
for f in $(find $ES -name scores_windowed.txt | sort); do
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
echo E2_SEEDRUN_DONE
