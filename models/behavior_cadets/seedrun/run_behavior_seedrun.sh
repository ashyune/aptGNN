#!/bin/bash
# Full D1-revised experiment (2026-07-14, user-approved):
# 3 seeds x {mem-on (max-norm 100), memory-off ablation (max-norm 0)}
# at w5000, 30 epochs, then uncalibrated + within-type-calibrated
# evaluation against both GT variants.
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
SR=../models/behavior_cadets/seedrun
RAWGT=../models/windowed_cadets/groundtruth_raw_global_id.txt
for s in 101 202 303; do
  for arm in mem nomem; do
    if [ $arm = mem ]; then MN=100.0; else MN=0; fi
    d=$SR/s$s/$arm
    mkdir -p $d
    echo "===== seed $s arm $arm train (max-norm $MN) ====="
    python train_behavior.py --seed $s --window-size 5000 --max-norm $MN --out-dir $d || { echo "TRAIN_FAILED s$s $arm"; continue; }
    echo "===== seed $s arm $arm test (max-norm $MN) ====="
    python test_behavior.py --window-size 5000 --max-norm $MN --checkpoint $d/best_model.pt --out-dir $d/eval || echo "TEST_FAILED s$s $arm"
  done
done
echo "===== metrics ====="
for f in $(find $SR -name scores_windowed.txt | sort); do
  d=$(dirname $f)
  echo "--- $d uncalibrated raw GT ---"
  python evaluate_windowed.py --scores-file $f --groundtruth-file $RAWGT
  echo "--- $d uncalibrated expanded GT ---"
  python evaluate_windowed.py --scores-file $f --groundtruth-file $d/groundtruth_global_id.txt
  python calibrate_within_type.py --scores-file $f
  echo "--- $d calibrated raw GT ---"
  python evaluate_windowed.py --scores-file $d/scores_windowed_caltype.txt --groundtruth-file $RAWGT
  echo "--- $d calibrated expanded GT ---"
  python evaluate_windowed.py --scores-file $d/scores_windowed_caltype.txt --groundtruth-file $d/groundtruth_global_id.txt
done
echo BEHAVIOR_SEEDRUN_DONE
