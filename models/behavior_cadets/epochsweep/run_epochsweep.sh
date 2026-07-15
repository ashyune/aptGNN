#!/bin/bash
# D1-revised epoch/checkpoint sweep (2026-07-15, user-approved).
# Purpose: evaluate the PREREGISTERED checkpoint-selection rule (delta-max;
# see PREREGISTRATION.md in this directory) -- NOT to pick the best epoch
# from the raw-GT curve.
#
# 3 seeds x mem-on (--max-norm 100) at w5000, 30 epochs with
# --save-every-epoch (per-epoch checkpoints + delta logging), then a full
# test pass at grid epochs {1,2,3,5,10,15,20,30} and uncalibrated +
# within-type-calibrated evaluation against both GT variants.
set -u
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
ES=../models/behavior_cadets/epochsweep
RAWGT=../models/windowed_cadets/groundtruth_raw_global_id.txt
GRID="001 002 003 005 010 015 020 030"

for s in 101 202 303; do
  d=$ES/s$s
  mkdir -p $d
  echo "===== seed $s train (mem-on, --save-every-epoch) ====="
  python train_behavior.py --seed $s --window-size 5000 --max-norm 100.0 \
    --out-dir $d --save-every-epoch || { echo "TRAIN_FAILED s$s"; continue; }
  for e in $GRID; do
    echo "===== seed $s test @ epoch $e ====="
    python test_behavior.py --window-size 5000 --max-norm 100.0 \
      --checkpoint $d/model_epoch$e.pt --out-dir $d/eval_e$e \
      || echo "TEST_FAILED s$s e$e"
  done
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
echo EPOCHSWEEP_DONE
