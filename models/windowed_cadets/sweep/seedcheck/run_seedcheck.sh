#!/bin/bash
# Seed-stability check for the w5000 memory-ablation finding (2026-07-14).
# 3 seeds x {mem-trained (max-norm 100), trainnomem (max-norm 0)};
# mem arm additionally scored with test-time memory off.
cd /home/tetsuya/aptGNN/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate threatrace
SC=../models/windowed_cadets/sweep/seedcheck
RAWGT=../models/windowed_cadets/groundtruth_raw_global_id.txt
for s in 101 202 303; do
  for arm in mem trainnomem; do
    d=$SC/s$s/$arm
    mkdir -p $d
    if [ $arm = mem ]; then MN=100.0; else MN=0; fi
    echo "===== seed $s arm $arm train (max-norm $MN) ====="
    python $SC/seed_train.py $s --window-size 5000 --max-norm $MN --out-dir $d || { echo "TRAIN_FAILED s$s $arm"; continue; }
    echo "===== seed $s arm $arm test (max-norm $MN) ====="
    python test_windowed.py --window-size 5000 --max-norm $MN --checkpoint $d/best_model.pt --out-dir $d/eval || echo "TEST_FAILED s$s $arm"
    if [ $arm = mem ]; then
      echo "===== seed $s arm mem test with memory off ====="
      python test_windowed.py --window-size 5000 --max-norm 0 --checkpoint $d/best_model.pt --out-dir $d/eval_nomem || echo "TEST_FAILED s$s mem nomem"
    fi
  done
done
echo "===== metrics ====="
for f in $(find $SC -name scores_windowed.txt | sort); do
  echo "--- $f (raw GT) ---"
  python evaluate_windowed.py --scores-file $f --groundtruth-file $RAWGT
  echo "--- $f (expanded GT) ---"
  python evaluate_windowed.py --scores-file $f --groundtruth-file $(dirname $f)/groundtruth_global_id.txt
done
echo SEEDCHECK_DONE
