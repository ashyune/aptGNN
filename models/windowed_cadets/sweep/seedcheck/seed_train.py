"""Seed-check wrapper: fixes RNG seeds, then runs train_windowed.main()
unmodified. Usage (from scripts/): python seed_train.py <seed> [train_windowed args...]"""
import os, sys, random
import numpy as np
import torch

seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

sys.path.insert(0, os.getcwd())
sys.argv = ['train_windowed.py'] + sys.argv[2:]
import train_windowed
train_windowed.main()
