import os
from pathlib import Path

#step1: define environment variable GRAPHCHI_ROOT
#############################################
graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
os.environ['GRAPHCHI_ROOT'] = graphchi_root



#step2: clean models directory ../models
#############################################
model_dir = Path('../models')
model_dir.mkdir(parents=True, exist_ok=True)

for path in model_dir.iterdir():
    if path.is_file():
        path.unlink()

for path in Path('.').glob('result_*'):
    if path.is_file():
        path.unlink()