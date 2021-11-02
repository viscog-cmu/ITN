# use this script to delete all remnants of a model specified by its base_fn

import argparse
import os
import shutil
import sys
sys.path.append('.')
from topographic.config import SAVE_DIR, FIGS_DIR, ACTS_DIR, TENSORBOARD_DIR

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--base-fn', type=str)
parser.add_argument('--backup-dir', type=str, default=None, help='place to store model and results files, but not the rest of things')
args = parser.parse_args()
base_fn = args.base_fn

# specific files to delete
files = [
    os.path.join(SAVE_DIR, 'models', base_fn + '.pkl'),
    os.path.join(SAVE_DIR, 'results',  base_fn+'_losses.pkl'),
]

# folders to delete all contents of
folders = [
    os.path.join(SAVE_DIR, 'results/floc',  base_fn),
    os.path.join(SAVE_DIR, 'searchlights',  base_fn),
    os.path.join(ACTS_DIR, base_fn),
    os.path.join(TENSORBOARD_DIR, base_fn),
    os.path.join(FIGS_DIR, 'topographic/EI/summaries', base_fn)
]

for file in files:
    try:
        if args.backup_dir is not None:
            new_file = file.replace(SAVE_DIR, args.backup_dir)
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            shutil.copyfile(file, new_file)
        os.remove(file)
    except Exception as e:
        print(e)

for folder in folders:
    try:
        shutil.rmtree(folder)
    except Exception as e:
        print(e)
