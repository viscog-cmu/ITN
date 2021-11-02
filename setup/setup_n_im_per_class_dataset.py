import argparse 
import os
import glob
import sys
sys.path.append('.')
from topographic.config import IMS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--topdir', type=str, default=IMS_DIR)
    parser.add_argument('--phase', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--testset', action='store_true', 
        help='use to create an equal sized independent test set excluding all images in the dataset that would not have this flag',
        )
    args = parser.parse_args()

    new_dataset = f'{args.dataset}_{args.n}-ims-per-class'
    if args.testset:
        new_dataset += '_test'
    folders = glob.glob(args.topdir + '/' + args.dataset + f'/{args.phase}/*')
    for folder in folders:
        files = glob.glob(f'{folder}/*')
        if not args.testset:
            for ii in range(args.n):
                file = files[ii]
                new_dir = os.path.join(args.topdir, new_dataset, os.path.basename(folder))
                print(new_dir)
                print(file)
                os.makedirs(new_dir, exist_ok=True)
                os.symlink(file, os.path.join(new_dir, os.path.basename(file)))
        else:
            for ii in range(args.n,2*args.n):
                file = files[ii]
                new_dir = os.path.join(args.topdir, new_dataset, os.path.basename(folder))
                print(new_dir)
                print(file)
                os.makedirs(new_dir, exist_ok=True)
                os.symlink(file, os.path.join(new_dir, os.path.basename(file)))