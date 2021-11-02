import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse
import sys
sys.path.append('.')
from topographic.config import IMS_DIR


def crop_vggface2_bbs(imset='vggface2', ext=0.3, square=False):
    for phase in ['train', 'test']:
        df = pd.read_table(f'{IMS_DIR}/vggface2/bb_landmark/loose_bb_{phase}.csv', sep=',')
        # if we are using a pre-divided version of vggface2, we need to search over train and val. do this lazily.
        if phase == 'train':
            try_phases = ['train', 'val']
        else:
            try_phases = [phase]
        for try_phase in try_phases:
            for im_i, name in enumerate(tqdm(df['NAME_ID'])):
                found = False
                for ext_ in ['.jpg', '.png', '.jpeg']:
                    try:
                        img = Image.open(f'{IMS_DIR}/{imset}/{try_phase}/{name}{ext_}')
                        new_fn = f'{IMS_DIR}/{imset}-crop-{ext}/{try_phase}/{name}{ext_}'
                        found = True
                        break
                    except:
                        pass
                if not found or os.path.exists(new_fn):
                    continue
                os.makedirs(os.path.dirname(new_fn), exist_ok=True)
                box = [df['X'][im_i], df['Y'][im_i], df['W'][im_i], df['H'][im_i]]
                box_ = np.array([box[0] -.5*box[2]*ext, box[1] - .5*box[3]*ext, (1+ext)*box[2], (1+ext)*box[3]]).astype(int)
                img = img.crop((box_[0], box_[1], box_[0]+box_[2]+1, box_[1] + box_[3]))
                
                if square:
                    width, height = img.size 
                    dim = np.min([width, height])
                    left = (width - dim)/2
                    top = (height - dim)/2
                    right = (width + dim)/2
                    bottom = (height + dim)/2
                    img = img.crop((left, top, right, bottom))

                img.save(new_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='vggface2')
    parser.add_argument('--ext', type=float, default=0.3)
    parser.add_argument('--square', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'vggface2' or args.dataset == '100faces':
        crop_vggface2_bbs(imset=args.dataset, ext=args.ext, square=args.square)
    else:
        raise NotImplementedError()