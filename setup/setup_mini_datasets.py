
import os, glob, shutil, numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse
import sys

sys.path.append('.')
from topographic.config import IMS_DIR

def crop_vggface2_subset_bbs(subset_ims_no_exts, ext=0.3, square=False, ims_dir=IMS_DIR):
    for phase in ['train']:
        df = pd.read_table(f'{ims_dir}/vggface2/bb_landmark/loose_bb_{phase}.csv', sep=',')
        for im_i, name in enumerate(tqdm(df['NAME_ID'])):
            if name not in subset_ims_no_exts:
                continue
            for ext_ in ['.jpg', '.png', '.jpeg']:
                try:
                    img = Image.open( f'{ims_dir}/vggface2/{phase}/{name}{ext_}')
                    new_fn = f'{ims_dir}/100faces-crop-{ext}/{phase}/{name}{ext_}'
                    break
                except:
                    pass
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

def setup_mini_faces_dset(ims_dir=IMS_DIR):
    #setup a mini VGGFace2 - 100 categories, first 500 ims/categ go to train, next 100 go to test, beyond 600 go to val
    if os.path.exists(f'{ims_dir}/100scenes'):
        print('removing existing directory')
    shutil.rmtree(f'{ims_dir}/100faces')
    classes = np.array(glob.glob(f'{ims_dir}/vggface2/train/*'))
    n_per_class = [len(glob.glob(face+'/*')) for face in classes]
    classes = np.array([categ for ii, categ in enumerate(classes) if n_per_class[ii]>=600])
    inds = np.arange(len(classes))
    np.random.shuffle(inds)
    print(f'we have{len(inds)} faces of sufficient # images')
    ii=0
    subset_ims = []
    for categ in classes[inds[-100:]]:
        train_files = glob.glob(categ+'/*')
    #     if len(train_files) < 700:
    #         print('not enough files...')
        os.makedirs(f'{ims_dir}/100faces/train/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100faces/val/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100faces/test/{os.path.basename(categ)}', exist_ok=True)
        for file in train_files[:500]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')
            os.symlink(file, f'{ims_dir}/100faces/train/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[500:600]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')
            os.symlink(file, f'{ims_dir}/100faces/test/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[600:]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')
            os.symlink(file, f'{ims_dir}/100faces/val/{os.path.basename(categ)}/{os.path.basename(file)}')


    face_classes = [os.path.basename(cls_) for cls_ in glob.glob(f'{ims_dir}/100faces-crop-2.0/train/*')]
    subset_ims = []
    for categ in face_classes:
        train_files = glob.glob(f'{ims_dir}/vggface2/train/{categ}/*')
        for file in train_files[:500]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')
        for file in train_files[500:600]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')
        for file in train_files[600:]:
            subset_ims.append(f'{os.path.basename(categ)}/{os.path.splitext(os.path.basename(file))[0]}')

    ext = 4.0
    crop_vggface2_subset_bbs(subset_ims, ext=ext, square=True)
    classes = glob.glob(f'{ims_dir}/100faces-crop-{ext}/train/*')
    # got to move into correct folders now
    for categ in classes:
        train_files = glob.glob(categ+'/*')
        for file in train_files[500:600]:
            os.makedirs(os.path.dirname(file.replace('/train/', '/val/')), exist_ok=True)
            os.rename(file, file.replace('/train/', '/val/'))
        for file in train_files[600:]:
            os.makedirs(os.path.dirname(file.replace('/train/', '/test/')), exist_ok=True)
            os.rename(file, file.replace('/train/', '/test/'))

def setup_mini_objects_dset(ims_dir=IMS_DIR):
    #setup a comparable mini ImageNet - 100 categories, first 500 ims/categ go to train, next 100 go to test, next 100 go to val
    np.random.seed(1)
    if os.path.exists(f'{ims_dir}/100objects'):
        print('removing existing directory')
    shutil.rmtree(f'{ims_dir}/100objects')
    classes = np.array(glob.glob('/lab_data/plautlab/ILSVRC/Data/CLS-LOC/train/*'))
    n_per_class = [len(glob.glob(categ+'/*')) for categ in classes]
    classes = np.array([categ for ii, categ in enumerate(classes) if n_per_class[ii]>=700])
    inds = np.arange(len(classes))
    np.random.shuffle(inds)
    print(f'we have {len(inds)} objects of sufficient # images')
    ii=0
    for categ in classes[inds[-100:]]:
        train_files = glob.glob(categ+'/*')
        os.makedirs(f'{ims_dir}/100objects/train/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100objects/val/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100objects/test/{os.path.basename(categ)}', exist_ok=True)
        for file in train_files[:500]:
            os.symlink(file, f'{ims_dir}/100objects/train/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[500:600]:
            os.symlink(file, f'{ims_dir}/100objects/test/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[600:700]:
            os.symlink(file, f'{ims_dir}/100objects/val/{os.path.basename(categ)}/{os.path.basename(file)}')
        
def setup_mini_scenes_dset(ims_dir=IMS_DIR):
    #setup a comparable mini Places - 100 categories, first 500 ims/categ go to train, next 100 go to test, next 100 go to val
    np.random.seed(1)
    if os.path.exists(f'{ims_dir}/100scenes'):
        print('removing existing directory')
        shutil.rmtree(f'{ims_dir}/100scenes')
    classes = np.array(glob.glob(f'{ims_dir}/places365_standard/train/*'))
    n_per_class = [len(glob.glob(categ+'/*')) for categ in classes]
    classes = np.array([categ for ii, categ in enumerate(classes) if n_per_class[ii]>=700])
    inds = np.arange(len(classes))
    np.random.shuffle(inds)
    print(f'we have {len(inds)} scenes of sufficient # images')
    ii=0
    for categ in classes[inds[-100:]]:
        train_files = glob.glob(categ+'/*')
        os.makedirs(f'{ims_dir}/100scenes/train/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100scenes/val/{os.path.basename(categ)}', exist_ok=True)
        os.makedirs(f'{ims_dir}/100scenes/test/{os.path.basename(categ)}', exist_ok=True)
        for file in train_files[:500]:
            os.symlink(file, f'{ims_dir}/100scenes/train/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[500:600]:
            os.symlink(file, f'{ims_dir}/100scenes/test/{os.path.basename(categ)}/{os.path.basename(file)}')
        for file in train_files[600:700]:
            os.symlink(file, f'{ims_dir}/100scenes/val/{os.path.basename(categ)}/{os.path.basename(file)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', type=str, nargs='*', default=['faces', 'objects', 'scenes'])
    parser.add_argument('--ims-dir', type=str, default=IMS_DIR)
    args = parser.parse_args()

    for domain in args.domains:
        exec(f'setup_mini_{domain}_dset(ims_dir=args.ims_dir)')