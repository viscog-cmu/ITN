
import torch
import torchvision
from torch.utils import data
from torchvision import datasets
from scipy.io import loadmat
import numpy as np
from torchvision.datasets.folder import default_loader
import glob
from PIL import Image
import pdb
import os
import pickle

from topographic.config import IMS_DIR, SCRATCH_DIR
from topographic.utils.commons import ImageSubsetFolder, get_data_transforms, MyImageFolder, FlexibleCompose
from topographic import transforms
from topographic.utils.gabor import GaborDataset

VPNL_FLOC_CATEG_DOMAIN_REF = np.array([0,1,2,0,3,3,2,1,4,5,4])
VPNL_FLOC_DOMAINS = ['face','body','object','scene','character','scrambled']
KONK_FLOC_CATEG_DOMAINS = ['body', 'face', 'object', 'scene', 'scrambled']
KONK_TF_CATEG_DOMAINS = sorted([os.path.basename(cc) for cc in glob.glob(f'{IMS_DIR}/konk_animsize_texform/*')])
OF_CLASSES = [os.path.basename(cc) for cc in sorted(glob.glob(f'{IMS_DIR}/objects_and_faces/train/*'))]
OFS_CLASSES = [os.path.basename(cc) for cc in sorted(glob.glob(f'{IMS_DIR}/objfacescenes/train/*'))]
BAO_DOMAINS = ['body', 'face', 'noise', 'spiky', 'stubby']
DOMAIN_CLASSES = {}
DOMAIN_CLASSES['object'] = [cc for cc in OFS_CLASSES if len(cc) == 9 and cc[0:2] in ['n0','n1']]
DOMAIN_CLASSES['face'] = [cc for cc in OFS_CLASSES if len(cc) == 7 and cc[0:2] in ['n0','n1']]
DOMAIN_CLASSES['scene'] = [cc for cc in OFS_CLASSES if cc[0:2] not in ['n0','n1']]

def get_domain_inds(domain, dataset):
    if domain not in ['object', 'face', 'scene', 'word'] and 'floc' not in dataset and 'konk' not in dataset:
        raise ValueError()
    if dataset not in ['objects_and_faces', 'objfacescenes', 'miniOFS', 'miniOFS2', 'miniOFS3', 'miniOFSW', 'vpnl_floc', 'konk_floc', 'vpnl+konk_floc', 'baotsao_floc', 'konk_animsize_texform', 'konk_anim', 'miniO', 'miniF', 'miniS']:
        raise NotImplementedError()
    if domain == 'scene' and dataset == 'objects_and_faces':
        raise ValueError()
    if 'miniOFS'  in dataset:
        domain_inds = {'object':np.arange(0,100), 'face':np.arange(100, 200), 'scene':np.arange(200,300)}
        if 'miniOFSW' in dataset:
            domain_inds['word'] = np.arange(300,400)
        domain_inds = domain_inds[domain]
    elif dataset == 'miniO':
        if domain == 'object':
            domain_inds = np.arange(0,100)
        else:
            domain_inds = None
    elif dataset == 'miniF':
        if domain == 'face':
            domain_inds = np.arange(0,100)
        else:
            domain_inds = None    
    elif dataset == 'miniS':
        if domain == 'object':
            domain_inds = np.arange(0,100)
        else:
            domain_inds = None
    elif dataset == 'vpnl_floc':
        domain_inds = [ii for ii, val in enumerate(VPNL_FLOC_CATEG_DOMAIN_REF)  if val == VPNL_FLOC_DOMAINS.index(domain)]
    elif dataset == 'konk_floc':
        domain_inds = [KONK_FLOC_CATEG_DOMAINS.index(domain)]
    elif dataset == 'vpnl+konk_floc':
        domain_inds = [ii for ii, val in enumerate(VPNL_FLOC_CATEG_DOMAIN_REF)  if val == VPNL_FLOC_DOMAINS.index(domain)]
        if domain in KONK_FLOC_CATEG_DOMAINS:
            konk_inds = [len(VPNL_FLOC_DOMAINS) + KONK_FLOC_CATEG_DOMAINS.index(domain)]
            domain_inds += konk_inds
    elif dataset == 'baotsao_floc':
        domain_inds = [BAO_DOMAINS.index(domain)]
    elif dataset == 'konk_animsize_texform':
        domain_inds = [KONK_TF_CATEG_DOMAINS.index(domain)]
    else:
        dataset_classes = sorted([os.path.basename(cc) for cc in glob.glob(f'{IMS_DIR}/{dataset}/train/*')])
        domain_classes = DOMAIN_CLASSES[domain]
        domain_inds = [dataset_classes.index(class_) for class_ in domain_classes]
    return domain_inds

def get_experiment_dataset(trainset):
    dataset_name =  trainset if trainset in ['objfacescenes', 'miniOFS', 'miniOFSW', 'miniOFS2', 'miniOFS3', 'miniF2', 'miniF3', 'objects_and_faces'] else 'miniOFS'
    # if trainset in ['miniF', 'miniO', 'miniS']:
    #     dataset_name = 'miniOFS'
    # else:
    #     dataset_name = trainset
    return dataset_name


def get_dataset(dataset_name, phase, 
                data_transform=None, 
                n_earlier_classes=0,
                subset_frac=None, # select a fraction of classes; only used for some datasets
                ims_dir=IMS_DIR,
               ):
    """
    load a torchvision dataset by a string name and training phase

    inputs:
        dataset_name: string identifier for the dataset
        phase: training phase (train, val, or test)
        data_transform: optional transform to apply to data during loading
        n_earlier_classes: nonzero iff you are concatenating multiple datasets and this is not the first one in the list.
    returns:
        image_dataset: the dataset
        n_classes: the # of classes in the dataset
    """

    if dataset_name.lower() in ['cifar10', 'cifar100', 'emnist']:
        assert subset_frac is None
        if dataset_name.lower() == 'cifar10':
            dset = datasets.CIFAR10
            dset_args = {'target_transform': transforms.Lambda(
                lambda x: n_earlier_classes+x)}
            n_classes = 10
        elif dataset_name.lower() == 'cifar100':
            dset = datasets.CIFAR100
            dset_args = {'target_transform': transforms.Lambda(
                lambda x: n_earlier_classes+x)}
            n_classes = 100
        elif dataset_name.lower() == 'emnist':
            dset = datasets.EMNIST
            dset_args = {'split': 'letters', 'target_transform': transforms.Lambda(
                lambda x: n_earlier_classes+x-1)}
            n_classes = 10
        try:
            image_dataset = dset(SCRATCH_DIR, train=(phase == 'train'),
                                 transform=data_transform,
                                 download=True,
                                 **dset_args)
        except:
            image_dataset = dset(ims_dir, train=(phase == 'train'),
                                 transform=data_transform,
                                 download=True,
                                 **dset_args)
    elif dataset_name.lower() == 'facegen':
        image_dataset = FaceGenImageFolder(f'{ims_dir}/rotated_faces',
                        transform=data_transform, 
                        target_transform=transforms.Lambda(lambda x: n_earlier_classes+x),
                        )     
        n_classes = len(np.unique(image_dataset.targets))
    elif dataset_name.lower() == 'baotsao':
        image_dataset = BaoTsaoDataset(ims_dir, name='1224_orig', transform=data_transform, target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
        n_classes = 51
    elif dataset_name.lower() == 'baotsao_floc':
        # the 4 quadrant + noise localizer
        image_dataset = datasets.ImageFolder(f"{ims_dir}/baotsao_loc", transform=data_transform, target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
        n_classes = 5
    elif dataset_name.lower() == 'baotsao_19300':
        # a set of 19300 images
        image_dataset = BaoTsaoDataset(ims_dir, name='19300', transform=data_transform, target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
        n_classes = 1 # we don't have any metadata as far as i know

    elif dataset_name.lower() == 'svhn':
        assert subset_frac is None
        try:
            image_dataset = datasets.SVHN(SCRATCH_DIR,
                                          split=phase if phase != 'val' else 'test',
                                          transform=data_transform,
                                          target_transform=transforms.Lambda(
                                              lambda x: n_earlier_classes+x),
                                          download=True)
        except:
            image_dataset = datasets.SVHN(ims_dir,
                                          split=phase if phase != 'val' else 'test',
                                          transform=data_transform,
                                          target_transform=transforms.Lambda(
                                              lambda x: n_earlier_classes+x),
                                          download=True)
        n_classes = 10
#     elif dataset_name.lower() in ['100words', '100faces']:
#         for folder in [SCRATCH_DIR, IMS_DIR]:
#             try:
#                 image_dataset = datasets.ImageFolder(f"{folder}/{dataset_name}/{phase if phase!= 'val' else 'test'}",
#                                             transform=data_transform,
#                                             target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
#                 break
#             except:
#                 print(f'did not find {dataset_name} in {folder}')
#                 continue
#         n_classes = 100
    elif dataset_name.lower() in ['objects_and_faces', 'objfacescenes', '100objects', '100scenes', '100words', '100faces-crop-2.0', '100faces-crop-4.0', '100faces-crop-0.3', 'saycam-tc-ego-s', 'saycam-tc-ego-s-300c', 'imagenet']:
        for folder in [SCRATCH_DIR, ims_dir]:
            try:
                subset_classes = subset_from_frac(f"{folder}/{dataset_name}/{phase}", subset_frac)
                image_dataset = ImageSubsetFolder(f"{folder}/{dataset_name}/{phase}",
                                            transform=data_transform,
                                            subset_classes=subset_classes,
                                            target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
                break
            except Exception as e:
                #print(f'did not find {dataset_name} in {folder}')
               print(e) 
               continue
        n_classes = len(np.unique(image_dataset.classes))
    elif 'ims-per-class' in dataset_name.lower() or dataset_name.lower() in ['vpnl_floc', 'konk_floc', 'konk_animsize_texform', 'baotsao_floc']:
        for folder in [SCRATCH_DIR, ims_dir]:
            try:
                subset_classes = subset_from_frac(f"{folder}/{dataset_name}", subset_frac)
                image_dataset = ImageSubsetFolder(f"{folder}/{dataset_name}",
                                            transform=data_transform,
                                            subset_classes=subset_classes,
                                            target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
                break
            except:
                # print(f'did not find {dataset_name} in {folder}')
                continue
        n_classes = len(np.unique(image_dataset.classes))
    elif dataset_name.lower() == 'nsd_shared1000':
        if n_earlier_classes > 0:
            raise NotImplementedError()
        image_dataset = NSDImageDataset(ims_dir+'/nsd_shared1000', transform=data_transform)
        n_classes = len(image_dataset)
    elif dataset_name.lower() in ['objects', 'faces', 'scenes']:
        domain = dataset_name.lower()[:-1]
        n_classes = len(DOMAIN_CLASSES[domain])
        for folder in [SCRATCH_DIR, ims_dir]:
            try:
                subset_classes = subset_from_frac(f"{folder}/objfacescenes/{phase}", subset_frac, larger_subset=DOMAIN_CLASSES[domain])
                image_dataset = ImageSubsetFolder(f"{folder}/objfacescenes/{phase}",
                                            transform=data_transform,
                                            subset_classes=subset_classes,
                                            target_transform=transforms.Lambda(lambda x: n_earlier_classes+x))
                break
            except Exception as e:
               print(e) 
               continue
    elif dataset_name.lower() == 'gabor':
        assert n_earlier_classes == 0
        if data_transform is not None:
            gab_transforms = FlexibleCompose(
                [t for t in data_transform.transforms if type(t) is not transforms.RandomHorizontalFlip]
            )
        else:
            gab_transforms = None
        thetas = list(np.linspace(0,180,10))
        target_transform = transforms.Lambda(lambda x: thetas.index(x['theta'])) # just return the theta parameter for discrimination
        image_dataset = GaborDataset(thetas=thetas, 
                            sigmas=np.linspace(0.1,1,10), 
                            pxs=np.linspace(0,1,10), 
                            pys=np.linspace(0,1,10), 
                            transform=gab_transforms,
                            target_transform=target_transform,
                        )
        n_classes = len(thetas)

    else:
        raise NotImplementedError(f'{dataset_name} not configured')
                
    return image_dataset, n_classes

def subset_from_frac(topdir, subset_frac, larger_subset=None):
    classes = np.array([os.path.basename(class_) for class_ in sorted(glob.glob(f"{topdir}/*"))])
    if larger_subset is not None:
        classes = np.array([class_ for class_ in classes if class_ in larger_subset])
    if subset_frac is not None:
        inds = np.linspace(0,len(classes)-1,num=int(subset_frac*len(classes)), dtype=int)
        subset_classes = list(classes[inds])
    else:
        subset_classes = None
    return subset_classes

def expand_datasets(datasets):
    dataset_name = make_dsets_name(datasets)
    datasets = expand_datasets_from_name(dataset_name)
    return dataset_name, datasets

def expand_datasets_from_name(dataset_name):
    if 'miniOFS' in dataset_name:
        if dataset_name == 'miniOFS':
            datasets = ['100objects', '100faces-crop-2.0', '100scenes']
        elif dataset_name == 'miniOFS2':
            datasets = ['100objects', '100faces-crop-4.0', '100scenes']
        elif dataset_name == 'miniOFS3':
            datasets = ['100objects', '100faces-crop-0.3', '100scenes']
        elif dataset_name == 'miniOFSW':
            datasets = ['100objects', '100faces-crop-0.3', '100scenes', '100words']
        else:
            raise ValueError()
    elif dataset_name == 'miniO':
        datasets = ['100objects']
    elif dataset_name == 'miniF':
        datasets = [f'100faces-crop-2.0']
    elif dataset_name == 'miniF2':
        datasets = [f'100faces-crop-4.0']
    elif dataset_name == 'miniF3':
        datasets = [f'100faces-crop-0.3']
    elif dataset_name == 'miniS':
        datasets = ['100scenes']
    elif dataset_name == 'miniW':
        datsets = ['100words']
    elif dataset_name == 'vpnl+konk_floc':
        datasets = ['vpnl_floc', 'konk_floc']
    else:
        datasets = unmake_dsets_name(dataset_name)
    return datasets

def make_dsets_name(datasets):
    name = datasets[0]
    for dataset in datasets[1:]:
        name += f'+{dataset}'
    return name

def unmake_dsets_name(dataset_name):
    datasets = dataset_name.split('+')
    return datasets


def get_multi_dataset_loaders(datasets, data_transforms,
                              num_workers=6,
                              batch_size=64,
                              ims_dir=IMS_DIR,
                              return_n_classes=False,
                              transforms_per_dataset = False,
                              subset_frac=None,
                              ):
    """
    concatenate a list of datasets, and build loaders using desired transforms
    inputs:
        datasets: list of string dataset names
        data_transforms: dict of data transforms indexed by the training phase
        num_workers: how many parallel workers to use
        batch_size: the batch size to use during training
        ims_dir: directory containing image sets
        return_n_classes: whether to additionally return the total number of classes
        transforms_per_dataset: whether data_transforms is a dict of dicts, indexed first by dataset and then phase.
            allows for different transforms per dataset, e.g., shrinking one dataset of images.
    returns:
        dataloaders: dict of torch dataloader objects indexed by training phase
    """
    full_dsets = {}
    if transforms_per_dataset:
        phases = data_transforms[datasets[0]].keys()
    else:
        phases = data_transforms.keys()
    for phase in phases:
        dsets = []
        n_class_accum = 0
        for dset in datasets:
            dset, n_classes = get_dataset(
                dset, phase, 
                data_transforms[dset][phase] if transforms_per_dataset else data_transforms[phase], 
                n_earlier_classes=n_class_accum, 
                ims_dir=ims_dir,
                subset_frac=subset_frac,
                )
            dsets.append(dset)
            n_class_accum += n_classes
        full_dsets[phase] = data.dataset.ConcatDataset(dsets)
    dataloaders = {x: data.DataLoader(full_dsets[x], batch_size=batch_size,
                                      shuffle=True if x == 'train' else False,
                                      pin_memory=True,
                                      num_workers=num_workers) for x in phases}
    if return_n_classes:
        return dataloaders, n_class_accum
    else:
        return dataloaders

def get_domain_padding_options(padding, datasets):
    options = {
        '0': {'face': 0, 'object': 0, 'scene': 0, 'word': 0},
        'sf': {'face': 1, 'object': 0, 'scene': 0, 'word': 0},
        'tf': {'face': 2, 'object': 0, 'scene': 0, 'word': 0},
        'so': {'face': 0, 'object': 1, 'scene': 0, 'word': 0},
        'to': {'face': 0, 'object': 2, 'scene': 0, 'word': 0},
        'ss': {'face': 0, 'object': 0, 'scene': 1, 'word': 0},
        'ts': {'face': 0, 'object': 0, 'scene': 2, 'word': 0},
        'sw': {'face': 0, 'object': 0, 'scene': 0, 'word': 1},
        'tw': {'face': 0, 'object': 0, 'scene': 0, 'word': 2},
        'twf': {'face': 2, 'object': 1, 'scene': 0, 'word': 2},
    }
    padding_options_ = options[padding]
    face_dset = [dset for dset in datasets if 'face' in dset][0]
    scene_dset = [dset for dset in datasets if 'scene' in dset][0]
    obj_dset = [dset for dset in datasets if 'object' in dset][0]

    padding_options = {}
    padding_options[face_dset] = padding_options_['face']
    padding_options[scene_dset] = padding_options_['scene']
    padding_options[obj_dset] = padding_options_['object']

    if sum(['words' in dset for dset in datasets]):
        word_dset = [dset for dset in datasets if 'word' in dset][0]
        padding_options[word_dset] = padding_options_['word']

    return padding_options


def get_all_data_transforms(im_dim, padding=None, dataset_names=None, **kwargs):
    if padding:
        transforms_per_dataset = True
        domain_padding = get_domain_padding_options(padding, dataset_names)
        data_transforms = {}
        for dset in dataset_names:
            data_transforms[dset] = get_data_transforms(im_dim, pad_factor=domain_padding[dset], **kwargs)
    else:
        transforms_per_dataset = False
        data_transforms = get_data_transforms(im_dim, **kwargs)
    return data_transforms, transforms_per_dataset

class FaceGenImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, meta) where target is class_index of the target class,
            and meta is a dict of attributes about the image
        """
        path, target = self.samples[index]
        # might end up changing the file naming to add other metadata at some point
        basename = os.path.splitext(os.path.basename(path))[0]
        meta = {}
        for pair in basename.split('_'):
            key, val = pair.split('-', 1)
            meta[key] = val

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, meta

class BaoTsaoDataset(datasets.vision.VisionDataset):
    """
    Dataset for loading stimuli from Bao et. al, 2020 Nature
    Contains 1224 images of 51 objects from 6 broad categories, at 24 random orientations
    """
    def __init__(self, root=IMS_DIR, name='1224_orig', **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        assert self.name in ['1224_orig', '19300']
        mat_file = loadmat(f'{root}/baotsao_mat_files/baotsao_{name}.mat')
        if self.name == '1224_orig':
            self.samples = mat_file['imgall'].transpose(2,0,1)
            self.category = []
            self.orientation = []
            self.object_num = []
            category_nums = {'animal':10, 'vehicle':11, 'face':9, 'veg/fruit':7, 'house':4, 'manmade object': 10}
            object_num = -1
            for category, nums in category_nums.items():
                for num in range(nums):
                    object_num += 1
                    for ori in range(24):
                        self.orientation.append(ori)
                        self.category.append(category)
                        self.object_num.append(object_num)
        elif self.name == '19300':
            self.samples = mat_file['im2all'].transpose(2,0,1)
            self.object_num = np.zeros(len(self.samples))
        
    def __getitem__(self, index):
        sample = Image.fromarray(np.tile(self.samples[index], [3,1,1]).transpose(1,2,0), mode='RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.object_num[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.name == '1224_orig':
            meta = {targ: getattr(self, targ)[index] for targ in ['category', 'orientation']}
        else:
            meta = {}
        return sample, target, meta

    def __len__(self):
        return len(self.samples)

class NSDImageDataset(datasets.vision.VisionDataset):
    """
    To-do
    """
    def __init__(self, root, loader=default_loader, **kwargs):
        super().__init__(root, **kwargs)
        self.root = root
        self.loader = default_loader
        self.samples = glob.glob(root+'/*.png')
        self.index_72k = [int(os.path.basename(sample)[14:19]) for sample in self.samples]  
        
    def __getitem__(self, index):
        sample = self.loader(self.samples[index])
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.index_72k[index]

        return sample, target

    def __len__(self):
        return len(self.samples)