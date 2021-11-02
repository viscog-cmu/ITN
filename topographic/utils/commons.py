
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from datetime import datetime
import numpy as np
import os
import sys
import torch
from torch.utils.data import Sampler
from torchvision.transforms import functional as F
import numbers
import platform
from contextlib import contextmanager

from topographic import transforms

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

DATA_TRANSFORM = transforms.Compose([
    transforms.Pad(padding=(0, 0), padding_mode='square_constant'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

DATACROP_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            NORMALIZE])

pltv = platform.version()
ON_CLUSTER = False if 'Darwin' in pltv or 'Ubuntu' in pltv else True


def reproducible_results(seed=1):
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    

def get_data_transforms(im_dim, 
                        logpolar=False, polar=False, 
                        fivecrop=False, 
                        centercrop_train=False, 
                        max_rot_deg=0,
                        pad_factor=0,
                        ):
    assert not (logpolar and polar), 'only one of logpolar and polar can be set to True'
    if fivecrop:
        tensor_maker = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        normalizer = transforms.Lambda(lambda crops: torch.stack([NORMALIZE(crop) for crop in crops]))
    else:
        tensor_maker = transforms.ToTensor()
        normalizer = NORMALIZE
    if logpolar:
        if fivecrop:
            to_lp = transforms.Lambda(lambda crops: torch.stack([transforms.LogPolar(im_dim)(crop) for crop in crops]))
        else:
            to_lp = transforms.LogPolar(im_dim)
    elif polar:
        if fivecrop:
            to_lp = transforms.Lambda(lambda crops: torch.stack([transforms.Polar(im_dim)(crop) for crop in crops]))
        else:
            to_lp = transforms.Polar(im_dim)
    else:
        to_lp = None

    data_transforms = {
            'train': FlexibleCompose([
            transforms.Resize(int((256/224)*im_dim)),
            transforms.Pad(int(pad_factor*im_dim/2)) if pad_factor is not 0 else None,
            transforms.Resize(int((256/224)*im_dim)) if pad_factor is not 0 else None,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(max_rot_deg) if max_rot_deg else None,
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            transforms.FiveCrop(im_dim) if fivecrop else transforms.CenterCrop((im_dim,im_dim)) if centercrop_train else transforms.RandomCrop((im_dim,im_dim)),
            to_lp,
            tensor_maker,
            normalizer,
            ]),
            'val': FlexibleCompose([
            transforms.Resize(int((256/224)*im_dim)),
            transforms.Pad(int(pad_factor*im_dim/2)) if pad_factor is not 0 else None,
            transforms.Resize(int((256/224)*im_dim)) if pad_factor is not 0 else None,
            transforms.FiveCrop(im_dim) if fivecrop else transforms.CenterCrop((im_dim,im_dim)),
            to_lp,
            tensor_maker,
            normalizer,
            ])}
    
    return data_transforms

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class,
            and path is the file name from which sample is derived
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class FlexibleCompose(transforms.Compose):
    """
    Extends Compose to allow for specification of "None" steps
    which are ignored. Makes composing transforms based on flexible conditions
    much easier.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if t is not None:
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            if t is not None:
                format_string += '\n'
                format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
        

def make_dataset(rootdir, class_to_idx, extensions):
    images = []
    rootdir = os.path.expanduser(rootdir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(rootdir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class ImageSubsetFolder(MyImageFolder):
    """
    Extends ImageFolder to  allow for:
    1) selection of arbitrary subsets of classes (folders)
    2) arbitrary reassignment of images within a (subset) class to a broader class name (e.g.
     golden retriever and terrier to dog)
    """

    def __init__(self, root, subset_classes=None, subset_reassignment=None,
                 loader=default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        classes, class_to_idx = self._find_classes(
            root, subset_classes, subset_reassignment)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, root, subset_classes, subset_reassignment):
        """
        Finds the class folders in a dataset.

        Args:
            root (string): Root directory path.
            subset_classes (None or list of strings): class names to include
            subset_reassignment (None or list of strings): reassignment of class names to (most definitely) broader classes

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (root), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(
                root) if os.path.isdir(os.path.join(root, d))]
        if subset_classes is not None:
            classes = [c for c in classes if (c in subset_classes)]
            d = [(c, subset_classes.index(c))
                 for c in classes if c in subset_classes]
        inds, classes = np.argsort(classes), np.sort(classes).tolist()
        if subset_reassignment is not None:
            subset_reassignment = [subset_reassignment[i] for (_, i) in d]
            subset_reassignment = [subset_reassignment[i] for i in inds]
            self.named_classes = list(set(subset_reassignment))
            class_to_idx = {classes[i]: self.named_classes.index(
                subset_reassignment[i]) for i in range(len(classes))}
        else:
            self.named_classes = classes
            class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx


class UnlabeledDataset(datasets.DatasetFolder):
    def __init__(self, fns, transform=None, loader=default_loader,
                 extensions=IMG_EXTENSIONS):
        self.samples = fns
        if len(self.samples) == 0:
            raise RuntimeError('0 files specified in fns')
        self.loader = loader
        self.extensions = extensions
        self.classes = None
        self.class_to_idx = None
        self.targets = None
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Unlabeled Dataset ' + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ExpTimes():
    """
    define the timing of an experiment
    """
    def __init__(self, dt=0.02, stim_on=0, stim_off=10, loss_on=5, exp_end=20, jitter=0, loss_jitter=0):
        self.dt = dt
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.loss_on = loss_on
        self.exp_end = exp_end
        self.jitter = jitter # stimulus will turn off t_j = Uniform(0,jitter) time steps after stim_off, and ends t_j + exp_end. loss_on is unaffected
        self.loss_jitter = loss_jitter # loss turns on t_lj = Uniform(0,loss_jitter) time steps after loss_on, and plateaus at t_lj + stim_off.

        if stim_on != 0:
            raise NotImplementedError()

exp_times = {
    'v0': ExpTimes(dt=0.02, stim_on=0, stim_off=5, loss_on=4, exp_end=5, jitter=0),
    'v1': ExpTimes(dt=0.02, stim_on=0, stim_off=10, loss_on=5, exp_end=20, jitter=0),
    'v2': ExpTimes(dt=0.02, stim_on=0, stim_off=10, loss_on=5, exp_end=10, jitter=5),
    'v3': ExpTimes(dt=0.02, stim_on=0, stim_off=10, loss_on=5, exp_end=15, jitter=5),
    'v4': ExpTimes(dt=0.02, stim_on=0, stim_off=10, loss_on=5, exp_end=15, jitter=5, loss_jitter=5),
    'v5': ExpTimes(dt=0.02, stim_on=0, stim_off=5, loss_on=5, exp_end=10, jitter=2),
    'ff': ExpTimes(dt=0.02, stim_on=0, stim_off=1, loss_on=0, exp_end=1, jitter=0, loss_jitter=0),
    'v6': ExpTimes(dt=0.02, stim_on=0, stim_off=5, loss_on=1, exp_end=5, jitter=0),
}

def get_name(opts, exclude_keys=[], include_keys=[], bool_keys=[], no_datetime=False, ext='.pkl', short_names=None, bools_as_ints=False):
    """
    Get a file-name based on a dictionary, set of dict keys to exclude from the name,
    whether to add the datetime string, and what extension, if any
    """
    name = None
    if not no_datetime:
        name = str(datetime.now()).replace(' ', '-').split('.')[0]

    if len(exclude_keys) > 0 and len(include_keys) > 0:
        raise ValueError('you tried to use both exclude_keys and include_keys, but they are mutually exclusive')

    for key in sorted(opts):
        if len(exclude_keys) > 0:
            if key in bool_keys:
                include = key not in exclude_keys and opts[key]
            else:
                include = key not in exclude_keys
        elif len(include_keys) > 0:
            if key in bool_keys:
                include = key in include_keys and opts[key]
            else:
                include = key in include_keys
        else:
            include = True
        if include:
            if short_names is not None:
                if key in short_names:
                    use_key = short_names[key]
                else:
                    use_key = key
            else:
                use_key = key
            if type(opts[key]) is bool and bools_as_ints:
                use_val = int(opts[key])
            else:
                use_val = opts[key]
            if name is None:
                name = '-'.join((use_key, str(use_val)))
            else:
                name = '_'.join((name, '-'.join((use_key, str(use_val)))))
    if ext is not None:
        name += ext
    return name


def actual_indices(idx, n):
    """
    Get the i,j indices from an idx (or vector idx) of
    corresponding to the squareform of an nxn matrix

    this is a code snip-it from https://code.i-harness.com/en/q/c7940b
    """
    n_row_elems = np.cumsum(np.arange(1, n)[::-1])
    ii = (n_row_elems[:, None] - 1 < idx[None, :]).sum(axis=0)
    shifts = np.concatenate([[0], n_row_elems])
    jj = np.arange(1, n)[ii] + idx - shifts[ii]
    return ii, jj


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class FixedRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (float or int): rotation angle
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            self.degrees = degrees
        else:
            raise ValueError('Degrees must be a single number')

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        return F.rotate(img, self.degrees, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
