import os

def X_is_running():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if not X_is_running() and not isnotebook():
    print('no X server detected. changing default matplotlib backend to Agg for compatibility.')
    import matplotlib
    matplotlib.use('Agg')

from topographic.ITN import EncoderITN, ITN, unparse_model_args, parse_model_args
from topographic import config
import numpy as np


def get_model_from_args(opt, arch_kwargs, base_fn=None):
    if base_fn is not None:
        mod_fn = os.path.join(config.SAVE_DIR, 'models', base_fn+'.pkl')
        assert os.path.exists(mod_fn)
    else:
        mod_fn = None
    if opt.encoder is not None:
        if opt.trainset == 'SAYCAM-TC-ego-S' or opt.trainset == 'SAYCAM-TC-ego-S-300c' or 'say' in opt.encoder:
            trainset = 'SAYCAM-TC-ego-S'
        elif (opt.trainset == 'miniOFS' or 'ofs' in opt.encoder) and not np.sum([a in opt.encoder for a in ['obj', 'scene', 'face', 'imgnet']]):
            trainset = 'objfacescenes'
        elif (opt.trainset == 'miniO' or 'obj' in opt.encoder) and not np.sum([a in opt.encoder for a in ['scene', 'face', 'imgnet']]):
            trainset = 'objects'
        elif (opt.trainset == 'miniS' or 'scene' in opt.encoder) and np.sum([a in opt.encoder for a in ['face', 'imgnet']]):
            trainset = 'scenes'
        elif (opt.trainset == 'miniF' or 'face' in opt.encoder) and not np.sum([a in opt.encoder for a in ['imgnet']]):
            trainset = 'faces'
        elif 'imgnet' in opt.encoder:
            trainset = 'imagenet'
        else:
            trainset = 'objfacescenes'
        print(f'using {trainset}-pretrained encoder')
        model = EncoderITN(n_classes=opt['n_classes'], trainset=trainset, weights_path=mod_fn, **arch_kwargs)
    else:
        model = ITN(n_classes=opt['n_classes'], weights_path=mod_fn, **arch_kwargs)
    return model


def get_model_from_fn(base_fn, load=True):
    kwargs = unparse_model_args(base_fn, use_short_names=True)
    _, opt, arch_kwargs = parse_model_args(get_n_classes=True, bools_as_ints=True, **kwargs)
    model = get_model_from_args(opt, arch_kwargs, base_fn=base_fn if load else None)
    return model, opt, kwargs, arch_kwargs
