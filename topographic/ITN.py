import torch
import numpy as np
from torch import nn
import copy
import os
from easydict import EasyDict as edict

from topographic import encoder
import topographic.IT as IT 
import topographic.readout as ITR

from topographic.utils.lesion import it_lesion, corblockz_lesion
from topographic.utils.commons import exp_times
from topographic.utils.dataset import expand_datasets_from_name, get_multi_dataset_loaders
from topographic.dl.cornet_z import CORblock_Z, CORblock_RZ
from topographic.dl.cornet_rt import CORblock_RT
from topographic.utils.commons import get_name, exp_times

# default values for model parameters
DEFAULT_KWARGS = edict(
    rs = 1,
    imdim = 112,
    cell = 'EI4',
    encoder = None, # we can specify either an encoder or a block
    block = None,
    noise = 'linear',
    fsig = 5.0, 
    rsig = 5.0,
    squashn = False,
    threshn = False,
    l2=5e-4,
    l1=0,
    jjlam = 0.0,
    conn = 'full',
    rwid = 0.0,
    fwid = 0.0,
    i2eratio = None,
    cs = False,
    maxepochs = 300,
    trainset = 'objfacescenes',
    subsetfrac = None,
    sched = 'plateau',
    lr0 = 0.01,
    id = None,
    ms = 32,
    polar = False,
    logpolar = False,
    mini = False,
    cIT = False,
    gc = None,
    cgc = None,
    optim = 'sgd',
    timing = 'v3',
    alpha = 0.2,
    e2ialpharatio = None,
    ereadout = False,
    erreadout = False,
    fivecrop = False,
    dropo = False,
    dropov4 = False,
    rampv4=False,
    norec = False,
    nl = None,
    nonretinotopic=False,
    pretrainedconvs=False,
    nowrap = False,
    centercroptrain = False,
    padding = False,
    itnarch=None,
    topoconv=None,
    rot=0,
)
KWARG_TYPES = edict(
    rs = int,
    imdim = int,
    cell = str,
    encoder = str,
    block = str,
    noise = str,
    fsig = float, 
    rsig = float,
    squashn = bool,
    threshn = bool,
    l2=float,
    l1=float,
    jjlam = float,
    conn = str,
    rwid = float,
    fwid = float,
    i2eratio = float,
    cs = bool,
    maxepochs = int,
    trainset = str,
    subsetfrac = float,
    sched = str,
    lr0 = float,
    id = str,
    ms = int,
    polar = bool,
    logpolar = bool,
    mini = bool,
    cIT = bool,
    gc = float,
    cgc = float,
    optim = str,
    timing = str,
    alpha = float,
    e2ialpharatio = float,
    ereadout = bool,
    erreadout = bool,
    fivecrop = bool,
    dropo = float,
    dropov4 = float,
    rampv4=bool,
    norec = bool,
    nl = str,
    nonretinotopic=bool,
    pretrainedconvs=bool,
    nowrap = bool,
    centercroptrain = bool,
    padding = str,
    itnarch=int,
    topoconv=bool,
    rot=int,
)
BOOL_KEYS = [
    'alpha',
    'id', 
    'encoder',
    'block',
    'novalnoise', 
    'polar',
    'logpolar', 
    'mini', 
    'gc',
    'cgc',
    'fivecrop',
    'rwid',
    'fwid',
    'dropo',
    'dropov4',
    'rampv4',
    'norec',
    'nonretinotopic',
    'nowrap',
    'i2eratio',
    'e2ialpharatio',
    'ereadout',
    'erreadout',
    'cs',
    'squashn',
    'threshn',
    'cIT',
    'jjlam',
    'nl',
    'pretrainedconvs',
    'centercroptrain',
    'padding',
    'itnarch',
    'topoconv',
    'subsetfrac',
    'l1',
    'rot',
]

# dictionary of shortened key names for certain longer keys
SHORT_NAME_DICT = dict(
    imdim = 'imd',
    encoder = 'enc',
    block = 'bl',
    noise = 'noi',
    squashn = 'sq',
    threshn = 'thr',
    jjlam = 'jj',
    timing = 't',
    i2eratio = 'i2e',
    maxepochs = 'me',
    trainset = 'tr',
    subsetfrac = 'sf',
    sched = 'sch',
    polar='po',
    logpolar = 'lp',
    fivecrop = 'fc',
    dropov4 = 'dpov4',
    rampv4='rv4',
    nonretinotopic='nret',
    pretrainedconvs='ptc',
    e2ialpharatio='ar',
    ereadout='er',
    erreadout='err',
    alpha='a',
    nowrap='nwr',
    centercroptrain='cct',
    padding='pad',
    itnarch='ma',
    topoconv='tc',
)

SHORT_NAME_INVERSE_DICT = {val:key for key, val in SHORT_NAME_DICT.items()}

def unparse_model_args(base_fn, use_short_names=False):
    """
    acquire the specified kwargs from a base filename. Can be combined with parse_model_args to fully reconstruct a model from its base_fn
    """
    kwarg_list = base_fn.split('_')
    kwargs = {}
    for kwarg in kwarg_list:
        key, val = kwarg.split('-', 1)
        if use_short_names and key in SHORT_NAME_INVERSE_DICT.keys():
            key = SHORT_NAME_INVERSE_DICT[key]
        if KWARG_TYPES[key] is bool:
            kwargs[key] = False if val == '0' or val == 'False' else True
        else:
            kwargs[key] = KWARG_TYPES[key](val)
    kwargs['datasets'] = expand_datasets_from_name(kwargs['trainset'])
    kwargs['use_short_names'] = use_short_names
    return kwargs

def parse_model_args(ext='', directory=None, get_n_classes=False, use_short_names=False, bools_as_ints=False, **kwargs):
    """
    take default kwargs and overwrite with any specified kwargs
    returns the filename, full opt (kwargs), and the arch_kwargs to be passed to the model
    """
    opt = copy.deepcopy(DEFAULT_KWARGS)
    for kw, arg in kwargs.items():
        opt[kw] = arg
    
    include_keys = list(DEFAULT_KWARGS.keys())

    if opt.l2 == DEFAULT_KWARGS.l2:
        include_keys.remove('l2')


    if opt.encoder is not None:
        # EncoderITN
        opt.block = None
        assert (not opt.dropov4)
        backend_kwargs = dict(
            encoder_arch = opt.encoder,
            no_stride = not opt.nonretinotopic,
            ramp_enc_outputs = opt.rampv4,
        )
        if opt.itnarch is None:
            if opt.mini:
                cell_mapsides={'IT': opt.ms}
            elif opt.cIT:
                cell_mapsides={'pIT': opt.ms, 'cIT':opt.ms, 'aIT': opt.ms}
            else:
                cell_mapsides={'pIT': opt.ms, 'aIT': opt.ms}
        else: 
            raise NotImplementedError()
    else:
        # ITN
        if opt.itnarch is None:
            if opt.nonretinotopic:
                block_details = {'image': [None, 3,1,1], 'V1': [7,64,2,2], 'V2': [3,128,2,1], 'V3': [3,128,2,1], 'V4': [3,256,2,1]} # (kernel_size, channels, stride, pool_stride)
            else:
                # for retinotopic version, we remove V1 pooling stride, remove V4 conv stride, and reduce V4 channels by 2x
                block_details = {'image': [None, 3,1,1], 'V1': [7,64,2,1], 'V2': [3,128,2,1], 'V3': [3,128,2,1], 'V4': [3,128,1,1]} # (kernel_size, channels, stride, pool_stride)
            if opt.mini:
                cell_mapsides={'IT': opt.ms}
            elif opt.cIT:
                cell_mapsides={'pIT': opt.ms, 'cIT':opt.ms, 'aIT': opt.ms}
            else:
                cell_mapsides={'pIT': opt.ms, 'aIT': opt.ms}
            if opt.nonretinotopic:
                block_details['V4'][1] = block_details['V4'][1]//2
                block_details['V4'][3] = 1 # no pool stride
        else: 
            assert not opt.nonretinotopic
            if opt.itnarch == 1:
                block_details = {'image': (None,3,1,1), 'retina':(7,32,1,1), 'LGN': (1,3,1,1)}
                cell_mapsides = {'IT1':2*opt.ms, 'IT2':opt.ms}
            elif opt.itnarch == 2:
                block_details = {'image': (None,3,1,1), 'retina':(7,32,2,1), 'LGN': (1,12,1,1)}
                cell_mapsides = {'IT1':opt.ms, 'IT2':opt.ms}
            elif opt.itnarch == 3:
                block_details = {'image': (None,3,1,1), 'retina':(7,32,2,1), 'LGN': (1,3,1,1), 'V1': (3,24,2,1)}
                cell_mapsides = {'IT1':opt.ms, 'IT2':opt.ms}
            elif opt.itnarch == 4:
                block_details = {'image': (None,3,1,1), 'retina':(9,32,2,1), 'LGN': (9,3,1,1)}
                cell_mapsides = {'IT1':2*opt.ms, 'IT2':opt.ms}
            elif opt.itnarch == 5:
                block_details = {'image': (None,3,1,1), 'retina':(3,32,2,1), 'LGN': (3,3,1,1)}
                cell_mapsides = {'IT1':2*opt.ms, 'IT2':opt.ms}
            else:
                raise NotImplementedError()

        backend_kwargs = dict(
            block = opt.block,
            block_kwargs=dict(polar=opt.logpolar or opt.polar),
            block_details = block_details,
            v4_dropout=opt.dropov4,
            v4_ramp=opt.rampv4,
        )

    rcell_kwargs = dict(
        recurrent_cell=opt.cell, 
        noise=opt.noise,
        connectivity=opt.conn, 
        grad_clip_val=opt.cgc,
        no_recurrence=opt.norec,
        rs=opt.rs,
        wrap_x = True if (opt.logpolar or opt.polar) else not opt.nowrap, 
        wrap_y = not opt.nowrap, # defaults, first layer will not be wrapped at all if opt.poolV4 is False
        cell_kwargs=dict(
            bias=True if 'SRN' in opt.cell or 'EI' in opt.cell or opt.cell == 'GM' else False, 
            nonlinearity=opt.nl if opt.nl is not None else 'relu',
            alpha=opt.alpha if opt.alpha is not None else 0.2,
            id = opt.id,
            ), 
        noise_kwargs=dict(f_sigma=opt.fsig, r_sigma=opt.rsig, squash_noise=opt.squashn, threshold_noise=opt.threshn), 
        connectivity_kwargs={} if opt.conn == 'full' else dict(
            f_sigma=opt.fwid, 
            r_sigma=opt.rwid, 
        )
        )
    if 'EI' in opt.cell:
        rcell_kwargs['cell_kwargs']['ei_alpha_ratio'] = opt.e2ialpharatio if opt.e2ialpharatio is not None else 1
        if opt.conn != 'full': 
            rcell_kwargs['connectivity_kwargs']['i2e_ratio'] = 1 if opt.i2eratio is None else opt.i2eratio
            rcell_kwargs['connectivity_kwargs']['center_surround'] = opt.cs if opt.conn == 'circ' else None

    arch_kwargs = dict(
        im_dim=opt.imdim,
        times=exp_times[opt.timing].stim_off,
        cell_mapsides=cell_mapsides,
        use_dropout=opt.dropo,
        E_readout=opt.ereadout,
        E_ramp_readout=opt.erreadout,
        nonretinotopic=opt.nonretinotopic,
        polar=opt.logpolar or opt.polar,
        **backend_kwargs,
        **rcell_kwargs,
    )

    if get_n_classes:
        _, n_classes = get_multi_dataset_loaders(opt.datasets, {'train':None}, subset_frac=opt.subsetfrac, return_n_classes=True)
        opt['n_classes'] = n_classes
    
    base_fn = get_name(opt, include_keys=include_keys, bool_keys=BOOL_KEYS, bools_as_ints=bools_as_ints, no_datetime=True, short_names=SHORT_NAME_DICT if use_short_names else None, ext=ext)
    if directory is not None:
        base_fn = os.path.join(directory, base_fn)

    return base_fn, opt, arch_kwargs

class BaseITN(nn.Module):
    """
    base class from which ITN and EncoderITN inherit. Defines the operations of IT
    """
    def __init__(self, 
                 n_classes=1000,
                 weights_path=None,
                 im_dim = 112,
                 times = 5,
                 use_dropout = False,
                 E_readout = False,
                 E_ramp_readout = False,
                 nonretinotopic = False,
                 polar = False,
                 cell_mapsides = {'pIT': 32, 'aIT':32},
                 device = 'cuda',
                 **rcell_kwargs,
                ):
        super().__init__()
        self.im_dim = im_dim
        self.times = times
        self.device = device
        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            if type(state_dict) is tuple:
                state_dict = state_dict[0]
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
            self._state_dict = state_dict
            self._weights_path = weights_path
            n_classes = len(state_dict['decoder.bias'])
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.E_readout = E_readout 
        self.E_ramp_readout = E_ramp_readout
        self.nonretinotopic = nonretinotopic
        self.polar = polar
        self.cells = list(cell_mapsides.keys())
        self.cell_mapsides = cell_mapsides
        self.rcell_kwargs = rcell_kwargs

    def __init_cells__(self, 
                 final_block_sl,
                 final_block_channels,
                 final_block=None,
                 ):
        self.final_block = final_block
        self.final_block_sl= final_block_sl
        self.final_block_channels= final_block_channels
        mult = self.im_dim/224
        rcell_kwargs = self.rcell_kwargs
        # set spatial coords for final block
        final_block_coords = IT.get_conv_spatial_coords(self.final_block_sl,self. final_block_channels)
        if final_block is not None:
            self.final_block.coords = final_block_coords
        # map-like recurrent cells
        # need to store default wrapping behavior, which will be overwritten for first map if using unpooled V4
        try:
            wrap_x, wrap_y = rcell_kwargs['wrap_x'], rcell_kwargs['wrap_y']
        except:
            print('wrap_x or wrap_y not specified \n setting default wrapping to True for x and y')
            wrap_x, wrap_y = True, True
        for ii, cell in enumerate(self.cells):
            if ii == 0:
                input_size = self.final_block_channels*(self.final_block_sl**2)
                if self.nonretinotopic:
                    rcell_kwargs['wrap_x'] = wrap_x
                    rcell_kwargs['wrap_y'] = wrap_y
                    input_coords = None
                else:
                    # no wrapping in the first map cell if V4 is unpooled                    
                    rcell_kwargs['wrap_y'] = False
                    # we wrap in the row dimension if using log polar, since this corresponds to the angular dimension
                    if self.polar:
                        rcell_kwargs['wrap_x'] = True
                    else:
                        rcell_kwargs['wrap_x'] = False                    
                    input_coords = final_block_coords
            else:
                input_size = self.cell_mapsides[self.cells[ii-1]]**2
                rcell_kwargs['wrap_x'] = wrap_x
                rcell_kwargs['wrap_y'] = wrap_y
                input_coords = getattr(getattr(self, self.cells[ii-1]), 'coords') 
            if ii > 0 and 'IT' in self.cells[ii-1] and rcell_kwargs['recurrent_cell'] == 'EI5':
                print('doubling input for EI5')
                cell_ff_factor = 2
                input_size = cell_ff_factor*input_size
            else:
                cell_ff_factor = 1

            dropo = self.use_dropout and (ii < len(self.cells) or len(self.cells) == 1)
            rcell = IT.ITArea(input_size, self.cell_mapsides[cell]**2,
                    input_coords=input_coords,
                    device=self.device,
                    use_dropout=dropo,
                    **rcell_kwargs,
                   )
            setattr(self, cell, rcell)
            
        if self.E_readout and self.E_ramp_readout:
            raise ValueError('Set True for at most one of E_readout and E_ramp_readout')
        if rcell_kwargs['recurrent_cell'] == 'EI5':
            print('automatically changing readout to EI ramping readout for cell EI5')
            self.decoder = ITR.EIRampReadout(self.cell_mapsides[self.cells[-1]]**2, self.n_classes, alpha=rcell_kwargs['cell_kwargs']['alpha'], device=self.device)
        else:
            if self.E_readout:
                self.decoder = ITR.EReadout(cell_ff_factor*self.cell_mapsides[self.cells[-1]]**2, self.n_classes)
            elif self.E_ramp_readout:
                self.decoder = ITR.ERampReadout(cell_ff_factor*self.cell_mapsides[self.cells[-1]]**2, self.n_classes, alpha=rcell_kwargs['cell_kwargs']['alpha'])
            else:
                self.decoder = nn.Linear(cell_ff_factor*self.cell_mapsides[self.cells[-1]]**2, self.n_classes)

    def _load_state_dict(self):
        if hasattr(self, '_weights_path'):
            # for compatibility
            if f'{self.cells[0]}.cell.hh.weight' in self._state_dict.keys():
                for cell in self.cells:
                    for conns in ['ih', 'hh']:
                        for w in ['weight', 'bias']:
                            dat = self._state_dict.pop(f'{cell}.cell.{conns}.{w}')
                            self._state_dict[f'{cell}.cell.{w}_{conns}'] = dat

            self.load_state_dict(self._state_dict)
            del self._state_dict
        else:
            self._initialize_weights()


    def spatial_params(self, subcells=['ff', 'rec']):
        for cell in self.cells:
            if cell == 'image':
                continue
            rcell = getattr(self, cell)
            if not hasattr(rcell, 'cell'):
                # ramping V4, not a real cell
                continue
            for subcell in subcells:
                if subcell == 'ff':
                    if torch.sum(rcell.ff_distances) == 0:
                        continue
                    if hasattr(self, 'v4_dropout') and self.v4_dropout:
                        yield (rcell.cell.weight_ih_raw, rcell.ff_distances)
                    else:
                        yield (rcell.cell.weight_ih, rcell.ff_distances)
                else:
                    if torch.sum(rcell.rec_distances) == 0:
                        continue
                    if hasattr(rcell.cell, 'weight_hh'):
                        yield (rcell.cell.weight_hh, rcell.rec_distances)
                    else:
                        # an FFCell
                        continue

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def lesion_network(self, lesions):
        seed = lesions.pop('seed')
        EI = False if 'EI' not in lesions else lesions.pop('EI')
        lesion_E = False if 'lesion_E' not in lesions else lesions.pop('lesion_E')
        lesion_I = False if 'lesion_I' not in lesions else lesions.pop('lesion_I')
        metric_type = None if 'metric_type' not in lesions else lesions.pop('metric_type')
        for layer_name, layer_lesions in lesions.items():
            layer = getattr(self, layer_name)
            p = layer_lesions['p']
            if 'IT' in layer_name:
                # we can have either a metric-ordered lesion, or a topographic/random one
                if 'metric' in layer_lesions.keys():
                    it_lesion(layer, p, False, metric=layer_lesions['metric'], seed=seed, EI=EI, lesion_E=lesion_E, lesion_I=lesion_I, metric_type=metric_type)
                else:
                    it_lesion(layer, p,  layer_lesions['topographic'], seed=seed, EI=EI, lesion_E=lesion_E, lesion_I=lesion_I, metric_type=metric_type)
            else:
                corblockz_lesion(layer, p, seed=seed)

    def enforce_connectivity(self):
        for ii, cell in enumerate(self.cells):
            if hasattr(getattr(self, cell), 'cell'):
                if hasattr(getattr(getattr(self, cell), 'cell'), 'enforce_connectivity'):
                    getattr(getattr(getattr(self, cell), 'cell'), 'enforce_connectivity')()
        if hasattr(self.decoder, 'enforce_connectivity'):
            self.decoder.enforce_connectivity()

class ITN(BaseITN):
    """
    a ITN model with a backend made of a series of blocks, followed by a series of RecurrentMapCells
    """
    def __init__(self,
                block='Z', 
                block_kwargs = {},
                block_details = {'image': (None, 3,1,1), 'V1': (7,64,2,2), 'V2': (3,128,2,2), 'V3': (3,128,2,2), 'V4': (3,128,2,2)}, # (kernel_size, channels, stride, pool_stride)
                v4_dropout = False,
                v4_ramp=False,
                **kwargs,
                    ):

            # do some basic initialization
            super().__init__(**kwargs)

            # set up blocks
            if block == 'Z':
                self.block = CORblock_Z
            elif block == 'RT':
                self.block = CORblock_RT
                raise NotImplementedError('The forward function needs to be optimized for conv-recurrent blocks')
            elif block == 'Zei':
                raise NotImplementedError('To implement CORblock-Z with separate E and I neurons')
            else:
                raise NotImplementedError()
            self.blocks = list(block_details.keys())
            self.block_details = block_details
            self.v4_dropout = v4_dropout
            self.v4_ramp = v4_ramp

            if self.v4_ramp and block != 'Z':
                raise ValueError('v4_ramp only used when block is otherwise non-recurrent')

            self.output_shapes = {}
            curr_size = self.im_dim
            for block, (_,_, stride, pool_stride) in block_details.items():
                curr_size =int(np.round(curr_size/(stride*pool_stride)))
                self.output_shapes[block] = curr_size

            if self.blocks[0] != 'image':
                raise ValueError(
                    f'first block must be image, was {self.blocks[0]}')

            for ii, block in enumerate(self.block_details):
                if block == 'image':
                    continue
                if block == self.blocks[-1] and self.v4_ramp:
                    block_func = CORblock_RZ
                    block_kwargs['alpha'] = self.rcell_kwargs['cell_kwargs']['alpha'] #getattr(getattr(self, self.cells[0]), 'alpha')
                else:
                    block_func = self.block
                    if 'alpha' in block_kwargs.keys():
                        block_kwargs.pop('alpha')
                setattr(self, block, 
                        block_func(self.block_details[self.blocks[ii-1]][1], self.block_details[block][1],
                                                kernel_size=self.block_details[block][0],
                                                stride=self.block_details[block][2],
                                                pool_stride=self.block_details[block][3],
                                                dropout = self.v4_dropout if block == self.blocks[-1] else False,
                                                out_shape=self.output_shapes[block],
                                                **block_kwargs,
                                            ),
                    )

            # set up cells
            final_block = list(block_details.keys())[-1]
            final_block_channels = block_details[final_block][1]
            final_block_sl = self.output_shapes[final_block]
            super().__init_cells__(final_block_sl, final_block_channels, getattr(self, final_block))

            # move V4 to "cells" if necessary to allow recurrence in forward pass
            if self.v4_ramp:
                self.blocks.remove(final_block)
                self.cells.insert(0, final_block)

            # load state dict if it is set and finish setting up
            if hasattr(self, '_weights_path'):
                self._load_state_dict()                
            self.to(self.device)
            self.eval()

    def forward(self, inputs,
                input_times=None, # by default, will be shown at every time step
                return_times=[], 
                bio_unrolling=False,
                return_layers=[], 
                out_device='cuda', 
                lesions=None,
                noise_off=False,
                get_preacts=False,
               ):
        # parse all time arguments
        if len(return_layers)>0 and len(return_times)==0 or return_times is None:
            #ensure that we return the final time step here
            return_times=[self.times-1]
        if return_times is not None and len(return_times) > 0 and np.max(return_times) > (self.times-1):
            max_time = np.max(return_times)+1
        else:
            max_time = self.times
        if input_times is None:
            input_times = max_time
        max_time = np.maximum(max_time, input_times)
        
        return_outputs = {layer: [] for layer in return_layers}
        if 'image' in return_layers:
            return_outputs['image'] = inputs
        outputs = {'image': inputs}
        states = {}   
        preact_states = {}
        if bio_unrolling:
            raise NotImplementedError()
        else:
            # multiple fixations of the same image?
            if len(inputs.shape) == 5:
                raise NotImplementedError('need to implement fivecropping')   
            else:
                for ii, block in enumerate(self.blocks[1:]):
                    prev_block = self.blocks[ii]
                    outputs[block] = getattr(self, block)(outputs[prev_block])
                    if block in return_layers:
                        return_outputs[block].append(outputs[block])
                for t in range(max_time):
                    for ii, cell in enumerate(self.cells):
                        if t == 0:
                            prev_state = None
                            prev_preact_state = None
                        else:
                            prev_state = states[cell]
                            prev_preact_state = preact_states[cell]
                        # if input should be off, first cell gets 0 as input
                        if t > input_times and ii == 0:
                            prev_output = torch.zeros_like(outputs[self.blocks[-1]])
                        else:
                            if ii == 0:
                                prev_output = outputs[self.blocks[-1]]
                            else:
                                prev_output = outputs[self.cells[ii-1]]
                        if 'IT' in cell:
                            # flatten the outputs for IT regions
                            prev_output = prev_output.view(inputs.shape[0], -1)
                        new_output = getattr(self, cell)(prev_output, prev_state, prev_preact_state, noise_off=noise_off)
                        # new output is the tuple (output, state, preact_state)
                        outputs[cell], states[cell], preact_states[cell] = new_output
                        if cell in return_layers and t in return_times:
                            if get_preacts:
                                return_outputs[cell].append(preact_states[cell])
                            else:
                                return_outputs[cell].append(states[cell]) # we use states (i.e same as outputs, but including ones not passed forward) to ensure I-cell activations are outputted 
                    if self.E_ramp_readout:
                        prev_state = None if t == 0 else states['output']
                        prev_output = outputs[self.cells[-1]]
                        outputs['output'], states['output']  = self.decoder(prev_output, prev_state)
                    else:
                        outputs['output'] = self.decoder(outputs[self.cells[-1]])                      
                    if 'output' in return_layers and t in return_times:
                        return_outputs['output'].append(outputs['output'])
        
        if len(return_layers) == 0:
            return outputs['output']
        else:
            for block in return_layers:
                output = return_outputs[block]
                if type(output) is list:
                    return_outputs[block] = torch.stack(output).transpose(0,1).to(out_device) # time dimension is 2nd dimension 
                else:
                    return_outputs[block] = output.to(out_device)
            return return_outputs

    def get_orderly_params(self):
        params = {}
        params['encoder'] = []
        for block in self.blocks:
            if block != 'image':
                params['encoder'] += [p for p in getattr(self, block).parameters()]
        params['IT'] = []
        for cell in self.cells:
            params['IT'] += [p for p in getattr(self, cell).parameters()]
        params['readout'] = [p for p in self.decoder.parameters()]
        return params

class EncoderITN(BaseITN):
    """
    a ITN model with a backend made of a pre-trained Encoder, followed by a series of RecurrentMapCells
    """
    def __init__(self,
                encoder_arch='resnet18',
                im_dim=112,
                no_stride=False,
                polar=False,
                trainset='objfacescenes',
                ramp_enc_outputs=False,
                **kwargs,
                ):
            super().__init__(**kwargs)
            str_tag = '_ns-1' if no_stride else ''
            polar_tag = '_po-1' if polar else ''
            for dset_name in ['say', 'ofs', 'obj', 'face', 'scene', 'imgnet']:
                encoder_arch = encoder_arch.replace(dset_name, '') # this is used elsewhere to get the specific pretrained model but now needs to be removed
            encoder_base_fn = f'enc-{encoder_arch}{str_tag}_imd-{im_dim}{polar_tag}_tr-{trainset}'
            mult = im_dim/224
            self.ramp_enc_outputs=ramp_enc_outputs
            if 'resnet' in encoder_arch:
                if no_stride:
                    sl = int(np.ceil(18*mult)) if polar else int(np.ceil(28*mult))
                    channels = 128
                else:
                    sl = 1 # int(np.ceil(7*mult))
                    channels = 512 if encoder_arch == 'resnet18' else 2048
                last_block_info = ('layer4', channels, sl)
            else:
                raise NotImplementedError()
            self.encoder = encoder.ITNEncoder(
                encoder_id=encoder_arch,
                base_fn=encoder_base_fn,
                no_stride=no_stride,
                im_dim=im_dim,
                polar=polar,
                dataset_name=trainset,
                )

            final_block = getattr(self.encoder.encoder.encoder, last_block_info[0])
            final_block_channels = last_block_info[1]
            final_block_sl = last_block_info[2]
            super().__init_cells__(final_block_sl, final_block_channels, final_block)

            if not hasattr(self, '_weights_path'):
                # no idea why, but loading this when we load the full state dict later makes the model perform terribly
                self.encoder.load_state_dict()
            self.encoder = self.encoder.encoder
            self.encoder._detach_features()
            for n,p in self.encoder.named_parameters():
                if 'output_downsample' not in n:
                    p.requires_grad = False
            if hasattr(self, '_weights_path'):
                self._load_state_dict()
                # self.encoder.load_state_dict(remove_encoder=True)
                
            self.to(self.device)
            self.eval()


    def train(self, mode=True):
        super().train(mode=mode)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, inputs,
                input_times=None, # by default, will be shown at every time step
                return_times=[], 
                bio_unrolling=False,
                return_layers=[], 
                out_device='cuda', 
                lesions=None,
                noise_off=False,
                get_preacts=False, # return pre-activations instead of activations
               ):
        # parse all time arguments
        if len(return_layers)>0 and len(return_times)==0 or return_times is None:
            #ensure that we return the final time step here
            return_times=[self.times-1]
        if return_times is not None and len(return_times) > 0 and np.max(return_times) > (self.times-1):
            max_time = np.max(return_times)+1
        else:
            max_time = self.times
        if input_times is None:
            input_times = max_time
        max_time = np.maximum(max_time, input_times)
        
        return_outputs = {layer: [] for layer in return_layers}
        outputs = {'image': inputs}
        if 'image' in return_layers:
            return_outputs['image'] = [inputs]
        states = {}   
        preact_states = {}
        alpha = self.rcell_kwargs['cell_kwargs']['alpha']
        if bio_unrolling:
            raise NotImplementedError()
        else:
            # multiple fixations of the same image?
            if len(inputs.shape) == 5:
                raise NotImplementedError('need to implement fivecropping')   
            else:
                outputs['encoder'] = self.encoder(inputs)
                if self.ramp_enc_outputs:
                    encoder_preramp = outputs['encoder']
                    encoder_state = 0
                    encoder_outputs = alpha*encoder_preramp
                if 'encoder' in return_layers:
                    return_outputs['encoder'].append(outputs['encoder'])
                for t in range(max_time):
                    for ii, cell in enumerate(self.cells):
                        if t == 0:
                            prev_state = None
                            prev_preact_state = None
                        else:
                            prev_state = states[cell]
                            prev_preact_state = preact_states[cell]
                        # if input should be off, first cell gets 0 as input
                        if t > input_times and ii == 0:
                            if self.ramp_enc_outputs:
                                encoder_preramp = 0
                                encoder_state = encoder_outputs
                                encoder_outputs = alpha*encoder_preramp + (1-alpha)*encoder_state
                                prev_output = encoder_outputs
                            else:
                                prev_output = torch.zeros_like(outputs['encoder'])
                        else:
                            if ii == 0:
                                if self.ramp_enc_outputs:
                                    states['encoder'] = outputs['encoder']
                                    outputs['encoder'] = self.rcell_kwargs
                                    encoder_state = encoder_outputs
                                    encoder_outputs = alpha*encoder_preramp + (1-alpha)*encoder_state
                                    prev_output = encoder_outputs
                                else:
                                    prev_output = outputs['encoder'] 
                            else:
                                prev_output = outputs[self.cells[ii-1]]
                        if cell[0] != 'V':
                            # flatten the outputs for IT regions
                            prev_output = prev_output.view(inputs.shape[0], -1)
                        new_output = getattr(self, cell)(prev_output, prev_state, prev_preact_state, noise_off=noise_off)
                        # new output is the tuple (output, state, preact_state)
                        outputs[cell], states[cell], preact_states[cell] = new_output
                        if cell in return_layers and t in return_times:
                            if get_preacts:
                                return_outputs[cell].append(preact_states[cell])
                            else:
                                return_outputs[cell].append(states[cell]) # we use states (i.e same as outputs, but including ones not passed forward) to ensure I-cell activations are outputted 

                    if self.E_ramp_readout:
                        prev_state = None if t == 0 else states['output']
                        prev_output = outputs[self.cells[-1]]
                        outputs['output'], states['output']  = self.decoder(prev_output, prev_state)
                    else:
                        outputs['output'] = self.decoder(outputs[self.cells[-1]])   
                    if 'output' in return_layers and t in return_times:
                        return_outputs['output'].append(outputs['output'])
        
        if len(return_layers) == 0:
            return outputs['output']
        else:
            for block in return_layers:
                output = return_outputs[block]
                if type(output) is list:
                    return_outputs[block] = torch.stack(output).transpose(0,1).to(out_device) # time dimension is 2nd dimension 
                else:
                    return_outputs[block] = output.to(out_device)
            return return_outputs

    def get_orderly_params(self):
        params = {}
        params['encoder'] = [p for p in self.encoder.parameters()]
        params['IT'] = []
        for cell in self.cells:
            params['IT'] += [p for p in getattr(self, cell).parameters()]
        params['readout'] = [p for p in self.decoder.parameters()]
        return params