import time
import os
import argparse
import torch
import pdb
from torch import optim
from torch import nn
import numpy as np
import sys
from tensorboardX import SummaryWriter
import torchsummary
import shutil
from easydict import EasyDict as edict

sys.path.append('.')
from topographic import get_model_from_args
from topographic.ITN import unparse_model_args, parse_model_args, DEFAULT_KWARGS as dkw
from topographic.utils.training import train, load_checkpoint, get_scheduler
from topographic.utils.activations import do_mini_objfacescenes_exp
from topographic.utils.experiments import run_lesion_experiment
from topographic.utils.commons import reproducible_results, exp_times
from topographic.utils.dataset import expand_datasets, get_multi_dataset_loaders, get_all_data_transforms
from topographic.config import SAVE_DIR, IMS_DIR, TENSORBOARD_DIR
from topographic.encoder import load_v1_to_v4_cornetz
from topographic.utils.plotting.EI import load_for_tests, plotting_pipeline

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# architecture kwargs
parser.add_argument('--pretrainedconvs', action='store_true')
parser.add_argument('--imdim', type=int, default=dkw.imdim, help=' ')
parser.add_argument('--cell', type=str, default=dkw.cell, help=' ')
parser.add_argument('--encoder', type=str, default=dkw.encoder, choices=['resnet18', 'resnet50', 'resnet50say', 'resnet50ofs', 'resnet50obj', 'resnet50scene', 'resnet50face', 'resnet50imgnet'], help='specify a pre-trained encoder to use')
parser.add_argument('--block', type=str, default=dkw.block, choices=['None', 'Z', 'RT'], help=' ')
parser.add_argument('--noise', type=str, default=dkw.noise, choices=['uniform', 'linear'], help='distance dependent scaling of noise')
parser.add_argument('--fsig', type=float, default=dkw.fsig, help='feedforward noise strength')
parser.add_argument('--rsig', type=float, default=dkw.rsig, help='recurrent lateral noise strength')
parser.add_argument('--squashn', action='store_true', help='squash the noise to lie in range (-1,1)')
parser.add_argument('--threshn', action='store_true', help='threshold the noise to lie in range (-1,inf) (no sign flips of weights)')
parser.add_argument('--jjlam', type=float, default=dkw.jjlam, help='jacobs/jordan lambda value (spatial cost)')
parser.add_argument('--l2', type=float, default=dkw.l2, help='l2 regularization strength')
parser.add_argument('--l1', type=float,default=dkw.l1, help='l1 regularization applied to the readout weights only')
parser.add_argument('--conn', type=str, default=dkw.conn, choices=['full', 'gaussian', 'circ', 'circ2'], help='type of initial connectivity')
parser.add_argument('--fwid', type=float, default=dkw.fwid, help='feedforward connection gaussian width')
parser.add_argument('--rwid', type=float, default=dkw.rwid, help='recurrent connection gaussian width')
parser.add_argument('--i2eratio', type=float, default=dkw.i2eratio, help='ratio of I to E wid; only valid for cell=EI*')
parser.add_argument('--cs', action='store_true', help='whether to use center-surround E/I RFs; only valid for cell=EI* with conn=circ')
parser.add_argument('--polar', action='store_true', help='transform images to polar coordinates')
parser.add_argument('--logpolar', action='store_true', help='transform images to log polar coordinates')
parser.add_argument('--ms', type=int, default=dkw.ms, help=' ')
parser.add_argument('--timing', type=str, default=dkw.timing, choices=list(exp_times.keys()))
parser.add_argument('--alpha', type=float, default=dkw.alpha, help='unitless inverse time constant')
parser.add_argument('--e2ialpharatio', type=float, default=dkw.e2ialpharatio, help='ratio of E to I alpha values, where alpha gives the I alpha value')
parser.add_argument('--ereadout', action='store_true', help='use excitatory-only readout/decoder layer')
parser.add_argument('--erreadout', action='store_true', help='use ramping excitatory-only readout/decoded layer')
parser.add_argument('--mini', action='store_true', help='use a shallower version of the model (one IT, no V3 (if CORnet encoder)')
parser.add_argument('--cIT', action='store_true', help='add cIT and remove V3 (if CORnet encoder)')
parser.add_argument('--dropo', action='store_true', help='use dropout in all but last IT layers')
parser.add_argument('--dropov4', type=float, default=False, help='set a dropout value for V4 inputs to IT')
parser.add_argument('--rampv4', action='store_true', help='ramp V4 activations over time, only relevant for block=Z')
parser.add_argument('--norec', action='store_true', help='turn off lateral connections')
parser.add_argument('--nowrap', action='store_true', help='use standard rather than wrap-around distance')
parser.add_argument('--nl', type=str, default='relu', choices=['sigmoid', 'relu', 'relu6', 'sig6'])
parser.add_argument('--nonretinotopic', action='store_true', help='strided pooling at V1, strided convolution at V4, no distance constraints on V4-pIT')
parser.add_argument('--itnarch', type=int, default=dkw.itnarch, help='MapNet version #, determines layout of blocks preceding cells')
# relevant training kwargs
parser.add_argument('--rs', type=int, default=dkw.rs, help='random seed')
parser.add_argument('--rot', type=int, default=dkw.rot, help='max rotation (in degrees) to use in data augmentation during training')
parser.add_argument('--datasets', nargs='*', default=['objfacescenes'], help='datasets to train on')
parser.add_argument('--subsetfrac', type=float, default=dkw.subsetfrac, help='fraction of classes per dataset to include')
parser.add_argument('--maxepochs', type=int, default=dkw.maxepochs, help='max # of epochs')
parser.add_argument('--gc', type=float, default=dkw.gc, help='grad clipping value (norm of all parameters)')
parser.add_argument('--cgc', type=float, default=dkw.cgc, help='grad clipping value (value for each time step of recurrent cell)')
parser.add_argument('--sched', type=str, default=dkw.sched, choices=['None', 'step', 'plateau', 'exponential'], help=' ')
parser.add_argument('--optim', type=str, default=dkw.optim, choices=['sgd', 'adam'], help=' ')
parser.add_argument('--lr0', type=float, default=dkw.lr0, help='initial learning rate')
parser.add_argument('--id', type=str, default=dkw.id, help='optional ID tag for results')
parser.add_argument('--centercroptrain', action='store_true', help='use center cropping rather than random cropping during training (same as validation)')
parser.add_argument('--padding', type=str, default=dkw.padding, help='type of expansion over domains')
# less relevant training kwargs (not to be included in name)
parser.add_argument('--old-reg', action='store_true', help='use old regularization where having spatial regularization turns off regularization in encoder and readout')
parser.add_argument('--batch-size', type=int, default=512, help=' ')
parser.add_argument('--num-workers', type=int, default=6, help=' ')
parser.add_argument('--patience', type=int, default=2, help=' ')
parser.add_argument('--amp', action='store_true', help='use mixed precision for faster training')
parser.add_argument('--save-epochs', type=int, nargs='*', default=[], help='specify a list of epochs at which to save statedict')
parser.add_argument('--save-weights', action='store_true', help=' ')
parser.add_argument('--ims-dir', type=str, default=IMS_DIR, help=' ')
parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help=' ')
parser.add_argument('--overwrite', action='store_true', help=' ')
parser.add_argument('--continue-from-checkpoint', action='store_true', help=' ')
parser.add_argument('--one-gpu', action='store_true', help=' ')
parser.add_argument('--no-tensorboard', action='store_true', help=' ')
parser.add_argument('--use-short-names', action='store_true')
parser.add_argument('--bools-as-ints', action='store_true')
# activations kwargs
parser.add_argument('--skip-training', action='store_true', help='just load model and acquire activations')
parser.add_argument('--novalnoise', action='store_true', help='turn off noise when acquiring activations')
# parser.add_argument('--testdsets', type=str, nargs='*', default=['objfacescenes_5-ims-per-class'], help='optional list of datasets to combine for saving activations')
# parser.add_argument('--testphase', type=str, default='val', help=' ')
parser.add_argument('--layers', type=str, nargs='*', default=['pIT', 'aIT'], help='layers for saving activations')
parser.add_argument('--no-overwrite-acts', action='store_true')
parser.add_argument('--no-overwrite-sl', action='store_true')
parser.add_argument('--no-searchlights', action='store_true')
#### USE THIS TO OVERWRITE ALL OTHER KEYWORD ARGS AND JUST REVERSE ENGINEER THE KWARGS FROM A FILE NAME
parser.add_argument('--base-fn', type=str, default=None, 
    help='Overrides all kwargs included in base_fn. Unparses kwargs from base filename. Only used to generate activations and plots for a trained model.',
    )

args = parser.parse_args()
opt = edict(args.__dict__)
opt.trainset, opt.datasets = expand_datasets(opt.datasets) # simplify name of potential list of training sets

if args.base_fn is None:
    base_fn, opt, arch_kwargs = parse_model_args(directory=None, is_activations=False, **opt)
else:
    if not args.skip_training:
        raise ValueError('base_fn can only be used when training is skipped to simply generate post-training results.')
    print(f'reconstructing kwargs from base_fn: {args.base_fn}')
    new_opt = unparse_model_args(args.base_fn, use_short_names=args.use_short_names)
    opt.update(new_opt)
    base_fn, opt, arch_kwargs = parse_model_args(directory=None, is_activations=False, **opt)
    assert base_fn == args.base_fn

reproducible_results(seed=opt['rs'])

node = os.getenv('SLURM_STEP_NODELIST')
if node is not None:
    print(f'working on node: {node}')

os.makedirs(f'{opt.save_dir}/models', exist_ok=True)
os.makedirs(f'{opt.save_dir}/results', exist_ok=True)

t0 = time.time()
data_transforms, transforms_per_dataset = get_all_data_transforms(opt.imdim, 
    logpolar=opt.logpolar, polar=opt.polar, 
    fivecrop=opt.fivecrop, 
    centercrop_train=opt.centercroptrain, 
    padding=opt.padding, 
    dataset_names=opt.datasets,
    max_rot_deg=opt.rot,
    )
print(data_transforms)

dataloaders, opt['n_classes'] = get_multi_dataset_loaders(opt.datasets, 
                data_transforms, 
                num_workers=opt.num_workers,
                batch_size=opt.batch_size,
                return_n_classes=True,
                subset_frac=opt.subsetfrac,
                transforms_per_dataset = transforms_per_dataset,
                )
print(opt['n_classes'])
t1 = time.time()
print(f'setting up dataloaders took {t1-t0} s')

model = get_model_from_args(opt, arch_kwargs, base_fn=None)

with torch.no_grad():
    print(torchsummary.summary(model, (3, opt.imdim, opt.imdim), batch_size=opt.batch_size))
# sys.exit()

if torch.cuda.device_count() > 1 and not opt.one_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
if opt.pretrainedconvs and not opt.skip_training:
    if opt.encoder is not None:
        raise ValueError('specifying an encoder already requires pretraining so this flag is redundant')
    if opt.block != 'Z':
        raise NotImplementedError('pretrained model does not exist. try training a simple model and updating this file with the path, or use a non-pretrained or resnet-encoder version of the model')
    nret_tag = '_nonretinotopic-True' if args.nonretinotopic else ''
    pretrained_fn = f"{opt.save_dir}/models/block-{opt.block}_cell-SRN_conn-full_fsig-0.0_gc-10.0_id-sep1_imdim-{opt.imdim}_lr0-0.01_maxepochs-300_ms-32_noise-linear{nret_tag}_optim-sgd_rs-1_rsig-0.0_sched-plateau_times-5_trainset-objfacescenes.pkl"
    if not os.path.exists(pretrained_fn):
        raise NotImplementedError('pretrained model does not exist. try training a simple model and updating this file with the path, or use a non-pretrained or resnet-encoder version of the model')
    print('loading pre-trained weights for V1-V4')
    model = load_v1_to_v4_cornetz(model, pretrained_fn, device='cuda', freeze=False, no_v3 = False)

# get parameters and weight decay values

orderly_params = model.get_orderly_params()
if args.old_reg:
    # for reproducibility of models in which a small bug was present. it doesn't seem to matter much, but we keep this for reproducibility. for new research, don't specify --old-reg
    print('spatial regularization will wipe out standard l2 regularization in encoder and readout')
    param_groups = [
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['encoder'])), 'weight_decay':opt.l2 if not opt.jjlam else 0},
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['IT'])), 'weight_decay':opt.l2 if not opt.jjlam else 0},
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['readout'])), 'weight_decay':opt.l2 if not opt.jjlam else 0},    
    ]
else:
    param_groups = [
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['encoder'])), 'weight_decay':0.0005},
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['IT'])), 'weight_decay':opt.l2 if not opt.jjlam else 0},
        {'params': list(filter(lambda p: p.requires_grad, orderly_params['readout'])), 'weight_decay':0.0005},

    ]

if opt.optim == 'sgd':
    optimizer = optim.SGD(param_groups,
                          lr=opt.lr0, momentum=0.9, 
                          )
elif opt.optim == 'adam':
    optimizer = optim.Adam(param_groups,
                          lr=opt.lr0, 
                          )
else:
    raise NotImplementedError()
scheduler = get_scheduler(optimizer, scheduler_type=opt.sched, patience=opt.patience)

if opt.amp:
    raise NotImplementedError('if you want to use amp, comment out this error message and install apex')
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

losses = None
if os.path.exists(f'{opt.save_dir}/models/{base_fn}.pkl') and not opt.overwrite and not opt.skip_training:
    if opt['continue_from_checkpoint']:
        print(f'continuing from checkpoint \n working on \n {base_fn}')
        state_dict, optim_sd, scheduler_sd, losses = load_checkpoint(base_fn, topdir=opt.save_dir)
        opt['maxepochs'] = opt['maxepochs'] - len(losses['val']['acc'])
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(optim_sd)
        scheduler.load_state_dict(scheduler_sd)
        print('starting at learning rate {}'.format(optimizer.param_groups[0]['lr']))
        if len(losses['val']['acc']) > 0:
            print('best previous validation accuracy: {}'.format(np.max(losses['val']['acc'])))
        else:
            print('starting from the first epoch')
    else:
        print(base_fn)
        raise ValueError('Model already exists and you chose to save, not to overwrite, and not to resume from checkpoint')


print(f'working on: \n{base_fn}\n ...')
if not opt.continue_from_checkpoint and os.path.exists(f'{TENSORBOARD_DIR}/{base_fn}') and not opt.skip_training:
    shutil.rmtree(f'{TENSORBOARD_DIR}/{base_fn}', ignore_errors=True)

# experiment timing
timing = exp_times[opt.timing]

exp_kwargs = dict(
                imdim=opt.imdim,
                equate_domains=True,
                fivecrop=opt.fivecrop,
                logpolar=opt.logpolar,
                polar=opt.polar,
                padding=opt.padding,
                EI='EI' in opt.cell,
)

if not opt.skip_training:
    if opt.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(logdir=f'{TENSORBOARD_DIR}/{base_fn}')
    floc_layers = [cell for cell in model.cells if hasattr(getattr(model, cell),'map_side')]
    print(floc_layers)
    model = train(model, dataloaders, optimizer, base_fn,
                    epochs=opt['maxepochs'],
                    scheduler=scheduler,
                    save_checkpoints=True,
                    save_epoch_weights=False,
                    save_final_weights=True,
                    no_val_noise=opt.novalnoise,
                    save_dir=opt.save_dir,
                    checkpoint_losses=losses,
                    tensorboard_writer=writer,
                    jj_lambda=opt.jjlam,
                    l1_decoder_lambda=opt.l1,
                    floc_layers= floc_layers,
                    floc_dataset=opt.trainset if opt.trainset in ['objfacescenes', 'miniOFS', 'miniOFSW', 'miniOFS2', 'miniOFS3', 'miniF2', 'miniF3', 'objects_and_faces'] else 'miniOFS',
                    use_amp=opt.amp,
                    grad_clip=opt.gc,
                    timing=timing,
                    topo_convs=opt.topoconv,
                    **exp_kwargs,
                    )

    if not opt.no_tensorboard:
        writer.close()
else:
    state_dict, optim_sd, scheduler_sd, losses = load_checkpoint(base_fn, topdir=opt.save_dir)
    model.load_state_dict(state_dict)
    
#####---------RUN EXPERIMENTS FOLLOWING TRAINING------------########
exp_kwargs = dict(
                layers = [cell for cell in model.cells if 'IT' in cell],
                imdim=opt.imdim,
                normalize=False,
                input_times = timing.stim_off,
                times = np.arange(timing.exp_end+timing.jitter),
                return_activations=True,
                no_noise=opt.novalnoise,
                overwrite=not args.no_overwrite_acts,
                overwrite_sl=not args.no_overwrite_sl,
                equate_domains=True,
                fivecrop=opt.fivecrop,
                logpolar=opt.logpolar,
                polar=opt.polar,
                padding=opt.padding,
                EI='EI' in opt.cell,
)

if opt.trainset in ['objfacescenes',  'miniOFSW', 'miniOFS2', 'miniOFS3']:
    res, acc, acts, d_trials = do_mini_objfacescenes_exp(model,
                    dataset_name=opt.trainset,
                    accuracy_vector=False, # return trialwise vs. mean accuracy
                    ims_per_class= 5,
                    cache_base_fn=base_fn,
                    **exp_kwargs,
                    )

res, acc, acts, d_trials = do_mini_objfacescenes_exp(model,
                dataset_name='miniOFS',
                accuracy_vector=False, # return trialwise vs. mean accuracy
                ims_per_class=10,
                cache_base_fn=base_fn,
                run_searchlights=not args.no_searchlights,
                sl_ps=[0.1],
                sl_ts=[exp_times[opt['timing']].exp_end-1],
                **exp_kwargs,
                )

### LESIONING
if opt.trainset in ['miniOFS', 'objfacescenes', 'miniOFSW', 'miniOFS2', 'miniOFS3']:
    for lesion_type in ['topographic', 'metric']:
        run_lesion_experiment(model, res, opt, arch_kwargs, base_fn, 
                            lesion_type=lesion_type,
                            overwrite=True,
                            overwrite_indiv_results=False,
                            selectivity_t=timing.exp_end,
                            analysis_times=[timing.stim_off, timing.exp_end, timing.exp_end+timing.jitter],
                            )

### all plotting
t_main='exp_end'
save=True

# plotting experiments can be customized by changing the kwargs per experiment
kwargs_by_exp = dict(
    accuracy = dict(save=save),
    maps = dict(save=save),
    cross_measure_scatters = dict(save=save),
    dynamics = dict(save=save),
    lesioning = dict(save=save),
    spatial_corrs = dict(save=save),
    localizers = dict(save=save),
    ei_columns = dict(save=save),
    rfs = dict(save=save),
    saycam = dict(save=save),
    category_level = dict(save=save, plot_contours=True),
    clustering = dict(save=save, k_clusters=3),
    principal_components = dict(save=save),
    )

outputs = load_for_tests(base_fn, overwrite=False, as_dict=True)
    
plotting_pipeline(base_fn, kwargs_by_exp, t_main=t_main, outputs=outputs, overwrite=False)