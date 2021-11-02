import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--no-slurm', action='store_true', 
    help='run in the local terminal session, rather than submitting a slurm job')
parser.add_argument('--exp', type=str, help='which experiment we want to run')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--no-continue', action='store_true')
parser.add_argument('--no-tensorboard', action='store_true')
parser.add_argument('--no-searchlights', action='store_true')
parser.add_argument('--skip-training', action='store_true')
parser.add_argument('--new-reg', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

slurm = not args.no_slurm
exp = args.exp
dry_run = args.dry_run
skip_training = args.skip_training

# might need to change for your setup
resources='-p gpu --cpus-per-task=6 --gres=gpu:1 --mem=25GB'

#-----------------------------------------------------Set fixed default parameters--------------------------------------------------------------------------

no_tensorboard = args.no_tensorboard
one_gpu = True
overwrite = args.overwrite
continue_from_check = False if overwrite else not args.no_continue
num_workers = 6
max_epochs = 300
imdim = 112

id = 'jjd2'
gc = 10
batch_size = 1024
timing = 'v3'
dropout = False
lr0 = 0.01
full_ff_conn = False
full_r_conn = False
squash_noise = True
threshold_noise = False
mini = False
cIT = False
use_short_names = True
pretrained_convs = False # only applies to standard ITN models (i.e., not EncoderITN models with a pre-trained ResNet)
ramp_v4 = True
e_readout = False
e_ramp_readout = False
wrap = False
bools_as_ints = True
block = 'Z'
encoder = None
subset_frac = None
optim = 'sgd'
rsig = 5.0
fsig = 5.0
old_reg = not args.new_reg
rotation = None

#-----------------------------------------------------Set looped default parameters--------------------------------------------------------------------------
dsets = ['miniOFS']
seeds = [1]
fivecrops = [False]
jj_lambdas = [None]
no_recs = [False]
v4_dropouts = [False]
nonlins = [None]
retinotopics = [False]
i2e_ratios = [1]
alphas = [0.2]
ei_alpha_ratios = [1]
l2s = [0.0005]
l1s = [None]
wids =[None]
conns=['full']


if 'main' in exp:
    rsig = 5.0
    fsig = 5.0
    noises=['uniform']
    cells = ['EI4']
    mapsides = [32]
    retinotopics = [False]
    dsets = ['miniOFS']
    nonlins = ['relu']
    e_ramp_readout = True
    seeds = [2]
    encoder = 'resnet50'
    block = None
    pretrained_convs = False
    mini = False
    cIT = True
    mapsides = [32]
    ramp_v4 = True
    jj_lambdas = [0.5]
    l2s = [None]
    if 'robustness' in exp:
        seeds = [1,2,3,4]
    if 'datasets' in exp:
        dsets = ['miniO', 'miniF', 'miniS']
    if 'cornetz' in exp:
        cIT = False
        encoder = None
        block = 'Z'
        batch_size = 512
        pretrained_convs = False # can increase training efficiency by setting to True
        jj_lambdas = [0.01] # i found this was a good choice
        cells = ['EI4'] # can easily test other architectures
        rotation = 120 # data augmentation that should induce greater invariance to rotation
    if 'singleIT' in exp:
        mini = True
        cIT = False
        alphas = [1.0] # this is helpful for single-IT-area ITN model
    if 'fulldataset' in exp:
        dsets = ['objfacescenes']
        batch_size = batch_size // 1.5
    if 'constraintremoval' in exp:
        seeds = [1] # add a second seed if you want to cross-validate your lambda/arch selection
        jj_lambdas = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
        cells = ['SRN', 'SRNEFF', 'EI4', 'EI5']
    if 'norec' in exp:
        cells = ['EFFLN', 'SRN', 'EI5']
        no_recs = [True]
        alphas = [1]
        timing = 'ff'
        jj_lambdas = [10.0] #[0, 0.01, 0.1, 10.0] #[0.01, 0.1, 1.0, 10.0] #[0.1, 0.05]
        seeds = [2] #[1,2,3,4]
    if 'noln' in exp:
        cells = ['EI5', 'SRNEFF', 'EI4', 'SRN'] #['EI5', 'SRN'] # ['SRN', 'SRNEFF', 'EI5', 'EI4']
        timing = 'v3'
        jj_lambdas = [0.1, 1.0, 10.0]
        id = 'jjd2noln'
    if 'bestnonln' in exp:
        cells = ['EI5']
        jj_lambdas = [0.5]
        id = 'jjd2noln'
    if 'nonoise' in exp:
        rsig = 0.0
        fsig = 0.0
        alphas = [1.0]
        jj_lambdas = [10.0]
        id = 'jjd2noln'


if overwrite and continue_from_check:
    raise ValueError()

params=[
    '--continue-from-checkpoint' if continue_from_check else '',
    '--no-searchlights' if args.no_searchlights else '',
    '--no-tensorboard' if no_tensorboard else '',
    '--overwrite' if overwrite else '',
    '--use-short-names' if use_short_names else '',
    '--bools-as-ints' if bools_as_ints else '',
    '--skip-training' if skip_training else '',
    f'--lr0 {lr0}',
    f'--imdim {imdim}',
    f'--maxepochs {max_epochs}',
    f'--timing {timing}',
    f'--one-gpu' if one_gpu else '',
    f'--num-workers {num_workers}',
    f"--id {id}" if id is not None else '',
    f'--gc {gc}' if gc is not None else '',
    '--dropo' if dropout else '',
    '--mini' if mini else '',
    '--cIT' if cIT else '',
    '--squashn' if squash_noise else '',
    '--threshn' if threshold_noise else '',
    '--pretrainedconvs' if pretrained_convs else '',
    '--rampv4' if ramp_v4 else '',
    '--ereadout' if e_readout else '',
    '--erreadout' if e_ramp_readout else '',
    '--nowrap' if not wrap else '',
    f'--block {block}' if block is not None else '',
    f'--encoder {encoder}' if encoder is not None else '',
    f'--subsetfrac {subset_frac}' if subset_frac is not None else '',
    f'--optim {optim}',
    f'--rsig {rsig}',
    f'--fsig {fsig}',
    '--old-reg' if old_reg else '',
    f'--rot {rotation}' if rotation is not None else '',
]
if dry_run:
    print('\n DRY RUN \n')

for l1 in l1s:
    for l2 in l2s:
        for alpha in alphas:
            for ei_alpha_ratio in ei_alpha_ratios:
                for i2e_ratio in i2e_ratios:
                    for nonlin in nonlins:
                        for v4_dropout in v4_dropouts:
                            for no_rec in no_recs:
                                for jj_lambda in jj_lambdas:
                                    for retinotopic in retinotopics:
                                        for cell in cells:
                                            for fivecrop in fivecrops:
                                                for dset in dsets:
                                                    for rs in seeds:
                                                        for mapside in mapsides:
                                                            use_batch_size = batch_size // 2 if fivecrop else batch_size
                                                            use_batch_size = use_batch_size // 1.5 if cIT else use_batch_size
                                                            use_batch_size = use_batch_size // 8 if retinotopic else use_batch_size
                                                            if mapside <= 32:
                                                                pass
                                                            elif mapside == 48:
                                                                use_batch_size = use_batch_size // 4
                                                            else:
                                                                raise NotImplementedError(f'specify a batch size for ms={mapside}')
                                                            use_batch_size = np.maximum(use_batch_size, 64)
                                                            for conn in conns:
                                                                if conn == 'full' or (full_ff_conn and full_r_conn):
                                                                    use_wids=['']
                                                                else:
                                                                    full_conn_amt = 100 if conn == 'gaussian' else 1
                                                                    if full_ff_conn:
                                                                        use_wids = [f'--rwid {wid} --fwid {full_conn_amt}' for wid in wids]
                                                                    elif full_r_conn:
                                                                        use_wids = [f'--rwid {full_conn_amt} --fwid {wid}' for wid in wids]
                                                                    else:
                                                                        use_wids = [f'--rwid {wid} --fwid {wid} ' for wid in wids]
                                                                for wid in use_wids:
                                                                    for noise in noises:
                                                                        all_params = params + [
                                                                            f'--cell {cell}',
                                                                            f"{'--fivecrop' if fivecrop else ''}",
                                                                            f"--datasets {dset}",
                                                                            f"--rs {rs}",
                                                                            f"--ms {mapside}",
                                                                            f"--conn {conn} {wid}",
                                                                            f"--noise {noise}",
                                                                            f"--l2 {l2}" if l2 is not None else '',
                                                                            f'--l1 {l1}' if l1 is not None else '',
                                                                            f'--batch-size {int(use_batch_size)}',
                                                                            f"--patience {6 if dset in ['miniOFS', 'cifar10', 'cifar100'] and encoder is None else 2}",
                                                                            f'--jjlam {jj_lambda}' if jj_lambda is not None else '',
                                                                            f'--norec' if no_rec else '',
                                                                            f'--dropov4 {v4_dropout}' if v4_dropout else '',
                                                                            f'--nl {nonlin}' if nonlin is not None else '',
                                                                            f'--alpha {alpha}' if alpha is not None else '',
                                                                            f'--e2ialpharatio {ei_alpha_ratio}' if ei_alpha_ratio is not None else '',
                                                                            f'--i2eratio {i2e_ratio}' if i2e_ratio is not None else '',
                                                                            '--nonretinotopic' if not retinotopic else '',
                                                                        ]
                                                                        all_params = ' '.join([param for param in all_params if len(param) > 0])
                                                                        debug = '-m pdb ' if args.debug else ''
                                                                        COMMAND="python " + debug + "scripts/train_itn.py " + all_params
                                                                        print(COMMAND)
                                                                        if dry_run:
                                                                            continue
                                                                        if slurm:
                                                                            os.system(f"sbatch --export=\"COMMAND={COMMAND}\" --job-name ITN --time 14-00:00:00 {resources} --output=log/%j.log scripts/run_slurm.sbatch")
                                                                        else:
                                                                            os.system(COMMAND)
