import os

# configure directories here

if os.getenv('SERVERNAME') == 'mind':
    """
    example cluster setup where training is performed
    """
    IMS_DIR='/lab_data/plautlab/imagesets'
    SAVE_DIR='/user_data/nblauch/kilthub_topographic'
    SCRATCH_DIR='/scratch/nblauch'
    FIGS_DIR = SAVE_DIR + '/figures'
    ACTS_DIR = SAVE_DIR + '/activations'
    TENSORBOARD_DIR = SAVE_DIR + '/tensorboard'
elif os.getenv('SERVERNAME') == 'xps':
    """
    example local setup where training is not performed, hence the lack of IMS_DIR and SCRATCH_DIR
    """
    IMS_DIR = None
    SAVE_DIR = os.path.join(os.getenv('WHOME'), 'cmu/user_data/git/topographic/data/topographic')
    SCRATCH_DIR = None
    FIGS_DIR =  os.path.join(os.getenv('WHOME'), 'git/topographic-overleaf/figures')
    ACTS_DIR = os.path.join(SAVE_DIR, 'activations')
    TENSORBOARD_DIR = os.path.join(os.getenv('WHOME'), 'cmu/user_data/git/topographic/data/tensorboard')
else:
    # if you do not want to use an environment variable, you can configure the appropriate directory paths here
    raise NotImplementedError('Need to configure default directories')
