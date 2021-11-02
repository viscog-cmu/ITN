import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import scipy
from scipy.stats import spearmanr
import random
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import pickle
import os
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import get_cmap

from topographic.utils.commons import exp_times
from topographic.config import SAVE_DIR
from topographic.utils.activations import do_mini_objfacescenes_exp, normalize_floc_results

def load_checkpoint(base_fn, topdir=SAVE_DIR):
    (state_dict, optim_sd, scheduler_sd) = torch.load(
        '{}/models/{}.pkl'.format(topdir, base_fn))
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)
    with open('{}/results/{}_losses.pkl'.format(topdir, base_fn), 'rb') as f:
        losses = pickle.load(f)
    return state_dict, optim_sd, scheduler_sd, losses


def resume_model(model, base_fn):
    (state_dict, optimizer, scheduler, losses) = load_checkpoint(base_fn)
    model.load_state_dict(state_dict)
    return model, optimizer, scheduler, losses


def save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=True, dir=SAVE_DIR):
    with open('{}/results/{}_losses.pkl'.format(dir, base_fn), 'wb') as f:
        pickle.dump(losses, f)
    if save_state:
        torch.save((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()),
                   '{}/models/{}.pkl'.format(dir, base_fn))


def get_scheduler(optimizer, scheduler_type='plateau', patience=2):
    scheduler_type = scheduler_type.lower(
    ) if scheduler_type is not None else scheduler_type
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                         factor=0.1,
                                                         patience=patience,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=1,
                                                         min_lr=1e-5,
                                                         eps=1e-08)
    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = None
    else:
        raise NotImplementedError()
    return scheduler


def train(model, dataloaders, optimizer, base_fn,
            epochs=50,
            device='cuda',
            scheduler=None,
            criterion=nn.CrossEntropyLoss(),
            l1_decoder_lambda=0,
            jj_lambda=0,
            no_val_noise=False,
            save_checkpoints=True,
            save_epoch_weights=False,
            save_final_weights=True,
            save_dir='from_scratch',
            checkpoint_losses=None,
            tensorboard_writer=None,
            floc_layers=['pIT', 'aIT'],
            floc_dataset='objfacescenes',
            use_amp=False,
            grad_clip=None,
            timing=exp_times['v1'],
            EI=False,
            **exp_kwargs,
            ):
    """
    trains and validates the model, saving losses and model checkpoints after each epoch
    """
    if hasattr(dataloaders['train'].sampler, 'indices'):
        dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val']}
    else:
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    if checkpoint_losses is None:
        losses = {'train':{'loss': [], 'acc': [], 'top5': []},
                'val':{'loss': [], 'acc': [], 'top5':[] }}
        start_ep = 1
    else:
        losses = checkpoint_losses
        start_ep = len(losses['train']['acc'])+1
        
    if use_amp:
        from apex import amp
        
    for epoch in range(epochs):
        curr_epoch = epoch+start_ep
        if tensorboard_writer is not None:
            with torch.no_grad():
                # do floc experiment before every epoch (want to see even how initialized model does)
                model.train(False)
                if floc_layers is not None:
                    test_times = np.unique([timing.stim_off-1, timing.exp_end-1])
                    floc_results_unnorm, _, floc_acts, domain_trials = do_mini_objfacescenes_exp(model, floc_layers, 
                                                            dataset_name=floc_dataset,
                                                            normalize=False, 
                                                            as_dataframe=False,
                                                            return_activations=True, 
                                                            ims_per_class=5, 
                                                            input_times=timing.stim_off,
                                                            times = test_times,
                                                            EI=EI,
                                                            **exp_kwargs,
                        )
                    os.makedirs(f'{SAVE_DIR}/results/floc/{base_fn}', exist_ok=True)
                    with open(f'{SAVE_DIR}/results/floc/{base_fn}/epoch-{curr_epoch-1}.pkl', 'wb') as f:
                        pickle.dump(pd.DataFrame(floc_results_unnorm), f)
                    floc_results = pd.DataFrame(normalize_floc_results(floc_results_unnorm))
                    for layer in floc_layers:
                        for col in floc_results.columns:
                            for t in test_times:
                                if col == 'layer' or col == 't':
                                    continue
                                val = floc_results.loc[np.logical_and(floc_results.layer == layer, floc_results.t==t), col].to_numpy()[0]
                                if np.array(val).size == 1: # it is a scalar
                                    tensorboard_writer.add_scalar(f'{layer}/{col}_t-{t}', val, curr_epoch-1) 
                                else:
                                    cm = get_cmap('RdBu_r')
                                    val= cm(val)[:,:,0:3] #just want RGB, no alpha
                                    tensorboard_writer.add_image(f'{layer}/{col}_t-{t}', val, curr_epoch-1, dataformats='HWC')    
                        # get generic topography summary stat
                        celltypes = [''] if not EI else ['_E', '_I']
                        ms = getattr(model, layer).map_side
                        for kk, celltype in enumerate(celltypes):
                            rvals = []
                            t = timing.exp_end
                            activations = floc_acts[layer][:,t-1,:].squeeze()
                            # take mean over feature maps if necessary
                            if len(activations.shape) > 2:
                                activations = activations.reshape(activations.shape[0], activations.shape[1], -1).mean(-1)
                            if celltype == '' or 'E' in celltype:
                                activations = activations[:,:ms**2]
                            else:
                                activations = activations[:,ms**2::]
                            # get rid of nans and infinities
                            activations[np.isnan(activations)] = 0
                            activations[activations == np.inf] = np.nanmax(activations)
                            corrs = np.corrcoef(activations.transpose())
                            corrs = corrs.reshape(-1)
                            corrs[np.isnan(corrs)] = 0
                            rand_inds = np.random.choice(corrs.shape[0], size=corrs.shape[0]//10, replace=False)
                            dists = getattr(model, layer).rec_distances[:ms**2,:ms**2].cpu().numpy().reshape(-1)[rand_inds]
                            corrs = corrs[rand_inds]
                            corrs = corrs[dists != 0]
                            dists = dists[dists != 0]
                            data_r, _ = spearmanr(corrs, dists)
                            tensorboard_writer.add_scalar(f'{layer}/r-dists-resps{celltype}', data_r, curr_epoch-1)
                tensorboard_writer.flush()    
        
        print('Training: epoch {}/{}'.format(curr_epoch, start_ep+epochs-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            torch.cuda.empty_cache()
            running_loss = 0.0
            running_corrects = 0
            running_top5 = 0.0
            running_trials = 0
            is_train = 'train' in phase
            model.train(is_train)
            if no_val_noise and phase == 'val':
                kwargs = {'noise_off': True}
            else:
                kwargs={}
            kwargs['return_layers'] = ['output']
            with torch.set_grad_enabled(is_train):
                pbar = tqdm(dataloaders[phase], smoothing=0, desc=phase, bar_format='{l_bar}{r_bar}')
                for ii, batch in enumerate(pbar):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)

                    jitter = 0 if timing.jitter == 0 else np.random.randint(0, timing.jitter)
                    loss_jitter = 0 if timing.loss_jitter == 0 else np.random.randint(0, timing.loss_jitter)
                    kwargs['input_times'] = timing.stim_off+jitter
                    kwargs['return_times'] = np.arange(timing.loss_on, timing.exp_end+jitter)

                    optimizer.zero_grad()
                    outputs = model.forward(inputs, **kwargs)
                    outputs = outputs['output']
                    task_loss = 0
                    l1_loss = 0
                    jj_loss = 0
                    loss_mults = []
                    for t in range(outputs.shape[1]):
                        if t < loss_jitter:
                            # no loss before loss_on + loss_jitter
                            continue
                        if timing.loss_on + t + loss_jitter < timing.stim_off + loss_jitter:
                            # loss scaling linearly increases in period after loss starts before the stim is turned off, after which it plateaus
                            loss_mult = (t+1)/(timing.stim_off - timing.loss_on)
                        else:
                            # loss plateau value
                            loss_mult = 1
                        loss_mults.append(loss_mult)
                        task_loss += loss_mult*criterion(outputs[:,t,:], labels)
                    # keep loss magnitude invariant to scaled number of loss time steps
                    # import pdb; pdb.set_trace()
                    task_loss = task_loss/(torch.sum(torch.tensor(loss_mults)))

                    _, ranked = torch.sort(outputs.data[:,-1,:], 1, descending=True)
                    preds = ranked[:,0]
                    top5_preds = ranked[:,0:5]

                    if l1_decoder_lambda:
                        l1_loss += torch.sum(torch.abs(model.decoder.weight))
                    if jj_lambda:
                        modid = getattr(model, model.cells[-1]).cell.modid
                        if 'rec' in modid:
                            spatial_params = model.spatial_params(subcells=['rec'])
                        elif 'ff' in modid:
                            spatial_params = model.spatial_params(subcells=['ff'])
                        else:
                            spatial_params = model.spatial_params(subcells=['ff', 'rec'])
                        for param, dists in spatial_params:
                            if len(param.shape) == 4:
                                # take sum over channel of conv weight
                                use_param = param.sum(-1).sum(-1)
                            else:
                                use_param = param
                            if 'jjl2' in getattr(model, model.cells[-1]).cell.modid:
                                jj_loss += torch.sum((use_param**2)*dists/(1+use_param**2))
                            elif 'jjd2' in getattr(model, model.cells[-1]).cell.modid:
                                jj_loss += torch.sum((use_param**2)*(dists**2)/(1+use_param**2))
                            else:
                                jj_loss += torch.sum(torch.abs(use_param)*dists)
                    loss = task_loss + l1_decoder_lambda*l1_loss + jj_lambda*jj_loss
                    if torch.isnan(loss):
                        # going to quit but save checkpoint anyways
                        if save_checkpoints:
                            save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)
                        raise ValueError('loss went to NAN; try a different random seed')
                    if is_train:
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        if grad_clip is not None:
                            clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                        if hasattr(model, 'enforce_connectivity'):
                            model.enforce_connectivity()

                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data)
                    running_top5 += np.sum([label in top5_preds[ii] for ii, label in enumerate(labels.data)])
                    running_trials += len(labels)

                    pbar.set_description(f'{phase}: batch acc={ torch.sum(preds == labels.data).item()/len(labels):05f}, loss={loss.data.item():05f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            epoch_top5 = running_top5.item() / dataset_sizes[phase]
            losses[phase]['loss'].append(epoch_loss)
            losses[phase]['acc'].append(epoch_acc)
            losses[phase]['top5'].append(epoch_top5)
            print('Loss: {:.4f}\nTop1 Acc: {:.4f}\nTop5 Acc: {:.4f}'.format(
                epoch_loss, epoch_acc, epoch_top5))
            
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(f'Loss/{phase}', epoch_loss, curr_epoch)
                tensorboard_writer.add_scalar(f'Acc_top1/{phase}', epoch_acc, curr_epoch)
                tensorboard_writer.add_scalar(f'Acc_top5/{phase}', epoch_top5, curr_epoch)
                if is_train and int(torch.__version__[0]) == 1:
                    # adding histograms doesn't work in pytorch 0.4.1 (at least for me)
                    for name, vals in model.named_parameters():
                        try:
                            grad=vals.grad.to('cpu').numpy().reshape(-1)
                            tensorboard_writer.add_histogram(f'grad/{name}', grad, curr_epoch)
                        except:
                            continue

            if scheduler is not None and not is_train:
                old_lr = float(optimizer.param_groups[0]['lr'])
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('lr', old_lr, curr_epoch)
                scheduler.step(epoch_acc)
                new_lr = float(optimizer.param_groups[0]['lr'])
                if old_lr == new_lr and scheduler.cooldown_counter == 1:
                    # we have reached the end of Training
                    if save_checkpoints:
                        save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)
                    print('Training stopped at epoch {} due to failure to increase validation accuracy for last time'.format(curr_epoch))
                    return model
                

        # save losses and state_dict after every epoch, simply overwriting previous state
        if save_checkpoints:
            save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)

        if save_epoch_weights:
            torch.save(model.state_dict(), f'{save_dir}/models/{base_fn}_epoch-{curr_epoch}.pkl')

    return model


def train_encoder(model, dataloaders, optimizer, base_fn,
            epochs=50,
            device='cuda',
            scheduler=None,
            criterion=nn.CrossEntropyLoss(),
            save_checkpoints=True,
            save_final_weights=True,
            save_dir='from_scratch',
            checkpoint_losses=None,
            tensorboard_writer=None,
            imdim=224,
            use_amp=False,
            grad_clip=None,
            detach_features=False,
            ):
    """
    trains and validates the model, saving losses and model checkpoints after each epoch
    """
    if hasattr(dataloaders['train'].sampler, 'indices'):
        dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val']}
    else:
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    if checkpoint_losses is None:
        losses = {'train':{'loss': [], 'acc': [], 'top5': []},
                'val':{'loss': [], 'acc': [], 'top5':[] }}
        start_ep = 1
    else:
        losses = checkpoint_losses
        start_ep = len(losses['train']['acc'])+1
        
    if use_amp:
        from apex import amp

    for epoch in range(epochs):
        curr_epoch = epoch+start_ep
        
        print('Training: epoch {}/{}'.format(curr_epoch, start_ep+epochs-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            running_top5 = 0.0
            running_trials = 0
            is_train = 'train' in phase
            model.train(is_train)
            with torch.set_grad_enabled(is_train):
                pbar = tqdm(dataloaders[phase], smoothing=0, desc=phase, bar_format='{l_bar}{r_bar}')
                for ii, batch in enumerate(pbar):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)

                    optimizer.zero_grad()
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    _, ranked = torch.sort(outputs.data, 1, descending=True)
                    preds = ranked[:,0]
                    top5_preds = ranked[:,0:5]
                    if torch.isnan(loss):
                        # going to quit but save checkpoint anyways
                        if save_checkpoints:
                            save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)
                        raise ValueError('loss went to NAN; try a different random seed')
                    if is_train:
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        if grad_clip is not None:
                            clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data)
                    running_top5 += np.sum([label in top5_preds[ii] for ii, label in enumerate(labels.data)])
                    running_trials += len(labels)

                    pbar.set_description(f'{phase}: batch acc={ torch.sum(preds == labels.data).item()/len(labels):05f}, loss={loss:05f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            epoch_top5 = running_top5.item() / dataset_sizes[phase]
            losses[phase]['loss'].append(epoch_loss)
            losses[phase]['acc'].append(epoch_acc)
            losses[phase]['top5'].append(epoch_top5)
            print('Loss: {:.4f}\nTop1 Acc: {:.4f}\nTop5 Acc: {:.4f}'.format(
                epoch_loss, epoch_acc, epoch_top5))
            
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(f'Loss/{phase}', epoch_loss, curr_epoch)
                tensorboard_writer.add_scalar(f'Acc_top1/{phase}', epoch_acc, curr_epoch)
                tensorboard_writer.add_scalar(f'Acc_top5/{phase}', epoch_top5, curr_epoch)
                if is_train and int(torch.__version__[0]) == 1:
                    # adding histograms doesn't work in pytorch 0.4.1 (at least for me)
                    for name, vals in model.named_parameters():
                        try:
                            grad=vals.grad.to('cpu').numpy().reshape(-1)
                            tensorboard_writer.add_histogram(f'grad/{name}', grad, curr_epoch)
                        except:
                            continue

            if scheduler is not None and not is_train:
                old_lr = float(optimizer.param_groups[0]['lr'])
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('lr', old_lr, curr_epoch)
                scheduler.step(epoch_acc)
                new_lr = float(optimizer.param_groups[0]['lr'])
                if old_lr == new_lr and scheduler.cooldown_counter == 1:
                    # we have reached the end of Training
                    if save_checkpoints:
                        save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)
                        if detach_features:
                            model._detach_features()
                            save_checkpoint(model, optimizer, scheduler, losses, base_fn+'_features', save_state=save_final_weights, dir=save_dir)
                    print('Training stopped at epoch {} due to failure to increase validation accuracy for last time'.format(epoch))
                    return model
                

        # save losses and state_dict after every epoch, simply overwriting previous state
        if save_checkpoints:
            save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=save_final_weights, dir=save_dir)

    return model