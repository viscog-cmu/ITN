import numpy as np
import torch
from torch import nn
import os
from tqdm import tqdm
import pickle
from sklearn.decomposition import IncrementalPCA
import glob
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from topographic.dl.cornet_z import Flatten, Identity
from topographic.utils.dataset import get_multi_dataset_loaders, get_domain_inds, expand_datasets_from_name, get_all_data_transforms
from topographic.config import SAVE_DIR, ACTS_DIR
from topographic.utils.lesion import get_circular_neighborhood
from topographic.utils.decoding import do_mvpa

def load_batched_activations(net_name, layers,
                             batch_size=128,
                             imageset='objects_and_faces-5-ims-per-class',
                             phase=None,        
                             model=None, # necessary if we are saving them    
                             no_noise=False,
                             pooling='avg',
                             pca_comps=None,
                             acts_dir=ACTS_DIR,
                             overwrite=False,
                             timepoints=[-1],
                             **kwargs # passed to save batched activations if file doesn't exist
                             ):

    phase_tag = f'_phase-{phase}' if phase is not None else ''
    model_acts_dir = f'{acts_dir}/{net_name}/noise-{not no_noise}_{imageset}{phase_tag}'

    if pca_comps is not None:
        raise NotImplementedError('need to fix this functionality for multiple timepoints')

    if not os.path.exists(f'{model_acts_dir}/info.pkl') or overwrite:
        assert model is not None
        save_batched_activations(model, net_name, layers, 
                                    imageset=imageset,
                                    phase=phase, 
                                    batch_size=batch_size,
                                    no_noise=no_noise,
                                    acts_dir=acts_dir,
                                    overwrite=overwrite,
                                    **kwargs,
                                    )
    info = np.load(f'{model_acts_dir}/info.pkl', allow_pickle=True)
    all_activations = {}
    for layer in layers:
        n_files = len(glob.glob(f'{model_acts_dir}/layer-{layer}_batch-{batch_size}/*'))
        activations = []
        flattener = Flatten()
        # flattener = Identity()
        if 'IT' in layer:
            pooler = Identity()
        elif pooling == 'avg':
            pooler = torch.nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            pooler = torch.nn.AdaptiveMaxPool2d(1)
        elif pooling == None:
            pooler = Identity()
        else:
            raise NotImplementedError()
        if pca_comps is not None:
            pca = IncrementalPCA(pca_comps)
        for ii in range(n_files):
            batch = np.load(f'{model_acts_dir}/layer-{layer}_batch-{batch_size}/{ii}.pkl', allow_pickle=True)
            # with open(f'{model_acts_dir}/layer-{layer}_batch-{batch_size}/{ii}.pkl', 'rb') as f:
            #     batch = torch.tensor(pickle.load(f), dtype=torch.float32)
            full_batch = []
            for timepoint in timepoints:
                t_batch = batch[:,timepoint,:]
                t_batch = flattener(pooler(torch.tensor(t_batch, dtype=torch.float32)))
                if pca_comps is not None:
                    pca.partial_fit(t_batch)
                else:
                    full_batch.append(t_batch)
            activations.append(torch.stack(full_batch))
        if pca_comps is not None:
            for ii in range(n_files):
                batch = np.load(f'{model_acts_dir}/layer-{layer}_batch-{batch_size}/{ii}.pkl', allow_pickle=True)
                batch = flattener(pooler(batch))
                activations.append(pca.transform(batch))
        activations = torch.cat(activations, dim=1).transpose(1,0)
        all_activations[layer] = activations.numpy()
    
    return all_activations, info


def save_batched_activations(model, net_name, layers,
                             data_loader=None,
                             imdim=112,
                             batch_size=128,
                             non_mapnet_arch=False,
                             imageset='objects_and_faces-5-ims-per-class',
                             phase=None,
                             device='cuda',
                             no_noise=False,
                             input_times=None,
                             total_times=None,
                             acts_dir=ACTS_DIR,
                             overwrite=False,
                             progress_bar=True,
                             **transform_kwargs,
                             ):

    if input_times is None:
        print(f'automatically selecting input_times to {model.times}. If not desired, specify input_times.')
    input_times = model.times if input_times is None else input_times
    if total_times is None:
        print(f'automatically selecting total_times to {model.times}. If not desired, specify total_times.')
    total_times = model.times if input_times is None else total_times

    dsets = expand_datasets_from_name(imageset)

    if data_loader is None:
        data_transforms_, transforms_per_dataset = get_all_data_transforms(imdim, dataset_names=dsets, **transform_kwargs)
        if transforms_per_dataset:
            data_transforms = {dset:{phase: data_transforms_[dset][phase]} for dset in data_transforms_.keys()} 
        else:
            data_transforms = {phase: data_transforms_[phase]}
        data_loader = get_multi_dataset_loaders(dsets,
                            data_transforms,
                            transforms_per_dataset=transforms_per_dataset,
                            num_workers=6,
                            batch_size=batch_size)[phase]

    phase_tag = f'_phase-{phase}' if phase is not None else ''
    model_acts_dir = f'{acts_dir}/{net_name}/noise-{not no_noise}_{imageset}{phase_tag}'
    os.makedirs(model_acts_dir, exist_ok=True)

    if os.path.exists(f'{model_acts_dir}/info.pkl'):
        if overwrite:
            print('activations already exist, overwriting')
        else:
            print('activations already exist, not overwriting')
            return

    with torch.no_grad():
        model.to(device)
        acts_dict = {}
        outputs = []
        data = {}
        category_idx = []
        fnames = []
        other_info = {}
        print('acquiring inputs....')
        for ii, batch in enumerate(tqdm(data_loader, disable=not progress_bar)):
            try:
                (inputs, labels, etc) = batch
                if type(etc) == dict:
                    for key in etc.keys():
                        if type(etc[key]) is not list:
                            etc[key] = list(etc[key])
                        if ii == 0:
                            other_info[key] = etc[key]
                        else:
                            other_info[key] += etc[key]
                else:
                    fnames.append(etc)
            except:
                (inputs, labels) = batch
            inputs = inputs.to(device)
            if inputs.max() > 10:
                raise RuntimeError(f'max of inputs is {inputs.max()}\n should be 2.46; GPU node may have issues')
            if non_mapnet_arch:
                mini_acts = model.forward(inputs, return_layers=layers+['output'], out_device='cpu')
            else:
                mini_acts = model.forward(inputs, return_layers=layers+['output'], return_times=np.arange(total_times), out_device='cpu', noise_off=no_noise, input_times=input_times)
            outputs.append(mini_acts['output'])
            category_idx.append(labels)
            for layer in layers:
                acts_fn = f'{model_acts_dir}/layer-{layer}_batch-{batch_size}/{ii}.pkl'
                os.makedirs(os.path.dirname(acts_fn), exist_ok=True)
                X = mini_acts[layer]

                # extract timepoint
                # X = X[:,timepoint]

                with open(acts_fn, 'wb') as f:
                    pickle.dump(X.numpy(), f, protocol=pickle.HIGHEST_PROTOCOL)

        outputs = torch.cat(outputs)
        category_idx = torch.cat(category_idx).numpy()
        
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2,1)  # make time dimension last
        _, ranked = torch.sort(outputs.data, 1, descending=True)
        preds = ranked[:,0]

        if len(outputs.shape) == 3:
            correct = []
            accuracy = []
            for t in range(outputs.shape[-1]):
                correct.append([preds[ii,t] == category_idx[ii] for ii in range(len(category_idx))])
                accuracy.append(np.mean(correct))
        else:
            correct = [preds == category_idx[ii] for ii in range(len(category_idx))]
            accuracy = np.mean(correct)

        if len(fnames) > 0:
            fnames = np.concatenate(fnames)
            data['paths'] = fnames
            data['category_name'] = [os.path.basename(
                os.path.dirname(fn)) for fn in fnames]
        data['category_idx'] = category_idx
        data['predictions'] = np.array(preds)
        data['correct'] = np.array(correct)
        data['accuracy'] = np.array(accuracy)
        for key in other_info.keys():
            data[key] = np.array(other_info[key])

        with open(f'{model_acts_dir}/info.pkl', 'wb') as f:
            pickle.dump(data, f)

    
def do_mini_objfacescenes_exp(model, layers,
                 data_loader=None,
                 batch_size=128,
                 device='cuda',
                 imdim=112,
                 normalize=False,
                 progress_bar=True,
                 as_dataframe=True,
                 times=[-1],
                 input_times=None,
                 return_activations=True,
                 return_top5=False,
                 dataset_name='objfacescenes',
                 OFS_test_set=False, # use this to select an alternative miniOFS with separate images from the standard one
                 topo_convs=False,
                 EI=False,
                 conv_pooling='avg',
                 non_mapnet_arch=False,
                 ims_per_class=1,
                 no_noise=False,
                 compute_selectivity=True,
                 accuracy_vector=False, # return trialwise vs. mean accuracy
                 cache_base_fn=None,
                 overwrite=False,
                 overwrite_sl=False,
                 equate_domains=False,
                 cache_dir=ACTS_DIR,
                 run_searchlights=False,
                 get_preacts=False,
                 sl_ps=[0.01, 0.1],
                 sl_ts=[-1],
                 sl_save_dir=os.path.join(SAVE_DIR, 'searchlights'),
                 sl_mvpa_kwargs=dict(kfolds=3, classifier='ridge', k_percent_feats=100),
                 **transform_kwargs,
                 ):
    """
    not to be confused with miniOFS, this experiment runs through 1 image for each category of objfacescenes
    """
    assert dataset_name in ['objfacescenes', 'objects_and_faces', 'miniOFS', 'miniOFSW', 'miniOFS2', 'miniOFS3', 'miniF2', 'miniF3', 'miniO', 'miniF', 'miniS', 'vpnl_floc', 'konk_floc', 'vpnl+konk_floc', 'konk_animsize', 'konk_animsize_texform']
    if OFS_test_set:
        assert dataset_name in ['objfacescenes', 'objects_and_faces', 'miniOFS', 'miniOFSW', 'miniOFS2', 'miniOFS3', 'miniF2', 'miniF3', 'miniO', 'miniF', 'miniS']
    loc_tag = '_test' if OFS_test_set else ''
    if dataset_name in ['miniOFS', 'miniO', 'miniF', 'miniS', 'miniOFS2', 'miniOFS3', 'miniF2', 'miniF3']:
        # miniO, miniF, miniS, miniOFS all map to the same testing datasets. the point is to test selectivity after partial experience
        if '2' in dataset_name:
            expander = 'miniOFS2'
        elif '3' in dataset_name:
            expander = 'miniOFS3'
        else:
            expander = 'miniOFS'
        phase = 'val'
        dsets_ = expand_datasets_from_name(expander)
        dsets = [f'{ds}_{ims_per_class}-ims-per-class{loc_tag}' for ds in dsets_]
        nametag = f'{dataset_name}_{ims_per_class}-ims-per-class{loc_tag}'
        domains = ['face', 'object', 'scene']
    elif dataset_name == 'miniOFSW':
        phase = 'val'
        dsets_ = expand_datasets_from_name(dataset_name)
        dsets = [f'{ds}_{ims_per_class}-ims-per-class{loc_tag}' for ds in dsets_]
        nametag = f'{dataset_name}_{ims_per_class}-ims-per-class{loc_tag}'
        domains = ['face', 'object', 'scene', 'word']
    elif dataset_name == 'vpnl_floc':
        phase = 'val'
        dsets = [dataset_name]
        nametag = dataset_name
        domains = ['face','body','object','scene','character','scrambled']
    elif dataset_name == 'konk_floc':
        phase = 'val'
        dsets = [dataset_name]
        nametag = dataset_name
        domains = ['body', 'face', 'object', 'scene', 'scrambled']
    elif dataset_name == 'konk_animsize':
        phase = 'val'
        dsets = ['konk_animsize_texform']
        nametag = dataset_name
        domains = ['OrigBigAnimals', 'OrigBigObjects', 'OrigSmallAnimals', 'OrigSmallObjects']
        dataset_name = 'konk_animsize_texform'
    elif dataset_name == 'konk_animsize_texform':
        phase = 'val'
        dsets = ['konk_animsize_texform']
        nametag = dataset_name
        domains = ['TexformBigAnimals', 'TexformBigObjects', 'TexformSmallAnimals', 'TexformSmallObjects']
    elif dataset_name == 'vpnl+konk_floc':
        phase = 'val'
        dsets = ['vpnl_floc', 'konk_floc']
        nametag = dataset_name
        domains = ['body', 'character', 'face', 'object', 'scene', 'scrambled']
    elif dataset_name in ['objects', 'faces', 'scenes']:
        # as above, map the single domains to all 3 in order to test selectivity after partial experience.
        # the reason this is necessary is we often set dataset_name = opt.trainset
        phase = 'val'
        dsets = [f'objfacescenes_{ims_per_class}-ims-per-class{loc_tag}']
        nametag = f'objfacescenes_{ims_per_class}-ims-per-class{loc_tag}'
        domains = ['face', 'object', 'scene']
    else:
        phase = 'val'
        dsets = [f'{dataset_name}_{ims_per_class}-ims-per-class{loc_tag}']
        nametag = f'{dataset_name}_{ims_per_class}-ims-per-class{loc_tag}'
        domains = ['face', 'object'] if dataset_name == 'objects_and_faces' else ['face', 'object', 'scene']

    if hasattr(model, 'times'):
        max_time = np.maximum(np.max(times)+1, model.times)
    else:
        max_time = 0
    if input_times is None:
        print(f'automatically selecting input_times to {model.times}. If not desired, specify input_times.')
    input_times = model.times if input_times is None else input_times

    loaded=False
    if cache_base_fn is not None:
        acts_fn = os.path.join(cache_dir, cache_base_fn, f'{nametag}_noise-{not no_noise}.pkl') 
        if os.path.exists(acts_fn) and not overwrite:
            try:
                # print(acts_fn)
                with open(acts_fn, 'rb') as f: 
                    acts_dict, category_idx, outputs = pickle.load(f)
                loaded=True
            except:
                pass

    if not loaded:
        
        if data_loader is None:
            data_transforms_, transforms_per_dataset = get_all_data_transforms(imdim, dataset_names=dsets, **transform_kwargs)
            if transforms_per_dataset:
                data_transforms = {dset:{phase: data_transforms_[dset][phase]} for dset in data_transforms_.keys()} 
            else:
                data_transforms = {phase: data_transforms_[phase]}
            data_loader = get_multi_dataset_loaders(dsets,
                                data_transforms,
                                transforms_per_dataset=transforms_per_dataset,
                                num_workers=6,
                                batch_size=batch_size)[phase]
            
        model.to(device)
        acts_dict = {layer_name:[] for layer_name in layers}
        outputs = []
        category_idx = []
        print('acquiring inputs....')

        with torch.no_grad():
            for ii, batch in enumerate(tqdm(data_loader, disable=not progress_bar)):
                try:
                    (inputs, labels, fns) = batch
                except:
                    (inputs, labels) = batch
                inputs = inputs.to(device)
                if inputs.max() > 10:
                    raise RuntimeError(f'max of inputs is {inputs.max()}\n should be 2.46; GPU node may have issues')
                if non_mapnet_arch:
                    mini_acts = model.forward(inputs, return_layers=layers+['output'], out_device='cpu')
                else:
                    mini_acts = model.forward(inputs, return_layers=layers+['output'], return_times=np.arange(max_time), out_device='cpu', 
                                                noise_off=no_noise, 
                                                input_times=input_times,
                                                get_preacts=get_preacts,
                                                )
                outputs.append(mini_acts['output'])
                category_idx.append(labels)
                for layer_name in layers:
                    acts_dict[layer_name].append(mini_acts[layer_name].to('cpu').numpy())
        
        category_idx = torch.cat(category_idx).numpy()
        outputs = torch.cat(outputs)
        
        if cache_base_fn is not None:
            os.makedirs(os.path.dirname(acts_fn), exist_ok=True)
            with open(acts_fn, 'wb') as f: 
                pickle.dump((acts_dict, category_idx, outputs), f)

    domain_trials = {}
    domain_inds = {}
    for domain in domains:
        domain_inds[domain] = get_domain_inds(domain, dataset_name)
    for domain in domains:
        if equate_domains:
            min_within_domain_inds = np.min([len(inds) for inds in domain_inds.values()])
            domain_inds[domain] = domain_inds[domain][:min_within_domain_inds]
        domain_trials[domain] = np.nonzero([label in domain_inds[domain] for label in category_idx])[0]

    if len(outputs.shape) == 3:
        outputs = outputs.transpose(2,1)  # make time dimension last
    _, ranked = torch.sort(outputs.data, 1, descending=True)
    preds = ranked[:,0]
    top5_preds = ranked[:,0:5]

    if len(outputs.shape) == 3: # we got predictions over time
        accuracy = {domain:[] for domain in domains}
        top5 = {domain: [] for domain in domains}
        for t in range(outputs.shape[-1]):
            for domain in domains:
                if accuracy_vector:
                    accuracy[domain].append([preds[ii,t] == category_idx[ii] for ii in domain_trials[domain]])
                    top5[domain].append([category_idx[ii] in top5_preds[ii,:,t] for ii in domain_trials[domain]])
                else:
                    accuracy[domain].append(np.mean([preds[ii,t] == category_idx[ii] for ii in domain_trials[domain]]))
                    top5[domain].append(np.mean([category_idx[ii] in top5_preds[ii,:,t] for ii in domain_trials[domain]]))
    else:
        accuracy = {}
        for domain in domains:
            if accuracy_vector:
                accuracy[domain].append(np.mean([preds[ii] == category_idx[ii] for ii in domain_trials[domain]]))
                top5[domain].append([category_idx[ii] in top5_preds[ii] for ii in domain_trials[domain]])
            else:
                accuracy[domain] = np.mean([preds[ii] == category_idx[ii] for ii in domain_trials[domain]])
                top5[domain] = np.mean([category_idx[ii] in top5_preds[ii] for ii in domain_trials[domain]])

    all_activations = {}
    for layer_name in layers:
        all_activations[layer_name] = np.concatenate(acts_dict[layer_name], axis=0)
    
    if compute_selectivity:
        # set this to false to speed up acquisition of activations/accuracy
        results = {
            'layer':[], 
            't':[], 
        }
        for domain in domains:
            results[f'mean_{domain}'] = []
            results[f'{domain}_selectivity_neglogp'] = []
            if EI:
                results[f'mean_{domain}_I'] = []
                results[f'{domain}_selectivity_neglogp_I'] = []

        if conv_pooling == 'avg':
            pooler = nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                Flatten(),
            )
        if conv_pooling == 'max':
            pooler = nn.Sequential(
                torch.nn.AdaptiveMaxPool2d(1).
                Flatten(),
            )
        if conv_pooling == None or not conv_pooling:
            pooler = Identity()

        for layer_name in layers:
            if not non_mapnet_arch and layer_name in model.cells:
                use_times = times
            else:
                use_times = [0]
            for t in np.unique(use_times):
                results['layer'].append(layer_name)
                results['t'].append(t)
                activations = all_activations[layer_name][:,t].squeeze()
                if 'IT' in layer_name and not topo_convs:
                    if EI:
                        sl = int(np.sqrt(len(activations.mean(0).reshape(-1))/2))
                        celltype_inds = {'': np.arange(sl**2), '_I':np.arange(sl**2,2*(sl**2))}
                    else:
                        sl = int(np.sqrt(len(activations.mean(0).reshape(-1))))
                        celltype_inds = {'': np.arange(sl**2)}
                else:
                    if topo_convs:
                        sl = getattr(model, layer_name).map_side
                    celltype_inds = {'': None}
                dom_acts={}
                for celltype, cell_inds in celltype_inds.items():
                    for domain in domains:
                        if topo_convs:
                            # figure out if we are looking at flat or feature map representations
                            if len(activations[domain_trials[domain]].shape) > 2:
                                dom_acts[domain] = pooler(torch.tensor(activations[domain_trials[domain]]).to('cuda')).to('cpu').numpy()
                            else:
                                dom_acts[domain] = activations[domain_trials[domain]]
                            results[f'mean_{domain}'].append(dom_acts[domain].mean(0).reshape(sl,sl))
                        else:
                            if 'IT' in layer_name:
                                dom_acts[domain] = activations[domain_trials[domain]][:,cell_inds]
                                results[f'mean_{domain}{celltype}'].append(dom_acts[domain].mean(0).reshape(sl,sl))
                            else:
                                dom_acts[domain] = pooler(torch.tensor(activations[domain_trials[domain]]).to('cuda')).to('cpu').numpy()
                                results[f'mean_{domain}'].append(dom_acts[domain].mean(0))
                                if EI:
                                    results[f'mean_{domain}_I'].append(None)
                    for domain in domains:
                        other_domain_acts = np.vstack([dom_acts[dom] for dom in domains if dom != domain])
                        tstat,p = stats.ttest_ind(dom_acts[domain], other_domain_acts, axis=0)
                        dom_greater = np.greater(dom_acts[domain].mean(0), other_domain_acts.mean(0))
                        dom_greater[dom_greater == 0] = -1
                        dom_greater[np.equal(dom_acts[domain].mean(0), other_domain_acts.mean(0))] = 0
                        p[np.isnan(tstat)] = 10e-16
                        p[p<10e-16] = 10e-16 # ensure logp does not have NaNs due to p=0 values
                        tstat[np.isnan(tstat)] = dom_greater[np.isnan(tstat)]
                        neglogp = (-np.sign(tstat)*np.log10(p))
                        if 'IT' in layer_name or topo_convs:
                            neglogp = neglogp.reshape(sl,sl)
                        else:
                            if EI:
                                results[f'{domain}_selectivity_neglogp_I'].append(None)

                        results[f'{domain}_selectivity_neglogp{celltype}'].append(neglogp)
                
        if normalize:
            results = normalize_floc_results(results)
        
        if as_dataframe:
            results = pd.DataFrame(results)
    else:
        results = None

    if run_searchlights:
        for sl_p in sl_ps:
            for layer_name in layers:
                if 'IT' not in layer_name:
                    raise NotImplementedError()
                if EI:
                    celltype_inds = {'': np.arange(sl**2), '_I':np.arange(sl**2,2*(sl**2))}
                else:
                    celltype_inds = {'': np.arange(sl**2)}
                for celltype, cell_inds in celltype_inds.items():
                    out_fn = os.path.join(
                                sl_save_dir, 
                                cache_base_fn, 
                                f'{nametag}_noise-{not no_noise}', 
                                f'p-{sl_p}_layer-{layer_name}{celltype}_clf-ridge_accuracies.pkl',
                            )
                    if os.path.exists(out_fn) and not overwrite_sl:
                        continue

                    activations = np.mean(all_activations[layer_name][:,sl_ts,:], 1).squeeze()
                    activations = activations[:,cell_inds]
                    mod_layer = getattr(model, layer_name)
                    distances = mod_layer.rec_distances[cell_inds][:,cell_inds]
                    accuracies = {}
                    for domain in domains:
                        dom_trials = domain_trials[domain]
                        # torch.save(activations[dom_trials], f'/home/nblauch/{domain}_activations_tmp.pkl')
                        accuracies[domain] = []
                        for unit_idx in tqdm(range(activations.shape[1])):
                            sl_neighborhood = get_circular_neighborhood(sl_p, distances, unit_idx).cpu()
                            res = do_mvpa(activations[dom_trials][:,sl_neighborhood], category_idx[dom_trials], **sl_mvpa_kwargs)
                            accuracy = np.mean(res['accuracy'])
                            accuracies[domain].append(accuracy)
                        accuracies[domain] = np.array(accuracies[domain])
                        
                    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                    with open(out_fn, 'wb') as f:
                        pickle.dump(accuracies, f)

    if return_top5:
        accuracy = (accuracy, top5)
        
    if return_activations:
        return results, accuracy, all_activations, domain_trials
    else:
        return results, accuracy



def normalize_floc_results(results, undo=False):
    for key in results.keys():
        if 'mean' in key:
            if 'selectivity' in key:
                vmin, vmax = -6, 6
            else:
                if np.max(results[key][0]) < 1.1:
                    # sigmoid units on 0-1
                    vmin, vmax = 0,1
                else:
                    vmin, vmax = 0, 6
        elif 'fstat' in key:
            if 'selectivity' in key:
                vmin, vmax = -100, 100
            else:
                vmin, vmax = 0, 100
        elif 'tstat' in key:
            vmin, vmax = -6, 6
        elif 'neglogp' in key:
            vmin, vmax = -5, 5
        else:
            continue
        if undo:
            results[key] = [res*(vmax-vmin)+vmin for res in results[key]]
        else:
            results[key] = [(res-vmin)/(vmax-vmin)for res in results[key]]
    return results
            
    
def floc_selectivity(X, labels, 
                     label_names=np.array(['faces','bodies','objects','scenes','characters','scrambled']), 
                     alpha_FDR = 0.0001,
                    ):
    
    assert(X.ndim == 2)
    n_neurons_in_layer = X.shape[1]
    
    # create output dictionary
    pref_dict = dict()
    pref_dict['twoway_Ns'] = dict() # record of # sig neurons from all 2-sample t-test comparisons between domain pairs
    pref_dict['domain_counts'] = {} # final N selective neurons for each domain when contrasting all domain pairs
    pref_dict['domain_props'] = {} # same, but proportions of layer neurons
    pref_dict['domain_idx'] = {} # indices of the selective neurons
    pref_dict['domain_mask'] = {} # mask version of domain_idx
    pref_dict['domain_unit_rankings'] = {} # neurons ranked by selectivity for all domains
    pref_dict['mean_resp'] = {} # mean response to given domain
    pref_dict['selectivity_t'] = {} # 1 vs. all selectivity, t-test of domain vs. others
    pref_dict['selectivity_neglogp'] = {} # 1 vs. all selectivity, -log(p) of t-test of domain vs. others

    domain_idx = np.unique(labels)
    
    # iterate through domains 
    for domainA in domain_idx:
        if label_names is not None:
            label_ = label_names[domainA]
        else:
            label_ = str(domainA)
        pref_dict['twoway_Ns'][str(domainA)] = dict()
        # get data from curr domain
        X_curr = X[labels==domainA]
        pref_dict['mean_resp'][label_] = X_curr.mean(0)
        dom_pref_rankings = []
        dom_flag = False
        # get overall selectivity measured against all domains
        X_test = X[labels!=domainA]
        t,p = stats.ttest_ind(X_curr,X_test, axis=0)
        pref_dict['selectivity_t'][label_] = t
        pref_dict['selectivity_neglogp'][label_] = -np.sign(t)*np.log10(p)
        # iterate through domains (again)
        for domainB in domain_idx:
            X_test = X[labels==domainB]
            # calculate t and p maps
            t,p = stats.ttest_ind(X_curr,X_test, axis=0)
            # deal with nans
            t[np.isnan(t)] = 0
            p[np.isnan(p)] = 1
            # determine which neurons remain significant after FDR correction
            FDR = multipletests(p, alpha=alpha_FDR, method='FDR_by', is_sorted=False, returnsorted=False)
            # sort indices according to the t map
            # sort the neuron indices according to the t map
            dom_pref_ranking = np.flip(np.argsort(t))
            # assert that no indices are repeated
            assert(len(np.unique(dom_pref_ranking)) == len(dom_pref_ranking))
            # calculate the size of the significant ROI
            dom_nsig = np.sum(np.logical_and(FDR[0] == True, t > 0))
            # skip if test was domain vs. same domain
            if domainA != domainB:
                # save nsig in pref dict for plotting
                pref_dict['twoway_Ns'][str(domainA)][str(domainB)] = dom_nsig
                # store neuron selectivity rankings
                dom_pref_rankings.append(dom_pref_ranking)
                # if the first comparison...
                if dom_flag is False:
                    dom_neurons = dom_pref_ranking[:dom_nsig] # create deepnet selective region
                    dom_flag = True
                else: # slim down the region
                    dom_neurons = pd.Index.intersection(pd.Index(dom_neurons), pd.Index(dom_pref_ranking[:dom_nsig]), sort = False)

        dom_neurons = dom_neurons.to_numpy()
        dom_mask = np.zeros((n_neurons_in_layer,))
        dom_mask[dom_neurons] = 1
        # calculate the size of the significant ROI
        dom_nsig = len(dom_neurons)
        # figure out which neurons are most selective across all categ pair comparisons...
        dom_ranking_score = np.zeros(n_neurons_in_layer)
        # "score" each index using the sort function
        for j in range(len(domain_idx)-1):
            dom_score = np.argsort(dom_pref_rankings[j])
            # accumulate scores
            dom_ranking_score = dom_ranking_score + dom_score

        # get the final selectivity indices - lowest score = most selective
        dom_ranking_score_final = np.argsort(dom_ranking_score)
        # log
        pref_dict['domain_counts'][label_] = len(dom_neurons)
        pref_dict['domain_props'][label_] = len(dom_neurons) / n_neurons_in_layer
        pref_dict['domain_idx'][label_] = dom_neurons
        pref_dict['domain_mask'][label_] = dom_mask
        pref_dict['domain_unit_rankings'][label_] = dom_ranking_score_final
                
    return pref_dict


def local_average(x, distances, p=0.3, avg_over_trials=True, verbose=True, median=False):
    assert len(x.shape) == 2
    y = []
    mean_axis = None if avg_over_trials else 1

    if distances.shape[0] > x.shape[1]:
        if verbose:
            print('using E cells to compute distances for smoothing. only of concern if using different sampling for E and I cells.')
        distances = distances[:x.shape[1],:x.shape[1]]
            
    for unit_idx in range(x.shape[1]):
        neighborhood = get_circular_neighborhood(p, distances, unit_idx).cpu()
        if median:
            local_val = np.median(x[:,neighborhood], axis=mean_axis)
        else:
            local_val = np.mean(x[:,neighborhood], axis=mean_axis)
        y.append(local_val)
    y = np.stack(y, axis=0).transpose().squeeze()
    return y
