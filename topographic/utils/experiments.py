import torch
import torch.nn.utils.prune as prune
import numpy as np
import os
import pickle
import sys
import pdb

from topographic.config import SAVE_DIR, ACTS_DIR
from topographic import get_model_from_args, get_model_from_fn
from topographic.utils.activations import do_mini_objfacescenes_exp, local_average
import topographic.utils.dataset
from topographic.utils.commons import reproducible_results, exp_times


def run_lesion_experiment(model, res, opt, arch_kwargs, base_fn, 
                        lesion_type='topographic',
                        overwrite=False,
                        overwrite_indiv_results=False,
                        selectivity_t=15,
                        analysis_times=[1,5,10,15,20],
                        lesion_ps=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                        ):
    """
    run a topographic or selectivity-ordered lesion experiment
    """
    assert lesion_type in ['topographic', 'metric', 'smooth_metric', 'metric_pval', 'smooth_metric_pval']

    os.makedirs(f'{SAVE_DIR}/lesions/summaries/{base_fn}', exist_ok=True)
    l_tag = f'_{lesion_type}_lesion' if lesion_type != 'topographic' else ''
    lesion_fn = f'{SAVE_DIR}/lesions/summaries/{base_fn}/summary{l_tag}.pkl'

    times = exp_times[opt['timing']].exp_end + exp_times[opt['timing']].jitter # let's go up to the maximum time
    input_times = exp_times[opt['timing']].stim_off
    dataset_name = topographic.utils.dataset.get_experiment_dataset(opt.trainset)
    layers = [cell for cell in model.cells if 'IT' in cell] # make sure to exclude V4
    celltypes = ['', '_I'] if 'EI' in opt.cell else ['']
    mod_fn = os.path.join(SAVE_DIR, 'models', base_fn+'.pkl')

    if os.path.exists(lesion_fn) and not overwrite:
        with open(lesion_fn, 'rb') as f:
            all_acc = pickle.load(f)
    else:
        all_acc = {'layer':[], 'lesion_E':[], 'lesion_I':[], 'lesion_EI':[], 'targ_domain':[], 'domain':[], 't':[],  'accuracy':[], 'lesion_p':[]}
        for lesion_area in layers:
            for lesion_E, lesion_I in [(True, True)]: #[(True, False), (False, True), (True, True)]:
                for lesion_targ_domain in ['face', 'object', 'scene']:

                    # compute smoothed selectivity map
                    func_map = res.iloc[np.logical_and.reduce((res.t == selectivity_t-1, res.layer == lesion_area))][f'{lesion_targ_domain}_selectivity_neglogp'].iloc[0]
                    func_map[func_map == np.nan] = 0
                    if lesion_type in ['topographic', 'smooth_metric', 'smooth_metric_pval']:
                        func_map = local_average(func_map.reshape(1,-1), getattr(model, lesion_area).rec_distances, verbose=False, p=0.3)
                        if lesion_type == 'topographic':
                            lesion_seed = func_map.argmax()
                            kwargs = dict(topographic=True)
                            tag_fmt = f'/l-{lesion_area}-' + '{}' + f'-ix-{lesion_seed}-E-{int(lesion_E)}-I-{int(lesion_I)}'
                    if 'metric' in lesion_type:
                        kwargs = dict(metric=func_map)
                        tag_fmt = f'/l-{lesion_area}-' + '{}' + f'-{lesion_targ_domain}-{lesion_type}-E-{int(lesion_E)}-I-{int(lesion_I)}'
                        lesion_seed = 1 # just for compatibility, won't do anything

                    metric_type = 'threshold' if 'pval' in lesion_type else 'rank' if 'metric' in lesion_type else None
                    for lesion_p in lesion_ps:
                        if 'pval' in lesion_type:
                            ltag = tag_fmt.format(lesion_p)
                        else:
                            ltag = tag_fmt.format(lesion_p) if lesion_p > 0 else ''
                        loaded = False
                        if os.path.exists(os.path.join(ACTS_DIR, base_fn+ltag, 'accuracy.pkl')) and not overwrite_indiv_results:
                            try:
                                with open(os.path.join(ACTS_DIR, base_fn+ltag, 'accuracy.pkl'), 'rb') as f:
                                    accuracy = pickle.load(f)
                                    loaded = True
                            except:
                                pass
                        if not loaded: 
                            reproducible_results(seed=opt['rs']) # ensures connectivity masks are identical to the trained ones
                            model = get_model_from_args(opt, arch_kwargs, base_fn)
                            if lesion_p is not None and (lesion_p > 0 or 'pval' in lesion_type):
                                if 'EI' in opt.cell:
                                    model.lesion_network({'seed':lesion_seed, 'EI':True, 'lesion_E':lesion_E, 'lesion_I':lesion_I, 'metric_type':metric_type, lesion_area:dict(p=lesion_p, **kwargs)})
                                else:
                                    model.lesion_network({'seed':lesion_seed, 'metric_type':metric_type, lesion_area:dict(p=lesion_p, **kwargs)})
                            exp_outputs = do_mini_objfacescenes_exp(model, layers=layers,
                                                                input_times=input_times, 
                                                                times=np.arange(times), 
                                                                dataset_name=dataset_name, 
                                                                OFS_test_set=True,
                                                                cache_base_fn=None, 
                                                                ims_per_class= 10, #1 if opt.trainset == 'objfacescenes' else 5, 
                                                                equate_domains=True, 
                                                                return_activations=False, 
                                                                accuracy_vector=True, 
                                                                overwrite=overwrite_indiv_results,
                                                                progress_bar=False,
                                                                compute_selectivity=False,
                                                                batch_size=128,
                                                                fivecrop=opt.fivecrop,
                                                                logpolar=opt.logpolar,
                                                                polar=opt.polar,
                                                                padding=opt.padding,
                                                                EI='EI' in opt.cell,
                                                                )
                            accuracy = exp_outputs[1]
                            os.makedirs(os.path.join(ACTS_DIR, base_fn+ltag), exist_ok=True)
                            with open(os.path.join(ACTS_DIR, base_fn+ltag, 'accuracy.pkl'), 'wb') as f:
                                pickle.dump(accuracy, f)

                        for t in np.unique(analysis_times):
                            for domain in ['face', 'object', 'scene']:
                                for jj in range(len(accuracy[domain][t-1])):
                                    all_acc['domain'].append(domain)
                                    all_acc['t'].append(t)
                                    all_acc['accuracy'].append(int(accuracy[domain][t-1][jj].numpy()))
                                    all_acc['lesion_p'].append(lesion_p)
                                    all_acc['targ_domain'].append(lesion_targ_domain)
                                    all_acc['layer'].append(lesion_area)
                                    all_acc['lesion_E'].append(lesion_E)
                                    all_acc['lesion_I'].append(lesion_I)
                                    all_acc['lesion_EI'].append((lesion_E, lesion_I))
        with open(lesion_fn, 'wb') as f:
            pickle.dump(all_acc, f)

    return all_acc


def run_wiring_cost_experiment(base_fn, 
                        overwrite=False,
                        overwrite_indiv_results=False,
                        overwrite_indiv_exp=False,
                        analysis_times=[1,5,10,15,20],
                        sparsity_vals=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                        alg='l1',
                        local_pruning=False, # set to true to achieve same sparsity for each parameter
                        inputs_and_outputs=False,
                        ):
    """
    run a topographic or selectivity-ordered lesion experiment
    """

    os.makedirs(f'{SAVE_DIR}/wiring/summaries/{base_fn}', exist_ok=True)
    a_tag = f'alg-{alg}_' if alg != 'l1' else ''
    l_tag = f'lp_' if not local_pruning else ''
    i_tag = 'io_' if inputs_and_outputs else ''
    output_fn = f'{SAVE_DIR}/wiring/summaries/{base_fn}/{a_tag}{l_tag}{i_tag}summary.pkl'

    if inputs_and_outputs:
        parameters_to_prune=(
            ('pIT', ['weight_hh']),
            # ('pIT', ['weight_ih', 'weight_hh']),
            ('cIT', ['weight_ih', 'weight_hh']),
            ('aIT', ['weight_ih', 'weight_hh']),
            ('decoder', ['weight']),
        )
    else:
        parameters_to_prune=(
            ('pIT', ['weight_hh']),
            ('cIT', ['weight_ih', 'weight_hh']),
            ('aIT', ['weight_ih', 'weight_hh']),
        )

    if os.path.exists(output_fn) and not overwrite:
        with open(output_fn, 'rb') as f:
            results = pickle.load(f)
    else:
        model, opt, kwargs, arch_kwargs = get_model_from_fn(base_fn)
        times = exp_times[opt['timing']].exp_end + exp_times[opt['timing']].jitter # let's go up to the maximum time
        input_times = exp_times[opt['timing']].stim_off
        dataset_name = topographic.utils.dataset.get_experiment_dataset(opt.trainset)
        layers = [cell for cell in model.cells if 'IT' in cell] # make sure to exclude V4
        celltypes = ['', '_I'] if 'EI' in opt.cell else ['']
        mod_fn = os.path.join(SAVE_DIR, 'models', base_fn+'.pkl')

        results = {'domain':[], 't':[], 'accuracy':[], 'top5':[], 'wiring_cost':[], 'wiring_cost_ff':[], 'wiring_cost_rec':[], 'sparsity':[]}
        for sparsity in sparsity_vals:
            s_tag = f'_spars-{sparsity}'
            loaded = False
            if os.path.exists(os.path.join(ACTS_DIR, base_fn+s_tag, a_tag+l_tag+i_tag+'results.pkl')) and not overwrite_indiv_results:
                try:
                    with open(os.path.join(ACTS_DIR, base_fn+s_tag, a_tag+l_tag+i_tag+'results.pkl'), 'rb') as f:
                        accuracy, top5, wiring_cost = pickle.load(f)
                        loaded = True
                except:
                    pass
            if not loaded: 
                reproducible_results(opt['rs'])
                model = get_model_from_args(opt, arch_kwargs, base_fn=base_fn)
                params_to_prune = []
                for layer, params in parameters_to_prune:
                    for param in params:
                        if hasattr(getattr(model, layer), 'cell') and hasattr(getattr(getattr(model, layer), 'cell'), param):
                            params_to_prune.append((getattr(getattr(model, layer), 'cell'), param))
                            if local_pruning:
                                prune.l1_unstructured(getattr(getattr(model, layer), 'cell'), name=param, amount=1-sparsity)
                        elif hasattr(getattr(model, layer), param) and hasattr(getattr(model, layer), param):
                            params_to_prune.append((getattr(model, layer), param))
                            if local_pruning:
                                prune.l1_unstructured(getattr(model, layer), name=param, amount=1-sparsity)
                        else:
                            print(f'model does not have attribute {layer}.{param} \n skipping')
                if not local_pruning:
                    prune.global_unstructured(
                        params_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=1-sparsity,
                    )
                    for layer, param in params_to_prune:
                        print(f'{param} sparsity = {100. * float(torch.sum(getattr(layer, param) == 0))/float(getattr(layer, param).nelement())}')

                wiring_cost = {}
                for kind in ['rec', 'ff']:
                    wiring_cost[kind] = get_wiring_cost(model, kind=kind, alg=alg)

                exp_outputs = do_mini_objfacescenes_exp(model, layers=layers,
                                    input_times=input_times, 
                                    times=np.arange(times), 
                                    dataset_name=dataset_name, 
                                    OFS_test_set=True,
                                    cache_base_fn=None, 
                                    ims_per_class= 10, #1 if opt.trainset == 'objfacescenes' else 5, 
                                    equate_domains=True, 
                                    return_activations=False, 
                                    return_top5=True,
                                    accuracy_vector=True, 
                                    overwrite=overwrite_indiv_exp,
                                    progress_bar=False,
                                    compute_selectivity=False,
                                    batch_size=128,
                                    fivecrop=opt.fivecrop,
                                    logpolar=opt.logpolar,
                                    polar=opt.polar,
                                    padding=opt.padding,
                                    EI='EI' in opt.cell,
                                    )
                accuracy, top5 = exp_outputs[1]
                os.makedirs(os.path.join(ACTS_DIR, base_fn+s_tag), exist_ok=True)
                with open(os.path.join(ACTS_DIR, base_fn+s_tag, a_tag+l_tag+i_tag+'results.pkl'), 'wb') as f:
                    pickle.dump((accuracy, top5, wiring_cost), f)

            for t in np.unique(analysis_times):
                for domain in ['face', 'object', 'scene']:
                    for jj in range(len(accuracy[domain][t-1])):
                        results['domain'].append(domain)
                        results['t'].append(t)
                        results['accuracy'].append(int(accuracy[domain][t-1][jj].numpy()))
                        results['top5'].append(int(top5[domain][t-1][jj]))
                        results['sparsity'].append(sparsity)
                        results['wiring_cost'].append(np.sum([cost for cost in wiring_cost.values()]))
                        results['wiring_cost_ff'].append(wiring_cost['ff'])
                        results['wiring_cost_rec'].append(wiring_cost['rec'])

        with open(output_fn, 'wb') as f:
            pickle.dump(results, f)

    return results

def get_wiring_cost(model, kind='all', alg='l1'):
    """
    compute the binary wiring cost of a sparsely connected model.
    typically the model should be subject to pruning first to ensure sparsity
    we iterate over all spatial parameters to compute the total

    inputs:
        model
        kind: one of ('all', 'ff', 'rec') - which connections to count
        alg: one of ('l1', 'l2') - whether to use standard or squared distance
    """

    assert kind in ['all', 'ff', 'rec']
    assert alg in ['l1', 'l2']
    subcells = ['ff', 'rec'] if kind == 'all' else [kind]
    spatial_params = model.spatial_params(subcells)
    cost = 0.0
    param_count = 0
    for param, dists in spatial_params:
        if alg == 'l1':
            cost += torch.sum(dists[param != 0])
        else:
            cost += torch.sum(dists[param != 0]**2)
        param_count += torch.sum(param != 0)
    return (cost/param_count).item()

if __name__ == "__main__":
    from topographic.utils.training import exp_times
    from topographic.config import NAMED_MODELS
    from topographic.ITN import unparse_model_args

    for key in ['main_model_objects_to_ofs']: # ['resnet50_jj_ei4_rv4_nowrap_d2', 'resnet50_jj_ei4_rv4_nowrap_d2_srneff', 'jjd2_norec']: # ['resnet50_jj_ei4_rv4']: #, 'resnet50_circ2_ei4_norv4', 'resnet50_gaussian_ei4_rv4']:
        for rs in [2]: #range(2,5):
            base_fn = NAMED_MODELS[key]
            base_fn = base_fn.replace('_rs-1_', f'_rs-{rs}_')
            kwargs = unparse_model_args(base_fn, use_short_names=True)
            timing = exp_times[kwargs['timing']]
            results = run_wiring_cost_experiment(base_fn, 
                                        sparsity_vals=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
                                        analysis_times=[timing.stim_off, timing.exp_end, timing.exp_end+timing.jitter],
            )
            pdb.set_trace()