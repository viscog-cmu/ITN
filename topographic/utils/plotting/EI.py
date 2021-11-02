import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from PIL import Image, ImageChops
import glob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import cv2
import timeit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import imageio
from scipy.stats import sem, spearmanr
import pandas as pd
from itertools import combinations
import torch.nn
import cmasher as cmr
import os
import pdb
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import os
from statannot import add_stat_annotation
from sklearn.linear_model import RidgeClassifierCV
from matplotlib.cm import get_cmap

from topographic import get_model_from_args
from topographic.config import FIGS_DIR, IMS_DIR, SAVE_DIR
from topographic.ITN import unparse_model_args, parse_model_args
from topographic.utils.activations import do_mini_objfacescenes_exp, local_average, load_batched_activations
from topographic.utils.dataset import get_domain_inds, get_experiment_dataset, BaoTsaoDataset, get_dataset
from topographic.utils.experiments import run_lesion_experiment
from topographic.utils.plotting import COLORS, MidpointNormalize
from topographic.utils.commons import reproducible_results, exp_times


def dist_to_desired_corr(dist, lam=1/np.log(10), shift=0):
    corr = shift + np.exp(-lam*dist)
    return corr

def dist_to_desired_corr_noshift(dist, lam=1/np.log(10)):
    corr = np.exp(-lam*dist)
    return corr

def dist_to_desired_corr_simple(dist, scale=1):
    """
    the model of Lee et. al, 2020. we need to fit a scale because our map size is not in mm
    """
    corr = scale/(1+dist)
    return corr


def plotting_pipeline(base_fn, kwargs_by_exp, t_main='exp_end', outputs=None, overwrite=False):
    """
    Wrapper function to execute an arbitrary set of experiments contained in the plotting_pipeline

    base_fn: filename specifying the model and its parameters
    kwargs_by_exp: dict where keys are experiment names and values are dicts containing kwargs per experiment
        possible experiments:
        ['accuracy', 'maps', 'cross_measure_scatters', 'dynamics', 'lesioning', 'spatial_corrs', 'localizers', 'ei_columns',
        'rfs', 'saycam', 'category_level', 'clustering', 'principal_components']
        See ppl_{experiment_name} for details per experiment
    outputs: default None, can be used to specify pre-loaded model outputs for speed
    overwrite: whether to overwrite intermediate results
    """
    if outputs is None:
        outputs = load_for_tests(base_fn, overwrite=overwrite, as_dict=True)
    else:
        print('using pre-loaded model outputs provided to function for speed. could be inconsistent with provided base_fn. use with care.')

    for exp, kwargs in kwargs_by_exp.items():
        exec(f'ppl_{exp}(base_fn, outputs, t_main=t_main, **{kwargs})')

    return


def ppl_accuracy(base_fn, outputs, t_main='exp_end', show=False, save=True):
    """
    Plotting Pipeline: accuracy plots
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()

    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, None, base_fn)

    with open(f'{SAVE_DIR}/results/{base_fn}_losses.pkl', 'rb') as f:
        losses = pickle.load(f)

    plt.figure()  
    plt.plot(losses['train']['acc'], label='train', color='r')
    plt.plot(losses['val']['acc'], label='validation', color='k')
    plt.title('Learning curves')
    plt.ylabel('Top1 Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
    if save:
        plt.savefig(f'{fig_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    plt.figure()
    for domain in d_trials.keys():
        d_accs =  [np.mean(accs) for accs in outputs['acc'][domain]]
        d_se = [sem(accs) for accs in acc[domain]]
        print(f'{domain} acc: {d_accs[t_main-1]}')
        plt.errorbar(1+times, d_accs, d_se, label=domain)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1))
    plt.title('Fully trained temporal accuracy curves')
    plt.xlabel('Processing time step')
    plt.xticks(np.arange(2,2+timing.exp_end+timing.jitter,2))
    # plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Top1 Accuracy')
    plt.savefig(f'{fig_dir}/temporal_accuracy.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()   
    plt.close() 

    return

def ppl_maps(base_fn, outputs, t_main='exp_end', show_smoothed=False, domains=None, layers=None, equal_sl_bounds=True, robust=True, save=True):
    """
    Plotting Pipeline: mean/selectivity/searchlight/readout topographic maps
    """

    dataset_name = get_experiment_dataset(outputs['opt'].trainset)
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    celltypes = ['', '_I'] if 'EI' in opt.cell else ['']
    sl_p = 0.1

    include_smootheds = [False, True] if show_smoothed else [False]
    if domains is None:
        domains = outputs['d_trials'].keys()

    for include_smoothed in include_smootheds:
        sm_tag = '_nosmooth' if not include_smoothed else ''
        for layer in layers:
            for domain, output_ids in {domain: get_domain_inds(domain, dataset_name) for domain in domains}.items():
                for celltype in celltypes:
                    mean_kws = dict(vmin=0, cmap='cmr.rainforest', robust=robust)
                    sel_kws = dict(vmin=-15, vmax=15, cmap='RdBu_r')
                    mean_domain = outputs['res'].iloc[np.logical_and.reduce((outputs['res'].t == t_main-1, outputs['res'].layer == layer))][f'mean_{domain}{celltype}'].iloc[0]
                    smoothed_mean = local_average(mean_domain.reshape(1,-1), getattr(outputs['model'], layer).rec_distances, verbose=False, p=sl_p).reshape(opt.ms, opt.ms)
                    selectivity = outputs['res'].iloc[np.logical_and.reduce((outputs['res'].t == t_main-1, outputs['res'].layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
                    smoothed_selectivity = local_average(selectivity.reshape(1,-1), getattr(model, layer).rec_distances, verbose=False, p=sl_p).reshape(opt.ms, opt.ms)
                    
                    if layer in sl_accs[celltype] and domain in sl_accs[celltype][layer]:
                        sl_domain = sl_accs[celltype][layer][domain].reshape(opt.ms,opt.ms)
                        r,p = spearmanr(mean_domain.reshape(-1), sl_domain.reshape(-1))
                        n_panes = 5 if include_smoothed else 3
                        fig, axs = plt.subplots(1,n_panes,figsize=(4*n_panes,3))
                        sl_i = 1
                    else:
                        n_panes = 4 if include_smoothed else 2
                        fig, axs = plt.subplots(1,n_panes,figsize=(4*n_panes,3))
                        sl_i = 0
                    
                    subfigsize = (3.5,3) # for single plot versions of each heatmap
                    readout_ids = get_domain_inds(domain, opt.trainset)
                    if readout_ids is not None:
                        # single-domain heatmaps of readout weights (only if this domain is a trained one)
                        if celltype == '_I' and opt.cell != 'EI5':
                            continue
                        celltype_inds = np.arange(opt.ms**2) if celltype == '' else np.arange(opt.ms**2,2*opt.ms**2)
                        domain_weights = model.decoder.weight[readout_ids][:,celltype_inds].sum(0).reshape(opt.ms,opt.ms).detach().cpu()
                        fig2, ax = plt.subplots(figsize=subfigsize)
                        if opt.ereadout or opt.erreadout:
                            g = sns.heatmap(domain_weights, vmin=0, ax=ax, robust=True, cmap='cmr.rainforest', square=True)
                        else:
                            sns.heatmap(domain_weights, center=0,  ax=ax, robust=True, cmap='RdBu_r', square=True)
                        g.set_title(f'mean {domain} readout weights')
                        g.set_xticklabels([])
                        g.set_yticklabels([])
                        if save:
                            fig2.savefig(f'{fig_dir}/aIT{celltype}_{domain}_readout.png', dpi=300, bbox_inches='tight')
                        fig2.show()
                        plt.close(fig2)

                    ii = 0
                    fig2, ax2 = plt.subplots(figsize=subfigsize)
                    for ax in [axs[0], ax2]:
                        sns.heatmap(mean_domain, ax=ax, square=True, **mean_kws)
                        ax.set_title(f'{layer} mean {domain} response{celltype}')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                    if save:
                        fig2.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_mean.png', dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                    ii+=1
                    if include_smoothed:
                        fig2, ax2 = plt.subplots(figsize=subfigsize)
                        for ax in [axs[ii], ax2]:
                            sns.heatmap(smoothed_mean, ax=ax, square=True, **mean_kws)
                            ax.set_title(f'{layer} mean {domain} response{celltype} \n smoothed')
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                        if save:
                            fig2.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_mean_smoothed.png', dpi=300, bbox_inches='tight')
                        plt.close(fig2)
                        ii+=1
                    fig2, ax2 = plt.subplots(figsize=subfigsize)
                    for ax in [axs[ii], ax2]:
                        sns.heatmap(selectivity, ax=ax, square=True, **sel_kws)
                        ax.set_title(f'{layer} {domain} selectivity{celltype}')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                    if save:
                        fig2.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_selectivity.png', dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                    ii+=1
                    if include_smoothed:
                        fig2, ax2 = plt.subplots(figsize=subfigsize)
                        for ax in [axs[ii], ax2]:
                            sns.heatmap(smoothed_selectivity, ax=ax, square=True, **sel_kws)
                            ax.set_title(f'{layer} {domain} selectivity{celltype} \n smoothed')
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                        if save:
                            fig2.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_selectivity_smoothed.png', dpi=300, bbox_inches='tight')
                        plt.close(fig2)
                        ii+=1
                    all_sl_accs = np.concatenate([sl_accs[celltype][i][j].reshape(-1) for i in sl_accs[celltype].keys() for j in sl_accs[celltype][i].keys()])
                    if equal_sl_bounds:
                        sl_kws = dict(
                            vmax = np.percentile(all_sl_accs.reshape(-1), 99),
                            vmin = np.minimum(np.percentile(all_sl_accs.reshape(-1), 99)/2, np.percentile(all_sl_accs.reshape(-1), 1)),
                        )
                    else:
                        sl_kws = dict(robust=True)
                    if layer in sl_accs[celltype] and domain in sl_accs[celltype][layer]:
                        # vmax = sl_max if sl_max is not None else np.percentile(sl_domain.reshape(-1), 95)
                        # vmin = sl_min if sl_max is not None else np.percentile(sl_domain.reshape(-1), 95)/2
                        fig2, ax2 = plt.subplots(figsize=subfigsize)
                        for ax in [axs[ii], ax2]:
                            sns.heatmap(sl_domain, ax=ax,  
                                # cmap = 'tab20c',
                                cmap='cmr.rainforest', 
                                square=True,
                                **sl_kws,
                                )                  
                            ax.set_title(f'{layer} {domain} decoding (p={sl_p})')
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                        if save:
                            fig2.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_searchlight-p-{sl_p}.png', dpi=300, bbox_inches='tight')
                        plt.close(fig2)

                    if save:
                        fig.savefig(f'{fig_dir}/{layer}_{domain}{celltype}_heatmap_summaries{sm_tag}.png', dpi=300, bbox_inches='tight')
                    plt.show()

def ppl_cross_measure_scatters(base_fn, outputs, layers=None, t_main='exp_end', save=True):
    """
    Plotting Pipeline: scatter plots comparing readout weights and mean responses
    note: this encompasses pt2 and pt7 of earlier format
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)

    domains = d_trials.keys()
    dataset_name = get_experiment_dataset(opt.trainset)
    t = t_main
    ms = opt.ms

    readout_domain_ids = {domain: get_domain_inds(domain, opt.trainset) for domain in domains}
    # filter out irrelevant domains
    readout_domain_ids = {domain: inds for domain, inds in readout_domain_ids.items() if inds is not None}
    experiment_domain_ids = {domain: get_domain_inds(domain, opt.trainset) for domain in domains}

    # Comparison of readout weights and aIT activations
    for domain, output_ids in readout_domain_ids.items():
        if output_ids is None:
            continue
        mean_domain = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layers[-1]))][f'mean_{domain}'].iloc[0]
        domain_weights = model.decoder.weight[output_ids,:].sum(0).reshape(ms,ms).detach().cpu()
        r,p = spearmanr(mean_domain.reshape(-1), domain_weights.reshape(-1))
        fig, axs = plt.subplots(1,3,figsize=(12,3))
        if opt.nl == 'sigmoid':
            sns.heatmap(mean_domain, ax=axs[0], center=0.5, cmap='RdBu_r', square=True)
        else:
            sns.heatmap(mean_domain, ax=axs[0], vmin=0, robust=True, cmap='cmr.rainforest', square=True)
        if opt.ereadout or opt.erreadout:
            sns.heatmap(domain_weights, ax=axs[1], vmin=0, robust=True, cmap='cmr.rainforest', square=True)
        else:
            sns.heatmap(domain_weights, ax=axs[1], center=0, robust=True, cmap='RdBu_r', square=True)
        axs[2].plot(mean_domain.reshape(-1), domain_weights.reshape(-1), 'o', alpha=0.1)
        axs[2].set_xlabel(f'mean {domain} {layers[-1]} response')
        axs[2].set_ylabel(f'mean {domain} {layers[-1]} readout weights')
        axs[0].set_title(f'mean {domain} {layers[-1]} response')
        axs[1].set_title(f'mean {domain} {layers[-1]} readout weights')
        axs[2].set_title(f'r{r:0.03f}, p={p:.03e}')
        for ax in axs:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if save:
            plt.savefig(f'{fig_dir}/{domain}_readout_vs_responses.png', dpi=300, bbox_inches='tight')
        plt.show()

    for domain_1, domain_2 in combinations(readout_domain_ids.keys(), 2):
        weights_1 = model.decoder.weight[readout_domain_ids[domain_1],:].sum(0).reshape(ms,ms).detach().cpu()
        weights_2 = model.decoder.weight[readout_domain_ids[domain_2],:].sum(0).reshape(ms,ms).detach().cpu()
        r,p = spearmanr(weights_1.reshape(-1), weights_2.reshape(-1))
        fig, axs = plt.subplots(1,3,figsize=(10,3))
        if opt.ereadout or opt.erreadout:
            sns.heatmap(weights_1, ax=axs[0], vmin=0, robust=True, cmap='cmr.rainforest', square=True)
            sns.heatmap(weights_2, ax=axs[1], vmin=0, robust=True, cmap='cmr.rainforest', square=True)
        else:
            sns.heatmap(weights_1, ax=axs[0], center=0, cmap='RdBu_r', square=True)
            sns.heatmap(weights_2, ax=axs[1], center=0, cmap='RdBu_r', square=True)
        axs[2].plot(weights_1.reshape(-1), weights_2.reshape(-1), 'o', alpha=0.2)
        axs[2].set_xlabel(f'mean {domain_1} {layers[-1]} readout weights')
        axs[2].set_ylabel(f'mean {domain_2} {layers[-1]} readout weights')
        axs[0].set_title(f'mean {domain_1} {layers[-1]} readout weights')
        axs[1].set_title(f'mean {domain_2} {layers[-1]} readout weights')
        axs[2].set_title(f'r{r:0.03f}, p={p:.03e}')
        for ax in axs:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.tight_layout()
        if save:
            plt.savefig(f'{fig_dir}/{domain_1}_vs_{domain_2}_readout.png', dpi=300, bbox_inches='tight')
        plt.show()

    for domain_1, domain_2 in combinations(experiment_domain_ids.keys(), 2):
        mean_1 = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layers[-1]))][f'mean_{domain_1}'].iloc[0]
        mean_2 = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layers[-1]))][f'mean_{domain_2}'].iloc[0]
        r,p = spearmanr(mean_1.reshape(-1), mean_2.reshape(-1))
        fig, axs = plt.subplots(1,3,figsize=(10,3))
        if opt.nl == 'sigmoid':
            sns.heatmap(mean_1, ax=axs[0], center=0.5, cmap='RdBu_r', square=True)
            sns.heatmap(mean_2, ax=axs[1], center=0.5, cmap='RdBu_r', square=True)
        else:
            sns.heatmap(mean_1, ax=axs[0], vmin=0, robust=True, cmap='cmr.rainforest', square=True)
            sns.heatmap(mean_2, ax=axs[1], vmin=0, robust=True, cmap='cmr.rainforest', square=True)

        axs[2].plot(mean_1.reshape(-1), mean_2.reshape(-1), 'o', alpha=0.2)
        axs[2].set_xlabel(f'mean {domain_1} {layers[-1]} response')
        axs[2].set_ylabel(f'mean {domain_2} {layers[-1]} response')
        axs[0].set_title(f'mean {domain_1} {layers[-1]} response')
        axs[1].set_title(f'mean {domain_2} {layers[-1]} response')
        axs[2].set_title(f'r{r:0.03f}, p={p:.03e}')
        for ax in axs:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.tight_layout()
        if save:
            plt.savefig(f'{fig_dir}/{domain_1}_vs_{domain_2}_mean.png', dpi=300, bbox_inches='tight')
        plt.show()    

    div_cmap = sns.diverging_palette(235, 20, s=99, l=40, center="light", as_cmap=True)

    for comp_type in ['sl', 'readout']:

        if comp_type == 'readout':
            use_layers = [model.cells[-1]]
            domain_ids = readout_domain_ids
        else:
            use_layers = arch_kwargs['cell_mapsides'].keys()
            domains = experiment_domain_ids.keys()
            domain_ids = experiment_domain_ids

        domains = list(domain_ids.keys())
        for ii, area in enumerate(use_layers):
            domain_pairs = [pair for pair in combinations(domains,2)]
            fig, axs = plt.subplots(1,len(domain_pairs), figsize=(4*len(domain_pairs),3))
            for jj, domain_pair in enumerate(domain_pairs):
                try:
                    if comp_type == 'sl':
                        x = sl_accs[''][area][domain_pair[0]].reshape(-1)
                        y = sl_accs[''][area][domain_pair[1]].reshape(-1)
                        plt_string = '{} sl decoding accuracy'
                    elif comp_type == 'readout':
                        x = model.decoder.weight[domain_ids[domain_pair[0]],:].sum(0).reshape(-1).detach().cpu()
                        y = model.decoder.weight[domain_ids[domain_pair[1]],:].sum(0).reshape(-1).detach().cpu()
                        plt_string = 'mean {} readout weight'
                    else:
                        raise ValueError()
                    c = res[np.logical_and.reduce((res.layer==area, res.t==t_main-1))][f'mean_{domain_pair[0]}'].iloc[0].reshape(-1) - res[np.logical_and.reduce((res.layer==area, res.t==t_main-1))][f'mean_{domain_pair[1]}'].iloc[0].reshape(-1)
                except Exception as e:
                    print({'domains':domain_pair, 'layer':area})
                    print(e)
                    continue
                if jj == 0:
                    # let the first comparison determine the color bar
                    cbar_lims = (-np.quantile(c, .95), np.quantile(c, .95))
        #         axs[ii,jj].scatter(x, y, alpha=0.1, s=20)
                r, p = spearmanr(x,y)
                if p<0.0001:
                    p_state = 'p<0.0001'
                else:
                    p_state = f'p={p:.04f}'

                ax_ = axs[jj]
                g = sns.regplot(x=x, y=y, ax=ax_, fit_reg=False,
                            scatter_kws=dict(c=c, color=None, 
                                            cmap='RdBu_r',
                                            linewidth=0,
                                            alpha=0.15, s=60, norm=MidpointNormalize(midpoint=0, vmin=cbar_lims[0], vmax=cbar_lims[1])),
                        )
                cbar = plt.colorbar(g.get_children()[0], ax=ax_)
                cbar.set_label(f'Selectivity ({domain_pair[0]} - {domain_pair[1]})')
                cbar.solids.set_alpha(1)
                # if jj == 1 and comp_type != 'readout':
                #     ax_.set_title(f'area = {area}') #' \n' + rf'$\rho=${r:.03f}, {p_state}')
    #             else:
    #                 ax_.set_title(rf'$\rho=${r:.03f}, {p_state}')

                ax_.set_xlabel(plt_string.format(domain_pair[0]))
                ax_.set_ylabel(plt_string.format(domain_pair[1]))
                # ax_.set_aspect('equal')
                plt.axis('square')
                ax_.set_facecolor('darkgrey')
                ax_.grid(False)
        #                 ax_.set_xlim(left=-.03*np.quantile(x, 0.999), right = np.quantile(x, 0.999))
        #                 ax_.set_ylim(bottom=np.min(y)-.1*(iqr(y, rng=(0,100))), top = np.max(y)+.1*(iqr(y, rng=(0,100))))

            plt.tight_layout()
            if save:
                plt.savefig(f'{fig_dir}/{comp_type}_comps_{area}.png', dpi=300, bbox_inches='tight')
            plt.show()

    # comparing metrics across areas
    domain_ids = experiment_domain_ids
    domains = list(domain_ids.keys())
    metrics = (
        ('sl', '{} decoding accuracy', (1+2*len(domains),3)),
        ('response', 'mean {} response', (2+2*len(domains),3))
            )

    for metric, plt_string, figsize in metrics:
        for layer1, layer2 in combinations(arch_kwargs['cell_mapsides'].keys(), 2):
            fig, axs = plt.subplots(1,len(domains), figsize=figsize, sharey=True, sharex=True)
            for jj, domain in enumerate(domains):
                if metric == 'sl':
                    x = sl_accs[''][layer1][domain].reshape(-1)
                    y = sl_accs[''][layer2][domain].reshape(-1)   
                elif metric == 'response':
                    x = res[np.logical_and.reduce((res.layer==layer1, res.t==t_main-1))][f'mean_{domain}'].iloc[0].reshape(-1)    
                    y = res[np.logical_and.reduce((res.layer==layer2, res.t==t_main-1))][f'mean_{domain}'].iloc[0].reshape(-1)                    
                    
                r, p = spearmanr(x,y)
                if p<0.0001:
                    p_state = 'p<0.0001'
                else:
                    p_state = f'p={p:.04f}'
                ax_ = axs[jj]
                sns.regplot(x=x, y=y, ax=ax_, scatter_kws=dict(alpha=0.1, color=COLORS[jj]))
                ax_.set_title(f'{domain}s \n'+ rf'$\rho=${r:.03f}, {p_state}')
                ax_.set_xlabel(plt_string.format(layer1))
                ax_.set_ylabel(plt_string.format(layer2))
            #             ax_.set_xlim(left=-.03*np.quantile(x, 0.999), right = np.quantile(x, 0.999))
            #             ax_.set_ylim(bottom=np.min(y)-.1*(iqr(y, rng=(0,100))), top = np.max(y)+.1*(iqr(y, rng=(0,100))))
                if metric == 'sl':
                    ax_.set_aspect('equal')

            plt.figure(fig.number)      
        #     plt.suptitle(r'$\sigma_{rec}$' + f' = {sigs[0]}, ' + r'$\sigma_{ff}$' + f' = {sigs[1]}, ' + f'rs= {rs}', y=1.04)
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/layer_{metric}_comps_{layer1}-vs-{layer2}.png', dpi=300, bbox_inches='tight')
            plt.show()

    metrics = {'sl': '{} decoding accuracy',
            'response': 'mean {} response', 
            'readout': 'mean {} readout weights',
            }

    for metric1, metric2 in combinations(metrics.keys(), 2):
        if 'readout' in [metric1, metric2]:
            use_layers = [model.cells[-1]]
            domain_ids = readout_domain_ids
        else:
            domain_ids = experiment_domain_ids
            use_layers = arch_kwargs['cell_mapsides'].keys()
        domains = list(domain_ids.keys())
        for layer in use_layers:
            fig, axs = plt.subplots(1,len(domains), figsize=(3*len(domains),3)) #, sharey='row', sharex='row')
            for jj, domain in enumerate(domains):
                if metric1 == 'response':
                    x = res[np.logical_and.reduce((res.layer==layer, res.t==t_main-1))][f'mean_{domain}'].iloc[0].reshape(-1)    
    #                 x = res[np.logical_and.reduce((res.layer==layer, res.t==t_main-1))][f'{domain}_selectivity_neglogp'].iloc[0].reshape(-1)    
                elif metric1 == 'sl':
                    x = sl_accs[''][layer][domain].reshape(-1)
                elif metric1 == 'readout':
                    x = model.decoder.weight[domain_ids[domain],:].sum(0).reshape(-1).detach().cpu()
                if metric2 == 'response':
                    y = res[np.logical_and.reduce((res.layer==layer, res.t==t_main-1))][f'mean_{domain}'].iloc[0].reshape(-1)    
                elif metric2 == 'sl':
                    y = sl_accs[''][layer][domain].reshape(-1)
                elif metric2 == 'readout':
                    y = model.decoder.weight[domain_ids[domain],:].sum(0).reshape(-1).detach().cpu()   
                if metric1 == 'sl' and metric2 == 'readout':
                    scatter_kws = dict(
                        alpha=0.2, color = None, cmap='cmr.rainforest',
                        c=res[np.logical_and.reduce((res.layer==layer, res.t==t_main-1))][f'mean_{domain}'].iloc[0].reshape(-1)  )
                else:
                    scatter_kws = dict(alpha=0.1, color = COLORS[jj])
                                            
                r, p = spearmanr(x,y)
                if p<0.0001:
                    p_state = 'p<0.0001'
                else:
                    p_state = f'p={p:.04f}'
                ax_ = axs[jj]
                g = sns.regplot(x=x, y=y, ax=ax_, scatter_kws=scatter_kws)
                ax_.set_title(rf'$\rho=${r:.03f}, {p_state}')
                ax_.set_xlabel(metrics[metric1].format(domain))
                ax_.set_ylabel(metrics[metric2].format(domain))
    #             ax_.set_xlim(left=-.03*np.quantile(x, 0.999), right = np.quantile(x, 0.999))
                ax_.set_aspect('auto',)
                if metric1 == 'sl' and metric2 == 'readout':
                    cbar = plt.colorbar(g.get_children()[0], ax=ax_)
                    cbar.set_label(f'Mean {domain} response')

            plt.suptitle(layer, y=1.02)
        #     plt.suptitle(r'$\sigma_{rec}$' + f' = {sigs[0]}, ' + r'$\sigma_{ff}$' + f' = {sigs[1]}, ' + f'rs= {rs}', y=1.04)
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/{metric1}_{metric2}_comps_{layer}.png', dpi=300, bbox_inches='tight')
            plt.show() 

    

def ppl_dynamics(base_fn, outputs, t_main='exp_end', layers=None, save=True):
    """
    Plotting Pipeline: Temporal unfolding of topographic measures (mean and selectivity)
    note: this could be extended to perform more interesting analyses of temporal dynamics beyond visualization
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)

    domains = d_trials.keys()

    mean_cmap_kws = dict(vmin=0, vmax=np.percentile(acts[layers[-1]].reshape(-1), 90), cmap='cmr.rainforest')
    for fmt, kws in [('{}_selectivity_neglogp{}', dict(vmin=-6, vmax=6, cmap='RdBu_r')), ('mean_{}{}', mean_cmap_kws)]:
        for domain in domains:
            for celltype in celltypes:
#                     sig_min = 0.45 if 'I' in celltype else 0.3
#                     sig_max = 0.55 if 'I' in celltype else 0.7
                fig = plot_metric_times(res, fmt.format(domain, celltype), base_fn='', layers=layers, show=True, close=False, **kws)
                if save:
                    fig.savefig(f'{fig_dir}/{fmt.format(domain, celltype)}_times.png', bbox_inches='tight')
                plt.show()

def ppl_lesioning(base_fn, outputs, t_main='exp_end', lesion_types=['topographic', 'metric'], save=True, layers=None, overwrite=False):
    """
    Plotting Pipeline: lesion analyses
    """
    if 'miniOFS' not in outputs['opt'].trainset:
        print('cannot run lesion analyses as currently implemented for a single domain-trained model')
        print('returning')
        return 
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    domains = d_trials.keys()

    domains = d_trials.keys()

    valid_lesion_types = ['topographic', 'metric', 'smooth_metric', 'metric_pval']
    for l_ in lesion_types:
        if l_ not in valid_lesion_types:
            raise ValueError(f"invalid metric specified: {l_} \n should be one of {valid_lesion_types}")

    for lesion_type in lesion_types:
        if 'pval' in lesion_type:
            lesion_ps = [None,0,1,2,3,4,5,6,10,14]
        else:
            lesion_ps = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        all_acc = run_lesion_experiment(model, res, opt, arch_kwargs, base_fn, 
                        lesion_type=lesion_type,
                        overwrite=overwrite,
                        overwrite_indiv_results=overwrite,
                        selectivity_t=t_main,
                        lesion_ps=lesion_ps,
                        )
        if lesion_type == 'topographic' or 'pval' in lesion_type:
            hue = 'lesion target domain'
        else:
            hue = 'lesion sorting domain'
        all_acc[hue] = all_acc['targ_domain']

        for lesion_area in layers:
            t=t_main
            fig, axs = plt.subplots(1,len(domains), figsize=(4*len(domains),3), sharey=False)
            all_df = pd.DataFrame(all_acc)
            all_df = all_df[all_df.t == t]

            for ii, domain in enumerate(domains):
                g = sns.lineplot(data=all_df[np.logical_and.reduce((all_df.domain==domain, all_df.lesion_E == True, all_df.lesion_I == True, all_df.layer==lesion_area))], 
                    x='lesion_p', y='accuracy', hue=hue,  style=hue, legend='brief' if ii==2 else False, ax=axs[ii], markers=['o', 'o', 'o'], err_style='bars', dashes=False, palette=COLORS)
            #     sns.lineplot(data=fake_df[fake_df.domain==domain], x='lesion_p', y='accuracy', hue='t', legend=False, ax=axs[ii]) #, palette=['k'])
            #     ymax = 0.7 if domain == 'face' else 0.35 if domain == 'object' else 0.52
                if 'pval' in lesion_type:
                    ymax = 1.1*(all_df[np.logical_and(all_df.domain==domain, np.isnan(all_df.lesion_p))].accuracy.mean())
                    axs[ii].set_xlabel(r'lesion selectivity threshold (-log($p$))')
                else:
                    ymax = 1.1*(all_df[np.logical_and(all_df.domain==domain, all_df.lesion_p==0)].accuracy.mean())
                    axs[ii].set_xlabel(r'lesion size $p$')
                axs[ii].set_ylim(-0.01, ymax)
                axs[ii].set_title(f'{domain} recognition')
                if ii == 2:
                    axs[ii].legend(loc='upper left', bbox_to_anchor=(1.05,1), 
                    )
                    # axs[ii].legend().set_title('target domain' if lesion_type == 'topographic' else 'sorting domain')
            plt.tight_layout()
            # plt.savefig(f'{FIGS_DIR}/topographic/EI/lesions/peak_domains/{base_fn}/{lesion_area}_t-{t}.png', dpi=400)
            if save:
                plt.savefig(f'{fig_dir}/{lesion_type}_lesion_{lesion_area}_peak_t-{t}.png', dpi=400, bbox_inches='tight')
            print(f'lesion type: {lesion_type}, area: {lesion_area}')
            plt.show()

            # targeted analysis
            all_df = pd.DataFrame(all_acc)
            all_df = all_df[all_df.t == t_main]
            all_df = all_df[all_df.layer == lesion_area]
            if 'pval' in lesion_type:
                all_df = all_df[np.logical_or.reduce([np.isnan(all_df.lesion_p), all_df.lesion_p== 4])]
                all_df.loc[np.isnan(all_df.lesion_p), hue] = 'none'
                print(np.unique(all_df['lesion_p']))
            else:
                all_df = all_df[np.logical_or.reduce([all_df.lesion_p == p_ for p_ in [0.0, 0.2]])]
                all_df.loc[all_df.lesion_p == 0.0, hue] = 'none'
            g = sns.barplot(data=all_df, x='domain', y='accuracy', hue = hue, palette=['grey'] + COLORS, capsize=0.05, errwidth=2)
            g.legend(loc='upper left', bbox_to_anchor=(1,1), title=hue)
            g.set_xlabel('Recognition domain')
            g.set_axisbelow(True)
            test_results = add_stat_annotation(g, data=all_df, x='domain', y='accuracy', hue=hue,
                                            box_pairs = [
                                                    (('face', "none"), ('face', "face")), (('face', "none"), ('face', "object")),  (('face', "none"), ('face', "scene")),
                                                    (('object', "none"), ('object', "face")), (('object', "none"), ('object', "object")),  (('object', "none"), ('object', "scene")),
                                                    (('scene', "none"), ('scene', "face")), (('scene', "none"), ('scene', "object")),  (('scene', "none"), ('scene', "scene")),
                                                            ] 
                                                ,
                                            test='t-test_ind', text_format='star',
                                            loc='outside', verbose=2)
            if save:
                plt.savefig(f'{fig_dir}/{lesion_type}_lesion_{lesion_area}_peak_t-{t}_subset_bars.png', dpi=400, bbox_inches='tight')

            # flip the plotting style using accuracies relative to unlesioned domain-level performance

            all_df = pd.DataFrame(all_acc)
            all_df = all_df[all_df.t == t]
            all_df = all_df[all_df.layer == lesion_area]
            # all_df = all_df[np.logical_or.reduce([all_df.lesion_p == p_ for p_ in [0.0, 0.2]])]
            all_df_hue = all_df[hue]
            all_df.loc[all_df.lesion_p == 0.0, hue] = 'none'
            means = all_df.groupby([hue, 'domain', 'lesion_p'], as_index=False).mean()

            for targ_domain in ['none', 'face', 'object', 'scene']:
                for domain in ['face','object','scene']:
                    for lesion_p in np.unique(all_df.lesion_p):
                        if (lesion_p != 0.0 and targ_domain == 'none') or (lesion_p == 0 and targ_domain != 'none'):
                            continue
                        coef = means[np.logical_and.reduce((means[hue] == 'none', means['domain'] == domain, means['lesion_p'] == 0.0))]['accuracy']
                        acc_vals = all_df.loc[np.logical_and.reduce((all_df[hue] == targ_domain, all_df['domain'] == domain, all_df['lesion_p'] == lesion_p)), 'accuracy']
                        all_df.loc[np.logical_and.reduce((all_df[hue] == targ_domain, all_df['domain'] == domain, all_df['lesion_p'] == lesion_p)), 'accuracy'] = 100*(coef.iloc[0] - acc_vals)/coef.iloc[0]

            # replace 0 lesion values since they are nice in this plot
            all_df_hue = all_df[hue]

            # rename accuracy
            new_acc_name = '% deficit in accuracy \n, relative to unlesioned'
            all_df[new_acc_name] = all_df['accuracy']

            fig, axs = plt.subplots(1,len(domains), figsize=(4*len(domains),3), sharey=True)
            for ii, domain in enumerate(domains):
                g = sns.lineplot(data=all_df[np.logical_and.reduce((all_df[hue]==domain, all_df.lesion_E == True, all_df.lesion_I == True, all_df.layer==lesion_area))], 
                    x='lesion_p', y=new_acc_name, hue='domain',  style='domain', legend='brief' if ii==2 else False, ax=axs[ii], markers=['o', 'o', 'o'], err_style='bars', dashes=False, palette=COLORS)
            #     sns.lineplot(data=fake_df[fake_df.domain==domain], x='lesion_p', y='accuracy', hue='t', legend=False, ax=axs[ii]) #, palette=['k'])
            #     ymax = 0.7 if domain == 'face' else 0.35 if domain == 'object' else 0.52
                # ymax = 1.1*(all_df[np.logical_and(all_df['targ_domain']==domain, all_df.lesion_p==0)].accuracy.mean())
                axs[ii].set_xlabel(r'lesion size $p$')
                axs[ii].set_title(f'{hue}: {domain}')
                if ii == 2:
                    axs[ii].legend(loc='upper left', bbox_to_anchor=(1.05,1), 
                    )
                    # axs[ii].legend().set_title(hue if lesion_type == 'topographic' else 'sorting domain')
            plt.tight_layout()
            # plt.savefig(f'{FIGS_DIR}/topographic/EI/lesions/peak_domains/{base_fn}/{lesion_area}_t-{t}.png', dpi=400)
            if save:
                plt.savefig(f'{fig_dir}/{lesion_type}_lesion_{lesion_area}_peak_t-{t}_flipped.png', dpi=400, bbox_inches='tight')
            print(f'lesion type: {lesion_type}, area: {lesion_area}')
            plt.show()

            # targeted analysis

            if 'pval' in lesion_type:
                subset_sizes = [0,4,14]
                baseline = None
            else:
                subset_sizes = [0.1, 0.2, 0.3]
                baseline = 0
            for lesion_subset_size in subset_sizes:
                # get rid of these
                # all_df.loc[all_df.lesion_p == 0.0, hue] = 'none'
                subset_df = all_df[np.logical_or.reduce([all_df.lesion_p == p_ for p_ in [baseline, lesion_subset_size]])]
                g = sns.barplot(data=subset_df[subset_df[hue] != 'none'], x=hue, y=new_acc_name, hue = 'domain', palette=COLORS, capsize=0.05, errwidth=2)
                g.legend(loc='upper left', bbox_to_anchor=(1,1), title='recognition domain')
                g.set_axisbelow(True)
                g.set_ylim(0, 100)
                if 'pval' in lesion_type:
                    g.set_title(rf'lesioning {lesion_area} selective units ($p<10^{-lesion_subset_size}$')
                else:
                    g.set_title(f'lesioning {100*lesion_subset_size}% of {lesion_area} units')
                if save:
                    plt.savefig(f'{fig_dir}/{lesion_type}_lesion_{lesion_area}_peak_t-{t}_subset-{lesion_subset_size}_bars_flipped.png', dpi=400, bbox_inches='tight')
                plt.show()


def ppl_spatial_corrs(base_fn, outputs, t_main='exp_end', layers=None, by_domain=True, by_time=False, plot_summary_stat=False, save=True):
    """
    Plotting Pipeline: scatter plots of pairwise response correlations vs. pairwise distance
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    domains = list(d_trials.keys())

    if by_domain:
        use_shift = True
        t = t_main

        p0 = (5,0) if use_shift else (5)

        for kk, celltype in enumerate(celltypes):
            for ii, layer in enumerate(layers):
                fig, axs = plt.subplots(1,1+len(domains),figsize=(2+3*(1+len(domains)),4), sharex=True, sharey=True)
                for jj, domain in enumerate([None] + domains):
                    fig2, ax2 = plt.subplots(1,1,figsize=(3,3))
                    if domain is not None:
                        activations = acts[layer][d_trials[domain],t-1,:].squeeze()
                    else:
                        activations = acts[layer][:,t-1,:].squeeze()
                    if celltype == 'E' or celltype == '':
                        activations = activations[:,:opt.ms**2]
                    else:
                        activations = activations[:,opt.ms**2::]
                    # get rid of nans and infinities
                    activations[np.isnan(activations)] = 0
                    activations[activations == np.inf] = np.nanmax(activations)
                    corrs = np.corrcoef(activations.transpose())
                    corrs = corrs.reshape(-1)
                    corrs[np.isnan(corrs)] = 0
                    rand_inds = np.random.choice(corrs.shape[0], size=50000, replace=False)
                    dists = getattr(model, layer).rec_distances[:opt.ms**2,:opt.ms**2].cpu().numpy().reshape(-1)[rand_inds]
                    sl = getattr(model, layer).map_side
                    corrs = corrs[rand_inds]
                    corrs = corrs[dists != 0]
                    dists = dists[dists != 0]
                    axs[jj].plot(sl*dists, corrs, 'o', alpha=0.01)
                    ax2.plot(sl*dists, corrs, 'o', alpha=0.01)
                    # fit_inds = np.abs(corrs) < 10e-6 # ignore uncorrelated units in curve fitting
                    if use_shift:
                        popt, pcov = curve_fit(dist_to_desired_corr, dists, corrs, p0=p0)
                        all_targets = dist_to_desired_corr(dists, lam=popt[0], shift=popt[-1])
                        targets = dist_to_desired_corr(np.linspace(0, np.max(dists), 100), lam=popt[0], shift=popt[-1])
                        axs[jj].plot(sl*np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x) $-$ {-popt[1]:.03f}')
                    else:
                        popt, pcov = curve_fit(dist_to_desired_corr_noshift, dists, corrs, p0=p0)
                        all_targets = dist_to_desired_corr_noshift(dists, lam=popt[0]) #, shift=popt[-1])
                        targets = dist_to_desired_corr_noshift(np.linspace(0, np.max(dists), 100), lam=popt[0]) #, shift=popt[-1])
                        axs[jj].plot(sl*np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x)')
                    # popt, pcov = curve_fit(dist_to_desired_corr_simple, dists, corrs, p0=12)
                    # all_targets = dist_to_desired_corr_simple(dists, scale=popt[0])
                    # targets = dist_to_desired_corr_simple(np.linspace(0, np.max(dists), 100), scale=popt[0])
                    # axs[jj].plot(sl*np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x)')
                    r2 = r2_score(corrs, all_targets)
                    data_r, _ = spearmanr(corrs, dists)
                    # axs[jj].legend(loc='upper right')
                    for ax in [axs[jj], ax2]:
                        ax.set_xlim(0,sl*1.01*np.max(dists))
                        ax.axhline(0.0, 0, 1, color='r', linestyle='--')
                        ax.set_ylim(-.5,1.1)
                        ax.set_xlabel('Pairwise unit distance (unit lengths)')
                        if jj == 0 or ax == ax2:
                            ax.set_ylabel('Pairwise unit response correlation')
                    im_dom = domain if domain is not None else 'all'
                    axs[jj].set_title(f'{layer} - {im_dom} images, t={t} \n data r={data_r:.04f}\n fit r2={r2:.04f}')
                    fig2.savefig(f'{fig_dir}/dist-resp-correlations_t-{t}_domain-{domain}_layer-{layer}{celltype}.png', dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                # if len(celltype)>0:
                #     plt.suptitle(f'{celltype} cells')
                plt.tight_layout()
                if save:
                    plt.savefig(f'{fig_dir}/dist-resp-correlations-by-domain_layer-{layer}{celltype}.png', dpi=300, bbox_inches='tight')
                plt.show()

    if by_time and len(times)>1:
        use_shift = True
        domain = None

        p0 = (5,0) if use_shift else (5)

        for kk, celltype in enumerate(celltypes):
            fig, axs = plt.subplots(len(layers),6,figsize=(20,3.25*len(layers)), sharex=True, sharey=True)
            if len(layers) == 1:
                axs = axs.reshape(1,-1)
            for ii, layer in enumerate(layers):
                for jj, t in enumerate([1, 2, 5, 10, 15, 20]):
                    if domain is not None:
                        activations = acts[layer][d_trials[domain],t-1,:].squeeze()
                    else:
                        activations = acts[layer][:,t-1,:].squeeze()
                    if celltype == 'E' or celltype == '':
                        activations = activations[:,:opt.ms**2]
                    else:
                        activations = activations[:,opt.ms**2::]
                    # get rid of nans and infinities
                    activations[np.isnan(activations)] = 0
                    activations[activations == np.inf] = np.nanmax(activations)
                    corrs = np.corrcoef(activations.transpose())
                    corrs = corrs.reshape(-1)
                    corrs[np.isnan(corrs)] = 0
                    rand_inds = np.random.choice(corrs.shape[0], size=100000, replace=False)
                    dists = getattr(model, layer).rec_distances[:opt.ms**2,:opt.ms**2].cpu().numpy().reshape(-1)[rand_inds]
                    corrs = corrs[rand_inds]
                    corrs = corrs[dists != 0]
                    dists = dists[dists != 0]
                    axs[ii,jj].plot(dists, corrs, 'o', alpha=0.005)
                    if use_shift:
                        popt, pcov = curve_fit(dist_to_desired_corr, dists, corrs, p0=p0)
                        all_targets = dist_to_desired_corr(dists, lam=popt[0], shift=popt[-1])
                        targets = dist_to_desired_corr(np.linspace(0, np.max(dists), 100), lam=popt[0], shift=popt[-1])
                        axs[ii,jj].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x) $-$ {-popt[1]:.03f}')
                    else:
                        popt, pcov = curve_fit(dist_to_desired_corr_noshift, dists, corrs, p0=p0)
                        all_targets = dist_to_desired_corr_noshift(dists, lam=popt[0]) #, shift=popt[-1])
                        targets = dist_to_desired_corr_noshift(np.linspace(0, np.max(dists), 100), lam=popt[0]) #, shift=popt[-1])
                        axs[ii,jj].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x)')
                    data_r, _ = spearmanr(corrs, dists)
                    r2 = r2_score(corrs, all_targets)
                    axs[ii,jj].axhline(0.0, 0, 1.01*np.max(dists), color='r', linestyle='--')
                    axs[ii,jj].set_xlim(0,1.01*np.max(dists))
        #             axs[ii,jj].set_ylim(-.5,1.1)
                    axs[ii,jj].legend(loc='upper right')
                    if ii == len(layers)-1:
                        axs[ii,jj].set_xlabel('Pairwise unit distance (au)')
                    if jj == 0:
                        axs[ii,jj].set_ylabel('Pairwise unit response correlation')
                    im_dom = domain if domain is not None else 'all'
                    axs[ii,jj].set_title(f'{layer} - t={t} \n data r={data_r:.04f}\n fit r2={r2:.04f}')
            if len(celltype) > 0:
                plt.suptitle(f'{celltype} cells')
            plt.tight_layout()
            if save:
                plt.savefig(f'{fig_dir}/dist-resp-correlations-by-time{celltype}.png', dpi=300, bbox_inches='tight')
            plt.show()

    if plot_summary_stat:
        ## spearman correlation of pairwise (dists, response corrs) as summary statistic
        times = np.arange(1,timing.exp_end+1)
        domain = None

        # fig, axs = plt.subplots(len(layers),len(celltypes),figsize=(7,6), sharex=True, sharey=True)
        if len(axs.shape) == 1:
            axs = axs.reshape(1,-1)

        for ii, layer in enumerate(layers):
            for kk, celltype in enumerate(celltypes):
                rvals = []
                for jj, t in enumerate(times):
                    if domain is not None:
                        activations = acts[layer][d_trials[domain],t-1,:].squeeze()
                    else:
                        activations = acts[layer][:,t-1,:].squeeze()
                    if celltype == 'E' or celltype == '':
                        activations = activations[:,:opt.ms**2]
                    else:
                        activations = activations[:,opt.ms**2::]
                    # get rid of nans and infinities
                    activations[np.isnan(activations)] = 0
                    activations[activations == np.inf] = np.nanmax(activations)
                    corrs = np.corrcoef(activations.transpose())
                    corrs = corrs.reshape(-1)
                    corrs[np.isnan(corrs)] = 0
                    rand_inds = np.random.choice(corrs.shape[0], size=100000, replace=False)
                    dists = getattr(model, layer).rec_distances[:opt.ms**2,:opt.ms**2].cpu().numpy().reshape(-1)[rand_inds]
                    corrs = corrs[rand_inds]
                    corrs = corrs[dists != 0]
                    dists = dists[dists != 0]
                    data_r, _ = spearmanr(corrs, dists)
                    rvals.append(data_r)
                plt.plot(times, rvals, label=f'layer {layer}, I cells' if 'I' in celltype else f'layer {layer}',
                        linestyle = '--' if kk else '-',
                        color ='rbk'[ii] if len(layers)==3 else 'rk'[ii])
        plt.xticks(times[::2])
        plt.xlabel('$t$')
        plt.ylabel(r'$r_{spearman}(r,d)$')
        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=(1,1))
        plt.ylim([0.05,-1])
        if save:
            plt.savefig(f'{fig_dir}/dist-resp-correlations-stat.png', dpi=300, bbox_inches='tight')
        plt.show()

def ppl_setup(model, opt, t_main, layers, base_fn):
    fig_dir = os.path.join(FIGS_DIR, 'topographic', 'EI', 'summaries', base_fn)
    os.makedirs(fig_dir, exist_ok=True)
    timing = exp_times[opt['timing']]
    times = np.arange(timing.exp_end + timing.jitter) # let's go up to the maximum time
    if type(t_main) == str:
        t_main = getattr(timing, t_main)
    if layers is None:
        layers = [cell for cell in model.cells if 'IT' in cell] # make sure to exclude V4
    celltypes = ['', '_I'] if 'EI' in opt.cell else ['']

    return fig_dir, timing, times, t_main, layers, celltypes

def ppl_localizers(base_fn, outputs, layers=None, localizers=['konk_floc', 'vpnl_floc'], 
                    plot_ofs_contours=True,
                    contour_smoothing=0.05,
                    contour_thresh_neglogp=3,
                    t_main='exp_end',
                    save=True,
                    ):
    """
    Plotting Pipeline: topography to different localizer sets
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)

    valid_localizers = ['konk_floc', 'vpnl_floc', 'vpnl+konk_floc', 'konk_animsize', 'konk_animsize_texform']
    for localizer in localizers:
        if localizer not in valid_localizers:
            raise ValueError()

    c_tag = '_contours' if plot_ofs_contours else ''

    for floc in localizers: # ['konk']: #['vpnl+konk', 'konk', 'vpnl']: #'konk', 'vpnl', 
        res, acc, acts, d_trials = do_mini_objfacescenes_exp(outputs['model'], layers=layers,
                                                                input_times=exp_times[outputs['opt']['timing']].stim_off-1, 
                                                                times=exp_times[outputs['opt']['timing']].exp_end-1, 
                                                                dataset_name=floc, 
                                                                cache_base_fn=base_fn,
        #                                                         ims_per_class= ims_per_class, #1 if opt.trainset == 'objfacescenes' else 5, 
        #                                                         equate_domains=True, 
                                                                return_activations=True, 
                                                                accuracy_vector=True, 
                                                                overwrite=False,
                                                                run_searchlights=False,
                                                                progress_bar=False,
                                                                compute_selectivity=True,
                                                                batch_size=32,
                                                                EI='EI' in outputs['opt'].cell,
                                                                fivecrop=outputs['opt'].fivecrop,
                                                                logpolar=outputs['opt'].logpolar,
                                                            )


        for layer in layers:
            t = t_main - 1

        #     if 'scrambled' in d_trials:
        #         del(d_trials['scrambled'])
            for map_ in ['mean', 'selectivity']:
                fig, axs = plt.subplots(1, len(d_trials.keys()), figsize=(3.5*len(d_trials),3))
                for ii, domain in enumerate(sorted(d_trials.keys())):
                    if map_ == 'selectivity':
                        sns.heatmap(res[res.layer == layer][f'{domain}_selectivity_neglogp'].iloc[0], cmap='RdBu_r', center=0, ax=axs[ii], square=True)
                    else:
                        sns.heatmap(res[res.layer == layer][f'mean_{domain}'].iloc[0], cmap='cmr.rainforest', vmin=0, ax=axs[ii], square=True)
                    if plot_ofs_contours:
                        for cont_domain in ['object', 'face', 'scene']:
                            cont_sel = outputs['res'][np.logical_and(outputs['res'].layer == layer, outputs['res'].t == t)][f'{cont_domain}_selectivity_neglogp'].iloc[0]
                            cont_sel = local_average(cont_sel.reshape(1,-1), getattr(outputs['model'], layer).rec_distances, p=contour_smoothing, verbose=False).reshape(32,32)
                            axs[ii].contour(np.arange(32), np.arange(32), cont_sel , [contour_thresh_neglogp], cmap='copper_r' if map_ == 'mean' else 'bone', linestyles='dashed', linewidths=2); 
                        axs[ii].set_title(domain)
                        axs[ii].set_xticks([])
                        axs[ii].set_yticks([])
                plt.tight_layout()
                if save:
                    plt.savefig(f'{fig_dir}/{floc}_{layer}_{map_}{c_tag}.png', dpi=200, bbox_inches='tight')
                plt.show()

def ppl_ei_columns(base_fn, outputs, t_main='exp_end', layers=None, save=True,
                    robust=True,
                    smoothing_p=0.05,
                    neglogp_thresh=3,
                    ):
    """
    Plotting Pipeline: assess columnar functional nature of E/I columns of units (only in E/I ITNs)
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    domains = d_trials.keys()
    dataset_name = get_experiment_dataset(opt.trainset)

    if 'EI' not in opt.cell:
        print("EI columns can't be plotted in a non E/I model")
        print('returning')
        return
    
    mean_kws = dict(vmin=0, cmap='cmr.rainforest', robust=robust)
    sel_kws = dict(vmin=-6, vmax=6, cmap='RdBu_r')
    for metric, fmt, kws in [('selectivity', '{} {} cell {} selectivity', sel_kws), ('mean', '{} mean {} cell {} response', mean_kws)]:
        for layer in layers:
            if metric == 'selectivity':
                rgb_selectivity_plot(res, t_main, getattr(model, layer).rec_distances, [layer], 
                    ms=opt.ms, 
                    smoothing_p=smoothing_p, 
                    neglogp_thresh=neglogp_thresh, 
                    kind='binary', 
                    fn=f'{fig_dir}/{layer}_EI_columns_RGB_selectivity.png',
                )
            for domain, output_ids in {domain: get_domain_inds(domain, dataset_name) for domain in domains}.items():
                fig, axs = plt.subplots(1,len(domains), figsize=(4*len(domains),3.5))
                metrics = {}
                if metric == 'mean':
                    metrics['E'] = res.iloc[np.logical_and.reduce((res.t == t_main-1, res.layer == layer))][f'mean_{domain}'].iloc[0]
                    metrics['I'] = res.iloc[np.logical_and.reduce((res.t == t_main-1, res.layer == layer))][f'mean_{domain}_I'].iloc[0]
                elif metric == 'selectivity':
                    metrics['E'] = res.iloc[np.logical_and.reduce((res.t == t_main-1, res.layer == layer))][f'{domain}_selectivity_neglogp'].iloc[0]
                    metrics['I'] = res.iloc[np.logical_and.reduce((res.t == t_main-1, res.layer == layer))][f'{domain}_selectivity_neglogp_I'].iloc[0]
                else:
                    raise NotImplementedError()

                for ii, celltype in enumerate(['E', 'I']):
                    sns.heatmap(metrics[celltype], ax=axs[ii], square=True, **kws)
                    axs[ii].set_title(fmt.format(layer, celltype, domain))
                    axs[ii].set_xticklabels([])
                    axs[ii].set_yticklabels([])
                
                activations = acts[layer][d_trials[domain],t_main-1,:].squeeze()
                E_activations = activations[:,:opt.ms**2]
                I_activations = activations[:,opt.ms**2::]
                bad_inds = np.unique(np.concatenate((np.argwhere(E_activations.sum(0)==0),np.argwhere(I_activations.sum(0)==0 ))))
                corrs = np.diagonal(spearmanr(E_activations, b=I_activations, axis=0)[0][:opt.ms**2][:,opt.ms**2::])
                axs[2].hist(corrs, bins=20)
                axs[2].set_xlim(-1,1)
                plt.axvline(x=0, ymin=0, ymax=1, c='r', linestyle='--')
                axs[2].set_title(f'{layer} correlations to {domain} images \nwithin E/I columns')

                # axs[2].plot(metrics['E'].reshape(-1), metrics['I'].reshape(-1), 'o', alpha=0.2)
                # axs[2].set_xlabel('E unit response')
                # axs[2].set_ylabel('I unit response')
                plt.tight_layout()
                if save:
                    plt.savefig(f'{fig_dir}/{layer}_{domain}_EI_columns_{metric}.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            # now plot E/I correlations over all images
            fig, ax = plt.subplots()
            activations = acts[layer][:,t_main-1,:].squeeze()
            E_activations = activations[:,:opt.ms**2]
            I_activations = activations[:,opt.ms**2::]
            corrs = np.diagonal(spearmanr(E_activations, b=I_activations, axis=0)[0][:opt.ms**2][:,opt.ms**2::])
            plt.hist(corrs, bins=20)
            ax.set_xlim(-1,1)
            plt.axvline(x=0, ymin=0, ymax=1, c='r', linestyle='--')
            ax.set_title(f'{layer} correlations to all images \nwithin E/I columns')
            if save:
                plt.savefig(f'{fig_dir}/{layer}_EI_columns_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()

    
def ppl_rfs(base_fn, outputs, t_main='exp_end', layers=None, save=True):
    """
    Plotting Pipeline: visualize receptive fields
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    mod_layers = [cell for cell in model.cells if hasattr(getattr(model, cell),'map_side')]
    for layer in layers:
        mod_layer = getattr(model, layer)
        # determine input and current map sides
        curr_ms = mod_layer.map_side
        if layer == mod_layers[0]:
            in_ms = opt.imdim
            # in_ms = model.output_shapes[list(model.output_shapes.keys())[-1]]
        else:
            in_ms = getattr(model, mod_layers[mod_layers.index(layer)-1]).map_side            
        rec_celltypes = ['E', 'I', 'diff'] if 'EI' in opt.cell else ['E']
        ff_celltypes = ['E', 'I', 'diff'] if opt.cell=='EI5' else ['E'] 
        try:
            ih_weights, hh_weights = mod_layer.cell.get_total_weights(ih_mask=mod_layer.ff_connection_mask, hh_mask=mod_layer.rec_connection_mask)
            conn_weights_cells = [('ff', ih_weights, ff_celltypes), ('rec', hh_weights, rec_celltypes)]
        except:
            ih_weights = mod_layer.cell.get_total_weights(ih_mask=mod_layer.ff_connection_mask)
            conn_weights_cells = [('ff', ih_weights, ff_celltypes)]
        for conns, weights, celltypes in conn_weights_cells:
            if layer == mod_layers[0] and conns == 'ff' and opt.nonretinotopic:
                continue
            if conns == 'rec' and opt['norec']:
                continue
            if conns == 'rec':
                ms = curr_ms
            elif conns == 'ff':
                ms = in_ms
                if layer == mod_layers[0]:
                    n_channels = weights.shape[1]//(in_ms**2)
                    weights = weights.reshape(curr_ms**2, n_channels, in_ms**2).mean(1) # take mean over input channel dimension for simplicity
            for celltype in celltypes:
                weights = weights.detach().cpu()
                re_weights = []
                all_weights = []
                skip_by = 2
                ind = -1
                indices = []
                for ind_i in range(int(curr_ms/skip_by)):
                    row_weights = []
                    for ind_j in range(int(curr_ms/skip_by)):
                        ind += 1
                        index = int(curr_ms*skip_by)*(ind_i) + skip_by*ind_j  #np.arange(0,opt.ms**2, skip_by)[ind]
                        indices.append(index)
                        if celltype == 'E':
                            row_weights.append(weights[index,:ms**2].reshape(ms,ms))
                        elif celltype == 'I':
                            row_weights.append(weights[index,ms**2::].reshape(ms,ms))
                        else:
                            row_weights.append((weights[index,:ms**2] + weights[index,ms**2::]).reshape(ms,ms))
                        all_weights.append(row_weights[-1])
                    row_weights = np.concatenate(row_weights, axis=1)
                    re_weights.append(row_weights)
                re_weights = np.concatenate(re_weights, axis=0)
                fig, ax = plt.subplots(figsize=(20,17))
                vmin, vmax = np.nanpercentile(re_weights, 1), np.nanpercentile(re_weights, 99)
                sns.heatmap(re_weights, ax=ax, square=True, vmin=vmin, vmax=vmax, cmap='RdBu_r', center=0)
                ax.set_xticks(np.arange(0,curr_ms*ms/skip_by, ms))
                ax.set_yticks(np.arange(0,curr_ms*ms/skip_by, ms))
                ax.grid('white')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                cell_tag = 'EI-diff' if celltype == 'diff' else f'{celltype}-weights'
                print(f'{layer}-{conns}_{cell_tag}')
                if save:
                    fig.savefig(f'{fig_dir}/{layer}-{conns}_{cell_tag}_skipby-{skip_by}.png', dpi=300, bbox_inches='tight')
                plt.show()

                # now plot mean profile
                # fig, ax = plt.subplots()
                # mean_weights = np.stack(all_weights).mean(0)
                # sns.heatmap(mean_weights, square=True, robust=True, cmap='RdBu_r', center=0)
                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # cell_tag = 'EI-diff' if celltype == 'diff' else f'{celltype}-weights'
                # fig.savefig(f'{fig_dir}/{layer}-{conns}_{cell_tag}_mean.png', dpi=300, bbox_inches='tight')
                # plt.show()

def ppl_saycam(base_fn, outputs, t_main='exp_end', overwrite=False, layers=None, save=True):
    """
    Plotting Pipeline: analyze responses to SAYCAM. currently it just looks at distance-dependent response correlations
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    saycam_acts, info = load_batched_activations(net_name=base_fn,
                                                layers=layers,
                                                imageset='SAYCAM-TC-ego-S-300c_10-ims-per-class',
                                                timepoints = np.arange(timing.exp_end+timing.jitter),
                                                phase='val',
                                                model=model,
                                                input_times=timing.stim_off,
                                                total_times=timing.exp_end+timing.jitter,
                                                progress_bar=False,
                                                overwrite=overwrite,
                                        )


    t = t_main
    for kk, celltype in enumerate(celltypes):
        for ii, layer in enumerate(layers):
            fig2, ax2 = plt.subplots(1,1,figsize=(3,3))
            activations = saycam_acts[layer][:,t-1,:].squeeze()
            if celltype == 'E' or celltype == '':
                activations = activations[:,:opt.ms**2]
            else:
                activations = activations[:,opt.ms**2::]
            # get rid of nans and infinities
            activations[np.isnan(activations)] = 0
            activations[activations == np.inf] = np.nanmax(activations)
            corrs = np.corrcoef(activations.transpose())
            corrs = corrs.reshape(-1)
            corrs[np.isnan(corrs)] = 0
            rand_inds = np.random.choice(corrs.shape[0], size=50000, replace=False)
            dists = getattr(model, layer).rec_distances[:opt.ms**2,:opt.ms**2].cpu().numpy().reshape(-1)[rand_inds]
            corrs = corrs[rand_inds]
            corrs = corrs[dists != 0]
            dists = dists[dists != 0]
            ax2.plot(dists, corrs, 'o', alpha=0.01)
            data_r, _ = spearmanr(corrs, dists)
            ax2.set_xlim(0,1.01*np.max(dists))
            ax2.axhline(0.0, 0, 1, color='r', linestyle='--')
#             ax2.set_ylim(-.5,1.1)
            ax2.set_xlabel('Pairwise unit distance (au)')
            ax2.set_ylabel('Pairwise unit response correlation')
            if save:
                fig2.savefig(f'{fig_dir}/dist-resp-correlations_t-{t}_saycam_layer-{layer}{celltype}.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)

    # by time
    use_shift = True
    p0 = (5,0) if use_shift else (5)

    for kk, celltype in enumerate(celltypes):
        fig, axs = plt.subplots(len(layers),6,figsize=(20,3.25*len(layers)), sharex=True, sharey=True)
        if len(layers) == 1:
            axs = axs.reshape(1,-1)
        for ii, layer in enumerate(layers):
            for jj, t in enumerate([1, 2, 5, 10, 15, 20]):
                activations = saycam_acts[layer][:,t-1,:].squeeze()
                if celltype == 'E' or celltype == '':
                    activations = activations[:,:opt.ms**2]
                else:
                    activations = activations[:,opt.ms**2::]
                # get rid of nans and infinities
                activations[np.isnan(activations)] = 0
                activations[activations == np.inf] = np.nanmax(activations)
                corrs = np.corrcoef(activations.transpose())
                corrs = corrs.reshape(-1)
                corrs[np.isnan(corrs)] = 0
                rand_inds = np.random.choice(corrs.shape[0], size=100000, replace=False)
                dists = getattr(model, layer).rec_distances[:opt.ms**2,:opt.ms**2].cpu().numpy().reshape(-1)[rand_inds]
                corrs = corrs[rand_inds]
                corrs = corrs[dists != 0]
                dists = dists[dists != 0]
                axs[ii,jj].plot(dists, corrs, 'o', alpha=0.005)
                if use_shift:
                    popt, pcov = curve_fit(dist_to_desired_corr, dists, corrs, p0=p0)
                    all_targets = dist_to_desired_corr(dists, lam=popt[0], shift=popt[-1])
                    targets = dist_to_desired_corr(np.linspace(0, np.max(dists), 100), lam=popt[0], shift=popt[-1])
                    axs[ii,jj].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x) $-$ {-popt[1]:.03f}')
                else:
                    popt, pcov = curve_fit(dist_to_desired_corr_noshift, dists, corrs, p0=p0)
                    all_targets = dist_to_desired_corr_noshift(dists, lam=popt[0]) #, shift=popt[-1])
                    targets = dist_to_desired_corr_noshift(np.linspace(0, np.max(dists), 100), lam=popt[0]) #, shift=popt[-1])
                    axs[ii,jj].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x)')
                data_r, _ = spearmanr(corrs, dists)
                r2 = r2_score(corrs, all_targets)
                axs[ii,jj].axhline(0.0, 0, 1.01*np.max(dists), color='r', linestyle='--')
                axs[ii,jj].set_xlim(0,1.01*np.max(dists))
    #             axs[ii,jj].set_ylim(-.5,1.1)
                axs[ii,jj].legend(loc='upper right')
                if ii == len(layers)-1:
                    axs[ii,jj].set_xlabel('Pairwise unit distance (au)')
                if jj == 0:
                    axs[ii,jj].set_ylabel('Pairwise unit response correlation')
                axs[ii,jj].set_title(f'{layer} - t={t} \n data r={data_r:.04f}\n fit r2={r2:.04f}')
        if len(celltype) > 0:
            plt.suptitle(f'{celltype} cells')
        plt.tight_layout()
        if save:
            plt.savefig(f'{fig_dir}/dist-resp-correlations-by-time{celltype}_saycam.png', dpi=300, bbox_inches='tight')
        plt.show()


def ppl_category_level(base_fn, outputs, t_main='exp_end', layer='aIT', plot_contours=True, save=True):
    """
    Plotting Pipeline: Analysis of category-level responses in the penultimate layer: readout weights and category means
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, None, base_fn)
    dataset_name = get_experiment_dataset(opt.trainset)
    domains = d_trials.keys()

    with open(f'/lab_data/plautlab/topographic/data/activations/{base_fn}/miniOFS_10-ims-per-class_noise-True.pkl', 'rb') as f:
        acts_dict, category_idx, outputs = pickle.load(f)
    decoder_weights = model.decoder.weight.detach().cpu().numpy()
    np.random.seed(1)

    readout_domain_inds = {domain: get_domain_inds(domain, opt.trainset) for domain in domains}
    exp_domain_inds = {domain: get_domain_inds(domain, dataset_name) for domain in domains}

    selective_maps = {}
    for domain, inds in readout_domain_inds.items():
        selective_map = res[np.logical_and(res.layer == layer, res.t==t_main-1)][f'{domain}_selectivity_neglogp'].iloc[0]
        selective_maps[domain] = local_average(selective_map.reshape(1,-1), getattr(model, layer).rec_distances, p=0.05).reshape(32,32)

    for domain, inds in readout_domain_inds.items():
        if inds is None:
            continue
        fig = plt.figure(figsize=(20, 4))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2,10), 
                        axes_pad=0.05,  # pad between axes in inch.
                        )
        inds = sorted(np.random.choice(inds, 10, replace=False))
        layer_acts = np.concatenate(acts_dict[layer])
        idx=-1
        for ii, ind in enumerate(inds):
            idx+=1
            ax = grid[idx]
        #     weights = model.decoder.weight[ind,:].reshape(32,32).detach().cpu().numpy()
        #     weights = scipy.stats.zscore(weights.reshape(-1)).reshape(32,32)
            categ_mean = layer_acts[category_idx == ind][:,timing.exp_end-1,:opt.ms**2].mean(0)
            sns.heatmap(categ_mean.reshape(32,32), ax=ax, cbar=False, cmap=cmr.rainforest, alpha=0.9, vmin=0)
            # for cont_domain in ['face', 'object', 'scene']:
            #     if cont_domain == domain:
            #         cont_cmap = 'cyan_copper_r'
            #     else:
            #         cont_cmap = 'cyan_copper'
            #     ax.contour(np.arange(32), np.arange(32), selective_maps[cont_domain] , [3], cmap=cont_cmap, linestyles='dashed', linewidths=1.5); 
            if plot_contours:
                ax.contour(np.arange(32), np.arange(32), selective_maps[domain] , [3], cmap='cyan_copper_r', linestyles='dashed', linewidths=1.5); 
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)

        for ii, ind in enumerate(inds):
            idx+=1
            ax = grid[idx]
            weights = decoder_weights[ind,:].reshape(32,32)
    #         weights = scipy.stats.zscore(weights).reshape(32,32)
            sns.heatmap(weights, ax=ax, cbar=False, cmap=cmr.rainforest, alpha=0.9, vmin=0)
            if plot_contours:
                ax.contour(np.arange(32), np.arange(32), selective_maps[domain] , [3], cmap='cyan_copper_r', linestyles='dashed', linewidths=1.5); 
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)
    #     for jj in range(len(grid)):
    #         fig.delaxes(grid[jj])
        fig.suptitle(f'{domain} category mean activations (top) and readout weights (bottom)', y=1)
        if save:
            fig.savefig(f'{fig_dir}/{domain}_category_mean_activations_and_weights.png', dpi=200, bbox_inches='tight')
    #     plt.tight_layout()
        plt.show()


def ppl_clustering(base_fn, outputs, t_main='exp_End', k_clusters=3, domains=['object','face','scene'], save=True):
    """
    Plotting Pipeline: clustering of readout weights (possibly to be replaced with activations or category-mean activations)
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()

    # fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    plot_OFS_clustering(model, opt, base_fn, k_clusters=k_clusters, domains=domains, save=save)


def ppl_principal_components(base_fn, outputs, layers=None, t_main='exp_end', ignore_untrained_domains=False, 
                                plot_more_obj_attributes=False, 
                                proj_faces_into_obj = False,
                                save=True, 
                                plot_contours=True,
                                contour_smoothing=0.05,
                                contour_thresh_neglogp=3,
                                ):
    """
    Plotting Pipeline: analysis of principal components of variation in responses
    """
    _, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = outputs.values()
    fig_dir, timing, times, t_main, layers, celltypes = ppl_setup(model, opt, t_main, layers, base_fn)
    readout_weights = model.decoder.weight.detach().cpu().numpy()

    orig_classes = {
    'object': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100objects/train/*')]),
    'face': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100faces-crop-2.0/train/*')]),
    'scene': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100scenes/train/*')]),
    }
    # domain_inds = {domain: get_domain_inds(domain, 'miniOFS') for domain in domains}

    obj_attributes = pd.read_csv('/home/nblauch/100objects_attributes.tsv', sep='\t')
    obj_attributes['inanimate'] = [1-animate for animate in obj_attributes['animate']]
    scene_attributes = pd.read_csv('/home/nblauch/100scene_attributes.tsv', sep='\t')
    face_attributes = pd.read_csv('/home/nblauch/100face_attributes.tsv', sep='\t')
    face_attributes['female'] = [1-male for male in face_attributes['male']]

    for layer in layers:

        domain_inds = d_trials
        domain_inds['all'] = np.arange(len(acts[layer]))

        domain_categ_inds = {'all': np.arange(300), 'object': np.arange(100), 'face': np.arange(100,200), 'scene': np.arange(200,300)}

        if len(acts[layer]) == 3000:
            # 10 ims per class
            labels = []
            for ind in range(300):
                labels += list(ind*np.ones(10, dtype=int))
        else:
            raise NotImplementedError()

        for domain, inds in domain_inds.items():
            if 'miniOFS' not in base_fn and ignore_untrained_domains and (domain not in opt.datasets[0]):
                continue
            save_dir = f'{fig_dir}/{domain}_layer-{layer}_acts_PCs'
            os.makedirs(save_dir, exist_ok=True)

            these_acts = acts[layer][inds,t_main-1,:opt.ms**2]
            pca_sol = PCA(n_components=None).fit(these_acts)
            loadings = pca_sol.transform(these_acts)
            components = pca_sol.components_
            expl_variance = pca_sol.explained_variance_ratio_

            these_images = []
            for ind in domain_categ_inds[domain]:
                for ii in range(10):
                    file = get_miniOFS_img_from_categ(ind, im_idx=ii)
                    these_images.append(file)

            if plot_contours:
                contour_domains = list(d_trials.keys()) if domain == 'all' else [domain]
                contour_maps = []
                for contour_domain in contour_domains:
                    contour_map = outputs['res'][np.logical_and(outputs['res'].layer == layer, outputs['res'].t == t_main-1)][f'{contour_domain}_selectivity_neglogp'].iloc[0]
                    contour_map = local_average(contour_map.reshape(1,-1), getattr(outputs['model'], layer).rec_distances, p=contour_smoothing, verbose=False).reshape(32,32)
                    contour_maps.append(contour_map)
            else:
                contour_maps = None

            plot_component_solution(components, loadings, expl_variance=expl_variance, n_ims=None, n_components=2, num_ims=200, images=these_images, save_dir=save_dir, 
                                    contour_maps = contour_maps, contour_levels=[contour_thresh_neglogp], scatter_pairs=[])

            markers = None
            if domain == 'object':
                colors = []
                if plot_more_obj_attributes:
                    palette = sns.color_palette('colorblind', 10)
                    cmap = {'dog': palette[0], 'non-dog mammal':palette[1], 'non-mammal animal':palette[2], 'inanimate object':palette[4]}
                    for im_i in inds:
                        ind = labels[im_i]
                        if obj_attributes['dog'][ind]:
                            colors.append(cmap['dog'])
                        elif obj_attributes['mammal'][ind]:
                            colors.append(cmap['non-dog mammal'])
                        elif obj_attributes['animate'][ind]:
                            colors.append(cmap['non-mammal animal'])
                        else:
                            colors.append(cmap['inanimate object'])
                    all_colors = colors
                else:
                    palette = sns.diverging_palette(90, 270, s=99, l=60, as_cmap=True)
                    palette = get_cmap('PuOr')
            #         palette = sns.diverging_palette(220, 20, as_cmap=True)
                    colors = [palette(0.15) if anim else palette(0.85) for anim in obj_attributes['animate']]
                    cmap = {'animal':palette(0.15), 'inanimate object':palette(0.85)}
                    these_labels = [labels[ind] for ind in inds]
                    all_colors = [colors[labels[ind]] for ind in inds]
                if proj_faces_into_obj:
                    cmap['face'] = 'r'
                    cmap['scene'] = 'k'
                    all_colors = ['r' for ii in range(len(domain_inds['face']))] + ['k' for ii in range(len(domain_inds['scene']))] + all_colors
                    these_acts = np.concatenate((acts[layer][domain_inds['face'],t_main-1,:opt.ms**2], acts[layer][domain_inds['scene'],t_main-1,:opt.ms**2], these_acts))
                    loadings = pca_sol.transform(these_acts)
            elif domain == 'face':
                palette = sns.diverging_palette(15, 195, s=99, l=50, as_cmap=True)
                palette = get_cmap('BrBG')
        #         palette = sns.diverging_palette(220, 20, as_cmap=True)
                colors = [palette(0.15) if male else palette(0.85) for male in face_attributes['male']]
                cmap = {'male face':palette(0.15), 'female face':palette(0.85)}
                these_labels = [labels[ind]-100 for ind in inds]
                all_colors = [colors[labels[ind]-100] for ind in inds]
            elif domain == 'scene':
                palette = sns.diverging_palette(30, 120, s=99,  l=75,as_cmap=True)
                palette = get_cmap('PRGn')
        #         palette = sns.diverging_palette(220, 20, as_cmap=True)
                colors = [palette(0.15) if indoor else palette(0.85) for indoor in scene_attributes['indoor']]
        #         colors = ['cyan' if indoor else 'brown' for indoor in scene_attributes['indoor']]
                cmap = {'indoor scene':palette(0.15), 'outdoor scene': palette(0.85)}
                these_labels = [labels[ind]-200 for ind in inds]
                all_colors = [colors[labels[ind]-200] for ind in inds]
            elif domain == 'all':
                palette = sns.color_palette("Paired")
                colors = [palette[0] if anim else palette[1] for anim in obj_attributes['animate']]
                colors += [palette[2] if male else palette[3] for male in face_attributes['male']]
                colors += [palette[4] if indoor else palette[5] for indoor in scene_attributes['indoor']]
                cmap = {'animal':palette[0], 'inanimate object':palette[1], 'male face':palette[2], 'female face':palette[3], 'indoor scene':palette[4], 'outdoor scene': palette[5]}
                all_colors = [colors[labels[ind]] for ind in inds]

            if domain == 'all' or (domain == 'object' and plot_more_obj_attributes):
                pc_pairs = [(1,2), (1,3), (2,3), (3,4)]
            else:
                pc_pairs = [(1,2)]
            for x_pc, y_pc in pc_pairs:
                # find separating hyperplane for 2-way comparisons
                if domain != 'all' and x_pc==1 and y_pc==2 and not (domain == 'object' and (plot_more_obj_attributes or proj_faces_into_obj)):
                    clf_labels = [0 if cc == palette(0.15) else 1 for cc in all_colors]
                    clf = RidgeClassifierCV(fit_intercept=False).fit(loadings[:,:2], clf_labels)
                    discriminability = clf.score(loadings[:,:2], clf_labels)
                    rotated_pc1 = np.dot(clf.coef_, components[:2,:])
                    figure = plt.figure()
                    g = sns.heatmap(rotated_pc1.reshape(32,32), center=0, 
                                cmap=palette, 
                                robust=True, square=True, cbar=False)
                    if contour_maps is not None and len(contour_maps) > 0:
                        styles = ['dotted', 'dashed', 'dashdot']
                        for ii, contour_map in enumerate(contour_maps):
                            g.contour(np.arange(32), np.arange(32), contour_map , [contour_thresh_neglogp], cmap='bone', linestyles=styles[ii], linewidths=2)

                    plt.xticks([])
                    plt.yticks([])
                    plt.title(f'Rotated PC-1, discriminability={discriminability:.02f}')
                    plt.savefig(f'{save_dir}/rotated_pc1.png', dpi=200, bbox_inches='tight')
                    plt.show()
                else:
                    clf = None
                plot_pca_scatter(loadings, None, 
                            num_ims=None, s=25 if proj_faces_into_obj and domain=='object' else 50,
                            x_pc=x_pc, y_pc=y_pc,
                            save_dir=save_dir, figsize=10 if colors is None else 6, 
                            imsize=20, colors=all_colors, cmap=cmap,
                            legend_title=None,
                                alpha=0.5,
                                edgecolors= 'k' if domain == 'all' else None,
                            clf_for_hyperplane=clf,
                                )

                plt.show()


def rgb_binary_selectivity_plot(res_df, t, distances, layer, ms=32, smoothing_p=0.05, neglogp_thresh=3, fn=None, legend=True, celltype=''):
    """
    2D spatial selectivity plot for visualizing 3-way binary selectivity (objects, faces, scenes)
    """
    colored_map, cmap = get_binary_selectivity(res_df, t, distances, layer, ms=ms, smoothing_p=smoothing_p, neglogp_thresh=neglogp_thresh, celltype=celltype)

    fig, ax = plt.subplots()
    plt.imshow(colored_map)
    plt.grid(False)
    patches = [ mpatches.Patch(color=color, label=domain) for domain, color in cmap.items()]
    # put those patched as legend-handles into the legend
    if legend:
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0., title='selectivity')
    # plt.title(f'Unit selectivity {dset_name}')
    plt.xticks([])
    plt.yticks([])
    if fn is not None:
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    return fig, ax


def get_binary_selectivity(res_df, t, distances, layer, ms=32, smoothing_p=0.05, neglogp_thresh=3, celltype=''):
    colored_map = np.zeros((ms,ms,3))
    masks = {}
    cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1], 'face & object': [1,1,0], 'object & scene':[0,1,1], 'face & scene':[1,0,1]}
    for domain in ['face', 'object', 'scene']:
        selective_map = res_df[np.logical_and.reduce((res_df.layer == layer, res_df.t==t-1))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
        if smoothing_p:
            selective_map = local_average(selective_map.reshape(1,-1), distances, p=smoothing_p, verbose=False).reshape(ms,ms)
        masks[domain] =  selective_map > neglogp_thresh
        colored_map[masks[domain],:] += cmap[domain]
    #     masks[domain] = masks[domain].reshape(-1)
    return colored_map, cmap


def rgb_selectivity_plot(res_df, t, distances, layers, celltypes=[''], 
                                    ms=32, 
                                    smoothing_p=0.05, 
                                    neglogp_thresh=3,
                                    fn=None, 
                                    legend=True,
                                    metric='selectivity',
                                    kind='continuous',
                                    transpose=False,
                                    ):
    """
    2D spatial selectivity plot for visualizing 3-way binary selectivity (objects, faces, scenes)
    """
    if transpose:
        fig, axs = plt.subplots(len(layers), len(celltypes), figsize=(3*len(celltypes), 3*len(layers)))
        layers = list(reversed(layers))
    else:
        fig, axs = plt.subplots(len(celltypes), len(layers), figsize=(3*len(layers), 3*len(celltypes)))
    if not(hasattr(axs, 'shape')):
        axs = np.array([axs])
    if len(axs.shape) == 1:
        axs = axs.reshape(len(celltypes), len(layers))
    for jj, layer in enumerate(layers):
        for ii, celltype in enumerate(celltypes):
            colored_map = np.zeros((ms,ms,3))
            if kind == 'continuous':
                cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1]}
                for ind, domain in enumerate(['face', 'object', 'scene']):
                    map_ = f'{domain}_selectivity_neglogp{celltype}' if metric == 'selectivity' else f'mean_{domain}{celltype}'
                    selective_map = res_df[np.logical_and.reduce((res_df.layer == layer, res_df.t==t-1))][map_].iloc[0]
                    # selective_map = res_df[np.logical_and.reduce((res_df.layer == layer, res_df.t==t-1))][f'mean_{domain}{celltype}'].iloc[0]
                    if smoothing_p:
                        selective_map = local_average(selective_map.reshape(1,-1), distances, p=smoothing_p, verbose=False).reshape(ms,ms)
                    colored_map[:,:,ind] = selective_map
            elif kind == 'binary':
                masks = {}
                cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1], 'face & object': [1,1,0], 'object & scene':[0,1,1], 'face & scene':[1,0,1]}
                for domain in ['face', 'object', 'scene']:
                    selective_map = res_df[np.logical_and.reduce((res_df.layer == layer, res_df.t==t-1))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
                    if smoothing_p:
                        selective_map = local_average(selective_map.reshape(1,-1), distances, p=smoothing_p, verbose=False).reshape(ms,ms)
                    masks[domain] =  selective_map > neglogp_thresh
                    colored_map[masks[domain],:] += cmap[domain]
            else:
                raise ValueError(f'kind must be either binary or continuous, was {kind}')

            # normalize
            colored_map = (colored_map - np.min(colored_map)) / (np.percentile(colored_map, 99) - np.percentile(colored_map, 1))

            if transpose:
                ax = axs[jj, ii]
            else:
                ax = axs[ii, jj]
            ax.imshow(colored_map)
            ax.grid(False)
            patches = [ mpatches.Patch(color=color, label=domain) for domain, color in cmap.items()]
            # put those patched as legend-handles into the legend
            if legend and ii == len(layers) - 1 and jj == len(celltypes)-1:
                ax.legend(handles=patches, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0., title='domain')
            # plt.title(f'Unit selectivity {dset_name}')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    return fig, axs


def load_for_tests(base_fn, overwrite=False, as_dict=False, no_bias=False, get_preacts=False, dataset_name=None, untrained=False):
    """
    Arguments:
        base_fn: the filename identifier of a given model
        overwrite: whether to overwrite experiments for the model or load previously existing ones
        as_dict: whether to return all the results as a single dictionary
        no_bias: whether to extract activations from a version of the model without unit biases (experimental)
        get_preacts: whether to extract activations without using the activation function (experimental)
        dataset_name: used to extract activations for a non-standard dataset, specified as a string
    Returns:
        (if as_dict=True, the following variables are stored as keys in a dictionary)
        base_fn: (str) the filename identifier
        opt: (dict) the full set of hyperparameters for the model
        arch_kwargs: (dict) a subset of hyperparameters passed specifically to the ITN model
        model: (ITN) the ITN model
        res: (pd.DataFrame) the selectivity experiment results 
        acc: (dict) the accuracy of the model in categorizing the images from the experiment
        acts: (dict) the activations of each tested IT layer of the model
        d_trials: (dict) the set of trials corresponding to each domain tested in the experiment
        sl_accs: (dict) searchlight decoding accuracies for each layer of the model
    """
    
    def_kwargs = unparse_model_args(base_fn, use_short_names=True)
    base_fn, opt, arch_kwargs = parse_model_args(get_n_classes=True, bools_as_ints=not 'True' in base_fn, **def_kwargs)
    reproducible_results(seed=opt['rs'])
    model = get_model_from_args(opt, arch_kwargs, base_fn=base_fn if not untrained else None)
    if no_bias:
        for cell in model.cells:
            if hasattr(getattr(model, cell), 'map_side'):
                getattr(model, cell).cell.bias = None
        b_tag = '_nobias'
    else:
        b_tag = ''
    p_tag = '_pre' if get_preacts else ''

    # fig_dir = os.path.join(FIGS_DIR, 'topographic', 'EI', 'summaries', base_fn)
    # os.makedirs(fig_dir, exist_ok=True)
    
    layers = [cell for cell in model.cells if 'IT' in cell] # make sure to exclude V4
    celltypes = ['', '_I'] if 'EI' in opt.cell else ['']
    if dataset_name is None:
        dataset_name = get_experiment_dataset(opt.trainset)

    # simple topographic visualizations (mean, readout, searchlights)
    overwrite_sl = False
    run_searchlights = False
    ims_per_class=10 if 'mini' in dataset_name else 5
    no_noise=False

    timing = exp_times[opt['timing']]
    times = np.arange(timing.exp_end + timing.jitter) # let's go up to the maximum time
    input_times = timing.stim_off
    res, acc, acts, d_trials = do_mini_objfacescenes_exp(model, layers=layers,
                                                            input_times=input_times, 
                                                            times=times, 
                                                            dataset_name=dataset_name, 
                                                            cache_base_fn=base_fn+b_tag+p_tag, 
                                                            ims_per_class= ims_per_class, #1 if opt.trainset == 'objfacescenes' else 5, 
                                                            equate_domains=True, 
                                                            return_activations=True, 
                                                            accuracy_vector=True, 
                                                            overwrite=overwrite,
                                                            overwrite_sl=overwrite_sl,
                                                            run_searchlights=run_searchlights,
                                                            sl_ps=[0.1, 0.3],
                                                            sl_ts= [timing.exp_end], #np.arange(timing.stim_off,timing.exp_end+timing.jitter),
                                                            progress_bar=False,
                                                            compute_selectivity=True,
                                                            batch_size=32,
                                                            EI='EI' in opt.cell,
                                                            fivecrop=opt.fivecrop,
                                                            logpolar=opt.logpolar,
                                                            get_preacts=get_preacts,
                                                        )
    sl_p = 0.1
    robust=False

    sl_accs = {celltype: {} for celltype in celltypes}
    for celltype in celltypes:
        for layer in layers:
            try:
                with open(f'/user_data/nblauch/git/topographic/data/topographic/searchlights/{base_fn}/{dataset_name}_10-ims-per-class_noise-True/p-{sl_p}_layer-{layer}{celltype}_clf-ridge_accuracies.pkl', 'rb') as f:
                    sl_accs[celltype][layer] = pickle.load(f)
            except Exception as e:
                print(e)
            
    if as_dict:
        return dict(
            base_fn=base_fn,
            opt=opt,
            arch_kwargs=arch_kwargs,
            model=model,
            res=res,
            acc=acc,
            acts=acts,
            d_trials=d_trials,
            sl_accs=sl_accs,
        )
    else:
        return base_fn, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs


def ei_ablations(seed=1, jj=False, ln=True, wrap=False, subset=False, **kwargs):

    jj_tag = '_jj' if jj else ''
    ln_tag = 'noln' if not ln else '' 
    wrap_tag = '_nwr-1' if not wrap else ''
    exp_name = f'ei_ablations{jj_tag}{ln_tag}{wrap_tag}_rs-{seed}'
    fig_dir = os.path.join(FIGS_DIR, 'topographic', 'EI', 'experiments', exp_name)
    os.makedirs(fig_dir, exist_ok=True)
    domains = ['object', 'face', 'scene']
    models = {}
    if jj:
        model_dict = {
            'E/I, EFF, RNN ($\lambda_w=0.5$)': f'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-0.5_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I, RNN ($\lambda_w=1.0$)': f'a-0.2_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-1.0_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I FNN ($\lambda_w=0.5$)': f'a-1.0_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-0.5_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS',
            'EFF, RNN ($\lambda_w=1.0$)': f'a-0.2_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-1.0_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'EFF, FNN ($\lambda_w=1.0$)': f'a-1.0_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-1.0_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS',
            'RNN ($\lambda_w=1.0$)': f'a-0.2_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-1.0_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'FNN ($\lambda_w=10.0$)': f'a-1.0_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2{ln_tag}_imd-112_jj-10.0_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS',
        }
        if not subset:
            if not ln:
                raise NotImplementedError()
            model_dict['EI, EFF, local-rec RNN'] = f'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2rec_imd-112_jj-0.5_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
            model_dict['EI, EFF, local-ff RNN'] = f'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2ff_imd-112_jj-0.5_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'

    else:
        if subset:
            raise NotImplementedError()
        model_dict = {
             'E/I, EFF, local RNN': f'a-0.2_cell-EI4_cIT-1_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I, local RNN': f'a-0.2_cell-EI5_cIT-1_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'EFF, local RNN': f'a-0.2_cell-SRNEFF_cIT-1_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'local RNN': f'a-0.2_cell-SRN_cIT-1_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I, EFF, local-rec RNN': f'a-0.2_cIT-1_cell-EI4_conn-circ_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-1.0_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I, EFF, local-ff RNN': f'a-0.2_cIT-1_cell-EI4_conn-circ_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_rwid-1.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'EFF, local-rec RNN': f'a-0.2_cIT-1_cell-SRNEFF_conn-circ_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_rwid-1.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'E/I, EFF, RNN': f'a-0.2_cIT-1_cell-EI4_conn-circ_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-1.0_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_rwid-1.0_sch-plateau_sq-1_t-v3_tr-miniOFS',
            'EFF, local-ff, RampFFLayNorm': f'a-0.2_cIT-1_cell-SRNEFF_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_norec-1_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS',
        }

    models = arbitrary_model_comparison(model_dict, exp_name, **kwargs)

    return models


def plot_metric_times(res, metric, base_fn='', show=True, close=False, layers=['pIT', 'aIT'], times = [1,2,4,5,10,15,20], **kwargs):
    fig, axs = plt.subplots(len(layers), len(times), figsize=(3.2*len(times), 3*len(layers)))
    if len(axs.shape) == 1:
        axs = axs.reshape(1,-1)
    for t_i, t in enumerate(times):
        for ii, layer in enumerate(layers):
            map_res = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][metric].iloc[0]
            
            g = sns.heatmap(map_res, ax=axs[ii,t_i], square=True, **kwargs)
            g.set_xticklabels([])
            g.set_yticklabels([])
            g.set_title(f'{layer}\n t={t}')

    plt.suptitle(metric, y=1.05)
    plt.tight_layout()
    if base_fn is not None and len(base_fn)>0:
        os.makedirs(f'{FIGS_DIR}/topographic/EI/metric_times/{base_fn}', exist_ok=True)
        plt.savefig(f'{FIGS_DIR}/topographic/EI/metric_times/{base_fn}/metric-{metric}.png', dpi=400)    
    if show:
        plt.show()
    if close:
        plt.close()
        
    return fig


def plot_mean_domains(res, time, inhibitory=False, base_fn='', layers=['pIT', 'aIT'], show=True, close=False, **kwargs):
    fig, axs = plt.subplots(len(layers),3, figsize=(10,3*len(layers)))
    if len(axs.shape) == 1:
        axs = axs.reshape(1,-1)
    celltype = '_I' if inhibitory else ''
    for ii, layer in enumerate(layers):
        for jj, domain in enumerate(['face', 'object', 'scene']):
            map_res = res.iloc[np.logical_and.reduce((res.t == time-1, res.layer == layer))][f'mean_{domain}{celltype}'].iloc[0]
            
            g = sns.heatmap(map_res, ax=axs[ii,jj], square=True, **kwargs)
            g.set_title(f'{layer} mean {domain}')

    plt.suptitle(f't={time}', y=1.05)
    plt.tight_layout()
    if base_fn is not None and len(base_fn)>0:
        os.makedirs(f'{FIGS_DIR}/topographic/EI/mean_domains/{base_fn}', exist_ok=True)
        plt.savefig(f'{FIGS_DIR}/topographic/EI/mean_domains/{base_fn}/t-{time}{celltype}.png', dpi=400)
    if show:
        plt.show()
    if close:
        plt.close()
        
    return fig

def plot_activity_domains(acts, d_trials, trial, time, inhibitory=False, base_fn='', layers=['pIT', 'aIT'], show=True, close=False, **kwargs):
    fig, axs = plt.subplots(len(layers),3, figsize=(10,3*len(layers)))
    if len(axs.shape) == 1:
        axs = axs.reshape(1,-1)
    for ii, layer in enumerate(layers):
        for jj, domain in enumerate(['face', 'object', 'scene']):
            if inhibitory:
                map_acts = acts[layer][d_trials[domain][trial],time-1,1024::]
            else:
                map_acts = acts[layer][d_trials[domain][trial],time-1,:1024]
            g = sns.heatmap(map_acts.reshape(32,32), ax=axs[ii,jj], square=True, **kwargs)
            g.set_title(domain + ' ' + layer)

    plt.tight_layout()
    if base_fn is not None and len(base_fn)>0:
        os.makedirs(f'{FIGS_DIR}/topographic/EI/activity_domains/{base_fn}', exist_ok=True)
        plt.savefig(f'{FIGS_DIR}/topographic/EI/activity_domains/{base_fn}/trial-{trial}_t-{time}.png', dpi=400)      
    if show:
        plt.show()
    if close:
        plt.close()

    return fig

def plot_unit_weights(pIT_hh_weight, aIT_hh_weight, unit_idx, sl=32, **kwargs):
    fig, axs = plt.subplots(2,4,figsize=(14,6))
    sns.heatmap(pIT_hh_weight[0:sl**2,unit_idx].reshape(sl,sl), ax=axs[0,0], **kwargs)
    sns.heatmap(pIT_hh_weight[sl**2::,unit_idx].reshape(sl,sl), ax=axs[0,1], **kwargs)
    sns.heatmap(pIT_hh_weight[0:sl**2,sl**2+unit_idx].reshape(sl,sl), ax=axs[0,2], **kwargs)
    sns.heatmap(pIT_hh_weight[sl**2::,sl**2+unit_idx].reshape(sl,sl), ax=axs[0,3], **kwargs)
#     sns.heatmap(aIT_ih_weight[:,unit_idx].reshape(sl,sl), ax=axs[2])
    sns.heatmap(aIT_hh_weight[0:sl**2,unit_idx].reshape(sl,sl), ax=axs[1,0], **kwargs)
    sns.heatmap(aIT_hh_weight[sl**2::,unit_idx].reshape(sl,sl), ax=axs[1,1], **kwargs)
    sns.heatmap(aIT_hh_weight[0:sl**2,sl**2+unit_idx].reshape(sl,sl), ax=axs[1,2], **kwargs)
    sns.heatmap(aIT_hh_weight[sl**2::,sl**2+unit_idx].reshape(sl,sl), ax=axs[1,3], **kwargs)
    
    axs[0,0].set_title(f'pIT E-{unit_idx} -> pIT E')
    axs[0,1].set_title(f'pIT E-{unit_idx} -> pIT I')
    axs[0,2].set_title(f'pIT I-{unit_idx} -> pIT E')
    axs[0,3].set_title(f'pIT I-{unit_idx} -> pIT I')   
    
    axs[1,0].set_title(f'aIT E-{unit_idx} -> aIT E')
    axs[1,1].set_title(f'aIT E-{unit_idx} -> aIT I')
    axs[1,2].set_title(f'aIT I-{unit_idx} -> aIT E')
    axs[1,3].set_title(f'aIT I-{unit_idx} -> aIT I')   
    plt.tight_layout()
    return fig

def plot_unit_weights_video(pIT_hh_weight, aIT_hh_weight, base_fn, 
            sl=32, fps=10, interval=2, 
            center=0, cmap='RdBu_r', vmin=-0.01, vmax=0.01, 
            **kwargs,
            ):
    vid_dir = f'../{FIGS_DIR}/topographic/EI/weights/{base_fn}'

    os.makedirs(vid_dir, exist_ok=True)

    writer = imageio.get_writer(f'{vid_dir}/video.mp4', fps=fps)
    for unit_idx in tqdm(range(0,sl**2,interval)):
        fig = plot_unit_weights_ia(unit_idx,center=center, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        fig.savefig(f'{vid_dir}/{unit_idx:04d}.png')
        plt.close()
        writer.append_data(imageio.imread(f'{vid_dir}/{unit_idx:04d}.png'))
        os.remove(f'{vid_dir}/{unit_idx:04d}.png')
    writer.close()


def full_ei_ablations(seed=1, ln=True, wrap=False, 
                        mod_types=['E/I EFF RNN', 'E/I RNN', 'E/I FNN', 'EFF RNN', 'EFF FNN', 'RNN', 'FNN'], 
                        lambdas=[0, 0.01, 0.1, 0.5, 1.0, 10.0], 
                        layers=['pIT', 'cIT', 'aIT'],
                        **kwargs):
    """
    plot a domain-level and generic summary statistic for various architectures at various lambda values 
    """
    model_dict = {'arch':[], 'lambda':[], 'base_fn':[], 'ms':[], 'acts':[], 'res':[], 'rec_distances':[], 'cell':[], 'timing':[]}
    wrap_tag = '_nwr-1' if not wrap else ''
    ln_tag = 'noln' if not ln else '' 
    mod_id = 'jjd2'+ln_tag
    for mod_type in mod_types:
        if mod_type == 'E/I EFF RNN':
            temp_fn = 'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'EFF RNN':
            temp_fn = 'a-0.2_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'E/I RNN':
            temp_fn = 'a-0.2_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'RNN':
            temp_fn = 'a-0.2_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'EFF FNN':
            temp_fn = 'a-1.0_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        elif mod_type == 'E/I FNN':
            temp_fn = 'a-1.0_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        elif mod_type == 'FNN':
            temp_fn = 'a-1.0_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        for lam in lambdas:
            jj_tag = f'_jj-{lam}' if lam else ''
            jj_name = r'$\lambda=$' + str(lam)

            full_fn = temp_fn.format(mod_id, jj_tag, wrap_tag, seed)
            model_dict['base_fn'].append(full_fn)
            model_dict['arch'].append(mod_type)
            model_dict['lambda'].append(lam)
            outputs = load_for_tests(base_fn, as_dict=True)
            model_dict['ms'].append(outputs['opt']['ms'])
            model_dict['acts'].append(outputs['acts'])
            model_dict['res'].append(outputs['res'])
            model_dict['rec_distances'].append(outputs['model'].getattr(layers[-1]).rec_distances.cpu()) # assuming they are the same for each layer, which is true for all current models
            for attr in ['cell', 'ms', 'timing']:
                model_dict[attr].append(outputs['opt'][attr])
    
     # plotting domain-level summary statistic 
    for stat_type in [1,2]:
        fig, ax = plt.subplots(figsize=(3,3))
        new_df = pd.DataFrame(dict(celltype=[], layer=[], domain_stat=[], arch=[], lam=[]))
        for ll, layer in enumerate(layers):
            for ii, celltype in enumerate(celltypes):
                n_panes = num_i if celltype == '_I' else len(models) 
                domain_stats = []
                x_vals = []
                for jj in range(len(model_dict)):
                    if 'EI' not in model_dict['cell'][jj] and celltype == '_I':
                        continue
                    t = getattr(exp_times[model_dict['timing'][jj]], time)
                    res = model_dict['res'][jj]
                    dists = model_dict['rec_distances'][jj][:model_dict['ms'][jj]**2,:model_dict['ms'][jj]**2]
                    selectivity_dict = {domain: res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0] for domain in ['object', 'face', 'scene']}
                    domain_stat = compute_domain_summary_stat(selectivity_dict, dists, model_dict['ms'][jj], celltype, stat_type=stat_type, binary_map_smoothing=binary_map_smoothing)
                    x_vals.append(jj+0.1 if ii==1 else jj )
                    domain_stats.append(domain_stat)
                    new_df = new_df.append(dict(layer = layer, celltype=celltype, domain_stat=domain_stat, arch=model_dict['arch'][jj], lam=model_dict['lambda'][jj]), ignore_index=True)
                # if stat_type == 1:
                #     title = 'Distance difference \n(different - same selectivity)'
                # else:
                #     title = 'Domain topography stat'
                # ax.plot(x_vals,domain_stats,label=layer+celltype, linestyle=['-','--'][ii], marker=['o','^'][ii], color=COLORS[ll])
                
        # now sum up over layers per model
        domain_stats = []
        for ii, arch in enumerate(mod_types):
            for jj, lam in enumerate(lambdas):
                these_domain_stats = new_df[np.logical_and.reduce((new_df['arch'] == arch, new_df['lambda'] == lam))].domain_stat
                if len(these_domain_stats) == 0:
                    continue
                else:
                    meaned = np.mean(these_domain_stats)
                    domain_stats.append(meaned)
            new_df = new_df.append(dict(layer = 'meaned', celltype='meaned', domain_stat=meaned, arch=arch, lam=lam), ignore_index=True)


        sns.pointplot()
        
        ax.plot(np.arange(len(domain_stats)),domain_stats,label='layer-avg', linestyle='-.', marker='>', color='r')
                
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(list(models.keys()), rotation=45)
        ax.set_ylabel(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1,0.8))
        fig.savefig(os.path.join(fig_dir, f'summary-{stat_type}_topography_t-{time}.png'),dpi=300,  bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
    # plotting generic summary statistic
    fig, ax = plt.subplots(figsize=(3,3))
    new_df = pd.DataFrame(dict(celltype=[], layer=[], generic_stat=[], model=[]))
    for ll, layer in enumerate(layers):
        for ii, celltype in enumerate(celltypes):
            generic_stats = []
            x_vals = []
            for jj, (model_name, outputs) in enumerate(models.items()):
                if 'EI' not in model_dict[jj]['cell'] and celltype == '_I':
                    continue
                dists = getattr(outputs['model'], layer).rec_distances[:model_dict[jj]['ms']**2,:model_dict[jj]['ms']**2]
                activations = outputs['acts'][layer][:,-1] #select last time point
                stat = compute_generic_summary_stat(activations, dists, model_dict[jj]['ms'], celltype, plot=False)
                generic_stats.append(stat)
                x_vals.append(jj+0.1 if ii==1 else jj )
                new_df = new_df.append(dict(layer = layer, celltype=celltype, generic_stat=stat, model=model_name), ignore_index=True)
            ax.plot(x_vals,generic_stats,label=layer+celltype, linestyle=['-','--'][ii], marker=['o','^'][ii], color=COLORS[ll])
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(list(models.keys()), rotation=45)
            ax.set_ylabel('Generic topography stat')

    # now sum up over layers per model
    generic_stats = []
    for jj, (model_name, outputs) in enumerate(models.items()):
        meaned = np.mean(new_df[new_df.model==model_name].generic_stat)
        generic_stats.append(meaned)
        new_df = new_df.append(dict(layer = 'meaned', celltype='meaned', generic_stat=meaned, model=model_name), ignore_index=True)
    ax.plot(np.arange(len(generic_stats)),generic_stats,label='layer-avg', linestyle='-.', marker='>', color='r')

    ax.legend(loc='upper left', bbox_to_anchor=(1,0.8))
    fig.savefig(os.path.join(fig_dir, f'summary-generic_topography_t-{time}.png'),dpi=300,  bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()



def ei_robustness(show=False, overwrite_res=False, jj=False, wrap=True, layers=['pIT', 'cIT', 'aIT'], **kwargs):
    model_dict = {}
    wrap_tag = '_nwr-1' if not wrap else ''

    if jj:
        n_seeds = 4
        jj_tag = '_jj'
        base_fn = 'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-jjd2_imd-112_jj-0.5_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
    else:
        n_seeds = 7
        jj_tag = ''
        base_fn = 'a-0.2_cell-EI4_cIT-1_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rs-{}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-miniOFS'

    for seed in range(1,n_seeds+1):
        # models[str(seed)] = load_for_tests(
        #     base_fn.format(wrap_tag, seed),
        #     as_dict=True,
        #     overwrite=overwrite_res,
        # )
        model_dict[f'seed-{seed}'] = base_fn.format(wrap_tag, seed)


    # first do generic stuff
    exp_name = 'ei_robustness'+jj_tag+wrap_tag
    models = arbitrary_model_comparison(model_dict, exp_name, layers=layers, show=False, overwrite_res=False, **kwargs)

    os.makedirs(os.path.join(FIGS_DIR, 'topographic', 'EI', 'experiments', exp_name), exist_ok=True)
        
    # ## domain topography
    # celltypes = ['', '_I']
    # for t in [10, 15]:
    #     for layer in ['pIT', 'cIT', 'aIT']:
    #         for domain in ['face', 'object', 'scene']:
    #             fig, axs = plt.subplots(2,n_seeds, figsize=((n_seeds/8)*29,6))
    #             for seed in range(1,n_seeds+1):
    #                 outputs = models[str(seed)]
    #                 for ii, celltype in enumerate(celltypes):
    #                     res = outputs['res']
    #                     selectivity = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
    #                     sns.heatmap(selectivity, ax=axs[ii,seed-1], square=True, center=0, cmap='RdBu_r')
    #                     axs[ii,seed-1].set_title(f'seed {seed}' + (' I units' if 'I' in celltype else ''))
    #                     axs[ii,seed-1].set_xticklabels([])
    #                     axs[ii,seed-1].set_yticklabels([])
    #             plt.tight_layout()
    #             fig.savefig(os.path.join(FIGS_DIR, 'topographic', 'EI', 'experiments', 'ei_robustness'+jj_tag+wrap_tag, f'{domain}_topography_layer-{layer}_t-{t}.png'),dpi=300,  bbox_inches='tight')
    #             if show:
    #                 plt.show()

    ## generic topography
    times = np.arange(20)
    for layer in ['pIT', 'cIT', 'aIT']:
        fig, ax = plt.subplots()
        df = pd.DataFrame({'seed':[], 'celltype':[], 'time':[], 'r':[]})
        for seed in range(1,n_seeds+1):
            outputs = models[f'seed-{seed}']
            acts = outputs['acts']
            model = outputs['model']
            d_trials = outputs['d_trials']
            celltypes = ['_E', '_I']
            for kk, celltype in enumerate(celltypes):
                for jj, t in enumerate(times):
                    if domain is not None:
                        activations = acts[layer][d_trials[domain],t,:].squeeze()
                    else:
                        activations = acts[layer][:,t,:].squeeze()
                    if 'E' in celltype or celltype == '':
                        activations = activations[:,:1024]
                    else:
                        activations = activations[:,1024::]
                    # get rid of nans and infinities
                    activations[np.isnan(activations)] = 0
                    activations[activations == np.inf] = np.nanmax(activations)
                    corrs = np.corrcoef(activations.transpose())
                    corrs = corrs.reshape(-1)
                    corrs[np.isnan(corrs)] = 0
                    rand_inds = np.random.choice(corrs.shape[0], size=100000, replace=False)
                    dists = getattr(model, layer).rec_distances[:1024,:1024].cpu().numpy().reshape(-1)[rand_inds]
                    corrs = corrs[rand_inds]
                    corrs = corrs[dists != 0]
                    dists = dists[dists != 0]
                    data_r, _ = spearmanr(corrs, dists)
                    df = df.append({'seed': seed, 'celltype': celltype, 'time': t, 'r':data_r}, ignore_index=True)
        sns.lineplot(data=df, x='time', y='r', hue='celltype')
        plt.xticks(times[::2])
        plt.xlabel('$t$')
        plt.ylabel(r'$r_{spearman}(r,d)$')
        plt.gca().invert_yaxis()
        plt.legend(['E cells', 'I cells'], bbox_to_anchor=(1,1))
        plt.ylim([0.05,-1])
        # plt.title('generic topographic organization in aIT')
        plt.savefig(os.path.join(FIGS_DIR, 'topographic', 'EI', 'experiments', exp_name, f'generic_topography_layer-{layer}.png'), dpi=300, bbox_inches='tight')
        plt.show()

def compute_generic_summary_stat(activations, distances, ms, celltype, domain_trials=None, domain=None, n_samples=100000, plot=False):
    """
    computes the inverse-distance-scaled sum of pairwise response correlations

    Args:
        activations: NxP matrix
        dists: PxP matrix
        ms: map side length, we assume ms**2 units per celltype
        celltype: either 'E', 'I', or ''
        domain_trials: if not None, a dict containing lists of trials for each named domain
        domain: if not None, a string naming the domain to select trials from. requires domain_trials
        n_samples: number of random pairs of units to sample
    Returns:
        stat: inverse-distance-scaled sum of pairwise response correlations
    """
    if domain is not None:
        assert domain_trials is not None
        activations = activations[domain_trials[domain]]
    
    if 'E' in celltype or celltype == '':
        activations = activations[:,:ms**2]
    else:
        activations = activations[:,ms**2:]
    # get rid of nans and infinities
    activations[np.isnan(activations)] = 0
    activations[activations == np.inf] = np.nanmax(activations)
    # compute pairwise correlations
    corrs = np.corrcoef(activations.transpose())
    corrs = corrs.reshape(-1)
    corrs[np.isnan(corrs)] = 0
    # subselect for efficiency
    rand_inds = np.random.choice(corrs.shape[0], size=100000, replace=False)
    dists = distances[:ms**2,:ms**2].cpu().numpy().reshape(-1)[rand_inds]
    corrs = corrs[rand_inds]
    # ensure we are not looking at same-unit pairs
    corrs = corrs[dists != 0]
    dists = dists[dists != 0]
    # standardize correlations so that networks with totally correlated units (and therefore a useless representational space) don't dominate
    corrs = (corrs - np.mean(corrs))/np.std(corrs)
    # finally compute summary stat
    stat = np.mean(corrs/dists)
    # stat, _ = spearmanr(corrs, dists)
    # stat = -stat

    if plot:
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(ms*dists, corrs, 'o', alpha=0.01)
        ax.set_xlim(0,ms*1.01*np.max(dists))
        ax.axhline(0.0, 0, 1, color='r', linestyle='--')
        ax.set_ylim(-.3,1.05)
        ax.set_xlabel('Pairwise unit distance (unit lengths)')
        ax.set_ylabel('Pairwise unit response correlation')
        ax.set_title(f'stat={stat:.04f}')
        # im_dom = domain if domain is not None else 'all'
        plt.show()


    return stat


def compute_domain_summary_stat(selectivity_dict, distances, ms, celltype, stat_type=2, binary_map_smoothing=0.05, binary_map_thresh=3):
    """
    computes the 1 of 4 statistics indexing domain-level functional topography

    Args:
        selectivity_dict: dict containing msxms selectivity for each domain
        dists: PxP matrix
        ms: map side length, we assume ms**2 units per celltype
        celltype: either 'E', 'I', or ''
        stat_type: which stat type. 1: distance difference, 2: dot product alignment, 3: cosine alignment, 4: correlation alignment
        binary_map_smoothing: how much smoothing to apply to the colored_maps; only used for stat_type==1
    Returns:
        stat: the domain-level summary statistic
    """
    colored_maps = np.zeros((ms, ms,3)) 
    full_selectivity = np.zeros((ms, ms,3))
    cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1], 'face & object': [1,1,0], 'object & scene':[0,1,1], 'face & scene':[1,0,1], 'none':[0,0,0]}

    for dd, domain in enumerate(['face', 'object', 'scene']):
        pane_i=-1
        pane_i+=1
        selectivity = selectivity_dict[domain] #res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
        if binary_map_smoothing:
            selectivity = local_average(selectivity.reshape(1,-1), distances, p=binary_map_smoothing).reshape(ms,ms)
        domain_mask =  selectivity > binary_map_thresh
        colored_maps[domain_mask] += cmap[domain]
        full_selectivity[:,:,dd] = selectivity

    dists = distances[:ms**2,:ms**2]
    pair_inds = list(combinations(range(ms**2), 2))
    v_colors = colored_maps.reshape(-1,3)
    if stat_type == 1:
        # difference between distances of units with identical or fully different selectivity
        sel_diffs = v_colors[pair_inds].std(1).sum(1) # where 0 < STD < 1, there is partial overlap of selectivity, and we will ignore these unit pairs
        domain_stat = dists[np.tri(ms**2) == 0][sel_diffs >= 1].mean().item() - dists[np.tri(ms**2) == 0][sel_diffs == 0].mean().item()
    elif stat_type in [2,3,4]:
        sel_pairs = full_selectivity.reshape(-1,3)[pair_inds]
        if stat_type == 2:
            # dot product
            sel_alignment = np.nansum(sel_pairs[:,0,:]*sel_pairs[:,1,:], 1)
        elif stat_type == 3:
            # cosine 
            sel_alignment = np.nansum(sel_pairs[:,0,:]*sel_pairs[:,1,:], 1)/(np.sqrt(np.nansum(sel_pairs[:,0,:]**2,1))*np.sqrt(np.nansum(sel_pairs[:,1,:]**2,1)) + 10e-9)
        elif stat_type == 4:
            # correlation
            sel_pairs = sel_pairs - np.nanmean(sel_pairs, 2).reshape(-1,2,1)
            sel_alignment = np.nansum(sel_pairs[:,0,:]*sel_pairs[:,1,:], 1)/(np.sqrt(np.nansum(sel_pairs[:,0,:]**2,1))*np.sqrt(np.nansum(sel_pairs[:,1,:]**2,1)) + 10e-9)
        # standardize the scores of selectivity alignment so that models with identical selectivity throughout don't simply dominate
        sel_alignment = (sel_alignment - np.mean(sel_alignment))/np.std(sel_alignment)
        # lastly weight the selectivity alignment by the inverse of the distance and take the mean 
        domain_stat = ((1/dists[np.tri(1024) == 0].cpu())*sel_alignment).mean().item()

    return domain_stat
                    
def arbitrary_model_comparison(model_dict, experiment_name, layers=['pIT', 'cIT', 'aIT'], time='exp_end', show=False, overwrite_res=False, 
                                do_domains=True, do_generic=True, do_combined=False, do_summary=False, do_performance=False,
                                plot_i_cells = True,
                                end_on_error=False,
                                binary_map_thresh=3,
                                binary_map_smoothing=0,
                                models=None,
):
    fig_dir = os.path.join(FIGS_DIR, 'topographic', 'EI', 'experiments', experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    # check to see if they all exist first
    for model_name, base_fn in model_dict.items():
        if not os.path.exists(f'{SAVE_DIR}/models/{base_fn}.pkl'):
            print(f'{model_name} does not exist at path \n{base_fn}')

    # performance (do before loading models since it is quicker)
    if do_performance:
        res = {'model':[], 'metric':[], 'accuracy':[]}
        for jj, (model_name, base_fn) in enumerate(model_dict.items()):
            with open(f'{SAVE_DIR}/results/{base_fn}_losses.pkl', 'rb') as f:
                losses = pickle.load(f)
            for metric in ['acc']:
                res['accuracy'].append(losses['val'][metric][-1])
                res['metric'].append(metric)
                res['model'].append(model_name)
        df = pd.DataFrame(res)
        fig, ax  = plt.subplots(figsize=(4,4))
        g = sns.pointplot(data=df, x='model', y='accuracy') #, hue='metric', style='metric', legend=False)
        plt.xticks(g.get_xticks(), rotation=45)
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'accuracy.png'),dpi=300,  bbox_inches='tight')
        if show:
            plt.show()

    if not do_domains and not do_generic and not do_combined and not do_summary:
        return model_dict

    if models is None:
        models = {}
        for model_name, base_fn in model_dict.items():
            try:
                models[model_name] = load_for_tests(
                    base_fn,
                    as_dict=True,
                    overwrite=overwrite_res,
                )
            except Exception as e:
                if end_on_error:
                    print(e)
                    raise ValueError(f'{model_name} does not exist at path \n{base_fn}')
                else:
                    print(e)
                    print(f'{model_name} does not exist at path \n{base_fn}\n continuing...')
    else:
        print('using pre-loaded models provided to function for speed. could be inconsistent with paths. use with care.')

    num_i = np.sum(['EI' in outputs['opt']['cell'] for model, outputs in models.items()])
    celltypes = ['', '_I'] if num_i > 0 and plot_i_cells else ['']

    ## domain topography
    if do_domains:
        for layer in layers:
            for ii, celltype in enumerate(celltypes):
                n_panes = num_i if celltype == '_I' else len(models) 

                colored_maps = {model_name: np.zeros((outputs['opt'].ms, outputs['opt'].ms,3)) for model_name, outputs in models.items()}
                cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1], 'face & object': [1,1,0], 'object & scene':[0,1,1], 'face & scene':[1,0,1], 'none':[0,0,0]}

                for domain in ['face', 'object', 'scene']:
                    fig, axs = plt.subplots(1,n_panes, figsize=(3.5*n_panes,3))
                    pane_i=-1
                    for jj, (model_name, outputs) in enumerate(models.items()):
                        if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                            continue
                        pane_i+=1
                        res = outputs['res']
                        t = getattr(exp_times[outputs['opt']['timing']], time)
                        selectivity = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
                        if binary_map_smoothing:
                            selectivity = local_average(selectivity.reshape(1,-1), getattr(outputs['model'], layer).rec_distances, p=binary_map_smoothing).reshape(outputs['opt'].ms,outputs['opt'].ms)
                        domain_mask =  selectivity > binary_map_thresh
                        colored_maps[model_name][domain_mask] += cmap[domain]
                    #     masks[domain] = masks[domain].reshape(-1)
                        sns.heatmap(selectivity, ax=axs[pane_i], square=True, center=0, cmap='RdBu_r')
                        axs[pane_i].set_title(model_name + (' I units' if 'I' in celltype else ''))
                        axs[pane_i].set_xticklabels([])
                        axs[pane_i].set_yticklabels([])
                    plt.tight_layout()
                    fig.savefig(os.path.join(fig_dir, f'{domain}_topography_layer-{layer}{celltype}_t-{time}.png'),dpi=300,  bbox_inches='tight')
                    if show:
                        plt.show()

                # now plot binary selectivity across all domains simultaneously
                fig, axs = plt.subplots(1,n_panes, figsize=(3.5*n_panes,3))
                pane_i=-1
                for jj, (model_name, outputs) in enumerate(models.items()):
                    if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                        continue
                    pane_i+=1
                    axs[pane_i].imshow(colored_maps[model_name])
                    axs[pane_i].grid(False)
                    axs[pane_i].set_xticklabels([])
                    axs[pane_i].set_yticklabels([])
                    axs[pane_i].set_title(model_name + (' I units' if 'I' in celltype else ''))
                    patches = [ mpatches.Patch(color=color, label=domain) for domain, color in cmap.items()]
                    # put those patched as legend-handles into the legend
                    if pane_i == n_panes - 1:
                        axs[pane_i].legend(handles=patches, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0., title='selectivity')
                # plt.tight_layout()
                fig.savefig(os.path.join(fig_dir, f'domains_topography_layer-{layer}{celltype}_t-{time}.png'),dpi=300,  bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.close()


    ## generic topography
    if do_generic:
        use_shift = True
        p0 = (5,0) if use_shift else (5)
        for ii, layer in enumerate(layers):
            for jj, domain in enumerate([None]): # + domains):
                n_panes = len(model_dict)
                fig, axs = plt.subplots(1,n_panes, figsize=(14.5*(n_panes/4),3), sharey=True)
                ii=-1
                for rcell, outputs in models.items():
                    if 'EI' in outputs['opt'].cell:
                        use_celltypes = ['_E']
                    else:
                        use_celltypes = ['']
                    for celltype in use_celltypes:
                        ii+=1
                        acts = outputs['acts']
                        d_trials = outputs['d_trials']
                        model = outputs['model']
                        t = getattr(exp_times[outputs['opt']['timing']], time)
                        if domain is not None:
                            activations = acts[layer][d_trials[domain],t-1,:].squeeze()
                        else:
                            activations = acts[layer][:,t-1,:].squeeze()
                        if celltype == '_E' or celltype == '':
                            activations = activations[:,:1024]
                        else:
                            activations = activations[:,1024::]
                        # get rid of nans and infinities
                        activations[np.isnan(activations)] = 0
                        activations[activations == np.inf] = np.nanmax(activations)
                        corrs = np.corrcoef(activations.transpose())
                        corrs = corrs.reshape(-1)
                        corrs[np.isnan(corrs)] = 0
                        rand_inds = np.random.choice(corrs.shape[0], size=50000, replace=False)
                        dists = getattr(model, layer).rec_distances[:1024,:1024].cpu().numpy().reshape(-1)[rand_inds]
                        sl = getattr(model, layer).map_side
                        corrs = corrs[rand_inds]
                        corrs = corrs[dists != 0]
                        dists = dists[dists != 0]
                        axs[ii].plot(sl*dists, corrs, 'o', alpha=0.01)
                        # if use_shift:
                        #     popt, pcov = curve_fit(dist_to_desired_corr, dists, corrs, p0=p0)
                        #     all_targets = dist_to_desired_corr(dists, lam=popt[0], shift=popt[-1])
                        #     targets = dist_to_desired_corr(np.linspace(0, np.max(dists), 100), lam=popt[0], shift=popt[-1])
                        #     axs[ii].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x) $-$ {-popt[1]:.03f}')
                        # else:
                        #     popt, pcov = curve_fit(dist_to_desired_corr_noshift, dists, corrs, p0=p0)
                        #     all_targets = dist_to_desired_corr_noshift(dists, lam=popt[0]) #, shift=popt[-1])
                        #     targets = dist_to_desired_corr_noshift(np.linspace(0, np.max(dists), 100), lam=popt[0]) #, shift=popt[-1])
                        #     axs[ii].plot(np.linspace(0, 0.75, 100), targets, '--', label=rf'y = exp(-{popt[0]:.02f} * x)')
                        # r2 = r2_score(corrs, all_targets)
                        # data_r, _ = spearmanr(corrs, dists)
                        # axs[ii].legend(loc='upper right')
                        for ax in [axs[ii]]:
                            ax.set_xlim(0,sl*1.01*np.max(dists))
                            ax.axhline(0.0, 0, 1, color='r', linestyle='--')
                            # ax.set_ylim(-.3,1.05)
                            ax.set_xlabel('Pairwise unit distance (unit lengths)')
                            if ii == 0:
                                ax.set_ylabel('Pairwise unit response correlation')
                        im_dom = domain if domain is not None else 'all'
                        axs[ii].set_title(f'{rcell} \n{celltype[-1]} units' if len(celltype)>0 else f'{rcell}')
                # fig.suptitle(f'{layer} - {im_dom} images, t={t}', y=1.05)
                fig.savefig(os.path.join(fig_dir, f'generic_topography_layer-{layer}_t-{time}.png'),dpi=300,  bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.close()

    if do_combined:
        # now plot binary selectivity across all domains simultaneously
        for layer in layers:
            for ii, celltype in enumerate(celltypes):
                n_panes = num_i if celltype == '_I' else len(models) 

                colored_maps = {model_name: np.zeros((outputs['opt'].ms, outputs['opt'].ms,3)) for model_name, outputs in models.items()}
                cmap = {'face': [1,0,0], 'object':[0,1,0], 'scene':[0,0,1], 'face & object': [1,1,0], 'object & scene':[0,1,1], 'face & scene':[1,0,1], 'none':[0,0,0]}

                for domain in ['face', 'object', 'scene']:
                    pane_i=-1
                    for jj, (model_name, outputs) in enumerate(models.items()):
                        if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                            continue
                        pane_i+=1
                        res = outputs['res']
                        t = getattr(exp_times[outputs['opt']['timing']], time)
                        selectivity = res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0]
                        if binary_map_smoothing:
                            selectivity = local_average(selectivity.reshape(1,-1), getattr(outputs['model'], layer).rec_distances, p=binary_map_smoothing).reshape(outputs['opt'].ms,outputs['opt'].ms)
                        domain_mask =  selectivity > binary_map_thresh
                        colored_maps[model_name][domain_mask] += cmap[domain]

                fig, axs = plt.subplots(2,n_panes, figsize=(4*n_panes,7), sharey='row')
                pane_i=-1
                for jj, (model_name, outputs) in enumerate(models.items()):
                    if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                        continue
                    pane_i+=1
                    axs[0,pane_i].grid(False)
                    axs[0,pane_i].set_xticklabels([])
                    axs[0,pane_i].set_yticklabels([])
                    axs[0,pane_i].imshow(colored_maps[model_name])
                    axs[0,pane_i].set_title(model_name + (' I units' if 'I' in celltype else ''))
                    patches = [ mpatches.Patch(color=color, label=domain) for domain, color in cmap.items()]
                    # put those patched as legend-handles into the legend
                    if pane_i == n_panes - 1:
                        axs[0,pane_i].legend(handles=patches, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0., title='selectivity')

                    # now add generic
                    t = getattr(exp_times[outputs['opt']['timing']], time)
                    acts = outputs['acts']
                    model = outputs['model']
                    activations = acts[layer][:,t-1,:].squeeze()
                    if celltype == '_E' or celltype == '':
                        activations = activations[:,:1024]
                    else:
                        activations = activations[:,1024:]
                    # get rid of nans and infinities
                    activations[np.isnan(activations)] = 0
                    activations[activations == np.inf] = np.nanmax(activations)
                    corrs = np.corrcoef(activations.transpose())
                    corrs = corrs.reshape(-1)
                    corrs[np.isnan(corrs)] = 0
                    rand_inds = np.random.choice(corrs.shape[0], size=50000, replace=False)
                    dists = getattr(model, layer).rec_distances[:1024,:1024].cpu().numpy().reshape(-1)[rand_inds]
                    sl = getattr(model, layer).map_side
                    corrs = corrs[rand_inds]
                    corrs = corrs[dists != 0]
                    dists = dists[dists != 0]
                    axs[1,pane_i].plot(sl*dists, corrs, 'o', alpha=0.01)
                    # axs[ii].legend(loc='upper right')
                    axs[1,pane_i].set_xlim(0,sl*1.01*np.max(dists))
                    axs[1,pane_i].axhline(0.0, 0, 1, color='r', linestyle='--')
                    axs[1,pane_i].set_ylim(-.5,1.05)
                    axs[1,pane_i].set_xlabel('Pairwise unit distance \n(unit lengths)')
                    if pane_i == 0:
                        axs[1,pane_i].set_ylabel('Pairwise unit response correlation')
                    # axs[ii].set_title(f'{rcell} \n{celltype[-1]} units' if len(celltype)>0 else f'{rcell}')

                # plt.tight_layout()
                fig.savefig(os.path.join(fig_dir, f'all_topography_layer-{layer}{celltype}_t-{time}.png'),dpi=300,  bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.close()

    if do_summary:
        # plotting domain-level summary statistic 
        for stat_type in [1,2]:
            fig, ax = plt.subplots(figsize=(3,3))
            new_df = pd.DataFrame(dict(celltype=[], layer=[], domain_stat=[], model=[]))
            for ll, layer in enumerate(layers):
                for ii, celltype in enumerate(celltypes):
                    n_panes = num_i if celltype == '_I' else len(models) 
                    domain_stats = []
                    x_vals = []
                    for jj, (model_name, outputs) in enumerate(models.items()):
                        if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                            continue
                        t = getattr(exp_times[outputs['opt']['timing']], time)
                        res = outputs['res']
                        dists = getattr(outputs['model'], layer).rec_distances[:outputs['opt'].ms**2,:outputs['opt'].ms**2]
                        selectivity_dict = {domain: res.iloc[np.logical_and.reduce((res.t == t-1, res.layer == layer))][f'{domain}_selectivity_neglogp{celltype}'].iloc[0] for domain in ['object', 'face', 'scene']}
                        domain_stat = compute_domain_summary_stat(selectivity_dict, dists, outputs['opt'].ms, celltype, stat_type=stat_type, binary_map_smoothing=binary_map_smoothing)
                        x_vals.append(jj+0.1 if ii==1 else jj )
                        domain_stats.append(domain_stat)
                        new_df = new_df.append(dict(layer = layer, celltype=celltype, domain_stat=domain_stat, model=model_name), ignore_index=True)
                    if stat_type == 1:
                        title = 'Distance difference \n(different - same selectivity)'
                    else:
                        title = 'Domain topography stat'
                    ax.plot(x_vals,domain_stats,label=layer+celltype, linestyle=['-','--'][ii], marker=['o','^'][ii], color=COLORS[ll])
                    
            # now sum up over layers per model
            domain_stats = []
            for jj, (model_name, outputs) in enumerate(models.items()):
                meaned = np.mean(new_df[new_df.model==model_name].domain_stat)
                domain_stats.append(meaned)
                new_df = new_df.append(dict(layer = 'meaned', celltype='meaned', domain_stat=meaned, model=model_name), ignore_index=True)
            ax.plot(np.arange(len(domain_stats)),domain_stats,label='layer-avg', linestyle='-.', marker='>', color='r')
                    
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(list(models.keys()), rotation=45)
            ax.set_ylabel(title)
            ax.legend(loc='upper left', bbox_to_anchor=(1,0.8))
            fig.savefig(os.path.join(fig_dir, f'summary-{stat_type}_topography_t-{time}.png'),dpi=300,  bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
                
        # plotting generic summary statistic
        fig, ax = plt.subplots(figsize=(3,3))
        new_df = pd.DataFrame(dict(celltype=[], layer=[], generic_stat=[], model=[]))
        for ll, layer in enumerate(layers):
            for ii, celltype in enumerate(celltypes):
                generic_stats = []
                x_vals = []
                for jj, (model_name, outputs) in enumerate(models.items()):
                    if 'EI' not in outputs['opt']['cell'] and celltype == '_I':
                        continue
                    t = getattr(exp_times[outputs['opt']['timing']], time)
                    dists = getattr(outputs['model'], layer).rec_distances[:outputs['opt'].ms**2,:outputs['opt'].ms**2]
                    activations = outputs['acts'][layer][:,t-1] 
                    stat = compute_generic_summary_stat(activations, dists, outputs['opt'].ms, celltype, plot=False)
                    generic_stats.append(stat)
                    x_vals.append(jj+0.1 if ii==1 else jj )
                    new_df = new_df.append(dict(layer = layer, celltype=celltype, generic_stat=stat, model=model_name), ignore_index=True)
                ax.plot(x_vals,generic_stats,label=layer+celltype, linestyle=['-','--'][ii], marker=['o','^'][ii], color=COLORS[ll])
                ax.set_xticks(np.arange(len(models)))
                ax.set_xticklabels(list(models.keys()), rotation=45)
                ax.set_ylabel('Generic topography stat')

        # now sum up over layers per model
        generic_stats = []
        for jj, (model_name, outputs) in enumerate(models.items()):
            meaned = np.mean(new_df[new_df.model==model_name].generic_stat)
            generic_stats.append(meaned)
            new_df = new_df.append(dict(layer = 'meaned', celltype='meaned', generic_stat=meaned, model=model_name), ignore_index=True)
        ax.plot(np.arange(len(generic_stats)),generic_stats,label='layer-avg', linestyle='-.', marker='>', color='r')

        ax.legend(loc='upper left', bbox_to_anchor=(1,0.8))
        fig.savefig(os.path.join(fig_dir, f'summary-generic_topography_t-{time}.png'),dpi=300,  bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    return models


def compare_dataset_training(seed=1, wrap=False, jj=True, lam=0.5, mod_type='E/I REC', mod_id='jjd2', **kwargs):
    model_dict = {}
    wrap_tag = '_nwr-1' if not wrap else ''
    for dset, dset_name in [('objects-trained', 'miniO'), ('faces-trained', 'miniF'), ('scenes-trained', 'miniS'), ('all-trained', 'miniOFS')]:
        if jj:
            jj_tag = f'_jj-{lam}' if lam else ''
            if mod_type == 'EFF-FNN':
                model_dict[dset] = f'a-1.0_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{mod_id}_imd-112{jj_tag}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-{dset_name}'
            elif mod_type == 'E/I-EFF-RNN':
                model_dict[dset] = f'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{mod_id}_imd-112{jj_tag}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-{dset_name}'
            elif mod_type == 'EFF-RNN':
                model_dict[dset] = f'a-0.2_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{mod_id}_imd-112{jj_tag}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{wrap_tag}_optim-sgd_rv4-1_rs-{seed}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-{dset_name}'
            else:
                raise NotImplementedError()
        else:
            model_dict[dset] = f'a-0.2_cIT-1_cell-EI4_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.3_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_optim-sgd_rs-{seed}_rsig-5.0_rwid-0.3_sch-plateau_sq-1_t-v3_tr-{dset_name}'
    models = arbitrary_model_comparison(model_dict, 'dataset_training', **kwargs)
    return models


def ei_columns(base_fn, show=False, overwrite_res=False):
    base_fn, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = load_for_tests(
        base_fn,
        as_dict=False,
        overwrite=overwrite_res,
    )
    raise NotImplementedError()

def ei_retinotopy(show=False, overwrite_res=False, layers=['pIT', 'cIT', 'aIT'], **kwargs):
    model_dict = {}
    for pad, pad_name in [('_pad-tf', 'small faces'), ('_pad-to', 'small objects')]:
        model_dict[pad_name] = f'a-0.2_cIT-1_cell-EI4_conn-circ2_ar-1.0_enc-resnet50_err-1_fsig-5.0_fwid-0.2_gc-10.0_i2e-1.0_id-nov12cr_imd-112_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nwr-1_optim-sgd{pad}_po-1_rv4-1_rs-1_rsig-5.0_rwid-0.2_sch-plateau_sq-1_t-v3_tr-miniOFS3'
    models = arbitrary_model_comparison(model_dict, 'retinotopy', layers=layers, show=show, overwrite_res=overwrite_res, **kwargs)
    return models

def compare_jj_lambdas(seed=1, mod_id='jjd2', mod_type='E/I REC', wrap=False, lambdas=[0, 0.01, 0.1, 0.5, 10.0], **kwargs):
    model_dict = {}
    wrap_tag = '_nwr-1' if not wrap else ''
    assert mod_type in ['FNN', 'RNN', 'EFF-FNN', 'EFF-RNN', 'E/I-FNN', 'E/I-RNN', 'E/I-EFF-RNN']
    for lam in lambdas:
        jj_tag = f'_jj-{lam}' if lam else ''
        jj_name = r'$\lambda=$' + str(lam)
        if mod_type == 'FNN':
            mod_tag = 'ff'
            temp_fn = 'a-1.0_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        elif mod_type == 'EFF-FNN':
            mod_tag = 'eff'
            temp_fn = 'a-1.0_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        elif mod_type == 'E/I-EFF-RNN':
            mod_tag = 'eirec'
            temp_fn = 'a-0.2_cIT-1_cell-EI4_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'EFF-FNN':
            mod_tag = 'srneff'
            temp_fn = 'a-0.2_cIT-1_cell-SRNEFF_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'RNN':
            mod_tag = 'srn'
            temp_fn = 'a-0.2_cIT-1_cell-SRN_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'E/I-RNN':
            mod_tag = 'ei5'
            temp_fn = 'a-0.2_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-v3_tr-miniOFS'
        elif mod_type == 'E/I-FNN':
            mod_tag = 'ei5ff'
            temp_fn = 'a-1.0_cIT-1_cell-EI5_conn-full_ar-1.0_enc-resnet50_err-1_fsig-5.0_gc-10.0_i2e-1.0_id-{}_imd-112{}_lr0-0.01_me-300_ms-32_nl-relu_noi-uniform_nret-1_norec-1{}_optim-sgd_rv4-1_rs-{}_rsig-5.0_sch-plateau_sq-1_t-ff_tr-miniOFS'
        else:
            raise ValueError()

        # mod_name = ' '.join((simp_tag, jj_name, str(seed)))
        full_fn = temp_fn.format(mod_id, jj_tag, wrap_tag, seed)
        model_dict[jj_name] = full_fn
    models = arbitrary_model_comparison(model_dict, f'jj_lambdas{wrap_tag}_mod-{mod_tag}_id-{mod_id}_seed-{seed}', **kwargs)
    return models

def plot_OFS_clustering(model, opt, base_fn, k_clusters=3, domains=['object', 'face', 'scene'], save=True):
    readout_weights = model.decoder.weight.detach().cpu().numpy()

    out_dir = os.path.join(FIGS_DIR, 'topographic/EI/summaries', base_fn)
    os.makedirs(out_dir, exist_ok=True)

    orig_classes = {
    'object': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100objects/train/*')]),
    'face': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100faces-crop-2.0/train/*')]),
    'scene': sorted([os.path.basename(categ) for categ in glob.glob('/lab_data/plautlab/imagesets/100scenes/train/*')]),
    }
    domain_inds = {domain: get_domain_inds(domain, opt.trainset) for domain in domains}

    obj_attributes = pd.read_csv('/home/nblauch/100objects_attributes.tsv', sep='\t')
    obj_attributes['inanimate'] = [1-animate for animate in obj_attributes['animate']]
    scene_attributes = pd.read_csv('/home/nblauch/100scene_attributes.tsv', sep='\t')
    face_attributes = pd.read_csv('/home/nblauch/100face_attributes.tsv', sep='\t')
    face_attributes['female'] = [1-male for male in face_attributes['male']]

    for domain in domains:
        if domain_inds[domain] is None:
            continue
        np.random.seed(1)

        kmeans = KMeans(n_clusters=k_clusters)
        labels = kmeans.fit_predict(readout_weights[domain_inds[domain]])

        clusters = []
        cluster_images = []
        for label in np.unique(labels):
            inds = np.where(labels == label)[0]
            this_cluster = [orig_classes[domain][ind] for ind in inds]
            clusters.append(this_cluster)
            these_images = []
            for ind in inds:
                fold = f'100{domain}s-crop-2.0' if domain == 'face' else f'100{domain}s' 

                files = glob.glob(os.path.join(f'/lab_data/plautlab/imagesets/{fold}', 'val', orig_classes[domain][ind], '*'))
                these_images.append(Image.open(files[0]).resize((100,100)))
            cluster_images.append(these_images)

        from mpl_toolkits.axes_grid1 import ImageGrid

        for cluster, these_images in enumerate(cluster_images):
            g = sns.heatmap(kmeans.cluster_centers_[cluster,:].reshape(32,32), square=True, cbar=False)
            g.set_xticklabels([])
            g.set_yticklabels([])
            g.set_title(f'aIT readout weight {domain} cluster {cluster} centroid')
            if save:
                plt.savefig(f'{out_dir}/readout_kmeans_{domain}-cluster-{cluster}-of-{k_clusters}_centroid.png', dpi=200, bbox_inches='tight')
            plt.show()
            fig = plt.figure(figsize=(8, 8))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(int(np.ceil(np.sqrt(len(these_images)))), int(np.ceil(np.sqrt(len(these_images))))), 
                            axes_pad=0,  # pad between axes in inch.
                            )
            for ii, (ax, im) in enumerate(zip(grid, these_images)):
                # Iterating over the grid returns the Axes.
                ax.imshow(im)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.grid(False)
            for jj in range(ii+1, len(grid)):
                fig.delaxes(grid[jj])
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-cluster-{cluster}-of-{k_clusters}_images.png', dpi=200, bbox_inches='tight')
            fig.suptitle(f'aIT readout weight {domain} cluster {cluster} exemplars', y=1.01)
            plt.tight_layout()
            plt.show()
            
        if domain == 'object':
            cluster_attributes = dict(attribute=[], cluster=[], p=[], p2=[])
            for attribute in ['inanimate', 'animate', 'dog', 'mammal']:
                for cluster_i, cluster in enumerate(clusters):
                    p = np.mean([obj_attributes[obj_attributes.synset == synset][attribute].values[0] for synset in clusters[cluster_i]])
                    p_of_attribute = np.sum([obj_attributes[obj_attributes.synset == synset][attribute].values[0] for synset in clusters[cluster_i]])/np.sum(obj_attributes[attribute])
                    cluster_attributes['p'].append(p)
                    cluster_attributes['p2'].append(p_of_attribute)
                    cluster_attributes['cluster'].append(cluster_i)
                    cluster_attributes['attribute'].append(attribute)
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p2', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n with attribute')
            g.legend(title='attribute', loc='center right', bbox_to_anchor=[1.3,0.8])
            sns.despine(top=True, right=True)
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n in cluster with attribute')
            g.legend(title='attribute',  loc='center right', bbox_to_anchor=[1.3,0.8])
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            sns.despine(top=True, right=True)
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes_2.png', dpi=200, bbox_inches='tight')
            plt.show()
                    
        elif domain == 'scene':
            cluster_attributes = dict(attribute=[], cluster=[], p=[], p2=[])
            for attribute in ['indoor', 'outdoor, man-made', 'outdoor, natural']:
                for cluster_i, cluster in enumerate(clusters):
                    p = np.mean([scene_attributes[scene_attributes.category == synset][attribute].values[0] for synset in clusters[cluster_i]])
                    p_of_attribute = np.sum([scene_attributes[scene_attributes.category == synset][attribute].values[0] for synset in clusters[cluster_i]])/np.sum(scene_attributes[attribute])
                    cluster_attributes['p'].append(p)
                    cluster_attributes['p2'].append(p_of_attribute)
                    cluster_attributes['cluster'].append(cluster_i)
                    cluster_attributes['attribute'].append(attribute.replace(', ','\n'))
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p2', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n with attribute')
            g.legend(title='attribute',  loc='center right', bbox_to_anchor=[1.3,0.8])
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            sns.despine(top=True, right=True)
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n in cluster with attribute')
            g.legend(title='attribute',  loc='center right', bbox_to_anchor=[1.3,0.8])
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            sns.despine(top=True, right=True)
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes_2.png', dpi=200, bbox_inches='tight')
            plt.show()

        elif domain == 'face':
            cluster_attributes = dict(attribute=[], cluster=[], p=[], p2=[])
            for attribute in ['male', 'female', 'blonde']:
                for cluster_i, cluster in enumerate(clusters):
                    p = np.mean([face_attributes[face_attributes.synset == synset][attribute].values[0] for synset in clusters[cluster_i]])
                    p_of_attribute = np.sum([face_attributes[face_attributes.synset == synset][attribute].values[0] for synset in clusters[cluster_i]])/np.sum(face_attributes[attribute])
                    cluster_attributes['p'].append(p)
                    cluster_attributes['p2'].append(p_of_attribute)
                    cluster_attributes['cluster'].append(cluster_i)
                    cluster_attributes['attribute'].append(attribute.replace(', ','\n'))
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p2', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n with attribute')
            g.legend(title='attribute',  loc='center right', bbox_to_anchor=[1.3,0.8])
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            sns.despine(top=True, right=True)
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            fig, ax = plt.subplots(figsize=(6,3))
            g = sns.barplot(data=cluster_attributes, x='cluster', y='p', hue='attribute', palette='cividis', dodge=True)
            g.set_xlabel('Cluster Index')
            g.set_ylim([0,1])
            g.set_ylabel('Fraction of categories \n in cluster with attribute')
            g.legend(title='attribute',  loc='center right', bbox_to_anchor=[1.3,0.8])
            g.set_axisbelow(True)
            # g.grid(False)
    #         plt.tight_layout()
            sns.despine(top=True, right=True)
            if save:
                fig.savefig(f'{out_dir}/readout_kmeans_{domain}-{k_clusters}-clusters_attributes_2.png', dpi=200, bbox_inches='tight')
            plt.show()


def plot_component_solution(components, loadings, expl_variance=None, n_ims=20, n_components=5, images=None, save_dir=None, 
                                contour_maps=None, contour_levels=[3],
                                scatter_pairs=[(1,2)], **scatter_kwargs):
    """
    plot a PCA solution on readout weights or activations from OFS trained models
    by default uses readout weights
    if images is specified, we use those instead of random images from the default OFS classes
    """

    for comp in range(n_components):
        component = loadings[:,comp]

        g = sns.heatmap(components[comp,:].reshape(32,32), center=0, robust=True, cmap='RdBu_r', square=True, cbar=False)
        if contour_maps is not None and len(contour_maps) > 0:
            styles = ['dotted', 'dashed', 'dashdot']
            for ii, contour_map in enumerate(contour_maps):
                g.contour(np.arange(32), np.arange(32), contour_map , contour_levels, cmap='bone', linestyles=styles[ii], linewidths=2); 

        g.set_yticklabels([])
        g.set_xticklabels([])
    #     sns.heatmap(components[comp,:].reshape(32,32), ax=axs[comp], vmin=0, robust=False, cmap='cmr.rainforest', square=True)
        if expl_variance is None:
            g.set_title(f'component {comp+1}')
        else:
            g.set_title(f'component {comp+1}, var ex.={expl_variance[comp]:.05f}')
        if save_dir is not None:
            start = timeit.timeit()
            plt.savefig(f'{save_dir}/component-{comp+1}_weights.png', bbox_inches='tight', dpi=300)
            end = timeit.timeit()
            print(f'saving fig took {start-end} s')
        plt.show()

        if n_ims is not None:
            sorted_components = np.sort(component)
            sorted_inds = np.argsort(component)
        #     inds = sorted_inds[np.linspace(0,299, num=n_ims, dtype=int)]
            inds = np.concatenate((sorted_inds[:n_ims//2], sorted_inds[-n_ims//2:]))
            subset_comps = component[inds]
            these_images = []
            if images is None:
                for ind in inds:
                    file = get_miniOFS_img_from_categ(ind, im_idx=0)
                    these_images.append(Image.open(file).resize((100,100)))
            else:
                for ind in inds:
                    if type(images) is list:
                        these_images.append(Image.open(images[ind]).resize((100,100)))
                    else:
                        these_images.append(cv2.resize(images[ind],(100,100)))
            for rank, rank_inds in {'highest': np.arange(n_ims//2,n_ims), 'lowest':np.arange(n_ims//2)}.items():
                # these_images[n_ims//2:], 'lowest':these_images[:n_ims//2]}.items():
                ranked_ims = [these_images[r] for r in rank_inds]
                fig = plt.figure(figsize=(3*(n_ims/2), 3))
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                nrows_ncols=(1, n_ims//2), 
                                axes_pad=0,  # pad between axes in inch.
                                )
                for ii, (ax, im) in enumerate(zip(grid, ranked_ims)):
                    # Iterating over the grid returns the Axes.
                    if type(im) == Image.Image:
                        im = np.array(im)
                    if len(im.shape) == 2:
                        ax.imshow(im, cmap='Greys_r')
                    else:
                        ax.imshow(im)
                    ax.set_title(subset_comps[rank_inds[ii]])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.grid(False)
                for jj in range(ii+1, len(grid)):
                    fig.delaxes(grid[jj])
                print(f'component {comp+1}, {rank} scoring category exemplars')
                if save_dir is not None:
                    fig.savefig(f'{save_dir}/component-{comp+1}_{rank}_examples.png', bbox_inches='tight', dpi=300)
                plt.show()

    # lastly just conveniently call plot_pca_scatter for requested pairs of PCs
    if images is None:
        images = []
        for ind in range(300):
            images.append(get_miniOFS_img_from_categ(ind, im_idx=0))
    for x_pc, y_pc in scatter_pairs:
        plot_pca_scatter(loadings, images, x_pc=x_pc, y_pc=y_pc, save_dir=save_dir, **scatter_kwargs)

def get_miniOFS_img_from_categ(categ_idx, im_idx=0, im_dim=IMS_DIR):
    """
    helper function to retrieve path of image
    """
    orig_classes = {
    'object': sorted([os.path.basename(categ) for categ in glob.glob(f'{IMS_DIR}/100objects/train/*')]),
    'face': sorted([os.path.basename(categ) for categ in glob.glob(f'{IMS_DIR}/100faces-crop-2.0/train/*')]),
    'scene': sorted([os.path.basename(categ) for categ in glob.glob(f'{IMS_DIR}/100scenes/train/*')]),
    }
    if categ_idx < 100:
        domain = 'object'
    elif categ_idx < 200:
        domain = 'face'
        categ_idx = categ_idx-100
    else:
        domain = 'scene'
        categ_idx = categ_idx-200
    fold = f'100{domain}s-crop-2.0' if domain == 'face' else f'100{domain}s' 

    files = glob.glob(os.path.join(f'{IMS_DIR}/{fold}', 'val', orig_classes[domain][categ_idx], '*'))
    return files[im_idx]
            

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    # diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im
    
def plot_pca_scatter(loadings, ims, x_pc=1, y_pc=2, num_ims=100, figsize=15, imsize=80, seed=None, 
                        s=100, colors=None, markers=None, edgecolors='k', alpha=1, cmap=None, crop=True, 
                        legend_title='category',
                        save_dir=None,
                        clf_for_hyperplane=None,
                        ):
    if seed is not None:
        np.random.seed(seed)
    if num_ims is not None:
        inds = np.random.choice(loadings.shape[0], num_ims)
    else:
        inds = np.arange(loadings.shape[0])
    x = loadings[inds,x_pc-1]
    y = loadings[inds,y_pc-1]
    if colors is not None:
        colors = [colors[ind] for ind in inds]
    if type(edgecolors) == list or type(edgecolors) == np.array:
        print('is array')
        edgecolors = [edgecolors[ind] for ind in inds]

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    if markers is not None:
        for marker in np.unique(markers):
            mark_mask = (markers == marker)
            ax.scatter(x[mark_mask], y[mark_mask], facecolors=colors[mark_mask], edgecolors=edgecolors[mark_mask], s=s, marker=marker, alpha=alpha) 
    else:
        ax.scatter(x, y, facecolors=colors, edgecolors=edgecolors, s=s, alpha=alpha) 
    ax.set_xlabel(f'PC {x_pc}')
    ax.set_ylabel(f'PC {y_pc}')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.axvline(x=0, color='r', linestyle='--')

    if clf_for_hyperplane is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf_for_hyperplane.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linewidths=[3],
                linestyles=['--'])

    if colors is None:
        rand_ims = [ims[ind] for ind in inds]
        for x0, y0, im in zip(x, y, rand_ims):
            if 'str' in str(type(im)):
                img = Image.open(im)
            else:
                img = Image.fromarray(im)
            if crop:
                img = trim(img)
            img = img.resize((imsize, imsize))
            ab = AnnotationBbox(OffsetImage(img, cmap='Greys_r'), (x0, y0), frameon=False)
            ax.add_artist(ab)
    elif cmap is not None:
        patches = [ mpatches.Patch(color=color, label=domain) for domain, color in cmap.items()]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, borderaxespad=0., title=legend_title) # bbox_to_anchor=(1.05, 0.8), loc=2,

    if save_dir is not None:
        tag = 'colors' if colors is not None else 'images'
        plt.savefig(f'{save_dir}/pc-{x_pc}_vs_pc-{y_pc}_scatter_{tag}.png', bbox_inches='tight', dpi=300)


def PCA_pipeline(base_fn, layers=['aIT'], do_readout_weights=False, do_OFS_acts=True, do_bao_tsao=True, show=False,
                    opt=None,
                    arch_kwargs=None,
                    model=None,
                    res=None,
                    acc=None,
                    acts=None,
                    d_trials=None,
                    sl_accs=None,
                ):
    """
    perform principal components analysis on topographic layers, using standard stimuli along with those from Bao and Tsao. addresses generic organization.
    """
    # check to see if we are providing these to save time
    if np.sum([val == None for val in [opt,arch_kwargs,model,res,acc,acts,d_trials,sl_accs]]) > 0:
        base_fn, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = load_for_tests(base_fn)

    bao_acts, bao_info = load_batched_activations(base_fn, ['pIT', 'cIT', 'aIT'],
                             batch_size=128,
                             imageset='baotsao',
                             phase='val',        
                             model=model,
                             input_times=10,
                             total_times=10,
                             overwrite=True,
                                     )

    if do_readout_weights:
        readout_weights = model.decoder.weight.detach().cpu().numpy()
        pca_sol = PCA(n_components=None).fit(readout_weights)
        loadings = pca_sol.transform(readout_weights)
        components = pca_sol.components_
        expl_variance = pca_sol.explained_variance_ratio_
        cum_var = [np.sum(expl_variance[:ii]) for ii in range(len(expl_variance))]
        plt.plot(cum_var)
        plt.xlabel('# PCs')
        plt.ylabel('Cumulative variance explained')
        if show:
            plt.show()
        plt.close()
        save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/readout_weight_PCs'
        os.makedirs(save_dir, exist_ok=True)
        plot_component_solution(components, loadings, expl_variance=expl_variance, save_dir=save_dir, n_ims=20, n_components=2, figsize=12, imsize=50, scatter_pairs=[(1,2)])

    if do_OFS_acts:
        for layer in layers:
            these_acts = acts[layer][:,0,:1024]
            ignore_nontrained_domains = False

            if ignore_nontrained_domains:
                files = [item[0] for item in np.concatenate([get_dataset(dataset_name+'_10-ims-per-class', 'val')[0].samples for dataset_name in opt['datasets']])]
                if 'miniOFS' not in base_fn:
                    these_acts = these_acts[d_trials['object']]
                use_colors = [None]
            else:
                files = [item[0] for item in np.concatenate([get_dataset(dataset_name+'_10-ims-per-class', 'val')[0].samples for dataset_name in ['100objects', '100faces-crop-2.0', '100scenes']])]
                colors = []
                cmap = {'face':'r', 'object': 'g', 'scene':'b'}
                for ii in range(len(files)):
                    colors.append(cmap['face'] if '100faces' in files[ii] else cmap['object'] if '100objects' in files[ii] else cmap['scene'])
                use_colors = [None, colors]

            pca_sol = PCA(n_components=None).fit(these_acts)
            loadings = pca_sol.transform(these_acts)
            components = pca_sol.components_
            expl_variance = pca_sol.explained_variance_ratio_
            cum_var = [np.sum(expl_variance[:ii]) for ii in range(len(expl_variance))]

            if show:
                fig, ax = plt.subplots()
                plt.plot(cum_var)
                plt.xlabel('# PCs')
                plt.ylabel('Cumulative variance explained')
                plt.show()
                plt.close()

            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs'
            os.makedirs(save_dir, exist_ok=True)

            plot_component_solution(components, loadings, expl_variance=expl_variance, n_ims=20, n_components=2, num_ims=200, images=files, save_dir=save_dir, scatter_pairs=[])
            #                         colors=color, save_dir=save_dir, images=files, figsize=12, imsize=35, scatter_pairs=[(1,2)])
            for color in use_colors:
                plot_pca_scatter(loadings, files, num_ims=200, save_dir=save_dir, figsize=10 if color is None else 6, 
                                imsize=35, colors=color, cmap=cmap)
                if show:
                    plt.show()
                plt.close()

    if do_bao_tsao:
        for layer in layers:
            baotsao = BaoTsaoDataset()
            #-----------------project bao tsao into OFS-------------------------------------

            these_acts = acts[layer][:,0,:1024]
                
            bao_trials = np.argwhere(bao_info['orientation'] == 0).squeeze()
            # bao_trials = np.arange(len(bao_info['orientation']))

            pca_sol = PCA(n_components=None).fit(these_acts)
            loadings = pca_sol.transform(bao_acts[layer][:,0,:1024][bao_trials])
            components = pca_sol.components_
            expl_variance = pca_sol.explained_variance_ratio_
            cum_var = [np.sum(expl_variance[:ii]) for ii in range(len(expl_variance))]


            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs_OFS-to-baotsao'
            os.makedirs(save_dir, exist_ok=True)

            colors = []
            cmap = {'face':'r', 'animal':'green', 'manmade object': 'purple', 'veg/fruit':'orange', 'vehicle': 'blue', 'house':'cyan' }
            colors = np.array([cmap[categ] for categ in baotsao.category])

            # plot_component_solution(components, loadings, expl_variance=expl_variance, n_ims=20, n_components=2, num_ims=200, save_dir=save_dir, 
            #                        scatter_pairs=None)
            # plt.show()

            for use_colors in [None, colors[bao_trials]]:
                plot_pca_scatter(loadings, baotsao.samples[bao_trials], num_ims=200, save_dir=save_dir, figsize=10 if use_colors is not None else 6, imsize=35, colors=use_colors)
                if show:
                    plt.show()
                plt.close()

            #--------------------color projections by baotsao K=4-means clustering solution---------------------------

            bao_trials = np.argwhere(bao_info['orientation'] == 0).squeeze()
            # bao_trials = np.arange(len(bao_info['orientation']))

            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs_OFS-to-baotsao_kmeans'
            os.makedirs(save_dir, exist_ok=True)

            pca_sol = PCA(n_components=None).fit(these_acts)
            loadings = pca_sol.transform(bao_acts[layer][:,0,:1024][bao_trials])
            kmeans_labels = KMeans(4, random_state=1).fit(bao_acts[layer][:,0,:1024]).predict(bao_acts[layer][:,0,:1024][bao_trials])
            cmap = ['red', 'green', 'cyan', 'purple']
            kmeans_colors = [cmap[label] for label in kmeans_labels]

            plot_pca_scatter(loadings, baotsao.samples[bao_trials], num_ims=200, save_dir=save_dir, figsize=6, imsize=35, colors=kmeans_colors)
            if show:
                plt.show()
            plt.close()

            #---------------PCA analysis purely on BaoTsao-----------------------------------------------------------
            trials = np.argwhere(bao_info['orientation'] == 0).squeeze()
        #     trials = np.arange(len(bao_info['orientation']))
            these_acts = bao_acts[layer].squeeze()[:,:1024]
            pca_sol = PCA(n_components=5).fit(these_acts)
            loadings = pca_sol.transform(these_acts[trials])
            components = pca_sol.components_
            expl_variance = pca_sol.explained_variance_ratio_
            
            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs_baotsao'
            os.makedirs(save_dir, exist_ok=True)

            plot_component_solution(components, loadings, expl_variance=expl_variance, n_ims=20, save_dir=save_dir, images=baotsao.samples[trials], scatter_pairs=[], n_components=2)
        #                             n_components=2, figsize=12, imsize=50, num_ims=200, )
            if show:
                plt.show()
            plt.close()

            colors = []
            cmap = {'face':'r', 'animal':'green', 'manmade object': 'purple', 'veg/fruit':'orange', 'vehicle': 'blue', 'house':'cyan' }
            colors = np.array([cmap[categ] for categ in baotsao.category])
            trials = np.argwhere(bao_info['orientation'] == 0).squeeze()
            # trials = np.arange(len(bao_info['orientation']))
            loadings = pca_sol.transform(these_acts[trials])
            ims = baotsao.samples[trials]
            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs_baotsao'
            os.makedirs(save_dir, exist_ok=True)

            for use_colors in [None, colors[trials]]:
                plot_pca_scatter(loadings, ims, num_ims=200, save_dir=save_dir, figsize=9, imsize=35, colors=use_colors, cmap=cmap if use_colors is not None else None)

            #--------------PCA analysis colored by K-means +  cluster selectivity maps---------------------------------------------------------

            # bao_trials = np.argwhere(bao_info['orientation'] == 0).squeeze()
            bao_trials = np.arange(len(bao_info['orientation']))

            save_dir = f'{FIGS_DIR}/topographic/EI/summaries/{base_fn}/layer-{layer}_acts_PCs_baotsao_kmeans'
            os.makedirs(save_dir, exist_ok=True)

            pca_sol = PCA(n_components=None).fit(these_acts)
            loadings = pca_sol.transform(bao_acts[layer][:,0,:1024][bao_trials])
            kmeans_labels = KMeans(4, random_state=1).fit(bao_acts[layer][:,0,:1024]).predict(bao_acts[layer][:,0,:1024][bao_trials])
            cmap = ['red', 'green', 'cyan', 'purple']
            kmeans_colors = [cmap[label] for label in kmeans_labels]

            plot_pca_scatter(loadings, baotsao.samples[bao_trials], num_ims=200, save_dir=save_dir, figsize=6, imsize=35, colors=kmeans_colors)
            if show:
                plt.show()
            plt.close()

            selectivities = []
            for label in np.unique(kmeans_labels):
                trials = [l == label for l in kmeans_labels]
                other_trials = [l != label for l in kmeans_labels]
                t, p = ttest_ind(these_acts[bao_trials][trials], these_acts[bao_trials][other_trials])
                selectivity = -np.sign(t)*np.log10(p)
                selectivity[np.isnan(selectivity)] = 0
                selectivities.append(selectivity)
            #     sns.heatmap(these_acts[bao_trials][trials].mean(0).reshape(32,32), cmap='cmr.rainforest', vmin=0)
            #     plt.show()
                sns.heatmap(selectivity.reshape(32,32), cmap='cmr.rainforest', vmin=0, vmax=15, square=True)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'{save_dir}/cluster-{label}_selectivity.png', dpi=300, bbox_inches='tight')
                if show:
                    plt.show()
                plt.close()
                
            max_label = np.argmax(selectivities, 0)
            sns.heatmap(max_label.reshape(32,32), cmap=cmap, vmin=0, vmax=4, square=True, cbar=False)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{save_dir}/cluster_selectivity.png', dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()

           


if __name__ == "__main__":
    # compare_jj_lambdas(seed=2, mod_id='jjd2', mod_type='E/I REC', wrap=False, lambdas=[0, 0.01, 0.1, 0.5, 1.0, 10.0], do_domains=False, do_generic=False, do_summary=True)
    # ei_ablations(seed=2, jj=True, wrap=False, overwrite_res=False, plot_i_cells=True, do_domains=False, do_generic=False, do_summary=True)
    from topographic.utils.training import exp_times
    from topographic.results import NAMED_MODELS

    # for key in ['main_model_objects_to_ofs']: # ['resnet50_jj_ei4_rv4_nowrap_d2', 'resnet50_jj_ei4_rv4_nowrap_d2_srneff', 'jjd2_norec']: # ['resnet50_jj_ei4_rv4']: #, 'resnet50_circ2_ei4_norv4', 'resnet50_gaussian_ei4_rv4']:
    #     for rs in [2]: #range(2,5):
    #         base_fn = NAMED_MODELS[key]
    #         base_fn = base_fn.replace('_rs-1_', f'_rs-{rs}_')
    #     # try:
    #         kwargs = unparse_model_args(base_fn, use_short_names=True)
    #         timing = exp_times[kwargs['timing']]
    #         # results = run_wiring_cost_experiment(base_fn, 
    #         #                             sparsity_vals=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    #         #                             analysis_times=[timing.stim_off, timing.exp_end, timing.exp_end+timing.jitter],
    #         # )
    #         # pdb.set_trace()

    #         base_fn, opt, arch_kwargs, model, res, acc, acts, d_trials, sl_accs = load_for_tests(base_fn)
    #         for lesion_type in ['metric_pval']: #['topographic', 'metric']:
    #             if 'pval' in lesion_type:
    #                 lesion_ps = [2,3,4,5,6,10,14,None,0,1]
    #             else:
    #                 lesion_ps = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    #             run_lesion_experiment(model, res, opt, arch_kwargs, base_fn, 
    #                         lesion_type=lesion_type,
    #                         overwrite=True,
    #                         overwrite_indiv_results=True,
    #                         selectivity_t=timing.exp_end,
    #                         analysis_times=[timing.stim_off, timing.exp_end, timing.exp_end+timing.jitter],
    #                         lesion_ps=lesion_ps,
    #                         )
    #     # except Exception as e:
    #     #     print(e)

    import matplotlib.pyplot as plt
    # for mod_type in ['E/I', 'E/I REC', 'SRN', 'REC', 'FF']: #, 'E/I REC']:
    #     plt.close('all')
    #     models = compare_jj_lambdas(seed=1, mod_id='jjd2', mod_type=mod_type, wrap=False, lambdas=[0, 0.01, 0.1, 0.5, 1.0, 10.0], 
    # #                                 do_generic=True, do_domains=True, do_combined=True, do_summary=True,
    #                                 do_performance=True, do_generic=True, do_domains=True, do_combined=True, do_summary=True, show=False,
    #                             )
        # plt.show()

    for mod_name in ['main_model', 'main_model_objects_only_1']:
        base_fn = NAMED_MODELS[mod_name]
        PCA_pipeline(base_fn, show=False)