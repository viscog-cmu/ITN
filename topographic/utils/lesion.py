
import torch.nn.utils.prune as prune
import numpy as np
import torch

from topographic.utils.commons import reproducible_results

class TopographicLesion(prune.BasePruningMethod):
    """
    Prune all connections from a neighborhood of topographically contiguous units
    """
    
    PRUNING_TYPE = 'structured' # we are pruning entire features

    def __init__(self, p, dists, center_idx=None, EI=False, lesion_I=True, lesion_E=True):
        """
        dists: the pairwise unit distances used to compute the lesion
        """
        self.p = p
        self.dists = dists
        self.center_idx = center_idx
        self.EI = EI
        self.lesion_I = lesion_I
        self.lesion_E = lesion_E
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        lesioned = get_circular_neighborhood(self.p, self.dists, self.center_idx, 
                                                EI=self.EI,
                                                include_I=self.lesion_I, 
                                                include_E=self.lesion_E,
                                                )
        mask[lesioned,:] = 0
        return mask

class MetricOrderedLesion(prune.BasePruningMethod):
    """
    Prune all connections from a pre-determined size set of units chosen to maximize a provided metric, i.e. (smoothed) selectivity
    """

    PRUNING_TYPE = 'structured' # we are pruning entire features

    def __init__(self, p, metric, EI=False, lesion_E=True, lesion_I=True):
        self.p = p
        self.metric = metric
        self.EI = EI
        self.lesion_E = lesion_E
        self.lesion_I = lesion_I

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        lesioned = get_metric_ordered_region(self.p, self.metric, EI=self.EI, include_I=self.lesion_I, include_E=self.lesion_E)
        mask[lesioned,:] = 0
        return mask

class MetricThresholdedLesion(prune.BasePruningMethod):
    """
    Prune all connections from a set of units with a metric value greater than some threshold, i.e. smoothed selectivity
    """
    PRUNING_TYPE = 'structured' # we are pruning entire features

    def __init__(self, threshold, metric, EI=False, lesion_E=True, lesion_I=True):
        self.threshold = threshold
        self.metric = metric
        self.EI = EI
        self.lesion_E = lesion_E
        self.lesion_I = lesion_I

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        lesioned = get_metric_thresholded_region(self.threshold, self.metric, EI=self.EI, include_I=self.lesion_I, include_E=self.lesion_E)
        mask[lesioned,:] = 0
        return mask


def get_metric_ordered_region(p, metric, EI=False, include_E=True, include_I=True):
    """
    helper function to compute a region chosen to maximize a provided metric, ie (smoothed) selectivity

    if network is EI, can be used to lesion EI columns. Only takes in one metric for either E or I units,
    so lesioning of E and I units together must be done as columns. 
    """
    ranked_units = np.argsort(-metric.reshape(-1)) # sorting negative returns in decreasing order of original metric
    region = ranked_units[:int(p*len(ranked_units))]
    if EI:
        if include_E and include_I:
            region = np.concatenate((region, len(ranked_units)+region),0)
        elif include_I:
            region = len(ranked_units)+region
    return region

def get_metric_thresholded_region(threshold, metric, EI=False, include_E=True, include_I=True):
    if threshold is None:
        return []
    metric = metric.reshape(-1)
    region = np.where(metric > threshold)[0]
    if EI:
        if include_E and include_I:
            region = np.concatenate((region, len(metric)+region),0)
        elif include_I:
            region = len(metric)+region
    return region

def get_circular_neighborhood(p, dists, center_idx=None, EI=False, include_E=True, include_I=False):
    """
    helper function to compute a topographically defined lesion or searchlight
    with inclusion probability p, in a map with distances defined by dist
    Inputs: 
        p: probability of inclusion
        dists: pairwise unit distance matrix
        center_idx: index of unit at center of neighborhood (optional)
        EI: whether the network has separate excitatory and inhibitory units
        include_I: only relevant if EI==True; whether to include I units in neighborhood
    Outputs:
        lesioned: an array of indexes into lesioned units
    """
    if EI:
        # we assume the dists are identical for symmetric E and I maps
        dists = dists[:dists.shape[0]//2,:dists.shape[1]//2]
        assert include_E or include_I
    indices = np.arange(len(dists)).reshape(int(np.sqrt(len(dists))),int(np.sqrt(len(dists))))
    if center_idx is None:
        x, y = np.random.randint(0, len(indices)-1), np.random.randint(0, len(indices)-1)
        center_idx = indices[x,y]
    sorted_dist_inds = torch.argsort(dists[center_idx,:])
    neighborhood = sorted_dist_inds[0:int(np.round(p*len(sorted_dist_inds)))]
    if include_E and include_I:
        neighborhood = torch.cat((neighborhood, sorted_dist_inds.shape[0]+neighborhood))
    elif include_I:
        neighborhood = sorted_dist_inds.shape[0]+neighborhood
    return neighborhood


def it_lesion(it_area, p, topographic, seed=None, metric=None, metric_type='rank', **topo_kwargs):
    """
    performs topographic or random lesioning of an ITArea layer
    lesioning is of specific units, and is implementing through pruning all connections into
    some set of units, along with the biases of those units. 
    regardless of the cell type (SRN, LSTM, GRU), the lesion occurs to all units at the specified
    location, across all neuron types (main, context, gates, etc.)
    """
    if not it_area.custom_cell:
        raise NotImplementedError()

    if seed is not None:
        reproducible_results(seed)
    if topographic:
        if seed is None or seed > it_area._h_sz or seed < 0:
            raise ValueError('seed must specify a valid unit idx for topographic lesion')
        TopographicLesion.apply(it_area.cell, 'weight_hh', p, it_area.rec_distances, center_idx=seed, **topo_kwargs)
    elif metric is not None:
        if metric_type == 'threshold':
            MetricThresholdedLesion.apply(it_area.cell, 'weight_hh', p, metric, **topo_kwargs)
        elif metric_type == 'rank':
            MetricOrderedLesion.apply(it_area.cell, 'weight_hh', p, metric, **topo_kwargs)
        else:
            raise ValueError()
    else:
        prune.random_structured(it_area.cell, 'weight_hh', amount=p, dim=0)

    mask = it_area.cell.weight_hh_mask
    good_units = torch.unique(torch.nonzero(mask)[:,0])
    bias_mask = torch.zeros((mask.shape[0]), device=mask.device)
    bias_mask[good_units] = 1
    map_mask = torch.zeros_like(it_area.cell.weight_ih)
    map_mask[good_units,:] = 1
    
    prune.custom_from_mask(it_area.cell, 'weight_ih', map_mask)
    prune.custom_from_mask(it_area.cell, 'bias', bias_mask)
    
    
def corblockz_lesion(conv_layer, p, seed=None):
    """
    random lesion of a CORBlockZ layer
    """
    if seed is not None:
        reproducible_results(seed)
    prune.random_structured(conv_layer.conv, 'weight', amount=p, dim=0)
    mask = conv_layer.conv.weight_mask
    good_units = torch.unique(torch.nonzero(mask)[:,0])
    bias_mask = torch.zeros((mask.shape[0]))
    bias_mask[good_units] = 1
    prune.custom_from_mask(conv_layer.conv, 'bias', bias_mask)