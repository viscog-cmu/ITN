
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from torch._six import inf
import scipy.spatial
import numpy as np
import sys
import pdb
from easydict import EasyDict as edict

from topographic.dl.cornet_rt import Identity
from topographic.utils.commons import reproducible_results


def get_conv_spatial_coords(side_length, channels, z_val=None):
    inds = np.tile(np.expand_dims(np.indices((side_length,side_length)), axis=1), (1,channels,1,1))
    coords = inds.reshape(2,-1).transpose()/(side_length)
    if z_val is not None:
        # add a Z coordinate of the provided value. ie, all neurons in a layer have the same z value.
        z_coord = z_val*np.ones((coords.shape[0],1))
        coords = np.hstack((coords, z_coord))
    return coords

def set_conv_spatial_coords(conv_layer, *args, **kwargs):
    coords = get_conv_spatial_coords(*args, **kwargs)
    conv_layer.coords = coords


def clip_grad_norm_hook_fn(grads, max_norm, norm_type=2):
    """
    
    A minor modification of torch.nn.utils.clip_grad_norm_ to use as a hook fn for the backward pass
    
    This allows the norm of gradients to be fixed for each time step in an RNN
    
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads.device
    if norm_type == inf:
        total_norm = grads.detach().abs().max().to(device)
    else:
        total_norm = torch.norm(grads.detach(), norm_type).to(device)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = grads.detach().mul_(clip_coef.to(grads.device))
    return grads
    
class ITArea(nn.Module):
    """
    A generalized Interactive Topography area 
    """
    def __init__(self, input_size, hidden_size, 
                 input_coords=None, 
                 recurrent_cell='RNN', 
                 noise='uniform',
                 connectivity='full', 
                 grad_clip_val=None,
                 no_recurrence=False,
                 use_dropout=False,
                 dropcon_inputs=False,
                 wrap_x=True, wrap_y=True, 
                 cell_kwargs={}, noise_kwargs={}, connectivity_kwargs={}, 
                 device='cuda',
                 rs=None,
                ):
        super().__init__()
        map_side = int(np.sqrt(hidden_size))
        # for gated cells, each weight matrix contains a horiz concatenation of different weight matrices (same cells, different connection types)
        # for ei cells, each weight matrix contains a vertical concatenation (different cells and connection types)
        if recurrent_cell == 'LSTM':
            self._weight_rep = (4,1) 
        elif recurrent_cell == 'GRU':
            self._weight_rep = (3,1)
        elif recurrent_cell == 'RNN':
            self._weight_rep = (1,1)
        elif 'EI' in recurrent_cell :
            self._weight_rep = (2,2)
        elif 'SRN' in recurrent_cell:
            self._weight_rep = (1,1)
        elif recurrent_cell in ['GM', 'FF', 'EFF', 'EFFLN', 'FFLN']:
            self._weight_rep = (1,1)
        else:
            raise NotImplementedError()

        if np.abs(np.sqrt(hidden_size) - map_side) > 0:
            raise ValueError('hidden size must be a perfect square')
        if no_recurrence and ('SRN' not in recurrent_cell and 'EI' not in recurrent_cell):
            raise NotImplementedError()
        
        if 'SRN' in recurrent_cell or 'EI' in recurrent_cell or recurrent_cell in ['GM', 'FF', 'EFF', 'EFFLN', 'FFLN']:
            self.custom_cell = True
            if 'EI' in recurrent_cell:
                cell_kwargs['device'] = device
        else:
            raise ValueError(f'{recurrent_cell} has been deprecated')
            self.custom_cell = False

        if dropcon_inputs:
            recurrent_cell = 'WeightDrop' + recurrent_cell
            cell_kwargs['weight_dropout'] = dropcon_inputs
        else:
            if 'weight_dropout' in cell_kwargs:
                cell_kwargs.pop('weight_dropout')

        if self.custom_cell:
            Cell = getattr(sys.modules[__name__], f'{recurrent_cell}Cell')
        else:
            Cell = getattr(nn, f'{recurrent_cell}Cell')
        self.cell = Cell(input_size, hidden_size, **cell_kwargs).to(device)
        self.connectivity = connectivity
        self.noise = noise
        self.grad_clip_val = grad_clip_val
        self.device = device
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.no_recurrence = no_recurrence
        self.input_coords = input_coords
        self.coords = (np.indices((map_side,map_side)).reshape(2,map_side**2,)).transpose()/(map_side)    
        self._i_sz = input_size
        self._h_sz = hidden_size
        self.map_side = map_side
        self.recurrent_cell = recurrent_cell
        self.rs = rs
            
        self._set_connection_distances(wrap_x, wrap_y, device=device)
        self._setup_connectivity_constraint(connectivity, **connectivity_kwargs)
        self._setup_noise_constraint(noise, **noise_kwargs)
        self._dropout = nn.Dropout(p=.2) if use_dropout else None
                
    def forward(self, inputs, state, preact_state, noise_off=False):
        """
        see https://discuss.pytorch.org/t/backpropagating-through-noise/27000/3
        """
        if noise_off:
            ih_noise, hh_noise = 0, 0
        else:
            ih_noise = self.ff_noise.sample((self._weight_rep[0]*self._h_sz, self._i_sz)).squeeze(-1)
            ih_noise = ih_noise*self.ff_noise_scaling
            hh_noise = self.rec_noise.sample((self._weight_rep[0]*self._h_sz, self._weight_rep[1]*self._h_sz)).squeeze(-1)
            hh_noise = hh_noise*self.rec_noise_scaling
            if self.squash_noise:
                # squash noise to range [-1,1] so that noisy weight is in range [0,2]*weight
                ih_noise = 2*torch.sigmoid(ih_noise)-1
                hh_noise = 2*torch.sigmoid(hh_noise)-1
            elif self.threshold_noise:
                # threshold noise -1 so that noisy weight is in range [0,inf]*weight
                ih_noise = torch.clamp(ih_noise, min=-1)
                hh_noise = torch.clamp(hh_noise, min=-1)

        if self.custom_cell:
            outputs = self.cell(inputs, state, preact_state,
                                ih_noise=self.cell.weight_ih_raw*ih_noise if hasattr(self.cell, 'weight_ih_raw') else self.cell.weight_ih*ih_noise, 
                                hh_noise=self.cell.weight_hh*hh_noise if hasattr(self.cell, 'weight_hh') else None, 
                                ih_mask=self.ff_connection_mask,
                                hh_mask=self.rec_connection_mask if hasattr(self.cell, 'weight_hh') else None,
                                no_recurrence=self.no_recurrence,
                                )
        else:
            # enforce connectivity constraint
            self.cell.weight_ih.data = self.cell.weight_ih.data*self.ff_connection_mask
            self.cell.weight_hh.data = self.cell.weight_hh.data*self.rec_connection_mask
            # enforce noise constraint
            self.orig_ih = self.cell.weight_ih.data.clone()
            self.orig_hh = self.cell.weight_hh.data.clone()
            self.cell.weight_ih.data += self.cell.weight_ih*ih_noise
            self.cell.weight_hh.data += self.cell.weight_hh*hh_noise
            
            outputs = self.cell(inputs, state, preact_state)

        if len(outputs) != 3:
            raise ValueError('outputs of cell should be a tuple of length 3: (outputs, state, preact_state)')

        # clip the values of gradients for each time step and optionally apply dropout
        new_outputs = []
        for ii, out in enumerate(outputs):
            if out.requires_grad and self.grad_clip_val is not None:
#                     out.register_hook(lambda x: x.clamp(min=-self.grad_clip_val, max=self.grad_clip_val) if x is not None else x)
                out.register_hook(lambda x: clip_grad_norm_hook_fn(x, self.grad_clip_val, 2))
            if self._dropout is not None:
                out = self._dropout(out)
            new_outputs.append(out)
        outputs = tuple(new_outputs)
        
        if not self.custom_cell:
            self.uncorrupt_weights()

        return outputs

    def uncorrupt_weights(self):
        """
        should be called after loss.backward() but just before optimizer.step() to remove noise added to weights
        (see https://discuss.pytorch.org/t/backpropagating-through-noise/27000/3)
        
        well, we are not going to do this because we want the 'weights' to change on each time step. 
        we will call this at each time step. 
        
        otherwise, the noise just accumulates in the weights through training, which has a different physical interpretation
        than axonal/dendritic conduction noise
        
        """
        self.cell.weight_ih.data = self.orig_ih
        self.cell.weight_hh.data = self.orig_hh
        
        
    def _set_connection_distances(self, *args, **kwargs):
        """
        helper function to set the connection distances, utilizes _get_connection_distances
        """
        if self.input_coords is not None:
            ff_dists = self._get_connection_distances(self.input_coords, self.coords, *args, **kwargs)
            self.ff_distances = ff_dists.transpose(0,1).repeat(self._weight_rep[0], self._weight_rep[1])
            self.ff_distances = self.ff_distances[:,:self._i_sz] # _i_sz takes care of eliminating FF for I-units in EI cells (besides EI5 for which _i_sz is doubled)
        else:
            self.ff_distances = torch.zeros(self._weight_rep[0]*self._h_sz, self._i_sz).to(self.device) # should this be 1 or 0.5? 
        rec_dists = self._get_connection_distances(self.coords, self.coords, *args, **kwargs)
        self.rec_distances = rec_dists.transpose(0,1).repeat(self._weight_rep[0], self._weight_rep[1])


    def _get_connection_distances(self, from_coords, to_coords, wrap_x, wrap_y, norm=2, device='cuda'):
        """
        returns euclidean connection distances between from_coords and to_coords, subject to potential wrapping in x and y dimensions
        """
       # dimensions must be normalized to be in 0,1, but the max-min coord should be nonzero for correct wrapping
        dx = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,0],1), np.expand_dims(to_coords[:,0],1), p=1)
        dy = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,1],1), np.expand_dims(to_coords[:,1],1), p=1)
        if to_coords.shape[1] == 3:
            dz = scipy.spatial.distance_matrix(np.expand_dims(from_coords[:,2],1), np.expand_dims(to_coords[:,2],1), p=1)
        else:
            dz = 0
        if wrap_x or wrap_y:
            wrapx = dx > .5
            wrapy = dy > .5
            if wrap_x: # probably false for log polar mapping
                dx[wrapx] = 1 - dx[wrapx]
            if wrap_y:
                dy[wrapy] = 1 - dy[wrapy]
        D = (dx**norm + dy**norm + dz**norm)**(1/norm)

        return torch.FloatTensor(D).to(device)
        
    def _setup_connectivity_constraint(self, connectivity, **kwargs):
        """
        sets the parameters for connectivity constraints
        """
        setup_constraint = getattr(self, f'_setup_{connectivity}_connectivity')
        setup_constraint(**kwargs)
        
    def _setup_full_connectivity(self):
        """
        full connectivity
        """
        self.ff_connection_mask=1
        self.rec_connection_mask=1

    def _logits_to_mask(self, logits):
        """
        simple helper function to turn connection logits into samples from a probability distribution where the max logit corresponds to probability of 1
        """
        connection_density = torch.exp(logits)/(1+torch.exp(logits))
        connection_probs = connection_density/torch.max(connection_density)
        mask = Bernoulli(connection_probs).sample().to(self.device) 
        return mask

    def _dists_to_circular_mask(self, dists, fraction):
        """
        helper function to select fraction of units with the smallest distance values
        """
        device = dists.device
        shape = dists.shape
        dists_ = dists.clone().to('cpu').reshape(-1)
        mask = torch.zeros_like(dists_)
        k = int(fraction*len(dists_))
        inds = torch.argsort(dists_)[:k]
        mask[inds] = 1
        return mask.reshape(shape).to(device)

    def _dists_to_nosquish_circular_mask(self, dists, fraction):
        """
        helper function to select fraction of units with the smallest distance values
        we assume the total grid side length is 1 and analytically compute the RF radius r
        Connections with distance > r are masked out, and vice versa 
        let Ac be area of RF circle, As be area of square, and f be fraction to include for a perfect circle (center of grid)
        we have:
        Ac = f*As
        pi*r**2 = f*s**2 ; s = 1
        r = sqrt(f/pi)
        """
        dists_ = dists.clone()
        r = np.sqrt(fraction/np.pi)
        mask = dists_ <= r

        return mask

    def _setup_gaussian_connectivity(self, r_sigma=0.1, f_sigma=0.1, i2e_ratio=None, center_surround=None):
        """
        gaussian probability density inputs bernoulli disribution to determine binary connectivity mask
        *_sigma : width of gaussian
        i2e_ratio : how much wider to make the I than E RFs, if using separate populations. ratio > 1 -> (I > E)

        we always check to make sure self.ff_distances is not all 0s for the pIT non-retinotopic case which
        has no distance-dependence on its feedforward inputs
        
        """
        reproducible_results(self.rs)
        assert hasattr(self, 'ff_distances')
        assert hasattr(self, 'rec_distances')
        assert center_surround is None
        if i2e_ratio is None or i2e_ratio==1:
            #recurrent
            logits = Normal(torch.Tensor([0.0]).to(self.device), r_sigma).log_prob(self.rec_distances)
            self.rec_connection_mask = self._logits_to_mask(logits)  
            # feedforward
            if self.ff_distances.sum() > 0:
                logits = Normal(torch.Tensor([0.0]).to(self.device), f_sigma).log_prob(self.ff_distances)
                self.ff_connection_mask = self._logits_to_mask(logits)
            else:
                self.ff_connection_mask = 1
        else:
            #recurrent
            celltype_dists = self.rec_distances[:, :self._h_sz] # same for e and i
            e_logits = Normal(torch.Tensor([0.0]).to(self.device), r_sigma).log_prob(celltype_dists)
            i_logits = Normal(torch.Tensor([0.0]).to(self.device), i2e_ratio*r_sigma).log_prob(celltype_dists)
            e_mask = self._logits_to_mask(e_logits)
            i_mask = self._logits_to_mask(i_logits)
            self.rec_connection_mask = torch.cat((e_mask, i_mask), dim=1)
            # ff (all excitatory)
            if self.ff_distances.sum() > 0:
                logits = Normal(torch.Tensor([0.0]).to(self.device), f_sigma).log_prob(self.ff_distances)
                self.ff_connection_mask = self._logits_to_mask(logits)
            else:
                self.ff_connection_mask = 1

    def _setup_circ_connectivity(self, r_sigma=0.1, f_sigma=0.1, i2e_ratio=None, center_surround=False):
        """
        circular connectivity, where the wid values represent the fraction of units to include. 

        *_wid : fraction of units to include for each connection type
        i2e_ratio : how much wider to make the I than E RFs, if using separate populations. thus i_wid = wid * i2e_ratio, e_wid = wid
        center_surround : if True, wider field contains only the surround of the narrower field, else, just overlapping circles

        we always check to make sure f_sigma is not None for the pIT non-retinotopic case which
        has no distance-dependence on its feedforward inputs

        """
        assert hasattr(self, 'ff_distances')
        assert hasattr(self, 'rec_distances')
        if i2e_ratio is None or i2e_ratio==1:
            self.rec_connection_mask = self._dists_to_circular_mask(self.rec_distances, r_sigma)  
            if self.ff_distances.sum() > 0:
                self.ff_connection_mask = self._dists_to_circular_mask(self.ff_distances, f_sigma)  
            else:
                self.ff_connection_mask = 1
        else:
            # rec
            celltype_dists = self.rec_distances[:, :self._h_sz] # same for e and i
            e_mask = self._dists_to_circular_mask(celltype_dists, r_sigma)  
            i_mask = self._dists_to_circular_mask(celltype_dists, i2e_ratio*r_sigma)  
            if center_surround:
                assert i2e_ratio > 1 # I field must be wider than E field. Implausible and pointless otherwise.
                i_mask[e_mask.to(torch.bool)] = 0
            self.rec_connection_mask = torch.cat((e_mask, i_mask), dim=1)
            # ff (all excitatory)
            if self.ff_distances.sum() > 0:
                self.ff_connection_mask = self._dists_to_circular_mask(self.ff_distances, f_sigma)  
            else:
                self.ff_connection_mask = 1

    def _setup_circ2_connectivity(self, r_sigma=0.1, f_sigma=0.1, i2e_ratio=None, center_surround=False):
        """
        Same as circ connectivity but does not squash at boundaries in unwrapped models.
        Uses a central seed to determine the max distance from a given unit of other units in RF
        """
        assert hasattr(self, 'ff_distances')
        assert hasattr(self, 'rec_distances')
        if i2e_ratio is None or i2e_ratio==1:
            self.rec_connection_mask = self._dists_to_nosquish_circular_mask(self.rec_distances, r_sigma)  
            if self.ff_distances.sum() > 0:
                self.ff_connection_mask = self._dists_to_nosquish_circular_mask(self.ff_distances, f_sigma)  
            else:
                self.ff_connection_mask = 1
        else:
            # rec
            celltype_dists = self.rec_distances[:, :self._h_sz] # same for e and i
            e_mask = self._dists_to_nosquish_circular_mask(celltype_dists, r_sigma)  
            i_mask = self._dists_to_nosquish_circular_mask(celltype_dists, i2e_ratio*r_sigma)  
            if center_surround:
                assert i2e_ratio > 1 # I field must be wider than E field. Implausible and pointless otherwise.
                i_mask[e_mask.to(torch.bool)] = 0
            self.rec_connection_mask = torch.cat((e_mask, i_mask), dim=1)
            # ff (all excitatory)
            if self.ff_distances.sum() > 0:
                self.ff_connection_mask = self._dists_to_nosquish_circular_mask(self.ff_distances, f_sigma)  
            else:
                self.ff_connection_mask = 1
        
    def _setup_noise_constraint(self, noise, squash_noise=False, threshold_noise=False, **kwargs):
        """
        sets the parameters for noise constraints
        """
        assert hasattr(self, 'ff_distances')
        assert hasattr(self, 'rec_distances')
        setup_constraint = getattr(self, f'_setup_{noise}_noise')
        setup_constraint(**kwargs)
        if squash_noise and threshold_noise:
            raise ValueError('cannot squash and threshold noise')
        self.squash_noise = squash_noise
        self.threshold_noise = threshold_noise
        
    def _setup_uniform_noise(self, r_sigma=0.0, f_sigma=0.0):
        """
        non-distance-dependent gaussian noise for recurrent and feedforward connectivity
        """
        self.rec_noise = Normal(torch.Tensor([0.0]).to(self.device), r_sigma)
        self.ff_noise = Normal(torch.Tensor([0.0]).to(self.device), f_sigma)
        self.rec_noise_scaling = 1
        self.ff_noise_scaling = 1

    def _setup_linear_noise(self, r_sigma=0.0, f_sigma=0.0):
        """
        gaussian noise scaled linearly with connection distance
        """
        self.rec_noise = Normal(torch.Tensor([0.0]).to(self.device), r_sigma)
        self.ff_noise = Normal(torch.Tensor([0.0]).to(self.device), f_sigma)
        self.rec_noise_scaling = self.rec_distances
        if len(self.ff_distances) == 1:
            self.ff_noise_scaling = 1
        else:
            self.ff_noise_scaling = self.ff_distances
        
    def _setup_inverse_gaussian_noise(self, r_sigma=0.0, f_sigma=0.0, r_width=0.1, f_width=0.1):
        """
        gaussian noise scaled with an additional gaussian distribution over connection distance
        """
        raise NotImplementedError()
        

class SRNCell(nn.Module):
    """
    a simple recurrent cell (arguably fully recurrent depending on the timing) without sign constraints
    """
    def __init__(self, in_size, hid_size, nonlinearity='sigmoid', bias=True, alpha=0.2, id=None):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.weight_ih = Parameter(torch.Tensor(hid_size, in_size))
        self.weight_hh = Parameter(torch.Tensor(hid_size, hid_size))
        self.modid = id
        if 'noln' in self.modid:
            print('no layer norm')
            self.norm = Identity()
        else: 
            print('using layer norm')
            self.norm = nn.LayerNorm(self.hid_size, elementwise_affine=False)
        if bias:
            self.bias = Parameter(torch.Tensor(hid_size))
        else:
            self.register_parameter('bias', None)
        self.nonlin = get_nonlin(nonlinearity)

        # unitless time constant (dt/tau)
        self.alpha = alpha

        self.reset_parameters()
    
    def forward(self, inputs, state, preact_state,
                ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1, no_recurrence=False,
    ):
        if state == None:
            state = torch.zeros((inputs.shape[0], self.hid_size), device=inputs.device)
            preact_state = torch.zeros((inputs.shape[0], self.hid_size), device=inputs.device)

        bias = 0 if self.bias is None else self.bias

        ih_weight, hh_weight = self.get_total_weights(ih_noise, hh_noise, ih_mask, hh_mask)
        if no_recurrence:
            new_preact_state = self.norm(inputs.matmul(ih_weight.t())) + bias
        else:
            new_preact_state = self.norm(inputs.matmul(ih_weight.t()) + state.matmul(hh_weight.t())) + bias

        state = self.nonlin((1-self.alpha)*preact_state + self.alpha*new_preact_state)
        outputs = state

        return outputs, state, new_preact_state

    def reset_parameters(self):
        nn.init.normal_(self.weight_ih, 0, 0.01)
        nn.init.normal_(self.weight_hh, 0, 0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def get_total_weights(self, ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1):
        ih_weight =  (self.weight_ih+ih_noise)*ih_mask
        hh_weight =  (self.weight_hh+hh_noise)*hh_mask
        return ih_weight, hh_weight

class SRNEFFCell(SRNCell):
    """
    SRN cell with excitatory-only feedforward connections. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.enforce_connectivity(init=True)

    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight_ih.data = torch.abs(self.weight_ih.data)
        else:
            self.weight_ih.data = torch.clamp(self.weight_ih.data, min=0)    

    def get_total_weights(self, ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1):
        """
        combine learned weight, random noise, connection masking, and fixed EI mask into a single weight for IH and HH
        """
        ih_weight = torch.clamp(ih_noise + self.weight_ih, min=0)*ih_mask
        hh_weight = (hh_noise + self.weight_hh)*hh_mask

        return (ih_weight, hh_weight)    

class SRNECell(SRNEFFCell):
    """
    SRNCell with excitatory-only recurrent and feedforward connections (inhibition mediated stricted through layer normalization)
    """
    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight_ih.data = torch.abs(self.weight_ih.data)
            self.weight_hh.data = torch.abs(self.weight_hh.data)
        else:
            self.weight_ih.data = torch.clamp(self.weight_ih.data, min=0)   
            self.weight_hh.data = torch.clamp(self.weight_hh.data, min=0)  

    def get_total_weights(self, ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1):
        """
        combine learned weight, random noise, connection masking, and fixed EI mask into a single weight for IH and HH
        """
        ih_weight = torch.clamp(ih_noise + self.weight_ih, min=0)*ih_mask
        hh_weight = torch.clamp(hh_noise + self.weight_hh, min=0)*hh_mask

        return (ih_weight, hh_weight)    

class SRNERECCell(SRNEFFCell):
    """
    SRN cell with excitatory-only recurrent connections and unrestricted feedforward connections (not a biological model, just for understanding the class of models)
    """
    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight_hh.data = torch.abs(self.weight_hh.data)
        else:
            self.weight_hh.data = torch.clamp(self.weight_hh.data, min=0)  

    def get_total_weights(self, ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1):
        """
        combine learned weight, random noise, connection masking, and fixed EI mask into a single weight for IH and HH
        """
        ih_weight = (ih_noise + self.weight_ih)*ih_mask
        hh_weight = torch.clamp(hh_noise + self.weight_hh, min=0)*hh_mask

        return (ih_weight, hh_weight)  

class FFCell(nn.Module):
    """
    Ramping feedforward control network. No layer normalization.
    """
    def __init__(self, in_size, hid_size, nonlinearity='sigmoid', bias=True, alpha=0.2, id=None):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.weight_ih = Parameter(torch.Tensor(hid_size, in_size))
        # self.norm = nn.LayerNorm(self.hid_size, elementwise_affine=False)
        self.norm = Identity()
        if bias:
            self.bias = Parameter(torch.Tensor(hid_size))
        else:
            self.register_parameter('bias', None)
        self.nonlin = get_nonlin(nonlinearity)

        # unitless time constant (dt/tau)
        self.alpha = alpha

        self.modid = id
        self.reset_parameters()
    
    def forward(self, inputs, state, preact_state,
                ih_noise=0, ih_mask=1,
                **kwargs, # this just so extra irrelevant arguments don't throw an error
    ):
        if state == None:
            state = torch.zeros((inputs.shape[0], self.hid_size), device=inputs.device)
            preact_state = torch.zeros((inputs.shape[0], self.hid_size), device=inputs.device)

        bias = 0 if self.bias is None else self.bias

        ih_weight = self.get_total_weights(ih_noise, ih_mask)
        new_preact_state = self.norm(inputs.matmul(ih_weight.t())) + bias

        state = self.nonlin((1-self.alpha)*preact_state + self.alpha*new_preact_state)
        outputs = state

        return outputs, state, new_preact_state

    def reset_parameters(self):
        nn.init.normal_(self.weight_ih, 0, 0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def get_total_weights(self, ih_noise=0, ih_mask=1):
        ih_weight =  (self.weight_ih+ih_noise)*ih_mask
        return ih_weight

class EFFCell(FFCell):
    """
    Excitatory-restricted version of the FFCell. No layer normalization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.enforce_connectivity(init=True)

    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight_ih.data = torch.abs(self.weight_ih.data)
        else:
            self.weight_ih.data = torch.clamp(self.weight_ih.data, min=0)    

    def get_total_weights(self, ih_noise=0, ih_mask=1):
        """
        combine learned weight, random noise, connection masking, and fixed EI mask into a single weight for IH and HH
        """
        ih_weight = torch.clamp(ih_noise + self.weight_ih, min=0)*ih_mask

        return ih_weight 

class EFFLNCell(EFFCell):
    """
    EFF cell with layer normalization. Equivalent to using SRNEFF with timing='ff' and no_recurrence=True in the call to forward
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(self.hid_size, elementwise_affine=False)
        self.reset_parameters()    


class FFLNCell(FFCell):
    """
    FF cell with layer normalization. Equivalent to using SRN with timing='ff' and no_recurrence=True in the call to forward
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(self.hid_size, elementwise_affine=False)
        self.reset_parameters() 

class EI4Cell(nn.Module):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects to both cell types, and both receive driving feedforward inputs. Only E cells send feedforward outputs.

    This is the base (E-feedforward) EI model because it has full recurrent connectivity, and full feedforward connectivity from an excitatory population. 
    Other models can trivially subclass it by modifying self.eimask_hh and self.eimask_ih. 
    """
    def __init__(self, in_size, hid_size, nonlinearity='sigmoid', bias=True, device='cpu', id=None, alpha=0.2, ei_alpha_ratio=1):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.modid = id
        self.weight_ih = Parameter(torch.Tensor(2*hid_size, in_size)) # E+I hidden units, E inputs
        self.weight_hh = Parameter(torch.Tensor(2*hid_size, 2*hid_size))
        if bias:
            self.bias = Parameter(torch.Tensor(2*hid_size))
        else:
            self.register_parameter('bias', None)
        self.nonlin = get_nonlin(nonlinearity)

        # unitless time constant (dt/tau). higher alpha means slower faster integration
        assert alpha is not None
        assert ei_alpha_ratio is not None
        self.alpha = alpha
        self.ei_alpha_ratio = ei_alpha_ratio # alpha_e = alpha*ei_alpha_ratio, alpha_i = alpha
        self.ei_alphas = torch.zeros((2*hid_size,), device=device)
        self.ei_alphas[:hid_size] = alpha*ei_alpha_ratio # E neurons
        self.ei_alphas[hid_size::] = alpha # I neurons

        # set up baseline full connectivity between E_in and (E,I)_hid, and (E,I)_hid and (I,E)_hid
        self.eimask_ih = torch.ones((2*hid_size, in_size), device=device) #, requires_grad=True)
        self.eimask_hh = torch.zeros((2*hid_size, 2*hid_size), device=device) #, requires_grad=True)
        self.eimask_hh[:,:hid_size] = 1
        self.eimask_hh[:,hid_size::] = -1 

        # the modid can be used for quick and dirty experimentation
        if 'noln' in self.modid:
            print('no layer norm')
            self.norm = Identity()
        else: 
            print('using layer norm')
            self.norm = nn.LayerNorm([2*hid_size], elementwise_affine=False)

        # no self-connections (per Song, Yang, Wang)
        self.eimask_hh.fill_diagonal_(0)

        self.to(device)
        self.reset_parameters()

    def forward(self, inputs, state, preact_state, 
        ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1, no_recurrence=False,
    ):

        # initialize states if first time step, with separate e and i populations
        if state == None:
            state = torch.zeros((inputs.shape[0], 2*self.hid_size), device=inputs.device)
            preact_state = torch.zeros((inputs.shape[0], 2*self.hid_size), device=inputs.device)

        bias = 0 if self.bias is None else self.bias

        (ih_weight, hh_weight) = self.get_total_weights(ih_noise, hh_noise, ih_mask, hh_mask)

        if no_recurrence:
            new_preact_state = self.norm(inputs.matmul(ih_weight.t())) + bias
        else:
            new_preact_state = self.norm(inputs.matmul(ih_weight.t()) + state.matmul(hh_weight.t())) + bias

        state = self.nonlin((1-self.ei_alphas)*preact_state + self.ei_alphas*new_preact_state)
        
        # set feedforward outputs to e state
        outputs = state[:,:self.hid_size]

        return outputs, state, new_preact_state

    def reset_parameters(self):
        nn.init.normal_(self.weight_ih, 0, 0.01)
        nn.init.normal_(self.weight_hh, 0, 0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        self.enforce_connectivity(init=True)

    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight_ih.data = torch.abs(self.weight_ih.data)
            self.weight_hh.data = torch.abs(self.weight_hh.data)
        else:
            self.weight_ih.data = torch.clamp(self.weight_ih.data, min=0)
            self.weight_hh.data = torch.clamp(self.weight_hh.data, min=0)

    def get_total_weights(self, ih_noise=0, hh_noise=0, ih_mask=1, hh_mask=1):
        """
        combine learned weight, random noise, connection masking, and fixed EI mask into a single weight for IH and HH
        """ 
        ih_weight = torch.clamp(ih_noise + self.weight_ih, min=0)*self.eimask_ih*ih_mask
        hh_weight = torch.clamp(hh_noise + self.weight_hh, min=0)*self.eimask_hh*hh_mask


        return (ih_weight, hh_weight)

class EI5Cell(EI4Cell):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects to both cell types, and both send and receive driving feedforward inputs.

    We subclass EI4Cell since the difference is merely in sending I feedforward outputs. ITN takes care of doubling the input size for cIT, aIT
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for EI5Cell, need to restrict the feedforward conns of I units to be inhibitory
        self.eimask_ih[self.hid_size:] = -1

    def forward(self, *args, **kwargs):
        outputs, state, new_preact_state = super().forward(*args, **kwargs)
        
        # outputs are the full state rather than just the E state
        outputs = state

        return outputs, state, new_preact_state

class EI1Cell(EI4Cell):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects only to the other cell type, and both receive driving feedforward inputs. Only E cells send feedforward outputs.

    We subclass EI4Cell since the difference is merely in refining the self.eimask_hh to enforce between-celltype-only recurrent connectivity
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for EI1Cell, we need to zero out E_hid->E_hid connections and I_hid->I_hid connections
        self.eimask_hh[:self.hid_size, :self.hid_size] = 0
        self.eimask_hh[self.hid_size::, self.hid_size::] = 0

        self.reset_parameters()
        
class EI3Cell(EI1Cell):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects only to the other cell type, and only E units receive driving feedforward inputs. Only E cells send feedforward outputs.

    We subclass EI1Cell since the difference is merely in refining the self.eimask_ih to not send I feedforward outputs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for EI3Cell, we need to additionally zero out in->I_hid connections
        self.eimask_ih[self.hid_size::, :] = 0

        self.reset_parameters()       

class EI6Cell(EI4Cell):
    """
    A recurrent cell with separate populations of excitatory and inhibitory units with recurrent connections to each other.
    Each cell type connects to both cell types, and only E units receive driving feedforward inputs. Only E cells send feedforward outputs.

    We subclass EI4Cell since the difference is merely in refining the self.eimask_ih

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for EI6Cell, we need to additionally zero out in->I_hid connections
        self.eimask_ih[self.hid_size::, :] = 0

        self.reset_parameters()              

class Sig6(nn.Sigmoid):
    """
    a sigmoidal activation function which has y=0 at x=0, reachs its midpoint y=3 at x=5, and plateaus at y=6
    """
    def forward(self, inputs):
        return 6/(1+torch.exp(-2*(inputs-5)))


def get_nonlin(nonlin_str):
    if nonlin_str.lower() == 'sigmoid':
        nonlin = nn.Sigmoid()
    elif nonlin_str.lower() == 'relu':
        nonlin = nn.ReLU()
    elif nonlin_str.lower() == 'relu6':
        nonlin = nn.ReLU6()
    elif nonlin_str.lower() == 'sig6':
        nonlin = Sig6()
    else:
        raise NotImplementedError()
    return nonlin