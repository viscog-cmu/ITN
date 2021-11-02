import torch
import torch.nn as nn
import torch.nn.functional as F

class EReadout(nn.Linear):
    """
    An excitatory-only readout/decoder (linear) layer. This allows E-cells in the penultimate layer to be truly E-cells, as in
    projecting only excitatory outputs.
    Inhibition is achieved strictly through the softmax. 
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.enforce_connectivity(init=True)

    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight.data = torch.abs(self.weight.data)
        else:
            self.weight.data = torch.clamp(self.weight.data, min=0)


class ERampReadout(EReadout):
    """
    Extends EReadout to ramp over time according to time constant alpha 
    """
    def __init__(self, in_features, out_features, alpha=0.2, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.alpha = alpha

    def forward(self, inputs, state=None):
        if state is None:
            state = 0
        outputs = self.alpha*super().forward(inputs) + (1-self.alpha)*state
        state = outputs
        return outputs, state

class EIRampReadout(ERampReadout):
    """
    Extends ERampReadout to allow for readout from both E and I cells. 

    We assume there are two cell types, E and I, with an equal number of units (features_per_celltype)
    """
    def __init__(self, features_per_celltype, out_features, alpha=0.2, bias=True, device='cuda'):
        super().__init__(2*features_per_celltype, out_features, bias=bias)
        self.alpha = alpha
        self.ei_mask = torch.ones((out_features, 2*features_per_celltype)).to(device)
        self.ei_mask[:,features_per_celltype:] = -1

    def forward(self, inputs, state=None):
        if state is None:
            state = 0
        outputs = F.linear(inputs, self.weight*self.ei_mask, self.bias) + (1-self.alpha)*state
        state = outputs
        return outputs, state

    def enforce_connectivity(self, init=False):
        """
        this should be called after optimizer.step(), to ensure that the next minibatch has correct weights

        we use abs at init to have a denser representation, and clamp everywhere else for stabler learning
        """
        if init:
            self.weight.data = torch.abs(self.weight.data)
        else:
            self.weight.data = torch.clamp(self.weight.data, min=0)
