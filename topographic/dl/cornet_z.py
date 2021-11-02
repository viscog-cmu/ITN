from collections import OrderedDict
from torch import nn
from torch import load

from topographic.dl.lc import Conv2dLocal

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x

class EConv2d(nn.Conv2d):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(min=0)
        return super().forward(inputs)

class IConv2d(nn.Conv2d):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(max=0)
        return super.forward(inputs)


class ELinear(nn.Linear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp(min=0)
        return super().forward(inputs)

class ILinear(nn.Linear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp_(max=0)
        super.forward(inputs)

class AdaptiveLinear(nn.Linear):
    """
        a linear layer that automatically determines the number of inputs on initial forward pass
        must remain the same for all future forward passes

        ** note that you must set the optimizer AFTER adapting this layer **
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapted = False
        self.flatten = Flatten()
    def forward(self, inputs):
        if not self.adapted:
            inputs = self.flatten(inputs)
            ndim_in = inputs.shape[1]
            new_units = nn.Linear(ndim_in, self.out_features)
            device = self._parameters['weight'].data.get_device()
            self._parameters['weight'].data = new_units._parameters['weight'].data.to(device)
            self.adapted = True
        return super().forward(inputs)

class AdaptiveELinear(AdaptiveLinear):
    def forward(self, inputs):
        for p in self.parameters():
            p.data.clamp(min=0)
        return super().forward(inputs)

# class EIConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         self.conv = EConv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
#         self.e2i = nn.AdaptiveELinear(10, 1)
#         self.i2e

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 pool_stride=2,
                 locally_connected=False,
                 do_pooling=True,
                 in_height=None,
                 in_width=None,
                 polar=False,
                 dropout=False,
                 out_shape=None, # irrelevant, just for compatibility with other blocks
                ):
        super().__init__()
        self.do_pooling = do_pooling
        self.polar = polar
        if polar:
            self.pad = (kernel_size)//2
            conv_pad = 0
            pool_pad = 0
        else:
            conv_pad = kernel_size//2
            pool_pad = 1
            
        if locally_connected:
            self.conv = Conv2dLocal(in_height, in_width, in_channels, out_channels,
                                kernel_size=kernel_size, padding=conv_pad,
                                stride=stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=conv_pad)
        self.nonlin = nn.ReLU(inplace=True)
        if do_pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=pool_pad)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = Identity()
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        if self.polar:
            inp = nn.functional.pad(inp, pad=(0, 0, self.pad, self.pad), mode='circular') # circular padding for angular dimension 
            inp = nn.functional.pad(inp, pad=(self.pad, self.pad, 0, 0), mode='constant') # normal padding for radial dimension
        x = self.conv(inp)
        x = self.nonlin(x)
        if self.do_pooling:
            if self.polar:
                x = nn.functional.pad(x, pad=(0,0,1,1), mode='circular')
                x = nn.functional.pad(x, pad=(1,1,0,0), mode='constant')
            x = self.pool(x)
        x = self.dropout(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

class CORblock_RZ(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 pool_stride=2,
                 alpha = 0.2,
                 locally_connected=False,
                 do_pooling=True,
                 in_height=None,
                 in_width=None,
                 polar=False,
                 dropout=False,
                 out_shape=None, # irrelevant, just for compatibility with other blocks
                ):
        super().__init__()
        self.do_pooling = do_pooling
        self.polar = polar
        if polar:
            self.pad = (kernel_size)//2
            conv_pad = 0
            pool_pad = 0
        else:
            conv_pad = kernel_size//2
            pool_pad = 1
            
        if locally_connected:
            self.conv = Conv2dLocal(in_height, in_width, in_channels, out_channels,
                                kernel_size=kernel_size, padding=conv_pad,
                                stride=stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=conv_pad)
        self.alpha = alpha
        self.nonlin = nn.ReLU(inplace=True)
        if do_pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=pool_pad)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = Identity()
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp, state=None, preact_state=None, **kwargs):
        """
        the state input is not used but is kept for consistency with other recurrent block architectures
        """
        if self.polar:
            inp = nn.functional.pad(inp, pad=(0, 0, self.pad, self.pad), mode='circular') # circular padding for angular dimension 
            inp = nn.functional.pad(inp, pad=(self.pad, self.pad, 0, 0), mode='constant') # normal padding for radial dimension
        if preact_state is None:
            preact_state = 0
        preact_state = (1-self.alpha)*preact_state + (self.alpha)*self.conv(inp)
        state = self.nonlin(preact_state)
        if self.do_pooling:
            if self.polar:
                state = nn.functional.pad(state, pad=(0,0,1,1), mode='circular')
                state = nn.functional.pad(state, pad=(1,1,0,0), mode='constant')
            state = self.pool(state)
        state = self.dropout(state)
        output = self.output(state)  # for an easy access to this block's output

        return output, state, preact_state

    
class CORnet_Z(nn.Sequential):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_debug(self, x0, return_layers, device='cpu'):
        acts = {}
        acts['x0'] = x0
        acts['x1'] = self.V1(x0)
        acts['x2'] = self.V2(acts['x1'])
        acts['x3'] = self.V4(acts['x2'])
        acts['x4'] = self.IT(acts['x3'])
        acts['x5'] = self.decoder(acts['x4'])
        acts['x6'] = nn.functional.softmax(acts['x5'], 1)

        out = {}
        for layer in return_layers:
            out['x{}'.format(layer)] = acts.pop('x{}'.format(layer)).to(device)
        del acts
        return out

    
def cornet_z(weights_path=None, n_classes=1000,
                half_filters_at_layer=None):
    n_filts = dict(V1=64, V2=128, V4=256, IT=512)
    if half_filters_at_layer:
        assert half_filters_at_layer in ['V1', 'V2', 'V4', 'IT', 'ALL'], f'invalid layer {half_filters_at_layer} specified for filter halfing'
        if half_filters_at_layer == 'ALL':
            n_filts = {key: int(val/2) for key, val in n_filts.items()}
        else:
            n_filts[half_filters_at_layer] = int(n_filts[half_filters_at_layer]/2)
    if weights_path:
        state_dict = load(weights_path)
        if type(state_dict) is tuple:
            state_dict = state_dict[0]
        state_dict_ = OrderedDict()
        for key, val in state_dict.items():
            state_dict_[key.replace('module.', '')] = val
        n_classes = len(state_dict_['decoder.linear.bias'])
    model = CORnet_Z(OrderedDict([
        ('V1', CORblock_Z(3, n_filts['V1'], kernel_size=7, stride=2)),
        ('V2', CORblock_Z(n_filts['V1'], n_filts['V2'])),
        ('V4', CORblock_Z(n_filts['V2'], n_filts['V4'])),
        ('IT', CORblock_Z(n_filts['V4'], n_filts['IT'])),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(n_filts['IT'], n_classes)),
            ('output', Identity())
        ])))]))
    if weights_path:
        model.load_state_dict(state_dict_)
    return model