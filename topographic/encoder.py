
import torch
import pdb
from torch import nn 
from torch import optim
import torch.nn.functional as F
import torchvision.models
from tensorboardX import SummaryWriter
from abc import abstractmethod 
import sys
import numpy as np
import os

from topographic import utils
from topographic.utils import commons, training, dataset
from topographic.config import TENSORBOARD_DIR, SAVE_DIR
from topographic.dl import resnet

class ITNEncoder(nn.Module):
    """
    an early visual cortex encoder intended to be used as input to a series of ITArea modules
    """
    def __init__(self, encoder_id='resnet18', base_fn=None, no_stride=False,
                im_dim=224, polar=False, logpolar=False, centercrop_train=False,
                dataset_name='objfacescenes', num_workers=6, batch_size=64,
                optim_name='sgd', schedule='plateau', patience=2, 
    ):
        super().__init__()
        self.base_fn = base_fn
        self.get_data_transforms(im_dim, polar=polar, logpolar=logpolar, centercrop_train=centercrop_train)
        self.get_data_loaders(dataset_name, im_dim=im_dim, num_workers=num_workers, batch_size=batch_size)
        if 'resnet' in encoder_id:
            self.encoder = ResNetEncoder(int(encoder_id.replace('resnet', '')), num_classes=self.num_classes, no_stride=no_stride, polar=polar or logpolar)
        else:
            # each model requires a separate implementation to remove the classifier
            raise NotImplementedError()   
        self.get_optimizer(optim_name=optim_name, schedule=schedule, patience=patience)

        self.encoder.to('cuda')

    def forward(self, inputs, *args, **kwargs):
        return self.encoder(inputs, *args, **kwargs)

    def get_data_transforms(self, *args, **kwargs):
        self.data_transforms = commons.get_data_transforms(*args, **kwargs)

    def get_data_loaders(self, dataset_name, im_dim=224, **kwargs):
        """
        extract the datasets from dataset_name, and data loaders.
        kwargs are passed to dataset.get_multi_dataset_loaders
        """
        assert hasattr(self, 'data_transforms'), 'must run get_data_transforms before get_data_loaders'
        self.dataset_name = dataset_name
        self.im_dim = im_dim
        self.datasets = dataset.expand_datasets_from_name(dataset_name)
        self.dataloaders, self.num_classes = dataset.get_multi_dataset_loaders(
                            self.datasets, 
                            self.data_transforms,
                            return_n_classes=True,
                            **kwargs,
        )

    def get_tensorboard_writer(self):
        self.writer = SummaryWriter(logdir=f'{TENSORBOARD_DIR}/{self.base_fn}')

    def load_state_dict(self):
        state_dict = torch.load(os.path.join(SAVE_DIR, 'models', self.base_fn + '.pkl'))[0]
        self.encoder.load_state_dict(state_dict)

    def get_optimizer(self, optim_name='sgd', lr0=0.01, jjlam=0, schedule='plateau', patience=2):
        model = self.encoder
        if optim_name == 'sgd':
            self.optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=lr0, momentum=0.9, 
                                weight_decay=5e-4 if not jjlam else 0,
                                )
        elif optim_name == 'adam':
           self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=lr0, 
                                weight_decay=5e-4 if not jjlam else 0,
                                )
        else:
            raise NotImplementedError()

        self.scheduler = training.get_scheduler(self.optimizer, scheduler_type=schedule, patience=patience)

    def pretrain(self, n_epochs=50):
        self.encoder = training.train_encoder(self.encoder, self.dataloaders, self.optimizer, self.base_fn,
            imdim=self.im_dim,
            scheduler=self.scheduler,
            tensorboard_writer=self.writer,
            device='cuda',
            epochs=n_epochs,
            criterion=nn.CrossEntropyLoss(),
            save_checkpoints=True,
            save_final_weights=True,
            save_dir=SAVE_DIR,
            checkpoint_losses=None,
            use_amp=False,
            detach_features=True,
            )

class BaseEncoder(nn.Module):
    """
    Abstract class
    """
    def __init__(self, n=18):
        super().__init__()
        self._detached_features = False

    @abstractmethod
    def _detach_features(self):
        # eliminate the classifier part of the model
        pass

class ResNetEncoder(BaseEncoder):
    def __init__(self, n=18, num_classes=1000, no_stride=False, polar=False):
        super().__init__()
        self.n = n
        self.no_stride = no_stride
        self.polar = polar
        if no_stride:
            self.encoder = getattr(resnet, 'resnetfcn' + str(n))(pretrained=False, num_classes=num_classes, polar=polar)
        else:
            if polar:
                raise NotImplementedError('polar coordinates not implemented for strided resnet')
            self.encoder = getattr(torchvision.models, 'resnet' + str(n))(pretrained=False, num_classes=num_classes)
        self._detached_features = False


    def _detach_features(self):
        # eliminate the classifier part of the model
        self.encoder.fc = None
        self._detached_features = True
        if self.no_stride:
            # we downsample by a factor of 2 spatially, and a factor of 4-16 channelwise
            channels = 512 if self.n == 18 else 2048
            self.encoder.output_downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2,2)),
                nn.Conv2d(channels, 128, kernel_size=(1,1), stride=1, bias=False),
                nn.ReLU(),
            )
    
    def _forward_impl(self, x, return_layers=[], out_device='cuda'):
        # See note [TorchScript super()]
        x_out = {}
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        if 'layer1' in return_layers:
            x_out['layer1'] = [x.to(out_device)]
        x = self.encoder.layer2(x)
        if 'layer2' in return_layers:
            x_out['layer2'] = [x.to(out_device)]
        x = self.encoder.layer3(x)
        if 'layer3' in return_layers:
            x_out['layer3'] = [x.to(out_device)]
        x = self.encoder.layer4(x)
        if 'layer4' in return_layers:
            x_out['layer4'] = [x.to(out_device)]

        if not self._detached_features:
            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.encoder.fc(x)
            if 'output' in return_layers:
                x_out['output'] = [x.to(out_device)]
        elif not self.no_stride:
            x = self.encoder.avgpool(x)
        else:
            x = self.encoder.output_downsample(x)

        for block in return_layers:
            output = x_out[block]
            if type(output) is list:
                x_out[block] = torch.stack(output).transpose(0,1).to(out_device) # time dimension is 2nd dimension 
            else:
                x_out[block] = output.to(out_device)

        if len(return_layers)>0:
            return x_out
        else:
            return x.to(out_device)

    def forward(self, x, *args, **kwargs):
        return self._forward_impl(x, *args, **kwargs)


def load_v1_to_v4_cornetz(model, pretrained_fn, device='cuda', freeze=False, no_v3=False):
    """
    load pre-trained convolutional layers for a 4-layer CORnet-z model
    """
    sd = torch.load(pretrained_fn, map_location=device)[0]
    model.V1.conv.weight = torch.nn.Parameter(sd['V1.conv.weight'], requires_grad=not freeze)
    model.V1.conv.bias = torch.nn.Parameter(sd['V1.conv.bias'], requires_grad=not freeze)
    model.V2.conv.weight = torch.nn.Parameter(sd['V2.conv.weight'], requires_grad=not freeze)
    model.V2.conv.bias = torch.nn.Parameter(sd['V2.conv.bias'], requires_grad=not freeze)
    if not no_v3:
        model.V3.conv.weight = torch.nn.Parameter(sd['V3.conv.weight'], requires_grad=not freeze)
        model.V3.conv.bias = torch.nn.Parameter(sd['V3.conv.bias'], requires_grad=not freeze)
    model.V4.conv.weight = torch.nn.Parameter(sd['V4.conv.weight'], requires_grad=not freeze)
    model.V4.conv.bias = torch.nn.Parameter(sd['V4.conv.bias'], requires_grad=not freeze)
    return model
