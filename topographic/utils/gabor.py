import torch
import numpy as np
from PIL import Image

def gabor(theta=0, lam=1/10, sigma=1, width=224, height=224, px=.5, py=.5):
    """
    theta: math angle orientation of the gabor patch in degrees
    lam: spatial wavelength relative to image width (on [0,1])
    Sigma: standard deviation of gaussian window relative to image width (on [0,1])
    width (256): width of generated image
    height (256): height of generated image
    px (rand): horizontal center of gabor patch, relative to the image width (must be between 0 and 1)
    py (rand): vertical center of gabor patch, relative to the image height (must be between 0 and 1)

    """

    # correct theta to be the math angle of the orientation, where 0 corresponds to horizontal
    theta = -(np.radians(theta) + np.pi/2)
#         -(theta + np.pi/2))

    # convert sigma and lambda to pixels
    sigma = width*sigma
    lam = width*lam
    
    # convert y to make the 0th row correspond to the top
    py = 1 - py
    
    # Generate mesh
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Center of gaussian window
    cx = px*width
    cy = py*height

    # Orientation
    x_theta = (x-cx)*np.cos(theta)+(y-cy)*np.sin(theta)
    y_theta = -(x-cx)*np.sin(theta)+(y-cy)*np.cos(theta)
    #     pdb.set_trace()

    # Generate gabor
    #     F = exp(-.5*(x_theta.^2/p.Results.Sigma^2+y_theta.^2/p.Results.Sigma^2)).*cos(2*pi/p.Results.lambda*x_theta);
    F = np.exp(-.5*(x_theta**2/sigma**2 + y_theta**2/sigma**2))*np.cos(2*np.pi/lam * x_theta)
    
    return F

class GaborDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 thetas=np.linspace(0,180,15), 
                 lams=np.linspace(.1, 1, 10), 
                 sigmas=np.linspace(0,1,20), 
                 pxs=np.linspace(0,1,20), 
                 pys=np.linspace(0,1,20), 
                 width=224, 
                 height=224,
                 transform=None,
                 target_transform=None,
                ):
        self.thetas = thetas
        self.lams = lams
        self.sigmas = sigmas
        self.pxs = pxs
        self.pys = pys
        self.width = width
        self.height = height
        self.transform = transform
        self.target_transform = target_transform
        self.targets = []
        for theta in self.thetas:
            for lam in self.lams:
                for sigma in self.sigmas:
                    for px in self.pxs:
                        for py in self.pys:
                            self.targets.append(dict(theta=theta,lam=lam,sigma=sigma,px=px,py=py))
        
    def __getitem__(self, index):
        target = self.targets[index]
        if self.transform is None:
            sample = np.tile(gabor(width=self.width, height=self.height, **target), [3,1,1])
        else:
            sample = Image.fromarray(gabor(width=self.width, height=self.height, **target), 'RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.targets)