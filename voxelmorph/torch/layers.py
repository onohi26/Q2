from numpy import nested_iters
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import time
import cv2
from skimage import measure

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class PointTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, points, flow):
        # new locations
        
        new_locs = self.grid + flow
        points = points.to('cpu').detach().numpy().copy()
        new_locs = new_locs.to('cpu').detach().numpy().copy()
        points_x=[]
        points_y=[]

        for i in range(12):
            a=new_locs[0,0,:,:]>points[0,0,i]-1
            b=new_locs[0,0,:,:]<points[0,0,i]+1
            new_x = np.logical_and(a,b)
            a=new_locs[0,1,:,:]>points[0,1,i]-1
            b=new_locs[0,1,:,:]<points[0,1,i]+1
            new_y = np.logical_and(a,b)
            new = np.logical_and(new_x,new_y)
            x_list,y_list = np.where(new)
            if len(x_list) == 0:
                points_x.append(points[0,0,i])
            else:
                points_x.append(np.average(x_list))
            if len(y_list) == 0:
                points_y.append(points[0,1,i])
            else:
                points_y.append(np.average(y_list))
            
     
        return np.round(points_x),np.round(points_y)
        
class PointSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='nearest'):
        super().__init__()

        self.mode = mode
        self.size = size

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self,points, flow):
        pt_image = np.zeros((self.size[0],self.size[1],3))
        points = points.to('cpu').detach().numpy().copy()

        points_x=[]
        points_y=[]

        for i in range(12):
            pixel = 20*(i+1)
            cv2.circle(pt_image, (int(points[0,0,i]),int(points[0,1,i])), 2, color=(pixel, 0, 0), thickness=-1)
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        pt_image = torch.tensor(pt_image.transpose(2,0,1),dtype=torch.float32,device='cuda').unsqueeze(0)
        warped_points = nnf.grid_sample(pt_image, new_locs, align_corners=True, mode=self.mode)
        warped_points = warped_points.squeeze(0).permute(1,2,0).to('cpu').detach().numpy().copy()

        for i in range(12):
            pixel = 20*(i+1)
            img = np.zeros((self.size[0],self.size[1]),dtype=np.uint8)
            img[warped_points[:,:,0]==pixel]=255
            props = measure.regionprops(img)
            try:
                center = props[0].centroid
                points_x.append(center[1])
                points_y.append(center[0])
            except:
                points_x.append(points[0,0,i])
                points_y.append(points[0,1,i])

        return points_x,points_y


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
