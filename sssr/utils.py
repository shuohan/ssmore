import numpy as np
import torch
import json
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from torch import nn

from .resize import resize_pt


def fwhm_to_std(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def calc_gaussian_slice_profie(scale):
    std = fwhm_to_std(scale)
    length = int(2 * round(scale) + 1)
    sp = gaussian(length, std)
    sp = sp / sp.sum()
    return sp


def get_axis_order(voxel_size):
    z = np.argmax(voxel_size)
    xy = list(range(len(voxel_size)))
    xy.remove(z)
    return int(xy[0]), int(xy[1]), int(z)


def save_args(args, filename):
    result = dict()
    for arg in vars(args):
        result[arg] = getattr(args, arg)
    with open(filename, 'w') as jfile:
        json.dump(result, jfile, indent=4)


def calc_patch_size(lr_patch_size, sp_len, scale1):
    ps0 = lr_patch_size[0] * scale1 + sp_len
    ps1 = lr_patch_size[1] # * scale1
    return [ps0, ps1, 1]


def pixel_shuffle(x, scale):
    """https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b"""
    num_batches, num_channels, nx, ny = x.shape
    num_channels = num_channels // scale
    out = x.contiguous().view(num_batches, num_channels, scale, nx, ny)
    out = out.permute(0, 1, 3, 2, 4).contiguous()
    out = out.view(num_batches, num_channels, nx * scale, ny)
    return out


class MultiL1Loss(nn.Module):
    def forward(self, outputs, truth):
        result = 0
        for output in outputs:
            errors = F.l1_loss(output, truth, reduction='none')
            weights = truth / torch.mean(truth, (2, 3), keepdim=True)
            result = result + torch.sum(errors / (weights + 1e-6))
        return result / len(outputs)


class L1SobelLoss(nn.Module):
    def __init__(self, sobel_lambda=0.5, l1_lambda=0.5, eps=1e-6):
        super().__init__()
        self.sobel_lambda = sobel_lambda
        self.l1_lambda = l1_lambda
        self.eps = eps
        self.register_buffer('kernel', self._get_sobel_kernel2d())

    def _get_sobel_kernel2d(self):
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]])
        kernel_y = kernel_x.transpose(0, 1)
        kernel = torch.stack([kernel_x, kernel_y]) 
        kernel = kernel.unsqueeze(1)
        return kernel

    def forward(self, input, target):
        input_grad = self._calc_sobel_grad(input)
        target_grad = self._calc_sobel_grad(target)
        grad_loss = F.l1_loss(input_grad, target_grad)
        l1_loss = F.l1_loss(input, target)
        loss = self.sobel_lambda * grad_loss + self.l1_lambda * l1_loss
        return loss

    def _calc_sobel_grad(self, image):
        grad = F.conv2d(image, self.kernel)
        gx = grad[:, 0, ...]
        gy = grad[:, 1, ...]
        result = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        return result
