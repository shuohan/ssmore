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
    return xy[0], xy[1], z


def save_args(args, filename):
    result = dict()
    for arg in vars(args):
        result[arg] = getattr(args, arg)
    with open(filename, 'w') as jfile:
        json.dump(result, jfile, pretty=4)


def calc_edsr_patch_size(lr_patch_size, sp_len, scale1):
    ps0 = lr_patch_size[0] * scale1 + sp_len
    ps1 = lr_patch_size[1]
    return [ps0, ps1, 1]


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
