"""Modified from https://github.com/yulunzhang/RCAN

"""
import torch
from torch import nn
from resize.pytorch import resize

from .edsr import Upsample


class CALayer(nn.Module):
    """Channel Attention Layer.

    """
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.num_channels = num_channels
        self.reduction = reduction

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        num_red_chan = num_channels // reduction
        self.conv0 = nn.Conv2d(num_channels, num_red_chan, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_red_chan, num_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv0(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class RCAB(nn.Module):
    """Residual Channel Attention Block.

    """
    def __init__(self, num_channels, kernel_size, reduction):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.reduction = reduction

        padding = (kernel_size - 1) // 2
        self.conv0 = nn.Conv2d(num_channels, num_channels, kernel_size,
                               padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size,
                               padding=padding)
        self.ca = CALayer(num_channels, reduction)

    def forward(self, x):
        res = self.conv0(x)
        res = self.relu(res)
        res = self.conv1(res)
        res = self.ca(res)
        out = res + x
        return out


class RG(nn.Module):
    """Residual Group.

    """
    def __init__(self, num_rcab, num_channels, kernel_size, reduction):
        super().__init__()
        self.num_rcab = num_rcab
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.reduction = reduction

        for i in range(num_rcab):
            rcab = RCAB(num_channels, kernel_size, reduction)
            self.add_module('rcab%d' % i, rcab)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size,
                              padding=(kernel_size - 1) // 2)

    def forward(self, x):
        res = x
        for i in range(self.num_rcab):
            res = getattr(self, 'rcab%d' % i)(res)
        res = self.conv(res)
        out = res + x
        return out


class RCAN(nn.Module):
    """Residual Channel Attention Network.

    """
    def __init__(self, num_rg, num_rcab, num_channels, reduction, scale):
        super().__init__()
        self.num_rg = num_rg
        self.num_rcab = num_rcab
        self.num_channels = num_channels
        self.reduction = reduction
        self.scale = scale
        self._scale1 = int(self.scale)
        self._scale0 = self.scale / float(self._scale1)

        kernel_size = 3
        act = nn.ReLU(True)

        padding = (kernel_size - 1) // 2
        self.conv0 = nn.Conv2d(1, num_channels, kernel_size, padding=padding)

        for i in range(num_rg):
            rg = RG(num_rcab, num_channels, kernel_size, reduction)
            self.add_module('rg%d' % i, rg)

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size,
                               padding=padding)
        self.up = Upsample(num_channels, self._scale1, use_padding=True)
        # self.rg_out = RG(2, num_channels, kernel_size, reduction)
        self.conv_out = nn.Conv2d(num_channels, 1, 1)

    def calc_out_patch_size(self, input_patch_size):
        x = torch.rand([1, 1] + input_patch_size).float()
        x = x.to(next(self.parameters()).device)
        out = self(x)
        patch_size = list(out.shape[2:])
        return patch_size

    def forward(self, x):
        x = resize(x, (1 / self._scale0, 1), order=3)
        x = self.conv0(x)
        res = x
        for i in range(self.num_rg):
            res = getattr(self, 'rg%d' % i)(res)
        res = self.conv1(res)
        out = res + x
        out = self.up(out)
        # out = self.rg_out(out)
        out = self.conv_out(out)
        return out
