"""EDSR network

This code is modified from 

    https://github.com/sanghyun-son/EDSR-PyTorch.git

"""
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, num_channels, res_scale=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.res_scale = res_scale
        self.conv0 = nn.Conv2d(num_channels, num_channels, 3)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3)

    def forward(self, x):
        result = self.conv0(x)
        result = self.relu(result)
        result = self.conv1(result)
        result = x[:, :, 2:-2, 2:-2]  + self.res_scale * result
        return result


class Upsample(nn.Module):
    def __init__(self, num_channels, scale):
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale
        self.conv0 = nn.Conv2d(num_channels, scale * num_channels, 3)
        self.conv1 = nn.Conv2d(num_channels, 1, 1)

    def forward(self, x):
        out = self.conv0(x)
        out = self._pixel_shuffle(out)
        out = self.conv1(out)
        return out

    def _pixel_shuffle(self, x):
        """https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b"""
        num_batches, num_channels, nx, ny = x.shape
        num_channels = num_channels // self.scale
        out = x.contiguous().view(num_batches, num_channels, self.scale, nx, ny)
        out = out.permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(num_batches, num_channels, nx * self.scale, ny)
        return out


class EDSR(nn.Module):
    def __init__(self, num_blocks=8, num_channels=256, scale=2, res_scale=0.1):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.scale = scale
        self.res_scale = res_scale

        self.conv = nn.Conv2d(1, num_channels, 3)
        for i in range(num_blocks):
            self.add_module('block%d' % i, ResBlock(num_channels, res_scale))
        self.up = Upsample(num_channels, scale)

    def forward(self, x):
        out = self.conv(x)
        for i in range(self.num_blocks):
            out = getattr(self, 'block%d' % i)(out)
        out = self.up(out)
        return out
