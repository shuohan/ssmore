import math
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .utils import pixel_shuffle


class WDSRB(nn.Module):
    def __init__(self, scale, num_channels=32, num_chan_multiplier=8,
                 num_blocks=16, use_padding=False):
        super().__init__()
        self.scale = scale
        self.num_channels = num_channels
        self.num_chan_multiplier = num_chan_multiplier
        self.num_blocks = num_blocks
        self.use_padding = use_padding

        padding = 1 if self.use_padding else 0
        self.conv0 = weight_norm(nn.Conv2d(1, num_channels, 3, padding=padding))
        init.ones_(self.conv0.weight_g)
        init.zeros_(self.conv0.bias)

        res_scale = 1 / math.sqrt(num_blocks)
        for i in range(num_blocks):
            block = Block(num_channels, num_chan_multiplier, res_scale,
                          use_padding=self.use_padding)
            self.add_module('block%d' % i, block)
        
        out_channels = scale
        conv = nn.Conv2d(num_channels, out_channels, 1)
        self.conv1 = weight_norm(conv)
        init.ones_(self.conv1.weight_g)
        init.zeros_(self.conv1.bias)

        skip_ks = 5
        skip_padding = (skip_ks - 1) // 2 if self.use_padding else 0
        conv = nn.Conv2d(1, out_channels, skip_ks, padding=skip_padding)
        self.skip_conv = weight_norm(conv)
        init.ones_(self.skip_conv.weight_g)
        init.zeros_(self.skip_conv.bias)

        if not self.use_padding:
            self.crop_size = self.num_blocks + 1
            self._skip_crop = self.crop_size - (skip_ks - 1) // 2

    def forward(self, x):
        out = self.conv0(x)
        for i in range(self.num_blocks):
            out = getattr(self, 'block%d' % i)(out)
        out = self.conv1(out)
        out = out + self._apply_skip_conv(x)
        out = pixel_shuffle(out, self.scale)
        return out

    def _apply_skip_conv(self, x):
        skip = self.skip_conv(x)
        if not self.use_padding:
            crop = self._skip_crop
            skip = skip[..., crop : -crop, crop : -crop]
        return skip

    def crop(self, batch):
        crop1 = self.crop_size
        crop0 = self.scale * self.crop_size
        return batch[:, :, crop0 : -crop0, crop1 : -crop1]


class Block(nn.Module):
    def __init__(self, num_channels, num_chan_multiplier=8, res_scale=1,
                 use_padding=False):
        super().__init__()
        self.num_channels = num_channels
        self.num_chan_multiplier = num_chan_multiplier
        self.res_scale = res_scale
        self.use_padding = use_padding

        out_channels = int(num_channels * num_chan_multiplier)
        self.conv0 = weight_norm(nn.Conv2d(num_channels, out_channels, 1))
        init.ones_(self.conv0.weight_g)
        init.zeros_(self.conv0.bias)

        self.relu = nn.ReLU()

        self.conv1 = weight_norm(nn.Conv2d(out_channels, out_channels, 1))
        init.ones_(self.conv1.weight_g)
        init.zeros_(self.conv1.bias)

        padding = 1 if self.use_padding else 0
        conv = nn.Conv2d(out_channels, num_channels, 3, padding=padding)
        self.conv2 = weight_norm(conv)
        init.constant_(self.conv2.weight_g, res_scale)
        init.zeros_(self.conv2.bias)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out + x if self.use_padding else out + x[:, :, 1:-1, 1:-1]
        return out
