#!/usr/bin/env python

import torch
from sssr.wdsr import WDSRB
from pytorchviz import make_dot
from pathlib import Path
from collections import namedtuple


# Params = namedtuple('Params', ['temporal_size', 'image_mean', 'num_channels',
#                                'scale', 'num_residual_units', 'num_blocks',
#                                'width_multiplier'])


def test_wdsr():
    # params = Params(temporal_size=0, image_mean=0, num_channels=1,
    #                 scale=2, num_residual_units=64, num_blocks=4,
    #                 width_multiplier=2)
    # net = MODEL(params).cuda()
    net = WDSRB(scale=4, num_blocks=3, use_padding=True).cuda()
    x = torch.rand(1, 1, 64, 64).cuda()
    print(x.shape, x.dtype)
    dot = make_dot(x, net)
    path = Path('results_models')
    path.mkdir(exist_ok=True)
    dot.render(path.joinpath('wdsr'))

    # x = torch.arange(8).reshape(1, 2, 2, 2).float()
    # up = Upsample(2, 2)
    # y = up._pixel_shuffle(x)

    # ref = torch.tensor([[0, 1], [4, 5], [2, 3], [6, 7]])[None, None, ...].float()
    # assert torch.equal(y, ref)


if __name__ == '__main__':
    test_wdsr()
