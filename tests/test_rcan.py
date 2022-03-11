#!/usr/bin/env python

import torch
from ssmore.models.rcan import RCAN
from pytorchviz import make_dot
from pathlib import Path


def test_wdsr():
    net = RCAN(2, 2, 64, 16, 4, 2).cuda()
    x = torch.rand(1, 1, 64, 64).cuda()
    print(x.shape, x.dtype)
    dot = make_dot(x, net)
    path = Path('results_models')
    path.mkdir(exist_ok=True)
    dot.render(path.joinpath('rcan'))
    print(net)


if __name__ == '__main__':
    test_wdsr()
