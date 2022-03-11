#!/usr/bin/env python

import torch
from ssmore.edsr import EDSR, Upsample
from pytorchviz import make_dot
from pathlib import Path


def test_edsr():
    net = EDSR(4, 32, 5, 0.1).cuda()
    x = torch.rand(1, 1, 32, 32).cuda()
    print(x.shape, x.dtype)
    dot = make_dot(x, net)
    path = Path('results_models')
    path.mkdir(exist_ok=True)
    dot.render(path.joinpath('edsr'))

    x = torch.arange(8).reshape(1, 2, 2, 2).float()
    up = Upsample(2, 2)
    y = up._pixel_shuffle(x)

    ref = torch.tensor([[0, 1], [4, 5], [2, 3], [6, 7]])[None, None, ...].float()
    assert torch.equal(y, ref)


if __name__ == '__main__':
    test_edsr()
