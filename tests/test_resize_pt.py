#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
from resize.pytorch import resize


def test_resize_cpu():
    a = torch.rand(256, 1, 128, 32).float().cuda()
    coords = None
    for i in range(10000):
        print(i, a.device)
        if coords is None:
            b, coords = resize(a, (4, 1), order=3, return_coords=True)
        else:
            b = F.grid_sample(a, coords, mode='bicubic', align_corners=True,
                              padding_mode='border')

if __name__ == '__main__':
    test_resize_cpu()
