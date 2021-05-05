#!/usr/bin/env python

import argparse

desc = 'Self-supervised super-resolution.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--image')
parser.add_argument('-o', '--output-directory')
parser.add_argument('-p', '--patch-size', default=[48, 48], type=int, nargs=2)
parser.add_argument('-s', '--slice-profile', default='gaussian')
parser.add_argument('-d', '--num-blocks', default=8, type=int)
parser.add_argument('-w', '--num-channels', default=256, type=int)
parser.add_argument('-r', '--residual-scale', default=0.1, type=float)
parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-e', '--num-epochs', default=100, type=int)
args = parser.parse_args()


import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import AdamW

from sssrlib.patches import Patches
from sssrlib.sample import Sampler

from sssr.train import Trainer
from sssr.edsr import EDSR
from sssr.utils import calc_gaussian_slice_profie, get_axis_order, save_args
from sssr.utils import calc_edsr_patch_size, L1SobelLoss


obj = nib.load(args.image)
voxel_size = obj.header.get_zooms()
x, y, z = get_axis_order(voxel_size)
voxel_size = voxel_size / voxel_size[x]
scale = voxel_size[z]
args.scale1 = int(scale)
args.scale0 = scale / float(args.scale1)

print(args)

if args.slice_profile == 'gaussian':
    slice_profile = calc_gaussian_slice_profie(scale)
else:
    slice_profile = np.load(args.slice_profile)
slice_profile = slice_profile.squeeze()[None, None, :, None]
slice_profile = torch.tensor(slice_profile).float().cuda()

sp_len = slice_profile.shape[2]
patch_size = calc_edsr_patch_size(args.patch_size, sp_len, args.scale1)
args.hr_patch_size = patch_size
image = obj.get_fdata(dtype=np.float32)
patches = Patches(patch_size, image, voxel_size=voxel_size).cuda()
sampler = Sampler(patches) # uniform sampling

net = EDSR(num_blocks=args.num_blocks, num_channels=args.num_channels,
           scale=args.scale1, res_scale=args.residual_scale).cuda()
optim = AdamW(net.parameters(), lr=args.learning_rate)
loss_func = L1SobelLoss().cuda()

trainer = Trainer(sampler, slice_profile, args.scale0, args.scale1,
                  net, optim, loss_func, batch_size=args.batch_size,
                  num_epochs=args.num_epochs)
trainer.train()
