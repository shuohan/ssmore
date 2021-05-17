#!/usr/bin/env python

import argparse

desc = 'Self-supervised super-resolution.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--image')
parser.add_argument('-o', '--output-dir')
parser.add_argument('-p', '--patch-size', default=[40, 40], type=int, nargs=2)
parser.add_argument('-s', '--slice-profile', default='gaussian')
parser.add_argument('-d', '--num-blocks', default=8, type=int)
parser.add_argument('-w', '--num-channels', default=256, type=int)
parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-e', '--num-epochs', default=10000, type=int)
parser.add_argument('-n', '--num-iters', default=3, type=int)
parser.add_argument('-I', '--image-save-step', default=50, type=int)
parser.add_argument('-W', '--num-channels-multiplier', default=8, type=int)
parser.add_argument('-P', '--use-padding', action='store_true')
parser.add_argument('-g', '--num-groups', default=4, type=int)
parser.add_argument('-f', '--following-num-epochs', default=100, type=int)
parser.add_argument('-S', '--iter-save-step', default=10, type=int)
args = parser.parse_args()


import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import AdamW, Adam
from improc3d import permute3d
from torch.nn import L1Loss, MSELoss

from sssr.models.edsr import EDSR
from sssr.models.wdsr import WDSRB
from sssr.models.rcan import RCAN
from sssr.utils import calc_gaussian_slice_profie, get_axis_order, save_args
from sssr.utils import calc_patch_size, L1SobelLoss
from sssr.build import build_sampler, build_trainer


Path(args.output_dir).mkdir(parents=True)
args.image_dirname = str(Path(args.output_dir, 'patches'))
args.log_filename = str(Path(args.output_dir, 'log'))
args.result_filename = str(Path(args.output_dir, 'result'))
args_filename = str(Path(args.output_dir, 'config.json'))

obj = nib.load(args.image)
voxel_size = obj.header.get_zooms()
x, y, z = get_axis_order(voxel_size)
voxel_size = voxel_size / voxel_size[x]
scale = voxel_size[z]
args.scale1 = int(scale)
args.scale0 = scale / float(args.scale1)

if args.slice_profile == 'gaussian':
    slice_profile = calc_gaussian_slice_profie(scale)
else:
    slice_profile = np.load(args.slice_profile)
slice_profile = slice_profile.squeeze()[None, None, :, None]
slice_profile = torch.tensor(slice_profile).float().cuda()

sp_len = slice_profile.shape[2]
patch_size = calc_patch_size(args.patch_size, sp_len, args.scale1)
args.hr_patch_size = patch_size

save_args(args, args_filename)

net = RCAN(args.num_groups, args.num_blocks, args.num_channels, 16, args.scale1)
net = net.cuda()
optim = Adam(net.parameters(), lr=args.learning_rate)
loss_func = L1Loss().cuda()
print(net)
print(optim)
print(loss_func)

image = obj.get_fdata(dtype=np.float32)
scale_mat = np.diag(np.array([1, 1, 1 / scale, 1])[[x, y, z, 3]])
affine = obj.affine @ scale_mat
perm_image, inv_x, inv_y, inv_z = permute3d(image, x=x, y=y, z=z)
tmp_image = image

for i in range(args.num_iters):
    print('Iteration', i)
    print('--------------------------------------')

    sampler = build_sampler(tmp_image, patch_size, (x, y, z), voxel_size)
    # print(sampler.patches)
    # print(tmp_image.sum(), tmp_image.shape)

    trainer = build_trainer(sampler, slice_profile, net, optim, loss_func, args, i)
    trainer.train()

    result = trainer.predict(perm_image).detach().cpu().numpy().squeeze()
    tmp_image = result

    if i % args.iter_save_step == 0 or i == args.num_iters - 1:
        result = permute3d(result, x=inv_x, y=inv_y, z=inv_z)[0]
        out = nib.Nifti1Image(result, affine, obj.header)
        out.to_filename(args.result_filename + '%d.nii.gz' % i)
