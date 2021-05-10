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
parser.add_argument('-r', '--residual-scale', default=0.1, type=float)
parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
parser.add_argument('-b', '--batch-size', default=100, type=int)
parser.add_argument('-e', '--num-epochs', default=100, type=int)
parser.add_argument('-I', '--image-save-step', default=50, type=int)
parser.add_argument('-W', '--num-channels-multiplier', default=8, type=int)
parser.add_argument('-P', '--use-padding', action='store_true')
parser.add_argument('-R', '--receptive-field', default=9, type=int)
args = parser.parse_args()


import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from improc3d import permute3d

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.sample import Sampler, SamplerCollection
from sssrlib.transform import Flip
from ptxl.save import ImageSaver 
from ptxl.log import EpochLogger, EpochPrinter, DataQueue

from sssr.train import Trainer
from sssr.edsr import EDSR
from sssr.wdsr import WDSRB
from sssr.utils import calc_gaussian_slice_profie, get_axis_order, save_args
from sssr.utils import calc_patch_size, L1SobelLoss


Path(args.output_dir).mkdir(parents=True)
image_dirname = Path(args.output_dir, 'patches')
log_filename = Path(args.output_dir, 'log.csv')
args_filename = Path(args.output_dir, 'config.json')
result_filename = Path(args.output_dir, 'result.nii.gz')

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
image = obj.get_fdata(dtype=np.float32)

flip0 = Flip((0, ))
flip1 = Flip((1, ))
flip01 = Flip((0, 1))

patches00 = Patches(patch_size, image, voxel_size=voxel_size, x=x, y=y, z=z).cuda()
patches01 = TransformedPatches(patches00, flip0)
patches02 = TransformedPatches(patches00, flip1)
patches03 = TransformedPatches(patches00, flip01)

patches10 = Patches(patch_size, image, voxel_size=voxel_size, x=y, y=x, z=z).cuda()
patches11 = TransformedPatches(patches10, flip0)
patches12 = TransformedPatches(patches10, flip1)
patches13 = TransformedPatches(patches10, flip01)

sampler00 = Sampler(patches00) # uniform sampling
sampler01 = Sampler(patches01) # uniform sampling
sampler02 = Sampler(patches02) # uniform sampling
sampler03 = Sampler(patches03) # uniform sampling

sampler10 = Sampler(patches10) # uniform sampling
sampler11 = Sampler(patches11) # uniform sampling
sampler12 = Sampler(patches12) # uniform sampling
sampler13 = Sampler(patches13) # uniform sampling

sampler = SamplerCollection(sampler00, sampler01, sampler02, sampler03,
                            sampler10, sampler11, sampler12, sampler13)

save_args(args, args_filename)

# net = EDSR(num_blocks=args.num_blocks, num_channels=args.num_channels,
#            scale=args.scale1, res_scale=args.residual_scale).cuda()
net = WDSRB(args.scale1, num_channels=args.num_channels,
            num_chan_multiplier=args.num_channels_multiplier,
            num_blocks=args.num_blocks, use_padding=args.use_padding,
            num_k3=(args.receptive_field - 1) // 2).cuda()
optim = AdamW(net.parameters(), lr=args.learning_rate)
loss_func = L1SobelLoss().cuda()
print(net)
print(optim)

trainer = Trainer(sampler, slice_profile, args.scale0, args.scale1,
                  net, optim, loss_func, batch_size=args.batch_size,
                  num_epochs=args.num_epochs)
queue = DataQueue(['loss'])
logger = EpochLogger(log_filename)
printer = EpochPrinter(print_sep=False)
queue.register(logger)
queue.register(printer)
attrs =  ['extracted', 'blur', 'lr', 'input_interp', 'output', 'hr_crop']
image_saver = ImageSaver(image_dirname, attrs=attrs,
                         step=args.image_save_step, zoom=4, ordered=True,
                         file_struct='epoch/sample', save_type='png_norm')
trainer.register(queue)
trainer.register(image_saver)
trainer.train()

image, inv_x, inv_y, inv_z = permute3d(image, x=x, y=y, z=z)
result = trainer.predict(image).detach().cpu().numpy().squeeze()
result = permute3d(result, x=inv_x, y=inv_y, z=inv_z)[0]
scale_mat = np.diag(np.array([1, 1, 1 / scale, 1])[[x, y, z, 3]])
affine = obj.affine @ scale_mat
out = nib.Nifti1Image(result, affine, obj.header)
out.to_filename(result_filename)
