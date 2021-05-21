import json
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import L1Loss
from pathlib import Path

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.sample import Sampler, SamplerCollection
from sssrlib.transform import Flip

from ptxl.save import ImageSaver 
from ptxl.log import EpochLogger, EpochPrinter, DataQueue
from ptxl.observer import Subject, Observer, SubjectObserver
from ptxl.utils import NamedData
from resize.pt import resize
from improc3d import permute3d

from .models.rcan import RCAN


def build_trainer(args):
    return Trainer(args)


class Trainer:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self._parse_image()
        self._load_slice_profile()
        self._create_net()
        self._create_optim()
        self._create_loss_func()

        self._calc_hr_patch_size()
        self._get_iter_pattern()
        self._specify_outputs()
        self._save_args()

    def train(self):
        for self._iter_ind in range(self.args.num_iters):
            self._sampler = self._build_sampler()
            self._trainer = self._build_trainer()
            self._trainer.train()
            self._predict()

    def _create_net(self):
        if self.args.network.lower() == 'rcan':
            self.net = RCAN(self.args.num_groups, self.args.num_blocks,
                            self.args.num_channels, 16, self.args.scale,
                            num_ag=self.args.num_groups_after).cuda()
        else:
            raise NotImplementedError

    def _create_optim(self):
        if self.args.optim.lower() == 'adam':
            self.optim = Adam(self.net.parameters(),
                              lr=self.args.learning_rate)
        else:
            raise NotImplementedError

    def _create_loss_func(self):
        if self.args.loss_func.lower() == 'l1':
            self.loss_func = L1Loss()
        else:
            raise NotImplementedError

    def _specify_outputs(self):
        Path(self.args.output_dir).mkdir(parents=True)
        self.args.patches_dirname = str(Path(self.args.output_dir, 'patches'))
        self.args.log_dirname = str(Path(self.args.output_dir, 'logs'))
        self.args.result_dirname = str(Path(self.args.output_dir, 'results'))
        self.args.config = str(Path(self.args.output_dir, 'config.json'))
        Path(self.args.patches_dirname).mkdir()
        Path(self.args.log_dirname).mkdir()
        Path(self.args.result_dirname).mkdir()

    def _parse_image(self):
        obj = nib.load(self.args.image)
        self.args.voxel_size = tuple(float(v) for v in obj.header.get_zooms())
        self._image = obj.get_fdata(dtype=np.float32)
        self._get_axis_order()
        self.args.scale = float(self.args.voxel_size[self.args.z])
        self._calc_output_affine(obj.affine)
        self._output_header = obj.header

    def _get_axis_order(self):
        z = np.argmax(self.args.voxel_size)
        xy = list(range(len(self.args.voxel_size)))
        xy.remove(z)
        self.args.x = int(xy[0])
        self.args.y = int(xy[1])
        self.args.z = int(z)

    def _calc_output_affine(self, orig_affine):
        scale = self.args.scale
        scale_mat = np.array([1, 1, 1 / scale, 1])
        mat_order = [self.args.x, self.args.y, self.args.z, 3]
        scale_mat = np.diag(scale_mat[mat_order])
        self._output_affine = orig_affine @ scale_mat

    def _load_slice_profile(self):
        if self.args.slice_profile == 'gaussian':
            slice_profile = calc_gaussian_slice_profile()
        else:
            slice_profile = np.load(self.args.slice_profile)
        slice_profile = slice_profile.squeeze()[None, None, :, None]
        self._slice_profile = torch.tensor(slice_profile).float().cuda()

    def _calc_gaussian_slice_profile(self, scale):
        std = self.args.scale / (2 * np.sqrt(2 * np.log(2)))
        length = int(2 * round(scale) + 1)
        slice_profile = gaussian(length, std)
        slice_profile = slice_profile / slice_profile.sum()
        return slice_profile

    def _calc_hr_patch_size(self):
        slice_profile_len = self._slice_profile.shape[2]
        hr_patch_size = list(self.net.calc_out_patch_size(self.args.patch_size))
        hr_patch_size[0] += slice_profile_len - 1
        self.args.hr_patch_size = tuple(hr_patch_size) + (1, )

    def _save_args(self):
        result = dict()
        for arg in vars(self.args):
            result[arg] = getattr(self.args, arg)
        with open(self.args.config, 'w') as jfile:
            json.dump(result, jfile, indent=4)

    def _build_sampler(self):
        samplers = list()
        for orient in ['xy', 'yx']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        sampler = SamplerCollection(*samplers)
        return sampler

    def _build_patches(self, orient='xy'):
        if orient == 'xy':
            patches = Patches(self.args.hr_patch_size, self._image,
                              voxel_size=self.args.voxel_size, x=self.args.x,
                              y=self.args.y, z=self.args.z).cuda()
        elif orient == 'yx':
            patches = Patches(self.args.hr_patch_size, self._image,
                              voxel_size=self.args.voxel_size, x=self.args.y,
                              y=self.args.x, z=self.args.z).cuda()
        return patches

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, flip)
                for flip in self._build_flips()]

    def _build_flips(self):
        return Flip((0, )), Flip((1, )), Flip((0, 1))

    def _build_trainer(self):
        num_epochs = self._get_num_epochs()
        save_step = min(self.args.image_save_step, num_epochs)

        trainer = _Trainer(self._sampler, self._slice_profile, self.args.scale,
                           self.net, self.optim, self.loss_func,
                           batch_size=self.args.batch_size, num_epochs=num_epochs)
        queue = DataQueue(['loss'])
        printer = EpochPrinter(print_sep=False)
        filename = (self._iter_pattern % (self._iter_ind + 1)) + '.csv'
        logger = EpochLogger(Path(self.args.log_dirname, filename))
        queue.register(logger)
        queue.register(printer)

        attrs =  ['extracted', 'blur', 'lr', 'lr_interp', 'output', 'hr_crop']
        filename = (self._iter_pattern % (self._iter_ind + 1))
        filename = Path(self.args.patches_dirname, filename)
        image_saver = ImageSaver(filename, attrs=attrs,
                                 step=self.args.image_save_step, zoom=4,
                                 ordered=True, file_struct='epoch/sample',
                                 save_type='png_norm')
        trainer.register(queue)
        trainer.register(image_saver)

        return trainer

    def _get_iter_pattern(self):
        self._iter_pattern = 'iter%%0%dd' % len(str(self.args.num_iters))

    def _get_num_epochs(self):
        if self._iter_ind == 0:
            return self.args.num_epochs
        else:
            return self.args.following_num_epochs

    def cont(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])

    def _predict(self):
        image, inv_x, inv_y, inv_z = permute3d(self._image, x=self.args.x,
                                               y=self.args.y, z=self.args.z)
        image = torch.tensor(image).float().cuda()[None, None, ...]

        result0 = list()
        with torch.no_grad():
            for i in range(image.shape[2]):
                batch = image[:, :, i, ...].permute(0, 1, 3, 2)
                sr = self.net(batch)
                result0.append(sr.permute(0, 1, 3, 2))
        result0 = torch.stack(result0, dim=2)

        result1 = list()
        with torch.no_grad():
            for i in range(image.shape[3]):
                batch = image[:, :, :, i, :].permute(0, 1, 3, 2)
                sr = self.net(batch)
                result1.append(sr.permute(0, 1, 3, 2))
        result1 = torch.stack(result1, dim=3)

        result = (result1 + result0) / 2

        result = result.detach().cpu().numpy().squeeze()
        result = permute3d(result, x=inv_x, y=inv_y, z=inv_z)[0]
        out = nib.Nifti1Image(result, self._output_affine, self._output_header)
        filename = (self._iter_pattern % (self._iter_ind + 1)) + '.nii.gz'
        filename = Path(self.args.result_dirname, filename)
        out.to_filename(filename)


class _Trainer(Subject):
    def __init__(self, sampler, slice_profile, scale,
                 net, optim, loss_func, batch_size=16, num_epochs=100):
        super().__init__()
        self.sampler = sampler
        self.slice_profile = slice_profile
        self.scale = scale
        self.net = net
        self.optim = optim
        self.loss_func = loss_func
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch_ind = -1
        self._batch_ind = -1

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_batches(self):
        return 1

    @property
    def epoch_ind(self):
        return self._epoch_ind + 1

    @property
    def batch_ind(self):
        return self._batch_ind + 1

    def train(self, start_ind=0):
        self.notify_observers_on_train_start()
        num_indices = self.batch_size * self.num_epochs
        self._indices = self.sampler.sample_indices(num_indices)
        for self._epoch_ind in range(start_ind, self.num_epochs):
            self.notify_observers_on_epoch_start()
            self.notify_observers_on_batch_start()
            self._train_on_batch()
            self.notify_observers_on_batch_end()
            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()

    def _train_on_batch(self):
        start_ind = self._epoch_ind * self.batch_size
        stop_ind = start_ind + self.batch_size 
        indices = self._indices[start_ind : stop_ind]
        batch = self.sampler.get_patches(indices)
        name = batch.name

        extracted = batch.data
        blur = F.conv2d(extracted, self.slice_profile)
        lr = resize(blur, (self.scale, 1), mode='bicubic')

        self.optim.zero_grad()
        output = self.net(lr)

        hr_crop = self._crop_hr(extracted, output.shape[2:])
        lr_interp = resize(lr, (1 / self.scale, 1), mode='bicubic',
                           target_shape=output.shape[2:])

        # print('extracted', extracted.shape)
        # print('blur', blur.shape)
        # print('lr', lr.shape)
        # print('lr_interp', lr_interp.shape)
        # print('output', output.shape)
        # print('hr', hr_crop.shape)

        loss = self.loss_func(output, hr_crop)
        loss.backward()
        self.optim.step()

        self._set_tensor_cuda('extracted', extracted, name=name)
        self._set_tensor_cuda('blur', blur, name=name)
        self._set_tensor_cuda('lr', lr, name=name)
        self._set_tensor_cuda('output', output, name=name)
        self._set_tensor_cuda('lr_interp', lr_interp, name=name)
        self._set_tensor_cuda('hr_crop', hr_crop, name=name)
        self._values['loss'] = loss

    def _crop_hr(self, hr_batch, output_shape):
        left_crop = (self.slice_profile.shape[2] - 1) // 2
        right_crop = self.slice_profile.shape[2] - 1 - left_crop
        result = hr_batch[:, :, left_crop : -right_crop, ...]
        return resize(result, (1, 1), target_shape=output_shape)
