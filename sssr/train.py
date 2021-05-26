import json
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import L1Loss
from pathlib import Path
from copy import deepcopy

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.sample import Sampler, SamplerCollection
from sssrlib.transform import Flip
from sssrlib.utils import calc_foreground_mask, calc_avg_kernel

from ptxl.save import ImageSaver, ThreadedSaver, ImageThread
from ptxl.log import EpochLogger, EpochPrinter, DataQueue
from ptxl.observer import Subject, Observer, SubjectObserver
from ptxl.utils import NamedData
from resize.pt import resize
from improc3d import permute3d

from .models.rcan import RCAN


class TrainerBuilder:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self._parse_image()
        self._load_slice_profile()
        self._create_net()
        self._create_optim()
        self._create_loss_func()
        self._calc_hr_patch_size()
        self._specify_outputs()
        self._save_args()

    def build(self):
        predictor = Predictor(self._image)
        xyz = (self.args.x, self.args.y, self.args.z)
        sampler_builder = SamplerBuilder(self.args.hr_patch_size, xyz)
        num_batches = self._calc_num_batches()
        builder = ContentsBuilder(self.model, self.optim, self._output_affine,
                                   self._output_header, num_batches, self.args)
        contents = builder.build().contents
        trainer = Trainer(self._image, self._slice_profile, self.args.scale,
                          sampler_builder, predictor, contents, self.loss_func,
                          batch_size=self.args.batch_size,
                          num_batches=num_batches,
                          valid_step=self.args.valid_step)
        return trainer

    def _create_model(self):
        if self.args.model.lower() == 'rcan':
            self.net = RCAN(self.args.num_groups, self.args.num_blocks,
                            self.args.num_channels, 16, self.args.scale).cuda()
        else:
            raise NotImplementedError

    def _create_optim(self):
        if self.args.optim.lower() == 'adam':
            self.optim = Adam(self.net.parameters(), lr=self.args.learning_rate)
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
        self.args.log_filename = str(Path(self.args.output_dir, 'log.csv'))
        self.args.result_dirname = str(Path(self.args.output_dir, 'results'))
        self.args.config = str(Path(self.args.output_dir, 'config.json'))
        Path(self.args.patches_dirname).mkdir()
        Path(self.args.log_dirname).parent().mkdir()
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

    def _calc_num_batches(self):
        num_batches = [self.args.num_batches]
        num_batches += [self.args.following_num_batches] * self.args.num_epochs
        return num_batches


class SamplerBuilder:
    def __init__(self, patch_size, xyz):
        self.patch_size = patch_size
        self.xyz = xyz
        self._sampler = None

    @property
    def sampler(self):
        return self._sampler

    def build(self, image, voxel_size):
        samplers = list()
        for orient in ['xy', 'yx']:
            patches = self._build_patches(image, voxel_size, orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        self._sampler = SamplerCollection(*samplers)
        return self.

    def _build_patches(self, image, voxel_size, orient='xy'):
        if orient == 'xy':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[0], y=self.xyz[1], z=self.xyz[2])
        elif orient == 'yx':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[1], y=self.xyz[2], z=self.xyz[2])
        return patches.cuda()

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, flip)
                for flip in self._build_flips()]

    def _build_flips(self):
        return Flip((0, )), Flip((1, )), Flip((0, 1))


class Predictor:
    def __init__(self, image, xyz):
        x, y, z = xyz
        image, self._ix, self._iy, self._iz = permute3d(image, x=x, y=y, z=z)
        self.image = torch.tensor(image).float().cuda()[None, None, ...]
        self.net = None

    def predict(self, network):
        network.eval()

        result0 = list()
        with torch.no_grad():
            for i in range(self.image.shape[2]):
                batch = self.image[:, :, i, ...].permute(0, 1, 3, 2)
                sr = network(batch)
                result0.append(sr.permute(0, 1, 3, 2))
        result0 = torch.stack(result0, dim=2)

        result1 = list()
        with torch.no_grad():
            for i in range(self.image.shape[3]):
                batch = self.image[:, :, :, i, :].permute(0, 1, 3, 2)
                sr = network(batch)
                result1.append(sr.permute(0, 1, 3, 2))
        result1 = torch.stack(result1, dim=3)

        result = (result1 + result0) / 2
        result = result.detach().cpu().numpy().squeeze()

        return permute3d(result, x=self._ix, y=self._iy, z=self._iz)[0]


class BatchCounter(Counter):
    def __init__(self, name, nums):
        self.name = name
        self._index = 0
        self._counters = [Counter(name, n) for n in nums]

    @property
    def num(self):
        return self._counters[self._index].num

    def has_reached_end(self):
        return self._counters[self._index].has_reached_end()

    def update(self):
        self._counters[self._index].update()
        if self.index == self.num:
            self._index = min(self._index + 1, len(self._counters))

    @property
    def index(self):
        return self._counters[self._index].index

    @property
    def named_index(self):
        return self._counters[self._index].named_index


class Contents(_Contents):
    def __init__(self, model, optim, counter):
        super().__init__(model, optim, counter)

        self.best_model_state = self.model.state_dict()
        self.best_optim_state = self.optim.state_dict()

        attrs = ['hr', 'blur', 'lr', 'lr_interp', 'output', 'hr_crop']
        for attr in attrs:
            self.set_tensor_cuda('train_' + attr, None, name=None)
            self.set_tensor_cuda('valid_' + attr, None, name=None)
        self.values['train_loss'] = float('nan')
        self.values['valid_loss'] = float('nan')
        self.values['min_valid_loss'] = float('inf')

    def get_model_state_dict(self):
        return self.best_model_state

    def get_optim_state_dict(self):
        return self.best_optim_state

    def update_valid_loss(self, valid_loss):
        self.values['valid_loss'] = valid_loss
        if valid_loss < self.values['min_valid_loss']:
            self.values['min_valid_loss'] = valid_loss
            self.best_model_state = self.model.state_dict()
            self.best_optim_state = self.optim.state_dict()


class ContentsBuilder:
    def __init__(self, model, optim, affine, header, num_batches, args):
        self.model = model
        self.optim = optim
        self.num_batches = num_batches
        self.args = args
        zoom = self.args.image_save_zoom
        self._save = SaveNifti(zoom=zoom, affine=affine, header=header)

    @property
    def contents(self):
        return self._contents

    def build(self):
        epoch_counter = Counter('epoch', self.args.num_epochs)
        batch_counter = BatchCounter('batch', self.num_batches)
        counter = Counters([epoch_counter, batch_counter])
        self._contents = Contents(self.model, self.optim, counter)
        self._set_observers()
        return self

    def _set_observers(self):
        printer = Printer()
        logger = Logger(self.args.log_filename)
        attrs = self._contents.get_tensor_attrs()
        image_saver = ImageSaver(self.args.patches_dirname, self._save,
                                 attrs=attrs, step=self.args.image_save_step)
        self._contents.register(printer, 'counter')
        self._contents.register(logger, 'counter')
        self._contents.register(image_saver, 'counter')


class Trainer:
    def __init__(self, image, slice_profile, scale, sampler_builder, predictor,
                 contents, loss_func, batch_size=16, valid_step=100):
        super().__init__()
        self.image = image
        self.slice_profile = slice_profile
        self.scale = scale
        self.contents = contents
        self.loss_func = loss_func
        self.valid_step = valid_step
        self.sampler_builder = sampler_builder
        self.predictor = predictor
        self.batch_size = batch_size

        self._pred = self.image

    @property
    def prediction(self):
        if self._has_predicted:
            return self._pred
        else:
            return self._predict()

    def cont(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        return self

    def train(self):
        self.contents.start_observers()
        counter = self.contents.counter
        while not counter[0].has_reached_end():
            self._build_sampler()
            while not counter[1].has_reached_end():
                self._has_predicted = False
                self._train_on_batch()
                if self._need_to_validate():
                    self._valid_on_batch()
                self.contents.notify_observers()
            self._pred = self._predict()
            self._has_predicted = True
        self.contents.close()

    def _build_sampler(self):
        num_indices = self.batch_size * self.num_batches
        self.sampler_builder.build(self._pred, self._voxel_size)
        self._sampler = self.sampler_builder.sampler
        self._valid_indices = self._select_valid_indices()
        self._train_indices = self._select_train_indices()
        self._valid_batch = self._sampler.get_patches(self._valid_indices)
        diff = set(self._valid_indices).intersection(set(self._train_indices))
        assert len(diff) == 0

    def _select_valid_indices(self):
        mask = calc_foreground_mask(self._pred)
        flat_mask = mask.flatten()
        mapping = np.where(flat_mask > 0)[0]
        tmp_indices = np.linspace(0, len(mapping) - 1, self.num_valid_samples)
        tmp_indices = np.round(tmp_indices).astype(int)
        valid_indices = mapping[tmp_indices]
        return valid_indices

    def _select_train_indices(self):
        return self._sampler.sample_indices(self.num_train_indices,
                                            exclude=self._valid_indices)

    def _predict(self):
        return self.predictor.predict(self.net)

    def _train_on_batch(self):
        start_ind = self._batch_ind * self.batch_size
        stop_ind = start_ind + self.batch_size
        indices = self._train_indices[start_ind : stop_ind]
        batch = self._sampler.get_patches(indices)

        self.contents.model.train()
        self.contents.optim.zero_grad()
        self._apply_model(batch, prefix='train_')
        output = self._tensors_cuda['train_output'].data
        hr_crop = self._tensors_cuda['train_hr_crop'].data
        loss = self.loss_func(output, hr_crop)
        loss.backward()
        self.optim.step()
        self._values['train_loss'] = loss

    def _apply_model(self, batch, prefix=''):
        blur = F.conv2d(batch.data, self.slice_profile)
        lr = resize(blur, (self.scale, 1), mode='bicubic')
        output = self.contents.model(lr)
        hr_crop = self._crop_hr(batch.data, output.shape[2:])
        lr_interp = resize(lr, (1 / self.scale, 1), mode='bicubic',
                           target_shape=output.shape[2:])

        self.contents.set_tensor_cuda(prefix + 'hr', batch.data, batch.name)
        self.contents.set_tensor_cuda(prefix + 'blur', blur, batch.name)
        self.contents.set_tensor_cuda(prefix + 'lr', lr, batch.name)
        self.contents.set_tensor_cuda(prefix + 'lr_interp', lr_interp, batch.name)
        self.contents.set_tensor_cuda(prefix + 'output', output, batch.name)
        self.contents.set_tensor_cuda(prefix + 'hr_crop', hr_crop, batch.name)

    def _crop_hr(self, hr, target_shape):
        left_crop = (self.slice_profile.shape[2] - 1) // 2
        right_crop = self.slice_profile.shape[2] - 1 - left_crop
        result = hr[:, :, left_crop : -right_crop, ...]
        return resize(result, (1, 1), target_shape=target_shape)

    def _need_to_validate(self):
        rule1 = self.batch_ind % self.valid_step == 0
        rule2 = self.batch_ind == self.num_batches
        return rule1 or rule2

    def _valid_on_batch(self):
        self.contents.model.eval()
        with torch.no_grad():
            self._apply_net(self._valid_batch, prefix='valid_')
            output = self.contents.get_tensor_cuda('valid_output').data
            hr_crop = self.contents.get_tensor_cuda('valid_hr_crop').data
            loss = self.loss_func(output, hr_crop)
            self.contents.update_valid_loss(loss)
