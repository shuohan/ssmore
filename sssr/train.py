import json
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import L1Loss
from pathlib import Path
from scipy.signal.windows import gaussian

from resize.pytorch import resize
from sssrlib.utils import calc_foreground_mask
from sssrlib.sample import SampleWeights

from .models.rcan import RCAN
from .contents import ContentsBuilder, ContentsBuilderDebug
from .predict import Predictor
from .sample import SamplerBuilder


class TrainerBuilder:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._trainer = None

    @property
    def trainer(self):
        return self._trainer

    def build(self):
        self._parse_image()
        self._load_slice_profile()
        self._create_model()
        self._create_optim()
        self._create_loss_func()
        self._calc_hr_patch_size()
        self._specify_outputs()
        self._save_args()
        xyz = (self.args.x, self.args.y, self.args.z)
        predictor = Predictor(self._image, xyz, self.args.pred_batch_size)
        sampler_builder = SamplerBuilder(self.args.hr_patch_size, xyz)
        contents_builder = self._create_contents_builder()
        self._trainer = Trainer(self._image, self._slice_profile,
                                contents_builder, sampler_builder, predictor,
                                self.loss_func, self.args)
        return self

    def _create_contents_builder(self):
        Builder = ContentsBuilderDebug if self.args.debug else ContentsBuilder
        return Builder(self.model, self.optim, self._out_affine,
                       self._out_header, self.args)

    def _create_model(self):
        if self.args.model.lower() == 'rcan':
            self.model = RCAN(self.args.num_groups, self.args.num_blocks,
                              self.args.num_channels, 16, self.args.scale)
            self.model = self.model.cuda()
        else:
            raise NotImplementedError

    def _create_optim(self):
        if self.args.optim.lower() == 'adam':
            self.optim = Adam(self.model.parameters(),
                              lr=self.args.learning_rate)
        else:
            raise notimplementederror

    def _create_loss_func(self):
        if self.args.loss_func.lower() == 'l1':
            self.loss_func = L1Loss()
        else:
            raise NotImplementedError

    def _specify_outputs(self):
        Path(self.args.output_dir).mkdir(parents=True)
        tp_dirname = str(Path(self.args.output_dir, 'train_patches'))
        self.args.train_patch_dirname = tp_dirname
        vp_dirname = str(Path(self.args.output_dir, 'valid_patches'))
        self.args.valid_patch_dirname = vp_dirname
        self.args.log_filename = str(Path(self.args.output_dir, 'log.csv'))
        self.args.result_dirname = str(Path(self.args.output_dir, 'results'))
        self.args.config = str(Path(self.args.output_dir, 'config.json'))
        cp_dirname = str(Path(self.args.output_dir, 'checkpoints'))
        self.args.output_checkpoint_dirname = cp_dirname

    def _parse_image(self):
        obj = nib.load(self.args.image)
        self.args.voxel_size = tuple(float(v) for v in obj.header.get_zooms())
        self._image = obj.get_fdata(dtype=np.float32)
        self._get_axis_order()
        self.args.scale = float(self.args.voxel_size[self.args.z])
        self._calc_output_affine(obj.affine)
        self._out_header = obj.header

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
        self._out_affine = orig_affine @ scale_mat

    def _load_slice_profile(self):
        if self.args.slice_profile == 'gaussian':
            slice_profile = self._calc_gaussian_slice_profile()
        else:
            slice_profile = np.load(self.args.slice_profile)
        self.args.slice_profile_values = slice_profile.astype(float).tolist()
        slice_profile = slice_profile.squeeze()[None, None, :, None]
        self._slice_profile = torch.tensor(slice_profile).float().cuda()

    def _calc_gaussian_slice_profile(self):
        std = self.args.scale / (2 * np.sqrt(2 * np.log(2)))
        length = int(2 * round(self.args.scale) + 1)
        slice_profile = gaussian(length, std)
        slice_profile = slice_profile / slice_profile.sum()
        return slice_profile

    def _calc_hr_patch_size(self):
        slice_profile_len = self._slice_profile.shape[2]
        hr_patch_size = self.model.calc_out_patch_size(self.args.patch_size)
        hr_patch_size[0] += slice_profile_len - 1
        self.args.hr_patch_size = tuple(hr_patch_size) + (1, )

    def _save_args(self):
        result = dict()
        for arg in vars(self.args):
            result[arg] = getattr(self.args, arg)
        with open(self.args.config, 'w') as jfile:
            json.dump(result, jfile, indent=4)


class Trainer:
    def __init__(self, image, slice_profile, contents_builder, sampler_builder,
                 predictor, loss_func, args):
        super().__init__()
        self.image = image
        self.slice_profile = slice_profile
        self.loss_func = loss_func
        self.contents_builder = contents_builder
        self.sampler_builder = sampler_builder
        self.predictor = predictor
        self.args = args
        self._voxel_size = self.args.voxel_size
        self._pred = self.image
        self._init_pred_batch_steps()

    def _init_pred_batch_steps(self):
        self._pred_batch_steps = [self.args.pred_following_batch_step] \
            * self.args.num_epochs
        self._pred_batch_steps[0] = self.args.pred_batch_step

    def train(self):
        self._build_contents()
        self.contents.start_observers()
        counter = self.contents.counter
        for i in counter['epoch']:
            self._build_sampler()
            self._update_pred_saver()
            for j in counter['batch']:
                self._has_predicted = False
                self._train_on_batch()
                if self._needs_to_validate():
                    self._valid_on_batch()
                if self._needs_to_predict():
                    self._predict()
                self.contents.notify_observers()
            counter['batch'].update()
            self.contents.revert_to_best()
            self._voxel_size = (min(self._voxel_size), ) * 3
            self.contents.set_value('valid_loss', float('nan'))
            self.contents.set_value('min_valid_loss', float('inf'))
            self.contents.set_value('min_valid_batch', float('nan'))
        self.contents.close_observers()

    def _build_contents(self):
        self.contents = self.contents_builder.build().contents
        if self.args.checkpoint:
            print('Train from checkpoint', self.args.checkpoint)
            assert Path(self.args.checkpoint).is_file()
            checkpoint = torch.load(self.args.checkpoint)
            self.contents.load_state_dicts(checkpoint)
            self._pred = checkpoint['pred'].squeeze(0).squeeze(0)
            self._voxel_size = checkpoint['voxel_size']

    def _build_sampler(self):
        self.sampler_builder.build(self._pred, self._voxel_size)
        self._train_sampler = self.sampler_builder.train_sampler
        self._valid_sampler = self.sampler_builder.valid_sampler
        self._valid_indices = self._select_valid_indices()
        self._train_indices = self._select_train_indices()
        self._valid_batch = self._valid_sampler.get_patches(self._valid_indices)

    def _update_pred_saver(self):
        pred_step = self._get_current_pred_batch_step()
        self.contents_builder.update_pred_batch_step(pred_step)

    def _select_valid_indices(self):
        valid_indices = list()
        num_valid_samples = self.args.num_valid_samples \
            // self._valid_sampler.num_samplers
        for i, patches in enumerate(self._valid_sampler.patches.sub_patches):
            mask = calc_foreground_mask(patches.image)
            flat_mask = SampleWeights(patches, (mask, )).weights_flat
            mapping = torch.where(flat_mask > 0)[0].cpu().numpy()
            indices = np.linspace(0, len(mapping) - 1, num_valid_samples)
            indices = np.round(indices).astype(int)
            i_array = np.ones(len(indices), dtype=int) * i
            indices = mapping[indices]
            indices = np.hstack((i_array[:, None], indices[:, None]))
            valid_indices.append(indices)
        valid_indices = np.vstack(valid_indices)
        return valid_indices

    def _select_train_indices(self):
        num_batches = self.contents.counter['batch'].num
        num_indices = self.args.batch_size * num_batches
        return self._train_sampler.sample_indices(num_indices)

    def _predict(self):
        self._pred = self.predictor.predict(self.contents.best_model)
        self.contents.set_tensor_cpu('pred', self._pred[None, None, ...], '')
        self.contents.set_value('voxel_size', self._voxel_size)

    def _train_on_batch(self):
        batch_ind = self.contents.counter['batch'].index0
        start_ind = batch_ind * self.args.batch_size
        stop_ind = start_ind + self.args.batch_size
        indices = self._train_indices[start_ind : stop_ind]
        batch = self._train_sampler.get_patches(indices)

        self.contents.model.train()
        self.contents.optim.zero_grad()
        self._apply_model(batch, prefix='train_')
        output = self.contents.get_tensor_cuda('train_output').data
        hr_crop = self.contents.get_tensor_cuda('train_hr_crop').data
        loss = self.loss_func(output, hr_crop)
        loss.backward()
        self.contents.optim.step()
        self.contents.set_value('train_loss', loss.item())

    def _apply_model(self, batch, prefix=''):
        blur = F.conv2d(batch.data, self.slice_profile)
        lr = resize(blur, (self.args.scale, 1), order=3)
        output = self.contents.model(lr)
        hr_crop = self._crop_hr(batch.data, output.shape[2:])
        lr_interp = resize(lr, (1 / self.args.scale, 1), order=3,
                           target_shape=output.shape[2:])

        name = batch.name
        self.contents.set_tensor_cuda(prefix + 'hr', batch.data, name)
        self.contents.set_tensor_cuda(prefix + 'blur', blur, name)
        self.contents.set_tensor_cuda(prefix + 'lr', lr, name)
        self.contents.set_tensor_cuda(prefix + 'lr_interp', lr_interp, name)
        self.contents.set_tensor_cuda(prefix + 'output', output, name)
        self.contents.set_tensor_cuda(prefix + 'hr_crop', hr_crop, name)

    def _crop_hr(self, hr, target_shape):
        left_crop = (self.slice_profile.shape[2] - 1) // 2
        right_crop = self.slice_profile.shape[2] - 1 - left_crop
        result = hr[:, :, left_crop : -right_crop, ...]
        return resize(result, (1, 1), target_shape=target_shape, order=3)

    def _needs_to_validate(self):
        counter = self.contents.counter['batch']
        rule1 = counter.index1 % self.args.valid_step == 0
        rule2 = counter.has_reached_end()
        return rule1 or rule2

    def _needs_to_predict(self):
        counter = self.contents.counter['batch']
        rule1 = counter.index1 % self._get_current_pred_batch_step() == 0
        rule2 = counter.has_reached_end()
        return rule1 or rule2

    def _get_current_pred_batch_step(self):
        epoch_ind = self.contents.counter['epoch'].index0
        return self._pred_batch_steps[epoch_ind]

    def _valid_on_batch(self):
        self.contents.model.eval()
        with torch.no_grad():
            self._apply_model(self._valid_batch, prefix='valid_')
            output = self.contents.get_tensor_cuda('valid_output').data
            hr_crop = self.contents.get_tensor_cuda('valid_hr_crop').data
            loss = self.loss_func(output, hr_crop)
            self.contents.update_valid_loss(loss)
