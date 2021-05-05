import math
import torch.nn.functional as F
from ptxl.observer import Subject, Observer, SubjectObserver
from ptxl.utils import NamedData

from .resize import resize_pt


class Trainer(Subject):
    def __init__(self, sampler, slice_profile, scale0, scale1,
                 net, optim, loss_func, batch_size=16, num_epochs=100):
        super().__init__()
        self.sampler = sampler
        self.slice_profile = slice_profile
        self.scale0 = scale0
        self.scale1 = scale1
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
        self._names = batch.name

        self._extracted = batch.data
        self._blur = F.conv2d(self._extracted, self.slice_profile)
        self._lr = resize_pt(self._blur, (self.scale0 * self.scale1, 1))
        self._input = resize_pt(self._lr, (1 / self.scale0, 1))
        self._input_interp = self._interp_input(self._input)
        self._hr_crop = self._crop_hr(self._extracted)

        # print('blur', self._blur.shape)
        # print('lr', self._lr.shape)
        # print('input', self._input.shape)
        # print('hr', self._hr.shape)

        self.optim.zero_grad()
        self._output = self.net(self._input)

        # print('output', self._output.shape)

        self._loss = self.loss_func(self._output, self._hr_crop)
        self._loss.backward()
        self.optim.step()

    def _crop_hr(self, batch):
        crop = (self.slice_profile.shape[2] - 1) // 2
        result = batch[:, :, crop : -crop, ...]
        size = self._input.shape[2] * self.scale1
        result = result[:, :, :size, ...]
        crop1 = 2 * (self.net.num_blocks + 1)
        crop0 = self.scale1 * crop1
        result = result[:, :, crop0 : -crop0, crop1 : -crop1]
        return result

    def _interp_input(self, batch):
        crop = 2 * (self.net.num_blocks + 1)
        result = batch[:, :, crop : -crop, crop : -crop]
        result = resize_pt(result, (1 / self.scale1, 1))
        pad = self.scale1 - 1
        result = F.pad(result, (0, 0, 0, pad), mode='replicate')
        return result

    def cont(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        self.train(start_ind=ckpt['epoch'])

    @property
    def extracted_cuda(self):
        return self._extracted

    @property
    def extracted(self):
        return NamedData(self._names, self._extracted.detach().cpu())

    @property
    def blur_cuda(self):
        return self._blur

    @property
    def blur(self):
        return NamedData(self._names, self._blur.detach().cpu())

    @property
    def lr_cuda(self):
        return self._lr

    @property
    def lr(self):
        return NamedData(self._names, self._lr.detach().cpu())

    @property
    def hr_crop_cuda(self):
        return self._hr_crop

    @property
    def hr_crop(self):
        return NamedData(self._names, self._hr_crop.detach().cpu())

    @property
    def input_cuda(self):
        return self._input

    @property
    def input(self):
        return NamedData(self._names, self._input.detach().cpu())

    @property
    def input_interp_cuda(self):
        return self._input_interp

    @property
    def input_interp(self):
        return NamedData(self._names, self._input_interp.detach().cpu())

    @property
    def output_cuda(self):
        return self._output

    @property
    def output(self):
        return NamedData(self._names, self._output.detach().cpu())

    @property
    def loss(self):
        return self._loss.item()
