import torch
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

        self.optim.zero_grad()
        self._output = self.net(self._input)

        # print('extracted', self._extracted.shape)
        # print('blur', self._blur.shape)
        # print('lr', self._lr.shape)
        # print('input', self._input.shape)
        # print('input_interp', self._input_interp.shape)
        # print('hr', self._hr_crop.shape)
        # print('output', self._output.shape)

        self._loss = self.loss_func(self._output, self._hr_crop)
        self._loss.backward()
        self.optim.step()

    def _crop_hr(self, hr_batch):
        crop = (self.slice_profile.shape[2] - 1) // 2
        result = hr_batch[:, :, crop : -crop, ...]
        size = self._input.shape[2] * self.scale1
        result = result[:, :, :size, ...]
        result = self.net.crop(result)
        return result

    def _interp_input(self, input_batch):
        result = resize_pt(input_batch, (1 / self.scale1, 1))
        result = F.pad(result, (0, 0, 0, self.scale1 - 1), mode='replicate')
        result = self.net.crop(result)
        return result

    def cont(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        self.train(start_ind=ckpt['epoch'])

    def predict(self, image):
        image = torch.tensor(image).float().cuda()[None, None, ...]
        padding = self.net.crop_size
        padding = (padding, padding, padding, padding)

        result0 = list()
        with torch.no_grad():
            for i in range(image.shape[2]):
                batch = image[:, :, i, ...].permute(0, 1, 3, 2)
                interp = resize_pt(batch, (1/ self.scale0, 1))
                padded = F.pad(interp, padding, mode='replicate')
                sr = self.net(padded)
                result0.append(sr.permute(0, 1, 3, 2))
        result0 = torch.stack(result0, dim=2)

        result1 = list()
        with torch.no_grad():
            for i in range(image.shape[3]):
                batch = image[:, :, :, i, :].permute(0, 1, 3, 2)
                interp = resize_pt(batch, (1/ self.scale0, 1))
                padded = F.pad(interp, padding, mode='replicate')
                sr = self.net(padded)
                result1.append(sr.permute(0, 1, 3, 2))
        result1 = torch.stack(result1, dim=3)

        result = (result1 + result0) / 2

        return result

    def _convert_data(self, data):
        # result = data / self._intensity_max
        result = data
        result = result.detach().cpu()
        result = NamedData(self._names, result)
        return result

    @property
    def extracted_cuda(self):
        return self._extracted

    @property
    def extracted(self):
        return self._convert_data(self._extracted)

    @property
    def blur_cuda(self):
        return self._blur

    @property
    def blur(self):
        return self._convert_data(self._blur)

    @property
    def lr_cuda(self):
        return self._lr

    @property
    def lr(self):
        return self._convert_data(self._lr)

    @property
    def hr_crop_cuda(self):
        return self._hr_crop

    @property
    def hr_crop(self):
        return self._convert_data(self._hr_crop)

    @property
    def input_cuda(self):
        return self._input

    @property
    def input(self):
        return self._convert_data(self._input)

    @property
    def input_interp_cuda(self):
        return self._input_interp

    @property
    def input_interp(self):
        return self._convert_data(self._input_interp)

    @property
    def output_cuda(self):
        return self._output

    @property
    def output(self):
        return self._convert_data(self._output)

    @property
    def loss(self):
        return self._loss.item()
