import torch
import torch.nn.functional as F
from ptxl.observer import Subject, Observer, SubjectObserver
from ptxl.utils import NamedData

from .resize import resize_pt


class TrainerSR(Subject):
    def __init__(self, sampler, slice_profile, net, optim, loss_func,
                 batch_size, num_epochs):
        super().__init__()
        self.sampler = sampler
        self.slice_profile = slice_profile
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
        self.net.train()
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
        self._get_batch()
        self._get_input()
        self._get_truth()

        self.optim.zero_grad()
        self._output = self.net(self._input)
        self._loss = self.loss_func(self._output, self._truth)
        self._loss.backward()
        self.optim.step()

        # print('blur', self._blur.shape)
        # print('lr', self._lr.shape)
        # print('input', self._input.shape)
        # print('input_interp', self._input_interp.shape)
        # print('hr', self._hr_crop.shape)
        # print('output', self._output.shape)

    def _get_batch(self):
        start_ind = self._epoch_ind * self.batch_size
        stop_ind = start_ind + self.batch_size
        indices = self._indices[start_ind : stop_ind]
        batch = self.sampler.get_patches(indices)
        self._hr = batch.data
        self._names = batch.name
        self._blur = F.conv2d(self._hr, self.slice_profile)

    def _get_input(self):
        self._input = self._blur

    def _get_truth(self):
        crop = (self.slice_profile.shape[2] - 1) // 2
        self._truth = self._hr[:, :, crop : -crop, ...]

    def cont(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        self.train(start_ind=ckpt['epoch'])

    def predict(self, image):
        image = torch.tensor(image).float().cuda()[None, None, ...]
        self.net.eval()

        result0 = list()
        # with torch.no_grad():
        #     for i in range(image.shape[2]):
        #         batch = image[:, :, i, ...].permute(0, 1, 3, 2)
        #         batch = self._predict(batch)
        #         result0.append(batch.permute(0, 1, 3, 2))
        # result0 = torch.stack(result0, dim=2)

        result1 = list()
        with torch.no_grad():
            for i in range(image.shape[3]):
                batch = image[:, :, :, i, :].permute(0, 1, 3, 2)
                batch = self._predict(batch)
                result1.append(batch.permute(0, 1, 3, 2))
        result1 = torch.stack(result1, dim=3)

        # result = (result1 + result0) / 2
        result = result1

        return result

    def _predict(self, image_slice):
        result = self.net(self.net.pad(image_slice))
        return result

    def _convert_data(self, data):
        # result = data / self._intensity_max
        result = data
        result = result.detach().cpu()
        result = NamedData(self._names, result)
        return result

    @property
    def hr_cuda(self):
        return self._extracted

    @property
    def hr(self):
        return self._convert_data(self._hr)

    @property
    def blur_cuda(self):
        return self._blur

    @property
    def blur(self):
        return self._convert_data(self._blur)

    @property
    def truth_cuda(self):
        return self._truth

    @property
    def truth(self):
        return self._convert_data(self._truth)

    @property
    def input_cuda(self):
        return self._input

    @property
    def input(self):
        return self._convert_data(self._input)

    @property
    def output_cuda(self):
        return self._output

    @property
    def output(self):
        return self._convert_data(self._output)

    @property
    def loss(self):
        return self._loss.item()


class TrainerAA(TrainerSR):
    def __init__(self, sampler, slice_profile, scale0, scale1,
                 net, optim, loss_func, batch_size, num_epochs):
        super().__init__(sampler, slice_profile, net, optim, loss_func,
                         batch_size, num_epochs)
        self.scale0 = scale0
        self.scale1 = scale1

    def _get_batch(self):
        super()._get_batch()
        self._lr = resize_pt(self._blur, (self.scale0 * self.scale1, 1))

    def _get_input(self):
        self._input = resize_pt(self._lr, (1 / self.scale0, 1))
        self._input_interp = self._interp_input(self._input)

    def _interp_input(self, input_batch):
        result = resize_pt(input_batch, (1 / self.scale1, 1))
        result = F.pad(result, (0, 0, 0, self.scale1 - 1), mode='replicate')
        result = self.net.crop(result)
        return result

    def _get_truth(self):
        size = self._input.shape[2] * self.scale1
        self._truth = self.net.crop(self._blur[:, :, :size, ...])

    def _predict(self, image_slice):
        interp = resize_pt(image_slice, (1/ self.scale0, 1))
        padded = self.net.pad(interp)
        return self.net(padded)

    @property
    def input_interp_cuda(self):
        return self._input_interp

    @property
    def input_interp(self):
        return self._convert_data(self._input_interp)

    @property
    def lr_cuda(self):
        return self._lr

    @property
    def lr(self):
        return self._convert_data(self._lr)
