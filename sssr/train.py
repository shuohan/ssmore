import torch
import torch.nn.functional as F
from ptxl.observer import Subject, Observer, SubjectObserver
from ptxl.utils import NamedData

from .resize import resize_pt


class Trainer(Subject):
    def __init__(self, sampler, slice_profile, scale0, scale1,
                 net, optim, loss_func, batch_size=16, num_epochs=100,
                 num_steps=1):
        super().__init__()
        self.sampler = sampler
        self.slice_profile = slice_profile
        self.scale0 = scale0
        self.scale1 = scale1
        self.net = net
        self.optim = optim
        self.loss_func = loss_func
        self.num_steps = num_steps
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
        lr = resize_pt(blur, (self.scale0 * self.scale1, 1))
        input = resize_pt(lr, (1 / self.scale0, 1))
        input_interp = self._interp_input(input)
        hr_crop = self._crop_hr(extracted, input.shape[2])

        self._set_tensor_cuda('extracted', extracted, name=name)
        self._set_tensor_cuda('blur', blur, name=name)
        self._set_tensor_cuda('lr', lr, name=name)
        self._set_tensor_cuda('input', input, name=name)
        self._set_tensor_cuda('input_interp', input_interp, name=name)
        self._set_tensor_cuda('hr_crop', hr_crop, name=name)

        self.optim.zero_grad()

        output = input
        outputs = list()
        for i in range(self.num_steps):
            if i > 0:
                self.net.apply_up = False
            else:
                self.net.apply_up = True
            output = self.net(output)
            self._set_tensor_cuda('output%d' % i, output, name=name)
            outputs.append(output)

        # print('extracted', extracted.shape)
        # print('blur', blur.shape)
        # print('lr', lr.shape)
        # print('input', input.shape)
        # print('input_interp', input_interp.shape)
        # print('hr', hr_crop.shape)
        # print('output', output.shape)

        loss = self.loss_func(outputs, hr_crop)
        loss.backward()
        self.optim.step()
        self._values['loss'] = loss

    def _crop_hr(self, hr_batch, length):
        crop = (self.slice_profile.shape[2] - 1) // 2
        result = hr_batch[:, :, crop : -crop, ...]
        size =  length * self.scale1
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

    def predict(self, image, apply_up=True):
        self.net.apply_up = apply_up
        image = torch.tensor(image).float().cuda()[None, None, ...]
        padding = self.net.crop_size
        padding = (padding, padding, padding, padding)

        result0 = list()
        with torch.no_grad():
            for i in range(image.shape[2]):
                batch = image[:, :, i, ...].permute(0, 1, 3, 2)
                if apply_up:
                    batch = resize_pt(batch, (1/ self.scale0, 1))
                batch = F.pad(batch, padding, mode='replicate')
                sr = self.net(batch)
                result0.append(sr.permute(0, 1, 3, 2))
        result0 = torch.stack(result0, dim=2)

        result1 = list()
        with torch.no_grad():
            for i in range(image.shape[3]):
                batch = image[:, :, :, i, :].permute(0, 1, 3, 2)
                if apply_up:
                    batch = resize_pt(batch, (1/ self.scale0, 1))
                batch = F.pad(batch, padding, mode='replicate')
                sr = self.net(batch)
                result1.append(sr.permute(0, 1, 3, 2))
        result1 = torch.stack(result1, dim=3)

        result = (result1 + result0) / 2

        return result
