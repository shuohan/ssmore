import torch
from improc3d import permute3d


class Predictor:
    def __init__(self, image, xyz, batch_size=2):
        x, y, z = xyz
        image, self._ix, self._iy, self._iz = permute3d(image, x=x, y=y, z=z)
        self.image = torch.tensor(image).float().cuda()[None, ...]
        self.batch_size = batch_size

    def predict(self, network):
        network.eval()
        with torch.no_grad():
            result = (self._predict_yz(network) + self._predict_xz(network)) / 2
        result = permute3d(result, x=self._ix, y=self._iy, z=self._iz)[0]
        result = torch.tensor(result)
        return result

    def _predict_yz(self, network):
        result = list()
        for start_ind in range(0, self.image.shape[1], self.batch_size):
            stop_ind = start_ind + self.batch_size
            batch = self.image[:, start_ind : stop_ind, :, :]
            batch = batch.permute(1, 0, 3, 2)
            sr = network(batch).cpu().permute(1, 0, 3, 2)
            result.append(sr)
        result = torch.cat(result, dim=1)
        return result.numpy().squeeze()

    def _predict_xz(self, network):
        result = list()
        for start_ind in range(0, self.image.shape[2], self.batch_size):
            stop_ind = start_ind + self.batch_size
            batch = self.image[:, :, start_ind : stop_ind, :]
            batch = batch.permute(2, 0, 3, 1)
            sr = network(batch).cpu().permute(1, 3, 0, 2)
            result.append(sr)
        result = torch.cat(result, dim=2)
        return result.numpy().squeeze()
