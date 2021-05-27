import torch
from improc3d import permute3d


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

        result = permute3d(result, x=self._ix, y=self._iy, z=self._iz)[0]
        result = torch.tensor(result)
        return result
