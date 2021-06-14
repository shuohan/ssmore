#!/usr/bin/env python

import torch
import numpy as np
from sssr.resize import _calc_old_fov_no_extra, _calc_new_fov_no_extra
from sssr.resize import _calc_new_shape_no_extra, resize_pt
from sssr.resize import _calc_sampling_indices_no_extra_pt


def test_resize_pt():
    old_shape = (4, 5, 6)
    dxyz = (0.7, 2, 2.3)
    old_fov = _calc_old_fov_no_extra(old_shape)
    new_shape = _calc_new_shape_no_extra(old_shape, dxyz)
    new_fov = _calc_new_fov_no_extra(new_shape, dxyz)
    assert new_fov == (2.8, 4.0, 4.6)

    # indices = _calc_sampling_indices_no_extra_pt(new_fov, dxyz)
    # assert indices.shape == (1, 5, 3, 3, 3)
    # assert np.allclose(np.unique(indices[0, ..., 0]), [0, 2, 4])
    # assert np.allclose(np.unique(indices[0, ..., 1]), [0., 0.7, 1.4, 2.1, 2.8])
    # assert np.allclose(np.unique(indices[0, ..., 2]), [0., 2.3, 4.6])

    old_shape = (4, 5)
    dxyz = (1.3, 1)
    image = np.arange(np.prod(old_shape)).reshape(old_shape)
    image = torch.tensor(image).float()[None, None, ...]
    image = torch.cat((image + 100, image), 0)
    result = resize_pt(image, dxyz)
    ref = torch.tensor([[0, 1, 2, 3, 4],
                        [6.5, 7.5, 8.5, 9.5, 10.5],
                        [13, 14, 15, 16, 17]]).float()
    assert torch.equal(result[1].squeeze(), ref)


if __name__ == '__main__':
    test_resize_pt()
