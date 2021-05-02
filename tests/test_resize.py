#!/usr/bin/env python

import numpy as np

from sssr.resize import _calc_old_fov, _calc_new_fov, _calc_new_shape 
from sssr.resize import _calc_sampling_indices, resize


def test_resize():
    old_shape = (4, 5, 6)
    old_fov = _calc_old_fov(old_shape)
    assert old_fov[0] == (0, 0, 0)
    assert old_fov[1] == old_shape

    dxyz = (0.28, 0.3, 0.12)
    new_shape = _calc_new_shape(old_shape, dxyz)
    assert new_shape == (14, 17, 50)

    new_lefts, new_rights = _calc_new_fov(old_fov, new_shape, dxyz)
    assert np.allclose(new_lefts, (0.04, -0.05, 0))
    assert np.allclose(new_rights, (3.96, 5.05, 6))

    indices = _calc_sampling_indices((new_lefts, new_rights), dxyz)
    ind0 = np.unique(indices[0])
    ind1 = np.unique(indices[1])
    ind2 = np.unique(indices[2])
    assert np.isclose(ind0[0], 0.18) \
        and np.isclose(ind0[-1], 3.82) \
        and len(ind0) == 14
    assert np.isclose(ind1[0], 0.1) \
        and np.isclose(ind1[-1], 4.9) \
        and len(ind1) == 17
    assert np.isclose(ind2[0], 0.06) \
        and np.isclose(ind2[-1], 5.94) \
        and len(ind2) == 50

    image = np.arange(np.prod(old_shape)).reshape(old_shape).astype(float)
    result = resize(image, dxyz)
    print(result[:, :, 1])

    print('successful')


if __name__ == '__main__':
    test_resize()
