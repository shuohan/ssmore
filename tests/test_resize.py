#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib

from sssr.resize import _calc_old_fov, _calc_new_fov, _calc_new_shape
from sssr.resize import _calc_sampling_indices, resize


def test_mid():
    old_shape = (4, 5, 6)
    dxyz = (0.28, 0.3, 0.12)

    old_fov = _calc_old_fov(old_shape)
    assert old_fov[0] == (-0.5, -0.5, -0.5)
    assert old_fov[1] == (3.5, 4.5, 5.5)

    new_shape = _calc_new_shape(old_shape, dxyz)
    assert new_shape == (14, 17, 50)

    new_lefts, new_rights = _calc_new_fov(old_fov, new_shape, dxyz)
    assert np.allclose(new_lefts, (-0.46, -0.55, -0.5))
    assert np.allclose(new_rights, (3.46, 4.55, 5.5))

    indices = _calc_sampling_indices((new_lefts, new_rights), dxyz)
    ind0 = np.unique(indices[0])
    ind1 = np.unique(indices[1])
    ind2 = np.unique(indices[2])
    assert np.isclose(ind0[0], -0.32) \
        and np.isclose(ind0[-1], 3.32) \
        and len(ind0) == 14
    assert np.isclose(ind1[0], -0.4) \
        and np.isclose(ind1[-1], 4.4) \
        and len(ind1) == 17
    assert np.isclose(ind2[0], -0.44) \
        and np.isclose(ind2[-1], 5.44) \
        and len(ind2) == 50


def test_resize():
    old_shape = (4, 5, 6)
    dxyz = (0.28, 0.3, 0.12)
    image = np.random.rand(*old_shape).astype(np.float32)
    result = resize(image, dxyz, order=1)

    os.system('rm -f old_image.nii.gz new_image.nii.gz')
    nib.Nifti1Image(image, np.eye(4)).to_filename('old_image.nii.gz')
    command = ['3dresample', '-inset', 'old_image.nii.gz', '-prefix',
               'new_image.nii.gz', '-rmode', 'Linear', '-dxyz',
               str(dxyz[0]), str(dxyz[1]), str(dxyz[2])]
    os.system(' '.join(command))
    afni = nib.load('new_image.nii.gz').get_fdata(dtype=np.float32)

    assert np.allclose(result[2:-2, 2:-2, 4:-4], afni[2:-2, 2:-2, 4:-4])

    print('successful')


if __name__ == '__main__':
    test_resize()
