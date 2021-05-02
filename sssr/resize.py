"""Resize with correct sampling step

"""
import numpy as np
from scipy.ndimage import map_coordinates


def resize(image, dxyz, order=3):
    """Resize the image with sampling step dx, dy, and dz.

    Args:
        image (numpy.ndarray): The image to resample.
        dxyz (tuple[float]): The sampling step. Less than 1 for upsampling.
        order (int): B-spline interpolation order.

    Returns:
        numpy.ndarray: The resampled image.

    """
    old_fov = _calc_old_fov(image.shape)
    new_shape = _calc_new_shape(image.shape, dxyz)
    new_fov = _calc_new_fov(old_fov, new_shape, dxyz)
    indices = _calc_sampling_indices(new_fov, dxyz)
    result = map_coordinates(image, indices, mode='nearest', order=order)
    result = result.reshape(new_shape)
    return result


def _calc_old_fov(old_shape):
    """Calculates the FOV of the original image.

    Suppose the left boundaries are at 0, then the first voxel is at
    (0.5, 0.5, 0.5). Assume the step size is 1.

    """
    step_size = 1
    lefts = (0, 0, 0)
    rights = tuple(l + s * step_size for l, s in zip(lefts, old_shape))
    return lefts, rights


def _calc_new_fov(old_fov, new_shape, dxyz):
    """Calculates the FOV of the resulting image.

    Assume the old and new FOV have the same center, then the new FOV is shifted
    from the old FOV by half of the size difference.

    """
    old_size = [r - l for l, r in zip(*old_fov)]
    new_size = [s * d for s, d in zip(new_shape, dxyz)]
    size_diff = [(o - n) / 2 for o, n in zip(old_size, new_size)]
    lefts = tuple(l + sd for l, sd in zip(old_fov[0], size_diff))
    rights = tuple(r - sd for r, sd in zip(old_fov[1], size_diff))
    return lefts, rights


def _calc_new_shape(old_shape, dxyz):
    """Calculates the shape of the interpolated image."""
    return tuple(round(s / d) for s, d in zip(old_shape, dxyz))


def _calc_sampling_indices(new_fov, dxyz):
    """Calculates the sampling indices in the original image.

    Note:
        The values should be sampled at pixel centers, so the first index should
        be shifted by half a pixel from the left boundaries.

    """
    indices = [np.arange(l + d/2, r - d/4, d)
               for l, r, d in zip(new_fov[0], new_fov[1], dxyz)]
    grid = np.meshgrid(*indices, indexing='ij')
    grid = [g.flatten() for g in grid]
    return np.array(grid)
