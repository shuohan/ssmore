import numpy as np
from improc3d import rotate3d

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import Sampler, SamplerCollection


class SamplerBuilder:
    def __init__(self, patch_size, xyz):
        self.patch_size = patch_size
        self.xyz = xyz
        self._train_sampler = None
        self._valid_sampler = None

    @property
    def train_sampler(self):
        return self._train_sampler

    @property
    def valid_sampler(self):
        return self._valid_sampler

    def build(self, image, voxel_size):
        self._train_sampler = self._build_train(image, voxel_size)
        self._valid_sampler = self._build_valid(image, voxel_size)

    def _build_train(self, image, voxel_size):
        samplers = list()
        for orient in ['xy', 'yx']:
            patches = self._build_patches(image, voxel_size, orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        return SamplerCollection(*samplers)

    def _build_valid(self, image, voxel_size):
        patches = self._build_patches(image, voxel_size, '45')
        trans_patches = self._build_trans_patches(patches)
        patches_list = [patches, *trans_patches]
        samplers = [Sampler(p) for p in patches_list]
        return SamplerCollection(*samplers)

    def _build_patches(self, image, voxel_size, orient='xy'):
        if orient == 'xy':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[0], y=self.xyz[1], z=self.xyz[2])
        elif orient == 'yx':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[1], y=self.xyz[0], z=self.xyz[2])
        elif orient == '45':
            angles = np.array((0, 0, 45))
            angles = angles[list(self.xyz)]
            image = rotate3d(image, *angles, order=3)
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[0], y=self.xyz[1], z=self.xyz[2])
            # import nibabel as nib
            # nib.Nifti1Image(image, np.eye(4)).to_filename('rot.nii.gz')
        return patches.cuda()

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, flip)
                for flip in self._build_flips()]

    def _build_flips(self):
        return Flip((0, )), Flip((1, )), Flip((0, 1))
