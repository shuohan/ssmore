from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import Sampler, SamplerCollection, SampleWeights
from sssrlib.utils import calc_foreground_mask


class SamplerBuilder:
    def __init__(self, patch_size, xyz):
        self.patch_size = patch_size
        self.xyz = xyz
        self._sampler = None

    @property
    def sampler(self):
        return self._sampler

    def build(self, image, voxel_size):
        samplers = list()
        for orient in ['xy', 'yx']:
            patches = self._build_patches(image, voxel_size, orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        self._sampler = SamplerCollection(*samplers)
        return self

    def _build_patches(self, image, voxel_size, orient='xy'):
        if orient == 'xy':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[0], y=self.xyz[1], z=self.xyz[2])
        elif orient == 'yx':
            patches = Patches(self.patch_size, image, voxel_size=voxel_size,
                              x=self.xyz[1], y=self.xyz[0], z=self.xyz[2])
        return patches.cuda()

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, flip)
                for flip in self._build_flips()]

    def _build_flips(self):
        return Flip((0, )), Flip((1, )), Flip((0, 1))
