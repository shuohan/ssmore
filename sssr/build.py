from ptxl.save import ImageSaver 
from ptxl.log import EpochLogger, EpochPrinter, DataQueue

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.sample import Sampler, SamplerCollection
from sssrlib.transform import Flip

from .train import Trainer


def build_sampler(image, patch_size, xyz, voxel_size):
    """Builds patches

    Args:
        xyz (tuple[int]): The indices of x, y, and z axes.

    """
    x, y, z = xyz

    flip0 = Flip((0, ))
    flip1 = Flip((1, ))
    flip01 = Flip((0, 1))

    patches00 = Patches(patch_size, image, voxel_size=voxel_size,
                        x=x, y=y, z=z).cuda()
    patches01 = TransformedPatches(patches00, flip0)
    patches02 = TransformedPatches(patches00, flip1)
    patches03 = TransformedPatches(patches00, flip01)

    patches10 = Patches(patch_size, image, voxel_size=voxel_size,
                        x=y, y=x, z=z).cuda()
    patches11 = TransformedPatches(patches10, flip0)
    patches12 = TransformedPatches(patches10, flip1)
    patches13 = TransformedPatches(patches10, flip01)

    sampler00 = Sampler(patches00) # uniform sampling
    sampler01 = Sampler(patches01) # uniform sampling
    sampler02 = Sampler(patches02) # uniform sampling
    sampler03 = Sampler(patches03) # uniform sampling

    sampler10 = Sampler(patches10) # uniform sampling
    sampler11 = Sampler(patches11) # uniform sampling
    sampler12 = Sampler(patches12) # uniform sampling
    sampler13 = Sampler(patches13) # uniform sampling

    sampler = SamplerCollection(sampler00, sampler01, sampler02, sampler03,
                                sampler10, sampler11, sampler12, sampler13)

    return sampler


def build_trainer(sampler, slice_profile, net, optim, loss_func, args, iter):
    num_epochs = args.num_epochs if iter == 0 else args.following_num_epochs
    step = min(args.image_save_step, num_epochs)

    trainer = Trainer(sampler, slice_profile, args.scale0, args.scale1,
                      net, optim, loss_func, batch_size=args.batch_size,
                      num_epochs=num_epochs, num_steps=args.num_net_steps)

    if iter % args.iter_save_step == 0 or iter == args.num_iters - 1:
        queue = DataQueue(['loss'])
        logger = EpochLogger(args.log_filename + '%d.csv' % iter)
        printer = EpochPrinter(print_sep=False)
        queue.register(logger)
        queue.register(printer)
        attrs =  ['extracted', 'blur', 'lr', 'input_interp']
        for i in range(args.num_net_steps):
            attrs.append('output%d' % i)
        attrs.append('hr_crop')
        image_saver = ImageSaver(args.image_dirname + '%d' % iter, attrs=attrs,
                                 step=step, zoom=4, ordered=True,
                                 file_struct='epoch/sample', save_type='png_norm')
        trainer.register(queue)
        trainer.register(image_saver)

    return trainer
