from copy import deepcopy

from ptxl.abstract import Contents as _Contents
from ptxl.utils import Counter_, Counter, Counters
from ptxl.log import Logger, Printer
from ptxl.save import ImageSaver, SaveNifti, SavePngNorm


class BatchCounter(Counter_):
    def __init__(self, name, nums):
        self._name = name
        self._counter_index = 0
        self._counters = [Counter(name, n) for n in nums]

    @property
    def name(self):
        return self._name

    @property
    def num(self):
        return self._counters[self._counter_index].num

    @property
    def index(self):
        return self._counters[self._counter_index].index

    @property
    def named_index(self):
        return self._counters[self._counter_index].named_index

    def __iter__(self):
        return self._counters[self._counter_index]

    def update(self):
        self._counter_index = self._counter_index + 1
        self._counter_index = min(self._counter_index, len(self._counters) - 1)

    def has_reached_end(self):
        return self._counters[self._counter_index].has_reached_end()


class Contents(_Contents):
    def __init__(self, model, optim, counter):
        super().__init__(model, optim, counter)

        self.best_model = deepcopy(self.model)
        self.best_optim_state = self.optim.state_dict()

        attrs = ['hr', 'blur', 'lr', 'lr_interp', 'output', 'hr_crop']
        for attr in attrs:
            self.set_tensor_cuda('train_' + attr, None, name=None)
            self.set_tensor_cuda('valid_' + attr, None, name=None)
        self.set_tensor_cpu('pred', None, name=None)
        self.set_value('train_loss', float('nan'))
        self.set_value('valid_loss', float('nan'))
        self.set_value('min_valid_loss', float('inf'))

    def get_model_state_dict(self):
        return self.best_model.state_dict()

    def get_optim_state_dict(self):
        return self.best_optim_state

    def update_valid_loss(self, valid_loss):
        valid_loss = valid_loss.item()
        self.set_value('valid_loss', valid_loss)
        if valid_loss < self.get_value('min_valid_loss'):
            self.set_value('min_valid_loss', valid_loss)
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_optim_state = self.optim.state_dict()

    def revert_to_best(self):
        self.model.load_state_dict(self.best_model.state_dict())
        self.optim.load_state_dict(self.best_optim_state)


class PatchSaver(ImageSaver):
    def _needs_to_update(self):
        rule1 = self.contents.counter['batch'].index % self.step == 0
        rule2 = self.contents.counter['batch'].has_reached_end()
        return rule1 or rule2


class PredSaver(ImageSaver):
    def _needs_to_update(self):
        erule1 = self.contents.counter['epoch'].index % self.step[0] == 0
        erule2 = self.contents.counter['epoch'].has_reached_end()
        epoch_rule = erule1 or erule2
        brule1 = self.contents.counter['batch'].index % self.step[1] == 0
        brule2 = self.contents.counter['batch'].has_reached_end()
        batch_rule = brule1 or brule2
        return batch_rule and epoch_rule
    def _get_filename(self, sind, aind, attr, num_samples):
        filename = super()._get_filename(sind, aind, attr, num_samples)
        min_valid_loss = self.contents.get_value('min_valid_loss')
        min_valid_loss = ('min-val-%.2e' % min_valid_loss).replace('.', 'p')
        filename = '_'.join([filename, min_valid_loss])
        return filename


class ContentsBuilder:
    def __init__(self, model, optim, affine, header, nums_batches, args):
        self.model = model
        self.optim = optim
        self.nums_batches = nums_batches
        self.args = args
        self._save_nii = SaveNifti(affine=affine, header=header)
        self._save_png = SavePngNorm(zoom=args.patch_save_zoom)

    @property
    def contents(self):
        return self._contents

    def build(self):
        epoch_counter = Counter('epoch', self.args.num_epochs)
        batch_counter = BatchCounter('batch', self.nums_batches)
        counter = Counters([epoch_counter, batch_counter])
        self._contents = Contents(self.model, self.optim, counter)
        self._set_observers()
        return self

    def _set_observers(self):
        attrs = self._contents.get_value_attrs()
        printer = Printer(attrs=attrs)
        logger = Logger(self.args.log_filename, attrs=attrs)
        self._contents.register(printer)
        self._contents.register(logger)

        attrs = self._contents.get_tensor_attrs()
        attrs = ['hr', 'blur', 'lr', 'lr_interp', 'output', 'hr_crop']
        train_saver = PatchSaver(self.args.train_patch_dirname, self._save_png,
                                 attrs=['train_' + a for a in attrs],
                                 step=self.args.patch_save_step)
        valid_saver = PatchSaver(self.args.valid_patch_dirname, self._save_png,
                                 attrs=['valid_' + a for a in attrs],
                                 step=self.args.patch_save_step)
        step = (self.args.pred_epoch_step, self.args.pred_batch_step)
        pred_saver = PredSaver(self.args.result_dirname, self._save_nii,
                               attrs=['pred'], step=step, use_new_folder=False)
        self._contents.register(train_saver)
        self._contents.register(valid_saver)
        self._contents.register(pred_saver)


