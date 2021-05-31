from copy import deepcopy

from ptxl.abstract import Contents as _Contents
from ptxl.utils import Counter_, Counter, Counters
from ptxl.log import Logger, Printer, MultiTqdmPrinter
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
    def index0(self):
        return self._counters[self._counter_index].index0

    @property
    def named_index0(self):
        return self._counters[self._counter_index].named_index0

    @property
    def index1(self):
        return self._counters[self._counter_index].index1

    @property
    def named_index1(self):
        return self._counters[self._counter_index].named_index1

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
        rule1 = self.contents.counter['batch'].index1 % self.step == 0
        rule2 = self.contents.counter['batch'].has_reached_end()
        return rule1 or rule2


class PredSaver(ImageSaver):
    def _needs_to_update(self):
        erule1 = self.contents.counter['epoch'].index1 % self.step[0] == 0
        erule2 = self.contents.counter['epoch'].has_reached_end()
        epoch_rule = erule1 or erule2
        brule1 = self.contents.counter['batch'].index1 % self.step[1] == 0
        brule2 = self.contents.counter['batch'].has_reached_end()
        batch_rule = brule1 or brule2
        return batch_rule and epoch_rule
    def _get_filename(self, sind, aind, attr, num_samples):
        filename = super()._get_filename(sind, aind, attr, num_samples)
        min_valid_loss = self.contents.get_value('min_valid_loss')
        min_valid_loss = ('min-val-%.4e' % min_valid_loss).replace('.', 'p')
        filename = '_'.join([filename, min_valid_loss])
        return filename


class ContentsBuilder:
    def __init__(self, model, optim, affine, header, args):
        self.model = model
        self.optim = optim
        self.args = args
        self._init_nums_batches()
        self._save_nii = SaveNifti(affine=affine, header=header)
        self._contents = None

    def _init_nums_batches(self):
        self._nums_batches = [self.args.following_num_batches] \
            * self.args.num_epochs
        self._nums_batches[0] = self.args.num_batches

    @property
    def contents(self):
        return self._contents

    def build(self):
        epoch_counter = Counter('epoch', self.args.num_epochs)
        batch_counter = BatchCounter('batch', self._nums_batches)
        counter = Counters([epoch_counter, batch_counter])
        self._contents = Contents(self.model, self.optim, counter)
        self._set_observers()
        return self

    def update_pred_batch_step(self, step):
        self._pred_saver.step[1] = step

    def _set_observers(self):
        self._printer = self._create_printer()
        self._logger = self._create_logger()
        self._pred_saver = self._create_pred_saver()
        self._contents.register(self._printer)
        self._contents.register(self._logger)
        self._contents.register(self._pred_saver)

    def _create_printer(self):
        attrs = self._contents.get_value_attrs()
        return MultiTqdmPrinter(attrs=attrs)

    def _create_logger(self):
        attrs = self._contents.get_value_attrs()
        return Logger(self.args.log_filename, attrs=attrs)

    def _create_pred_saver(self):
        step = [self.args.pred_epoch_step, self.args.pred_batch_step]
        return PredSaver(self.args.result_dirname, self._save_nii,
                         attrs=['pred'], step=step, use_new_folder=False)


class ContentsBuilderDebug(ContentsBuilder):
    def __init__(self, model, optim, affine, header, args):
        super().__init__(model, optim, affine, header, args)
        self._save_png = SavePngNorm(zoom=args.patch_save_zoom)

    def _set_observers(self):
        super()._set_observers()
        self._train_saver = self._create_train_saver()
        self._valid_saver = self._create_valid_saver()
        self._contents.register(self._train_saver)
        self._contents.register(self._valid_saver)

    def _create_printer(self):
        attrs = self._contents.get_value_attrs()
        return Printer(attrs=attrs)

    def _create_train_saver(self):
        attrs = ['train_' + a for a in self._get_patch_saver_attrs()]
        return PatchSaver(self.args.train_patch_dirname, self._save_png,
                          attrs=attrs, step=self.args.patch_save_step)

    def _create_valid_saver(self):
        attrs = ['valid_' + a for a in self._get_patch_saver_attrs()]
        return PatchSaver(self.args.valid_patch_dirname, self._save_png,
                          attrs=attrs, step=self.args.patch_save_step)

    def _get_patch_saver_attrs(self):
        return ['hr', 'blur', 'lr', 'lr_interp', 'output', 'hr_crop']
