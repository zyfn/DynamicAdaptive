from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmyolo.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class SetEpochInfoHook(Hook): 
    def before_train_epoch(self, runner:Runner):
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch)