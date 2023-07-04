# Copyright (c) OpenMMLab. All rights reserved.
from .ppyoloe_param_scheduler_hook import PPYOLOEParamSchedulerHook
from .switch_to_deploy_hook import SwitchToDeployHook
from .yolov5_param_scheduler_hook import YOLOv5ParamSchedulerHook
from .yolox_mode_switch_hook import YOLOXModeSwitchHook
###################################################################
from .set_epoch_hook import SetEpochInfoHook
__all__ = [
    'YOLOv5ParamSchedulerHook', 'YOLOXModeSwitchHook', 'SwitchToDeployHook',
    'PPYOLOEParamSchedulerHook','SetEpochInfoHook'
]
