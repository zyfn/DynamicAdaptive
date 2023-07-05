# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..utils import make_divisible, make_round
from .base_yolo_neck import BaseYOLONeck

import torch.nn.functional as F
from mmengine.model import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer

@MODELS.register_module()
class YOLOv5PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv5.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 1,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = num_csp_blocks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()


    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = ConvModule(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int):
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """

        if idx == 1:
            return CSPLayer(
                make_divisible(self.in_channels[idx - 1] * 2,
                               self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return nn.Sequential(
                CSPLayer(
                    make_divisible(self.in_channels[idx - 1] * 2,
                                   self.widen_factor),
                    make_divisible(self.in_channels[idx - 1],
                                   self.widen_factor),
                    num_blocks=make_round(self.num_csp_blocks,
                                          self.deepen_factor),
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    make_divisible(self.in_channels[idx - 1],
                                   self.widen_factor),
                    make_divisible(self.in_channels[idx - 2],
                                   self.widen_factor),
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(
            make_divisible(self.in_channels[idx] * 2, self.widen_factor),
            make_divisible(self.in_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()

########################SPPAttention############################
    def build_spp_attention(self,idx:int) -> nn.Module:
        """build SPPAttention."""
        in_out_channel =int(self.out_channels[idx]*self.widen_factor)
        return SPPAttention(in_out_channel,in_out_channel)


####################SPPAttention#############################################
class SPPAttention(nn.Module):
    def __init__(self, c1, c2, k=(1, 3, 5),reduction=16):
        super().__init__()
        ################SPP###################
        self.dyConv = torch.nn.Conv2d(3*c1, c2, 3, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        ##############Attention################
        self.gap=torch.nn.AdaptiveAvgPool2d(1)
        self.eca=eca_layer(c1)
        self.att=self_Attn(c1)
    def forward(self, x):
        b, c, h, w = x.size()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            max_pool=[m(x) for m in self.m]
            max_pool_cat=torch.cat([max_pool[i] for i,val in enumerate(max_pool)], 1)
            out_feature_selector = self.dyConv(max_pool_cat)
            gap=self.gap(out_feature_selector).view(b,c,h,w)
            att_adaptive=self.eca(gap)
            att_self=self.mlp(out_feature_selector).view(b,c,1,1)
            return torch.add(att_adaptive,att_self)


###########################################ECA################################################################https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        # Two different branches of ECA module
        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y=y_avg+y_max
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

###########################################spatialAttention################################################################https://blog.csdn.net/lzzzzzzm/article/details/123558175
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)
        return out


###########################################self-attention################################################################https://zhuanlan.zhihu.com/p/283125663
class self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=None):
        super(self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C/8
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N/8
        energy =  torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        attention = self.softmax(energy) # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        out = out.view(m_batchsize,C,width,height) # B*C*H*W
 
        out = self.gamma*out + x
        return out,attention

#############################################动态卷积#########################################################
class attention2d(torch.nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.att=torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden_planes, K, 1, bias=True),
        )
        self.temperature = temperature
    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x=self.att(x).view(x.size(0),-1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25,stride=1, padding=0, dilation=1, groups=1, bias=False, K=2,                 
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes,ratio, K, 34)
        self.now_epoch = 0

        self.weight = torch.nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            norm_channels = in_planes
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)  # type: ignore
            self.add_module(self.norm_name, norm)
            if self.bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None  # type: ignore
        
        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.activate = build_activation_layer(act_cfg_)
        
        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        
        if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
            a = self.act_cfg.get('negative_slope', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        nn.init.kaiming_normal_(
                self.weight, a=a, mode='fan_out', nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        if(hasattr(self,'epoch') and self.epoch == self.now_epoch):
            self.now_epoch+=1
            self.update_temperature()
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        x = self.norm(output) # batchnorm
        x = self.activate(x)  # 激活
        return output

