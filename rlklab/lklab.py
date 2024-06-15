# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers

from models.param_init import kaiming_normal_init, constant_init

from .backbone import LKResNet
from .modules import LKAA, SELayer, CDFSF


class LKALab(nn.Layer):
    """
    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self,num_classes,in_channels=3,aspp_ratios=(1, 6, 12, 18)):
        super().__init__()

        self.backbone = LKResNet(in_channels)
        self.aspp = layers.ASPPModule(aspp_ratios,1024,256,True,True,True)
        
        self.lkaa1 = LKAA(256)

        self.cd1 = CDFSF(256,256)
        self.cd2 = CDFSF(256, 64)
        # self.zipc1 = layers.ConvBNReLU(512, 256, 3)
        # self.zipc2 = layers.ConvBNReLU(256, 64, 3)

        self.zipc4 = layers.ConvBNReLU(64, 256, 3)
        self.zipc42 = layers.ConvBNReLU(256, 64, 3)
        self.classifier = nn.Conv2D(64, num_classes, 3, 1, 1)

        self.init_weight()
        
    def forward(self, x):
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.backbone(x)

        y1 = self.aspp(f5)
        # y1 = F.interpolate(y1, scale_factor=4, mode='bilinear')
        f3 = self.lkaa1(f3)
        # y1 = paddle.concat([y1, f3], 1)
        # y1 = self.zipc1(y1)
        # y1 = self.zipc2(y1)
        y1 = self.cd1(y1, f3)

        # y1 = F.interpolate(y1, scale_factor=4, mode='bilinear')
        # y1 = paddle.concat([y1, f1], 1)
        y1 = self.cd2(y1, f1)
        y1 = self.zipc4(y1)
        y1 = self.zipc42(y1)
        y1 = F.interpolate(y1, size=size, mode='bilinear')
        y1 = self.classifier(y1)
        return y1
        

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                kaiming_normal_init(layer.weight)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)


class LKALab_2(nn.Layer):
    def __init__(self,num_classes,in_channels=3,aspp_ratios=(1, 6, 12, 18)):
        super().__init__()

        self.backbone = LKResNet(in_channels)
        self.aspp = layers.ASPPModule(aspp_ratios,1024,256,True,True,True)
        
        self.lkaa1 = LKAA(256)
        self.zipc1 = layers.ConvBNReLU(512, 256, 3)
        self.se1 = SELayer(256, 16)
        self.zipc2 = layers.ConvBNReLU(256, 64, 3)

        self.lkaa2 = LKAA(64)
        self.zipc3 = layers.ConvBNReLU(128, 64, 3)
        self.se2 = SELayer(64, 8)
        self.zipc4 = layers.ConvBNReLU(64, 64, 3)
        self.classifier = nn.Conv2D(64, num_classes, 3, 1, 1)
        

    def forward(self, x):
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.backbone(x)

        y1 = self.aspp(f5)
        y1 = F.interpolate(y1, scale_factor=4, mode='bilinear')
        f3 = self.lkaa1(f3)
        y1 = paddle.concat([y1, f3], 1)
        y1 = self.zipc1(y1)
        y1 = self.se1(y1)
        y1 = self.zipc2(y1)

        y1 = F.interpolate(y1, scale_factor=4, mode='bilinear')
        f1 = self.lkaa2(f1)
        y1 = paddle.concat([y1, f1], 1)
        y1 = self.zipc3(y1)
        y1 = self.se2(y1)
        y1 = self.zipc4(y1)
        y1 = F.interpolate(y1, size=size, mode='bilinear')
        y1 = self.classifier(y1)
        return y1


class LKALab_3(nn.Layer):
    def __init__(self,num_classes,in_channels=3,aspp_ratios=(1, 6, 12, 18)):
        super().__init__()

        self.backbone = LKResNet(in_channels)
        self.aspp = layers.ASPPModule(aspp_ratios,1024,512,True,True,True)

        self.lkaa0 = LKAA(512)
        self.zipc01 = layers.ConvBNReLU(1024, 256, 1)
        self.zipc02 = layers.ConvBNReLU(256, 256, 3)
        self.zipc03 = layers.ConvBNReLU(256, 256, 3)
        
        self.lkaa1 = LKAA(256)
        self.zipc1 = layers.ConvBNReLU(512, 128, 1)
        self.zipc2 = layers.ConvBNReLU(128, 128, 3)
        self.zipc21 = layers.ConvBNReLU(128, 128, 3)

        self.lkaa2 = LKAA(128)
        self.zipf21 = layers.ConvBNReLU(256, 64, 1)
        self.zipf22 = layers.ConvBNReLU(64, 64, 3)
        self.zipf23 = layers.ConvBNReLU(64, 64, 3)

        self.zipc3 = layers.ConvBNReLU(128, 64, 1)
        self.zipc4 = layers.ConvBNReLU(64, 64, 3)
        self.classifier = nn.Conv2D(64, num_classes, 3, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.backbone(x)

        y1 = self.aspp(f5)

        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        f4 = self.lkaa0(f4)
        y1 = paddle.concat([y1, f4], 1)
        y1 = self.zipc01(y1)
        y1 = self.zipc02(y1)
        y1 = self.zipc03(y1)

        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        f3 = self.lkaa1(f3)
        y1 = paddle.concat([y1, f3], 1)
        y1 = self.zipc1(y1)
        y1 = self.zipc2(y1)
        y1 = self.zipc21(y1)

        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        f2 = self.lkaa2(f2)
        y1 = paddle.concat([y1, f2], 1)
        y1 = self.zipf21(y1)
        y1 = self.zipf22(y1)
        y1 = self.zipf23(y1)

        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        y1 = paddle.concat([y1, f1], 1)
        y1 = self.zipc3(y1)
        y1 = self.zipc4(y1)
        y1 = F.interpolate(y1, size=size, mode='bilinear')
        y1 = self.classifier(y1)
        return y1

