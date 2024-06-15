# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from numbers import Integral
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D

from paddleseg.utils import utils
from paddleseg.models import layers

from .modules import LKC, SELayer

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class LKB_Down(nn.Layer):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        mid_c = int(out_channels // 2)
        self.conv = layers.ConvBNReLU(in_channels , mid_c, 3)
        self.lkc = LKC(mid_c) #layers.DepthwiseConvBN(mid_c, mid_c, 7)
        self.cbr1 = layers.ConvBNReLU(mid_c, mid_c,3)
        self.cbr2 = layers.ConvBNReLU(mid_c, out_channels, 1)
        self.cbr3 = layers.ConvBNReLU(out_channels , out_channels, 3, stride = 2)

    def forward(self, x):
        y = self.conv(x)
        y = self.lkc(y)
        y = self.cbr1(y)
        y = self.cbr2(y)
        y = self.cbr3(y)
        return y

class ShortBlock(nn.Layer):
    expansion = 4
    def __init__(self,
                 ch_in,
                 ch_out):
        super(ShortBlock, self).__init__()
        mid_ch = ch_out // self.expansion
        self.cbrs1 = layers.ConvBNReLU(ch_in, ch_out, 1)
        self.cbrs2 = layers.ConvBNReLU(ch_out, ch_out, 3, stride=2)

        self.cbr1 = layers.ConvBNReLU(ch_in, mid_ch, 1)
        self.lk = LKC(mid_ch) #layers.DepthwiseConvBN(mid_ch, mid_ch, 7)
        self.cbrs = layers.ConvBNReLU(mid_ch, mid_ch, 3, stride=2)
        self.cbr2 = layers.ConvBNReLU(mid_ch, ch_out, 1)

    def forward(self, x):
        
        s = self.cbrs1(x)
        s = self.cbrs2(s)

        y = self.cbr1(x)
        y = self.lk(y)
        y = self.cbrs(y)
        y = self.cbr2(y)

        return s + y

class LKResBlock(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels=None):
        super(LKResBlock, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        self.short = Identity() if out_channels == in_channels else layers.ConvBNReLU(in_channels, out_channels, 1)
        mid_ch = in_channels // self.expansion
        self.cbr1 = layers.ConvBNReLU(in_channels, mid_ch, 1)
        self.lk = LKC(mid_ch)
        self.bn = nn.BatchNorm2D(mid_ch)
        self.se = SELayer(mid_ch)
        self.cbrs = layers.ConvBNAct(mid_ch, in_channels, 1, act_type='gelu')
        self.cbr2 = layers.ConvBNReLU(in_channels, out_channels, 3)

    def forward(self, x):
        y = self.cbr1(x)
        y = self.lk(y)
        y = self.bn(y)
        y = self.se(y)
        y = self.cbrs(y)
        y = self.cbr2(y)
        return self.short(x) + y


class LKResNet(nn.Layer):
  
    def __init__(self, in_channels):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385
        """
        super(LKResNet, self).__init__()
        self.down = LKB_Down(in_channels, 64)


        self.short1 = ShortBlock(64, 256)
        self.block11 = LKResBlock(256)
        self.block12 = LKResBlock(256, 128)

        self.short2 = ShortBlock(128, 512)
        self.block21 = LKResBlock(512)
        # self.block212 = LKResBlock(512)
        self.block22 = LKResBlock(512, 256)

        self.short3 = ShortBlock(256, 1024)
        self.block31 = LKResBlock(1024)
        # self.block312 = LKResBlock(1024)
        self.block32 = LKResBlock(1024, 512)

        self.short4 = ShortBlock(512, 2048)
        self.block41 = LKResBlock(2048)
        # self.block412 = LKResBlock(2048)
        self.block42 = LKResBlock(2048, 1024)

    def forward(self, x):
        y1 = self.down(x)
        
        y2 = self.short1(y1)
        y2 = self.block11(y2)
        y2 = self.block12(y2)

        y3 = self.short2(y2)
        y3 = self.block21(y3)
        y3 = self.block22(y3)

        y4 = self.short3(y3)
        y4 = self.block31(y4)
        y4 = self.block32(y4)

        y5 = self.short4(y4)
        y5 = self.block41(y5)
        y5 = self.block42(y5)
        
        return y1, y2, y3, y4, y5



