import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
import paddleseg.models.layers as layers
import numpy as np


class LargeKernelBlock(nn.Layer):
    def __init__(self, dims, inter, kernels=7):
        super().__init__()
        self.proj1 = layers.DepthwiseConvBN(dims, dims, kernels)
        self.sp1 = layers.SeparableConvBNReLU(dims, inter, 3)
        # self.proj2 = layers.DepthwiseConvBN(inter, inter, kernels)
        self.sp2 = layers.SeparableConvBNReLU(inter, dims, 3)
        self.activate = nn.ReLU()

    def forward(self, x):
        y = self.proj1(x)
        y = nn.GELU()(y)
        y = self.sp1(y)
        # y = self.proj2(y)
        y = self.sp2(y)
        y = x + y
        return self.activate(y)


class LKFF(nn.Layer):
    # Large kernels feature fusion
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = layers.DepthwiseConvBN(in_channels, in_channels, kernels)

        self.sig = layers.ConvBNAct(in_channels, 1, 3, act_type="sigmoid")
        self.tanh = layers.ConvBNAct(in_channels, 1, 3, act_type="tanh")
        self.sa = layers.ConvBNReLU(2,1,3)

        self.outcbn = layers.ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)
        
        max_feature = self.sig(y)
        mean_feature = self.tanh(y)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        # y1 = max_feature * y
        # y2 = mean_feature * y
        # att_feature = paddle.concat([y1, y2], axis=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y

class LKAFF(nn.Layer):
    # Large kernels feature fusion
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
    
        self.sa = layers.ConvBNAct(2, 1, 3, act_type='sigmoid')

        self.outcbn = layers.ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)
        
        max_feature = paddle.max(y, axis=1, keepdim=True)
        mean_feature = paddle.mean(y, axis=1, keepdim=True)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        # y1 = max_feature * y
        # y2 = mean_feature * y
        # att_feature = paddle.concat([y1, y2], axis=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y


class SFF(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.channels = in_channels
        self.se = SELayer(2*in_channels, 8)

        self.cbn = layers.ConvBNReLU(2 * in_channels, in_channels, 3)

        self.cam = layers.attention.CAM(in_channels)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], axis=1)
        
        x = self.se(x)

        x = self.cbn(x)
        y = self.cam(x)
        return y


class CFF(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.channels = in_channels
        self.cbn1 = layers.ConvBNReLU(2 * in_channels, in_channels, 3)

        self.se = SELayer(in_channels, 8)
        
        self.cam = layers.attention.CAM(in_channels)

        self.out_cbr = layers.ConvBNReLU(2 * in_channels, in_channels, 3)

    def forward(self, x1, x2):

        x = paddle.concat([x1, x2], axis=1)
        x = self.cbn1(x)

        y1 = self.se(x)

        y2 = self.cam(x)

        y = paddle.concat([y1, y2], 1)
        y = self.out_cbr(y)

        return y

class PSConv(nn.Layer):
    # refers: https://github.com/d-li14/PSConv
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super(PSConv, self).__init__()
        self.prim = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set)
        self.prim_shift = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set)
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, axis=1) for z in x.chunk(self.groups, axis=1))
        x_merge = paddle.concat(tuple(paddle.concat((x2, x1), 1) for (x1, x2) in x_split), 1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift

class PyramidPoolingModule(nn.Layer):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat  = feat + x
        return feat
    
class SFFUp(nn.Layer):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = layers.ConvBNReLU(in_channels, self.inter_dim, 1)

        self.native = layers.ConvBNReLU(self.inter_dim, self.inter_dim//2, 1, 1)

        self.ppmn = PyramidPoolingModule()

        self.aux = nn.Sequential(
            PSConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2D(self.inter_dim//2),
            nn.ReLU())

        self.outcbr = layers.ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x2, x1], axis=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        y1 = self.ppmn(y1)
        y = paddle.concat([y1, y2], 1)
        
        return self.outcbr(y)

class LSFFUp(nn.Layer):
    
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = layers.ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = nn.Sequential(layers.DepthwiseConvBN(self.inter_dim, self.inter_dim,kernels,3),
                                    nn.GELU(),
            layers.ConvBNReLU(self.inter_dim, self.inter_dim//2, 1))

        # self.ppmn = PyramidPoolingModule()

        self.aux = nn.Sequential(
            PSConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2D(self.inter_dim//2),
            nn.ReLU())

        self.outcbr = layers.ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x2, x1], axis=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        # y1 = self.ppmn(y1)
        y = paddle.concat([y1, y2], 1)
        
        return self.outcbr(y)

class SELayer(nn.Layer):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / np.sqrt(ch)
        c_ = ch // reduction_ratio
        weight_attr_1 = paddle.ParamAttr(initializer=Uniform(-stdv, stdv))
        self.squeeze = nn.Linear(ch, c_, weight_attr=weight_attr_1, bias_attr=None)

        stdv = 1.0 / np.sqrt(c_)
        weight_attr_2 = paddle.ParamAttr(initializer=Uniform(-stdv, stdv))
        self.extract = nn.Linear(c_,ch, weight_attr=weight_attr_2, bias_attr=None)

    def forward(self, x):
        out = self.pool(x)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        y = out * x
        return y

class SpikingNeuron(nn.Layer):
    def __init__(self, threshold=0.5, decay=0.25):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrance_potential = 0.
    
    def forward(self, x):
        self.membrance_potential += x
        spike = (self.membrance_potential >= self.threshold).float()
        self.membrance_potential = self.membrance_potential * (1 - spike)* self.decay
        return spike


if __name__ == "__main__":
    print("spp")
    x = paddle.rand([1, 16, 256, 256]).cuda()
    m = LKFF(16).to('gpu')
    y = m(x,x)
    print(y.shape)