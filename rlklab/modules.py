import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform

from paddleseg.models import layers

from .reparams import Reparams

class LKC(Reparams):
    def __init__(self, in_channels, small_kernels=7, dilation = 3, bias_attr=None):
        super(LKC, self).__init__()

        # self.kernel_sizes = [5, 9, 3, 3, 3] lk = 17
        # self.dilates = [1, 2, 4, 5, 7]

        self.in_channels = in_channels
        self.dilation = dilation
        self.sk = small_kernels
        self.lk = (self.sk -1)*self.dilation + 1

        self.conv1 = nn.Conv2D(in_channels,in_channels,self.sk,padding=self.sk//2,groups=in_channels, bias_attr=bias_attr)
        self.conv2 = nn.Conv2D(in_channels,in_channels,3,padding=1,groups=in_channels, bias_attr=bias_attr)
        self.conv3 = nn.Conv2D(in_channels,in_channels,3,padding=3,dilation=3,groups=in_channels, bias_attr=bias_attr)
        self.lkc = nn.Conv2D(in_channels,in_channels,self.sk,padding=self.lk//2,dilation=dilation,groups=in_channels, bias_attr=bias_attr)
    
    def forward(self, x):
        if not self.training and hasattr(self, "repc"):
            print("repc")
            y = self.repc(x)
            y = F.relu(y)
            return y
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.lkc(x)
        y = y1 + y2 + y3 + y4
        y = F.relu(y)
        return y

    def get_equivalent_kernel_bias(self):
        kernel1, bias1 = self._fuse_conv_bn(self.conv1)  
        kernel2, bias2 = self._fuse_conv_bn(self.conv2)
        kernel3, bias3 = self._fuse_conv_bn(self.conv3)
        klkc, biaslkc = self._fuse_conv_bn(self.lkc)
        # print(kernel1.shape, kernel2.shape, klkc.shape, bias1.shape, bias2.shape, biaslkc.shape)
        return klkc + kernel1 + kernel2 + kernel3,  bias1 + bias2 + biaslkc + bias3

class SELayer(nn.Layer):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(
            ch,
            c_,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(
            c_,
            ch,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        scale = out * inputs
        return scale

class LKAA(nn.Layer):
    # large kernel attention aggregation module
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = 2 * in_channels

        self.cbrs1 = layers.ConvBNReLU(in_channels, mid_channels, 1)
        self.lkc = LKC(mid_channels, 7, 3)
        self.cbrsa = layers.ConvBNAct(2, 1, 7,act_type='sigmoid')
        self.cbrs2 = layers.ConvBNReLU(mid_channels, in_channels, 1)

        s_channels = in_channels // 8
        self.apool = nn.AdaptiveAvgPool2D(1)
        self.cbrc1 = nn.Conv2D(in_channels, s_channels, 1)
        self.cbrc2 = nn.Conv2D(s_channels, in_channels, 1)

        self.cbrout = layers.ConvBNReLU(in_channels * 3, in_channels, 1)
    

    def forward(self, x):

        #sptial attention path
        sp = self.cbrs1(x)
        sp = self.lkc(sp)

        msp = paddle.max(sp, 1, True)
        asp = paddle.mean(sp, 1, True)
        csp = paddle.concat([msp, asp], 1)
        csp = self.cbrsa(csp)
        csp = sp * csp
        csp = self.cbrs2(csp)

        #channel attention path
        cp = self.apool(x)
        cp = self.cbrc1(cp)
        cp = F.relu(cp)
        cp = self.cbrc2(cp)
        cp = F.sigmoid(cp)
        cp = x * cp

        y = paddle.concat([x, csp, cp], 1)
        y = self.cbrout(y)
        return y


class CDFSF(nn.Layer):
    #cross dimension features shift fusion
    def __init__(self, in_c1, in_c2):
        super().__init__()
        dims = in_c1 + in_c2
        self.zip_channels = layers.ConvBNReLU(dims, in_c2, 1)
        self.lfc = layers.ConvBNReLU(in_c2, in_c2, 3)
    
        self.sa = layers.ConvBNAct(2, 1, 3, act_type='sigmoid')

        self.outcbr = layers.ConvBNReLU(in_c2, in_c2, 3)
        
    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        
        max_feature = paddle.max(y, axis=1, keepdim=True)
        mean_feature = paddle.mean(y, axis=1, keepdim=True)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbr(y)
        return y