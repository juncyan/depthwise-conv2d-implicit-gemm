import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from .blocks import *


class PSLKNet(nn.Layer):
    #large kernel pseudo siamese network
    def __init__(self, in_channels=3, kernels=7):
        super().__init__()
        self.conv = layers.ConvBNPReLU(2 * in_channels, in_channels, 3)
        self.stage1 = STAGE2(in_channels, 64, kernels)
        self.stage2 = STAGE2(64, 128, kernels)
        self.stage3 = STAGE2(128, 256, kernels)
        self.stage4 = STAGE2(256,512, kernels)

        self.ppm = layers.ASPPModule([1,2,4,6], 512, 1024, True)

        self.up = UpBlock(512*3, 512)
        self.up1 = UpBlock(256*3, 256)
        self.up2 = UpBlock(128*3, 128)
        self.up3 = UpBlock(64*3, 64)

        self.classiier = layers.ConvBNAct(64, 2, 3, act_type="sigmoid")
    
    def forward(self, x1, x2):
        f0 = paddle.concat([x1, x2], 1)
        f0 = self.conv(f0)
        y1, y2, f1 = self.stage1(x1, x2, f0)
        y1, y2, f2 = self.stage2(y1, y2, f1)
        y1, y2, f3 = self.stage3(y1, y2, f2)
        y1, y2, f4 = self.stage4(y1, y2, f3)

        f5 = self.ppm(f4)
        r0 = self.up(f4, f5)
        r1 = self.up1(r0, f3)
        r2 = self.up2(r1, f2)
        r3 = self.up3(r2, f1)
        y = F.interpolate(r3, scale_factor=2,mode='bilinear')
        y = self.classiier(y)
        return y
    

class PLKNet(nn.Layer):
    #large kernel pseudo siamese network
    def __init__(self, in_channels=3, kernels=7):
        super().__init__()
        self.stage1 = BII(in_channels, 64, kernels)#BFIB(2*in_channels, 64, kernels)
        self.stage2 = BFIB(64, 128, kernels)
        self.stage3 = BFIB(128, 256, kernels)
        
        self.stage4 = BFIB(256, 512, kernels)
        
        self.up1 = UpBlock(256*3, 256)
        self.up2 = UpBlock(128*3, 128)
        self.up3 = UpBlock(64*3, 64)

        self.classiier = layers.ConvBNAct(64, 2, 7, act_type="sigmoid")
    
    def forward(self, x1, x2):
        # f0 = paddle.concat([x1, x2], 1)
        f1 = self.stage1(x1, x2)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        # f3 = self.stage31(f3)
        f4 = self.stage4(f3)
        # f4 = self.stage41(f4)

        # f5 = self.ppm(f4)
        # r0 = self.up(f4, f5)
        r1 = self.up1(f4, f3)
        r2 = self.up2(r1, f2)
        r3 = self.up3(r2, f1)
        y = F.interpolate(r3, scale_factor=2,mode='bilinear')
        y = self.classiier(y)
        return y
    
class STAGE1(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ps = PSBFA(in_channels, out_channels)
        self.conv11 = layers.ConvBNReLU(in_channels, out_channels, 1)
        self.conv12 = layers.ConvBNAct(out_channels, out_channels, 3, stride=2, act_type='gelu')
        self.cbr = layers.ConvBNReLU(2 * out_channels, out_channels, 1)

    def forward(self, x1, x2, xm):
        y1, y2, af = self.ps(x1, x2)
        m = self.conv11(xm)
        m = self.conv12(m)

        y = paddle.concat([af, m], 1)
        y = self.cbr(y)
        return y1, y2, y

class STAGE2(nn.Layer):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self.ps = PSBFA(in_channels, out_channels)
        self.bfib = BFIB(in_channels, out_channels, kernels)
        self.cbr = layers.ConvBNReLU(2 * out_channels, out_channels, 1)

    def forward(self, x1, x2, xm):
        y1, y2, af = self.ps(x1, x2)
        m = self.bfib(xm)
        y = paddle.concat([af, m], 1)
        y = self.cbr(y)
        return y1, y2, y
