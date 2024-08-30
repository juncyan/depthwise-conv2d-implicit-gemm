import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers


class Fusion(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, out_dims=64):
        super(Fusion, self).__init__()

        self.deconv1 = layers.ConvBNReLU(320,256,3)
        self.deconv2 = layers.ConvBNReLU(256,256,3)
        self.deconv3 = layers.ConvBNReLU(256,128,3)
        
        self.deconv4 = layers.ConvBNReLU(256,128,3)
        self.deconv5 = layers.ConvBNReLU(128,128,3)
        self.deconv6 = layers.ConvBNReLU(128,out_dims,3)

    def forward(self, x1, x2):
        f = self.deconv1(x2)
        f = self.deconv2(f)
        f = self.deconv3(f)
        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)

        y = paddle.concat([x1, f], 1)
        y = self.deconv4(y)
        y = self.deconv5(y)
        y = F.interpolate(y, scale_factor=8, mode='bilinear', align_corners=True)
        y = self.deconv6(y)
        return y