import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers

class VFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()

        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
        self.conv3 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.adaptive_avg_pool2d(x, 1)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        y = x2 * x1
        y = self.conv3(y)
        y = F.relu(y)
        y = y + x
        return y

# class SMDM(nn.Layer):
#     def __init__(self, in_chs, kernel_size=3):
#         super().__init__()
#         self.c1 = layers.ConvBNReLU(2*in_chs, in_chs, kernel_size=kernel_size)
#         self.v1 = VFC(in_chs, kernel_size)
#         self.cbr = layers.ConvBNReLU(2*in_chs, in_chs, kernel_size=kernel_size)
#         self.v2 = VFC(in_chs, kernel_size)
#     def forward(self, x1, x2):
#         y = paddle.concat([x1, x2], 1)
#         y = self.c1(y)
#         y1 = self.v1(x1)
#         y2 = self.v1(x2)
#         y3 = paddle.concat([y1, y2], 1)
#         y3 = self.cbr(y3)
#         y3 = self.v2(y3)
#         y3 = y + y3
#         y3 = F.relu(y3)
#         return y3

class SMDM(nn.Layer):
    def __init__(self, in_chs, kernel_size=3):
        super().__init__()
        self.v1 = VFC(in_chs, kernel_size)
        self.cbr = layers.ConvBNReLU(in_chs, in_chs, 3)
        self.v2 = VFC(in_chs, kernel_size)
    def forward(self, x):
        y1 = self.v1(x)
        y3 = self.cbr(y1)
        y3 = self.v2(y3)
        y3 = x + y3
        y3 = F.relu(y3)
        return y3

class Decoder(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, out_dims=64):
        super().__init__()

        self.s1 = SMDM(64)
        self.cbr = layers.ConvBNReLU(128, 64, 1)
        self.s2 = SMDM(64)
    
        self.deconv6 = layers.ConvBNReLU(64,out_dims,1)

    def forward(self, x1, x2):
        f = self.s1(x2)
        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)

        y = paddle.concat([x1, f], 1)
        y = self.cbr(y)
        y = self.s2(y)
        y = F.interpolate(y, scale_factor=8, mode='bilinear', align_corners=True)
        y = self.deconv6(y)
        return y



class Fusion2(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, out_dims=64):
        super(Fusion2, self).__init__()

        self.deconv1 = layers.ConvBNReLU(64,128,1)
        self.deconv2 = layers.ConvBNReLU(128,128,3)
        self.deconv3 = layers.ConvBNReLU(128,64,1)
        
        self.deconv4 = layers.ConvBNReLU(128,128,3)
        self.deconv5 = layers.ConvBNReLU(128,128,3)
        self.deconv6 = layers.ConvBNReLU(128,out_dims,1)

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

class Fusion(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, out_dims=64):
        super(Fusion, self).__init__()

        self.deconv1 = layers.ConvBNReLU(64,128,1)
        self.deconv2 = layers.ConvBNReLU(128,128,3)
        self.deconv3 = layers.ConvBNReLU(128,64,4)
        
        self.deconv4 = layers.ConvBNReLU(128,128,3)
        self.deconv5 = layers.ConvBNReLU(128,128,3)
        self.deconv6 = layers.ConvBNReLU(128,out_dims,1)

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