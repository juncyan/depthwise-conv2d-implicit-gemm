import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers

from models.attention import ECA

class VFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()

        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
        self.conv3 = layers.DepthwiseConvBN(in_channels, in_channels, kernel_size=kernel_size)
        self.eca = ECA(kernel_size)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.adaptive_avg_pool2d(x, 1)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        y = x2 * x1
        y = self.conv3(y)
        y = F.relu(y)
        y = y + x
        y = self.eca(y)
        return y


class EFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.eca = ECA(kernel_size)
        self.cbr = layers.ConvBNReLU(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        y = self.eca(x)
        y = self.cbr(y)
        return y

class Decoder_SCAB(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 2*in_channels
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = EFC(in_channels)
        # self.st1conv3 = layers.ConvBNReLU(in_channels,in_channels,3)

        self.st2conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st2conv2 = EFC(in_channels)
        # self.st2conv3 = layers.ConvBNReLU(in_channels,in_channels,3)

        self.st3conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st3conv2 = EFC(in_channels)
        # self.st3conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2, x3):
        f = self.st1conv1(x3)
        f = self.st1conv2(f)

        f = paddle.concat([f, x2], 1)
        f = self.st2conv1(f)
        f = self.st2conv2(f)
        # f = self.st2conv3(f)
        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)

        f = paddle.concat([x1, f], 1)
        f = self.st3conv1(f)
        f = self.st3conv2(f)
        # f = self.st3conv3(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f
    

class Decoder_stage3(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 2*in_channels
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = layers.ConvBNReLU(in_channels,in_channels,3)
        # self.st1conv3 = layers.ConvBNReLU(in_channels,in_channels,3)

        self.st2conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st2conv2 = layers.ConvBNReLU(in_channels,in_channels,3)
        # self.st2conv3 = layers.ConvBNReLU(in_channels,in_channels,3)

        self.st3conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st3conv2 = layers.ConvBNReLU(in_channels,in_channels,3)
        # self.st3conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2, x3):
        f = self.st1conv1(x3)
        f = self.st1conv2(f)

        f = paddle.concat([f, x2], 1)
        f = self.st2conv1(f)
        f = self.st2conv2(f)
        # f = self.st2conv3(f)
        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)

        f = paddle.concat([x1, f], 1)
        f = self.st3conv1(f)
        f = self.st3conv2(f)
        # f = self.st3conv3(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f

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