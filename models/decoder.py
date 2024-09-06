import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers

from models.attention import ECA, RandFourierFeature


class EFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.eca = ECA(kernel_size)
        self.cbr = layers.ConvBNReLU(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        y = self.eca(x)
        y = self.cbr(y)
        return y


class HSDecoder(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 2*in_channels
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = EFC(in_channels)
        self.rff = RandFourierFeature(in_channels, in_channels)

        self.st2conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st2conv2 = EFC(in_channels)
        
        self.st3conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st3conv2 = EFC(in_channels)
        
        self.conv1 = layers.ConvBNReLU(3*in_channels, in_channels, 1)
        self.conv2 = layers.ConvBNReLU(in_channels, in_channels, 3)
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2, x3):
        f3 = self.st1conv1(x3)
        fr3 = self.rff(f3)
        f3 = f3 + fr3
        f3 = self.st1conv2(f3)

        f2 = x2 + f3
        f2 = self.st2conv1(f2)
        f2 = self.st2conv2(f2)

        f3 = F.interpolate(f3, x1.shape[-2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, x1.shape[-2:], mode='bilinear', align_corners=True)
        # print(x1.shape, f2.shape, f3.shape)
        f1 = x1 + f2 + f3
        f1 = self.st3conv1(f1)
        f1 = self.st3conv2(f1)

        f = paddle.concat([f1, f2, f3], 1)
        f = self.conv1(f)
        fd = self.conv2(f)
        f = f + fd
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f

class Decoder_SCABF_v2(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 2*in_channels
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = EFC(in_channels)
        # self.st1conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        self.rff1 = RandFourierFeature(in_channels, in_channels)

        self.st2conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st2conv2 = EFC(in_channels)
        # self.st2conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        self.rff2 = RandFourierFeature(in_channels, in_channels)

        # self.st3conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        # self.st3conv2 = EFC(in_channels)
        # self.st3conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2):
        f = self.st1conv1(x2)
        f = self.st1conv2(f)
        fr1 = self.rff1(f)
        f = f + fr1

        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)
        f = paddle.concat([x1, f], 1)
        f = self.st2conv1(f)
        f = self.st2conv2(f)
        fr2 = self.rff2(f)
        f = f + fr2
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f

class Decoder_SCAB_in2(nn.Layer):
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

        # self.st3conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        # self.st3conv2 = EFC(in_channels)
        # self.st3conv3 = layers.ConvBNReLU(in_channels,in_channels,3)
        
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2):
        f = self.st1conv1(x2)
        f = self.st1conv2(f)

        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)
        f = paddle.concat([x1, f], 1)
        f = self.st2conv1(f)
        f = self.st2conv2(f)
        # f = self.st3conv3(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f

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