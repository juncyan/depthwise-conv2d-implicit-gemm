import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


class BFIB(nn.Layer):
    #bi-temporal feature integrating block
    def __init__(self, in_channels, out_channels, kernels = 7):
        super().__init__()
        self.reduce = layers.ConvBNReLU(in_channels, in_channels, 3, stride = 2)
        self.fe = LKFE(in_channels, kernels)
        self.ce = LKCE(in_channels, out_channels, kernels)
        
    def forward(self, x):
        y = self.reduce(x)
        y = self.fe(y)
        y = self.ce(y)
        return y

class PSBFA(nn.Layer):
    #pseudo siamese bi-temporal feature assimilating module
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce1 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b1 = layers.ConvBNAct(in_channels, out_channels, 3, 1, act_type="gelu")

        self.reduce2 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b2 = layers.ConvBNAct(in_channels, out_channels, 3, 1, act_type="gelu")
    
    def forward(self, x1, x2):
        y1 = self.reduce1(x1)
        y1 = self.b1(y1)

        y2 = self.reduce2(x2)
        y2 = self.b2(y2)
        y = y1 + y2
        return y1, y2, y

class UpBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        
        x = self.double_conv(x)
        return x

class LKFE(nn.Layer):
    #large kernel feature extraction
    def __init__(self, in_channels, kernels = 7):
        super().__init__()
        self.conv1 = layers.ConvBNAct(in_channels, 4 * in_channels, 1, act_type="gelu")
        self.dwc = nn.Sequential(layers.DepthwiseConvBN(4 * in_channels, 4 * in_channels, kernels),
                                 nn.GELU())
        self.conv2 = nn.Conv2D(4 * in_channels, in_channels, 1)
        self.ba = nn.Sequential(nn.BatchNorm2D(in_channels), nn.GELU())

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwc(m)
        m = self.conv2(m)
        y = x + m
        return self.ba(y)
    
class LKCE(nn.Layer):
    #large kernel channel expansion
    def __init__(self, in_channels, out_channels, kernels = 7):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, out_channels, 3)

        self.conv2 = layers.ConvBNReLU(out_channels, out_channels // 2, 1)
        self.dwc = nn.Sequential(layers.DepthwiseConvBN(out_channels // 2, out_channels // 2, kernels),
                                 nn.GELU())
        self.conv3 = nn.Conv2D(out_channels // 2, out_channels, 1)
        self.ba = nn.Sequential(nn.BatchNorm2D(out_channels), nn.GELU())

    def forward(self, x):
        y = self.conv1(x)
        m = self.conv2(y)
        m = self.dwc(m)
        m = self.conv3(m)
        z = y + m
        return self.ba(z)