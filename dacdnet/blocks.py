import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


class BFIB(nn.Layer):
    #bi-temporal feature integrating block
    def __init__(self, in_channels, out_channels, kernels = 7, stride=2):
        super().__init__()
        self.fe = LKFE(in_channels, kernels)
        
        self.ce = LKCE(in_channels, out_channels, kernels, stride)
        
    def forward(self, x):
        y = self.fe(x)
        
        y = self.ce(y)
        return y

class PSBFA(nn.Layer):
    #pseudo siamese bi-temporal feature assimilating module
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce1 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b1 = layers.ConvBNReLU(in_channels, out_channels, 3)

        self.reduce2 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b2 = layers.ConvBNReLU(in_channels, out_channels, 3)
    
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
        self.cam = nn.Sequential(layers.DepthwiseConvBN(in_channels, in_channels, 7)#layers.attention.CAM(in_channels)
                                 ,nn.GELU())
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        x = self.cam(x)
        x = self.double_conv(x)
        return x

class LKFE(nn.Layer):
    #large kernel feature extraction
    def __init__(self, in_channels, kernels = 7):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2 * in_channels, 3,)
        self.dwc = nn.Sequential(layers.DepthwiseConvBN(2 * in_channels, 2 * in_channels, kernels),
                                 nn.GELU())
        self.se = SEModule(2 * in_channels, 8)
        self.conv2 = layers.ConvBNReLU(2 * in_channels, in_channels, 3,)
        self.ba = nn.Sequential(nn.BatchNorm2D(in_channels), nn.ReLU())

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwc(m)
        m = self.se(m)
        m = self.conv2(m)
        y = x + m
        return self.ba(y)
    
class LKCE(nn.Layer):
    #large kernel channel expansion
    def __init__(self, in_channels, out_channels, kernels = 7, stride=1):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2*in_channels, 3)

        self.dwc = nn.Sequential(layers.DepthwiseConvBN(2*in_channels, 2*in_channels, kernels, stride=stride),
                                 nn.GELU())
        self.conv3 = layers.ConvBNReLU(2*in_channels, out_channels, 3)
       

    def forward(self, x):
        y = self.conv1(x)
        m = self.dwc(y)
        m = self.conv3(m)
        return m

class BII(nn.Layer):
    #bi-temporal images integration
    def __init__(self, in_channels, out_channesl, kernels=7):
        super().__init__()
        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.reduce = nn.Sequential(nn.GELU(), layers.ConvBNReLU(2*in_channels, out_channesl, 3, stride=2))
        self.cbr1 = layers.ConvBNReLU(out_channesl, out_channesl*2, 1)
        self.dws = nn.Sequential(layers.DepthwiseConvBN(out_channesl*2, out_channesl*2, kernels), nn.GELU())
        self.se = SEModule(out_channesl*2)
        self.cbr2 = layers.ConvBNReLU(out_channesl*2, out_channesl, 3)
        
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        ym = paddle.concat([y1, y2], 1)

        ym = self.reduce(ym)
        # ym = self.pam(ym)
        y = self.cbr1(ym)
        y = self.dws(y)
        y = self.se(y)
        y = self.cbr2(y)
        return y

class SEModule(nn.Layer):
    def __init__(self, channels, reductions = 8):
        super(SEModule, self).__init__()
        reduction_channels = channels // reductions
        self.avg = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, reduction_channels, 1, bias_attr=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(reduction_channels, channels, 1, bias_attr=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c , _, _ = x.shape
        avg = self.avg(x)
        y = self.fc1(avg)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        y = y.expand_as(x)
        return x * y