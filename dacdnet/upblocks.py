import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


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

class UpBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        mid_c = out_channels * 2
        self.cbr1 = layers.ConvBNReLU(in_channels, out_channels, 3)
        self.cbr2 = layers.ConvBNReLU(out_channels, mid_c, 3)
        self.cbr3 = layers.ConvBNReLU(mid_c, out_channels, 3)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        y = self.cbr1(x) 
        y = self.cbr2(y)
        res = self.cbr3(y)
        return res

class UpPAM(nn.Layer):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__() 

        self.mid_channels = out_channels // 8
        self.out_channels = out_channels
        self.cbr1 = layers.ConvBNReLU(in_channels, out_channels, 3)
        self.dsw = nn.Sequential(layers.DepthwiseConvBN(out_channels, out_channels, kernels)
                                 ,nn.GELU())
        
        # self.pam = layers.attention.PAM(out_channels)
        self.dsw = nn.Sequential(layers.DepthwiseConvBN(out_channels, out_channels, kernels)
                                 ,nn.GELU())

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        y = self.cbr1(x)
        y = self.dsw(y)
        res = self.pam(y)
        return res

class UpLK(nn.Layer):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__() 
        mid_c = out_channels * 2
        self.cbr1 = layers.ConvBNReLU(in_channels, out_channels, 3)
        self.cbr2 = layers.ConvBNReLU(out_channels, mid_c, 3)
        self.dsw = nn.Sequential(layers.DepthwiseConvBN(mid_c, mid_c, kernels),nn.GELU())
        self.cbr3 = layers.ConvBNReLU(mid_c, out_channels, 3)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,paddle.shape(x2)[2:],mode='bilinear')
        x = paddle.concat([x1, x2], axis=1)
        y = self.cbr1(x) 
        y = self.cbr2(y)
        y = self.dsw(y)
        res = self.cbr3(y)
        return res


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

