import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from models.utils import MLPBlock
from models.attention import RandFourierFeature


class BFSGFM(nn.Layer):
    #Bitemporal Fourier Spatial Gate Fusion Module
    def __init__(self, dims, out_channels=64):
        super().__init__()
        self.cov1 = nn.Conv1D(2*dims, out_channels,3,padding=1,data_format='NLC')
        self.bn1 = nn.BatchNorm1D(out_channels, data_format='NLC')
        self.mlp1 = MLPBlock(out_channels, out_channels*2)

        self.fc = nn.Linear(2,1)
        self.rff = RandFourierFeature(out_channels, out_channels)
        self.mlp = MLPBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], -1)
        x = self.cov1(x)
        x = self.bn1(x)
        x = self.mlp1(x)

        xa = F.adaptive_avg_pool1d(x, 1)
        xm = F.adaptive_max_pool1d(x, 1)
        xt = paddle.concat([xa, xm], -1)
        xt = self.fc(xt)
        xt = F.relu(xt)
        y = x * xt
        y = y + x
        y = self.rff(y)
        y = self.mlp(y)
        y = F.relu(y)
        return y


class BSGFM(nn.Layer):
    #Bitemporal Spatial Gate Fusion Module
    def __init__(self, dims, out_channels=64):
        super().__init__()
        self.cov1 = nn.Conv1D(2*dims, out_channels,3,padding=1,data_format='NLC')
        self.bn1 = nn.BatchNorm1D(out_channels, data_format='NLC')
        self.mlp1 = MLPBlock(out_channels, out_channels*2)

        self.lamda = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(1.0), dtype='float16')
        self.fc = nn.Linear(2,1)
        self.mlp = MLPBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], -1)
        x = self.cov1(x)
        x = self.bn1(x)
        x = self.mlp1(x)

        xa = F.adaptive_avg_pool1d(x, 1)
        xm = F.adaptive_max_pool1d(x, 1)
        xt = paddle.concat([xa, xm], -1)
        xt = self.fc(xt)
        xt = F.relu(xt)
        y = x * xt
        y = y* self.lamda + x
        y = self.mlp(y)
        y = F.relu(y)
        return y


class BMF(nn.Layer):
    #Bitemporal Image Multi-level Fusion Module
    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        self.cbr1 = layers.ConvBNReLU(in_channels, 32, 3, stride=2)
        self.cbr2 = layers.ConvBNReLU(in_channels, 32, 3, stride=2)

        self.cond1 = nn.Conv2D(64, 64, 3, padding=1)
        self.cond3 = nn.Conv2D(64, 64, 3, padding=3, dilation=3)
        self.cond5 = nn.Conv2D(64, 64, 3, padding=5, dilation=5)

        self.bn = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()

        self.shift = layers.ConvBNReLU(64, out_channels, 3, 1, stride=2)

    def forward(self, x1, x2):
        y1 = self.cbr1(x1)
        y2 = self.cbr2(x2)

        y = paddle.concat([y1, y2], 1)

        y10 = self.cond1(y)
        y11 = self.cond3(y)
        y12 = self.cond5(y)
       
        yc = self.relu(self.bn(y10 + y11 + y12))
        return self.shift(yc)

class BDGF(nn.Layer):
    def __init__(self, in_channels=6, out_channels=64):
        super().__init__()
        midc = out_channels // 2
        self.cov1 = layers.ConvBNReLU(in_channels,midc,1)
        self.cov2 = layers.DepthwiseConvBN(midc,midc,3,stride=2)
        self.cov3 = layers.DepthwiseConvBN(midc,midc,3)
        self.cov4 = layers.DepthwiseConvBN(midc,midc,3,stride=2)
        self.cov5 = layers.DepthwiseConvBN(midc,midc,3)
        self.con6 = layers.ConvBNReLU(midc,out_channels,1)

    def forward(self, x):
        y = self.cov1(x)
        y = self.cov2(y)
        y = self.cov3(y)
        y = self.cov4(y)
        y = self.cov5(y)
        y = self.con6(y)
        return y


class DGF2D(nn.Layer):
    def __init__(self, dims, num_features=64):
        super().__init__()
        self.cov1 = layers.ConvBNReLU(2*dims, dims, 1)
        
        self.mconv = layers.DepthwiseConvBN(dims, dims, 3)
        
        self.fc = layers.ConvBNReLU(dims,dims, 3)
        self.bn1 = nn.BatchNorm2D(dims)
        self.mlp = layers.ConvBNReLU(dims, 64, 1)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)

        x = self.cov1(x)

        x1 = self.mconv(x)

        x2 = F.adaptive_avg_pool2d(x, 1)
        x2 = x * x2
        x2 = self.fc(x2)
        x2 = F.softmax(x2, axis=-1)

        y = x1 + x2
        y = self.bn1(y)
        y = y + x
        y = self.mlp(y)
        return y


class BF_PSP(nn.Layer):
    # Bitemporal Fusion based on Parall Shift Pattern 
    def __init__(self, lenk_size=64, channel=32):
        super().__init__()
        self.ln1 = nn.Linear(2*channel, channel)
        self.relu1 = nn.ReLU6()

        self.conv1 = nn.Conv1D(2*lenk_size, lenk_size, 1)
        self.relu2 = nn.ReLU6()

        self.ln21 = nn.Linear(4*channel, 2*channel)
        self.relu31 = nn.ReLU6()

        self.ln2 = nn.Linear(2*channel, 128)
        self.relu3 = nn.ReLU6()
        

    def forward(self, x1, x2):
        y1 = paddle.repeat_interleave(x1, 2, 1)
        y1[:, 1::2, :] = x2

        y1 = self.conv1(y1)
        y1 = self.relu2(y1)

        y2 = paddle.repeat_interleave(x1, 2, 2)
        y2[:, :, 1::2] = x2

        y2 = self.ln1(y2)
        y2 = self.relu1(y2)

        y = paddle.concat([y1, y2], -1)

        lmax = F.adaptive_max_pool1d(y, 1)
        lavg = F.adaptive_avg_pool1d(y, 1)
        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], -1)
        ya = self.ln21(ya)
        ya = self.relu31(ya)

        y = y + ya

        y = self.ln2(y)
        y = self.relu3(y)
        return y



class BF_PS2(nn.Layer):
    # Bitemporal Fusion based on Parall Shift Pattern 
    def __init__(self, lenk_size=64, channel=32):
        super().__init__()
        self.ln1 = nn.Linear(2*channel, channel)
        self.relu1 = nn.ReLU6()
        self.ln12 = nn.Linear(channel, 64)

        self.conv1 = nn.Conv1D(2*lenk_size, lenk_size, 1)
        self.ln21 = nn.Linear(channel, 64)
        self.relu2 = nn.ReLU6()

        self.ln2 = nn.Linear(2*channel, channel)
        self.relu21 = nn.ReLU6()
        self.ln22 = nn.Linear(channel, 64)

        self.ln3 = nn.Linear(64*3, 64)
        self.bn3 = nn.BatchNorm1D(64, data_format="NLC")
        self.relu3 = nn.ReLU6()
        self.mlp = MLPBlock(64, 128)
        

    def forward(self, x1, x2):
        y1 = paddle.repeat_interleave(x1, 2, 1)
        y1[:, 1::2, :] = x2

        y1 = self.conv1(y1)
        y1 = self.ln21(y1)
        y1 = self.relu2(y1)

        y2 = paddle.repeat_interleave(x1, 2, 2)
        y2[:, :, 1::2] = x2

        y2 = self.ln1(y2)
        y2 = self.relu1(y2)
        y2 = self.ln12(y2)

        y3 = paddle.concat([x1, x2], -1)
        y3 = self.ln2(y3)
        y3 = self.relu21(y3)
        y3 = self.ln22(y3)
        
        y = paddle.concat([y1, y2, y3], -1)
        y = self.ln3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.mlp(y)
        return y


class Bit_Fusion(nn.Layer):
    def __init__(self, in_channels=32, out_channels=64):
        super().__init__()
        self.lmax = nn.AdaptiveMaxPool1D(1)
        self.lavg = nn.AdaptiveAvgPool1D(1)
        dims = int(in_channels // 2)

        self.lc1 = nn.Linear(in_channels, dims)
        self.bn1 = nn.BatchNorm1D(dims, data_format="NLC")

        self.lc2 = nn.Linear(2*dims, dims)
        self.bn2 = nn.BatchNorm1D(dims, data_format="NLC")

        self.lc3 = nn.Linear(2*dims, dims)#nn.Linear(2*dims, dims)
        self.bn3 = nn.BatchNorm1D(dims, data_format="NLC")


    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], -1)
        y = self.lc1(x)
        y = self.bn1(y)
        y = F.relu(y)

        lavg = self.lavg(y)
        lmax = self.lmax(y)

        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], -1)

        ya = self.lc2(ya)
        ya = self.bn2(ya)
        ya = F.relu(ya)

        yb = paddle.concat([y, ya], -1)

        yb = yb + x

        yb = self.lc3(yb)
        yb = self.bn3(yb)
        yb = F.relu(yb)

        return yb