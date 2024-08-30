import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BF_PSP(nn.Layer):
    # Bitemporal Fusion based on Parall Shift Pattern 
    def __init__(self, lenk_size=64, channel=32):
        super().__init__()
        self.ln1 = nn.Linear(2*channel, channel)
        self.bn1 = nn.BatchNorm1D(channel, data_format="NLC")
        self.relu1 = nn.ReLU6()

        self.conv1 = nn.Conv1D(2*lenk_size, lenk_size, 1)
        self.bn2 = nn.BatchNorm1D(lenk_size)
        self.relu2 = nn.ReLU6()

        self.ln21 = nn.Linear(4*channel, 2*channel)
        self.bn31 = nn.BatchNorm1D(2*channel, data_format="NLC")
        self.relu31 = nn.ReLU6()

        self.ln2 = nn.Linear(2*channel, channel)
        self.bn3 = nn.BatchNorm1D(channel, data_format="NLC")
        self.relu3 = nn.ReLU6()
        

    def forward(self, x1, x2):
        y1 = paddle.repeat_interleave(x1, 2, 1)
        y1[:, 1::2, :] = x2

        y1 = self.conv1(y1)
        y1 = self.bn2(y1)
        y1 = self.relu2(y1)

        y2 = paddle.repeat_interleave(x1, 2, 2)
        y2[:, :, 1::2] = x2

        y2 = self.ln1(y2)
        y2 = self.bn1(y2)
        y2 = self.relu1(y2)

        y = paddle.concat([y1, y2], -1)

        lmax = F.adaptive_max_pool1d(y, 1)
        lavg = F.adaptive_avg_pool1d(y, 1)
        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], -1)
        ya = self.ln21(ya)
        ya = self.bn31(ya)
        ya = self.relu31(ya)

        y = y + ya

        y = self.ln2(y)
        y = self.bn3(y)
        y = self.relu3(y)
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