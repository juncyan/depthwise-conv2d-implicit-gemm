import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers


class FEModule(nn.Layer):
    def __init__(self, in_channels, mid_channels: list = [16, 32, 64, 128]):
        super(FEModule, self).__init__()
        self.layers = nn.LayerList()
        # self.layers.append(nn.Sequential(layers.ConvBNReLU(in_channels, mid_channels[0], 7, 3), layers.ConvBNReLU(mid_channels[0], mid_channels[0], 3)))
        self.layers.append(nn.Sequential(layers.ConvBNReLU(in_channels, mid_channels[0], 7, 3), layers.ConvBNReLU(mid_channels[0], mid_channels[0], 3)))
        in_channels = mid_channels[0]
        for c in mid_channels[1:]:
            self.layers.append(nn.Sequential(layers.ConvBN(in_channels, in_channels, 3, stride=2, padding=1),
                                             CBRGroup(in_channels, c)))
            in_channels = c

    def forward(self, x):
        y = x
        res = []
        for layer in self.layers:
            y = layer(y)
            res.append(y)
        return res


class FDModule(nn.Layer):
    # Feature Difference
    def __init__(self, in_channels=3, mid_channels=[32, 64, 128]):
        super(FDModule, self).__init__()
        self.channels = in_channels
        self.branch1 = FEModule(in_channels, mid_channels)
        # self.branch2 = FEModule(in_channels, mid_channels)

    def forward(self, x):
        x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch1(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i - j
            res.append(paddle.clip(z, 0., 1.0))
        return res

class FDModule_v2(nn.Layer):
    # Feature Difference
    def __init__(self, in_channels=3, mid_channels=[32, 64, 128]):
        super(FDModule_v2, self).__init__()
        self.channels = in_channels
        self.branch1 = FEModule(in_channels, mid_channels)
        self.branch2 = FEModule(in_channels, mid_channels)

    def forward(self, x):
        x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i - j
            res.append(paddle.clip(z, 0., 1.0))
        return res


class FAModule(nn.Layer):
    # Feature Assimilation
    def __init__(self, in_channels=3, mid_channels=[128, 256, 512]):
        super(FAModule, self).__init__()
        self.channels = in_channels
        self.branch1 = FEModule(in_channels, mid_channels)
        # self.branch2 = FEModule(in_channels, mid_channels)

    def forward(self, x):
        x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch1(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(paddle.clip(z, 0., 1.0))
        return res

class FAModule_v2(nn.Layer):
    # Feature Assimilation
    def __init__(self, in_channels=3, mid_channels=[128, 256, 512]):
        super(FAModule_v2, self).__init__()
        self.channels = in_channels
        self.branch1 = FEModule(in_channels, mid_channels)
        self.branch2 = FEModule(in_channels, mid_channels)

    def forward(self, x):
        x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(paddle.clip(z, 0., 1.0))
        return res


class CBRGroup(nn.Layer):
    def __init__(self, in_channels, out_channels, down=True):
        super(CBRGroup, self).__init__()
        if down:
            self.cbrg = nn.Sequential(
                layers.ConvBNReLU(in_channels, in_channels //2, 3, 1, stride=1),
                nn.Conv2D(in_channels // 2, in_channels, 3, 1, 1))
        else:
            self.cbrg = nn.Sequential(
                layers.ConvBNReLU(in_channels, out_channels // 2, 3, 1, stride=1),
                layers.ConvBNReLU(out_channels // 2, out_channels, 3, 1, stride=1))
            
        self.bnr = nn.Sequential(nn.BatchNorm2D(in_channels), nn.ReLU())
        self.lastcbr = layers.ConvBNReLU(in_channels, out_channels, kernel_size=3,padding=1)
    def forward(self, x):
        z = self.cbrg(x)
        y = z + x
        return self.lastcbr(self.bnr(y))

class LKBlock(nn.Layer):
    def __init__(self, in_channels, dw_channels, block_lk_size, drop_path, stride=1):
        '''
        Args:
            in_channels:
            dw_channels: in_channels
            block_lk_size: 29,27,21,13
            drop_path: 0.3
        '''
        super(LKBlock, self).__init__()
        self.pw1 = layers.ConvBNReLU(in_channels, dw_channels, 1, stride=1)
        self.pw2 = layers.ConvBN(dw_channels, in_channels, 1, stride=1)
        self.large_kernel = layers.DepthwiseConvBN(dw_channels, dw_channels, block_lk_size, stride=stride)
        self.lk_nonlinear = nn.ReLU()
        # self.prelkb_bn = nn.BatchNorm2D(in_channels)
        self.drop_path = nn.Dropout(drop_path)
        # print('drop path: ', self.drop_path.drop_path)

    def forward(self, x):
        # y = self.prelkb_bn(x)
        y = self.pw1(x)
        y = self.large_kernel(y)
        y = self.lk_nonlinear(y)
        y = self.pw2(y)
        return x + self.drop_path(y)


class ConvFFN(nn.Layer):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        '''
        Args:
            in_channels:
            internal_channels:  4 * in_channels
            out_channels:  in_channels
            drop_path:
        '''
        super().__init__()
        self.drop_path = nn.Dropout(drop_path)
        self.preffn_bn = nn.BatchNorm2D(in_channels)
        self.pw1 = layers.ConvBN(in_channels, internal_channels, 1, stride=1)
        self.pw2 = layers.ConvBN(internal_channels, out_channels, 1, stride=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        y = self.preffn_bn(x)
        y = self.pw1(y)
        y = self.nonlinear(y)
        y = self.pw2(y)
        return x + self.drop_path(y)


if __name__ == "__main__":
    print('features')
    x = paddle.rand([1, 64, 64, 64]).cuda()
    m = LKBlock(64, 18, 27, 0.3).to('gpu')
    y = m(x)
    print(y.shape)
