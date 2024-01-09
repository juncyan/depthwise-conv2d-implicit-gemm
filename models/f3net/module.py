import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


class FSA(nn.Layer):
    # Large kernels feature fusion
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
    
        self.sa = layers.ConvBNAct(2, 1, 3, act_type='sigmoid')

        self.outcbn = layers.ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)
        
        max_feature = paddle.max(y, axis=1, keepdim=True)
        mean_feature = paddle.mean(y, axis=1, keepdim=True)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        # y1 = max_feature * y
        # y2 = mean_feature * y
        # att_feature = paddle.concat([y1, y2], axis=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y


class CISConv(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super().__init__()
        self.prim = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set)
        self.prim_shift = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set)
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, axis=1) for z in x.chunk(self.groups, axis=1))
        x_merge = paddle.concat(tuple(paddle.concat((x2, x1), 1) for (x1, x2) in x_split), 1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class CFDF(nn.Layer):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = layers.ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = nn.Sequential(layers.DepthwiseConvBN(self.inter_dim, self.inter_dim,kernels,3),
                                    nn.GELU(),
            layers.ConvBNReLU(self.inter_dim, self.inter_dim//2, 1))

        self.aux = nn.Sequential(
            CISConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2D(self.inter_dim//2),
            nn.ReLU())

        self.outcbr = layers.ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x2, x1], axis=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        # y1 = self.ppmn(y1)
        y = paddle.concat([y1, y2], 1)
        
        return self.outcbr(y)


if __name__ == "__main__":
    print("spp")
    