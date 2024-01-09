import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from .SFF import *

class DoubleConv(nn.Layer):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Layer):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.bilinear = bilinear
        if bilinear:
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2DTranspose(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        if(self.bilinear):
            x1 =nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # print("x2.size():", x2.shape)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = paddle.concat([x2, x1], axis=1)
        return self.conv(x)

class Down(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2D(in_channels,in_channels,3,2,1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class backbone(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # factor = 2 #if bilinear else 1
        # self.down4 = Down(512, 512)
        
    def forward(self, x):
        res = []
        y = self.inc(x)
        res.append(y)
        y = self.down1(y)
        res.append(y)
        y = self.down2(y)
        res.append(y)
        y = self.down3(y)
        # res.append(y)
        # y = self.down4(y)
        res.append(y)
        return res

class F3Net(nn.Layer):
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.encode1 = backbone(3)#(3, 34)
        self.encode2 = backbone(3)

        self.lkff1 = LKAFF(64)
        self.lkff2 = LKAFF(128)
        self.lkff3 = LKAFF(256)
        self.lkff4 = LKAFF(512)

        self.ppm = layers.attention.PAM(512)

        self.up1 = LSFFUp(1024, 256)
        self.up2 = LSFFUp(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.classier = nn.Sequential(layers.ConvBN(64, num_classes,7), nn.Sigmoid())

    def forward(self, x):
        x1 ,x2 = x[:, :self.in_channels//2, :, :], x[:, self.in_channels//2:, :, :]
        self.feature1 = self.encode1(x1)
        self.feature2 = self.encode2(x2)

        # for i in self.feature1:
        #     print(i.shape)
            
        self.augf1 = self.lkff1(self.feature1[0], self.feature2[0])
        self.augf2 = self.lkff2(self.feature1[1], self.feature2[1])
        self.augf3 = self.lkff3(self.feature1[2], self.feature2[2])
        self.augf4 = self.lkff4(self.feature1[3], self.feature2[3])

        self.flast = self.ppm(self.augf4)

        y = self.up1(self.flast, self.augf4)
        y = self.up2(y, self.augf3)
        y = self.up3(y, self.augf2)
        y = self.up4(y, self.augf1)
        # y = nn.functional.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        return self.classier(y)


# class LKUChange(nn.Layer):
#     def __init__(self,in_channels=6, num_classes=2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.encode1 = backbone(3)#(3, 34)
#         self.encode2 = backbone(3)

#         self.lkff1 = LKFF(64)
#         self.lkff2 = LKFF(128)
#         self.lkff3 = LKFF(256)
#         self.lkff4 = LKFF(512)

#         self.ppm = layers.attention.CAM(512)
        
#         self.up1 = SFFUp(1024, 256)
#         self.up2 = Up(512, 128)
#         self.up3 = Up(256, 64)
#         self.up4 = Up(128, 64)

#         self.classier = nn.Sequential(layers.ConvBN(64, num_classes,3), nn.Sigmoid())

#     def forward(self, x):
#         x1 ,x2 = x[:, :self.in_channels//2, :, :], x[:, self.in_channels//2:, :, :]
#         self.feature1 = self.encode1(x1)
#         self.feature2 = self.encode2(x2)

#         self.augf1 = self.lkff1(self.feature1[0], self.feature2[0])
#         self.augf2 = self.lkff2(self.feature1[1], self.feature2[1])
#         self.augf3 = self.lkff3(self.feature1[2], self.feature2[2])
#         self.augf4 = self.lkff4(self.feature1[3], self.feature2[3])

#         self.flast = self.ppm(self.augf4)

#         y = self.up1(self.flast, self.augf4)
#         y = self.up2(y, self.augf3)
#         y = self.up3(y, self.augf2)
#         y = self.up4(y, self.augf1)
#         # y = nn.functional.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
#         return self.classier(y)


if __name__ == "__main__":
    print("test DACDNet")
    # from paddleseg.utils import op_flops_funs
    # x = paddle.rand([1,6,512,512]).cuda()
    # m = UChange(in_channels=6, num_classes=2).to("gpu")
    # y = m(x)
    # # for i in y:
    # print(y.shape)
    # _, c, h, w = x.shape
    # _ = paddle.flops(
    #     m, [1, c, h, w],
    #     custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
