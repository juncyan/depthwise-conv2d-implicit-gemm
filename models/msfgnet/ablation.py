import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers

from .modules import CDFSF, MRFE
from .blocks import BMF
from models.backbone.replknet import RepLKNet31B
from models.backbone.resnet import ResbackBone
from models.backbone.swin_transformer import SwinTransBackbone
from models.backbone.vit import ViTB_patch16_512


class MSFGNet_ViT(nn.Layer):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = ViTB_patch16_512(64,256,depth=12)

        self.pam = layers.attention.PAM(384)

        self.up1 = CDFSF(384, 384)
        self.up2 = CDFSF(384, 96)
        self.up3 = CDFSF(96, 24)
        self.up4 = CDFSF(24, 64)
        
        self.classier = layers.ConvBNAct(64, num_classes, 7, act_type='sigmoid')

    def forward(self, x):
        # f1, f2, f3, f4 = self.encode(x)
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x1 ,x2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.bmf(x1,x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)


class MSFGNet_RES50(nn.Layer):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = ResbackBone(50)

        self.pam = layers.attention.PAM(4*512)

        self.up1 = CDFSF(4*512, 4*512)
        self.up2 = CDFSF(4*512, 4*256)
        self.up3 = CDFSF(4*256, 4*128)
        self.up4 = CDFSF(4*128, 64)
        
        self.classier = layers.ConvBNAct(64, num_classes, 7, act_type='sigmoid')

    def forward(self, x):
        # f1, f2, f3, f4 = self.encode(x)
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x1 ,x2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.bmf(x1,x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)

class Up(nn.Layer):

    def __init__(self, in_c1, in_c2):
        super().__init__()
        in_channels = in_c1 + in_c2
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = layers.ConvBNReLU(in_channels, in_channels // 2, 3)
        self.conv2 = layers.ConvBNReLU(in_channels//2, in_c2, 3)
        
    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = paddle.concat([x1, x2], 1)
        
        return self.conv2(self.conv1(x))

class MSFGNet_noBMF(nn.Layer):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = layers.ConvBNPReLU(in_channels, 64, 3, stride=2)#BMF(3)
        self.encode =MRFE()

        self.pam = layers.attention.PAM(512)

        self.up1 = CDFSF(512, 512)
        self.up2 = CDFSF(512, 256)
        self.up3 = CDFSF(256, 128)
        self.up4 = CDFSF(128, 64)
        
        self.classier = layers.ConvBNAct(64, num_classes, 7, act_type='sigmoid')

    def forward(self, x):
        # x1 ,x2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.bmf(x)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)