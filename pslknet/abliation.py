import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
# from paddleseg.models.backbones import ResNet34_vd
from models.backbone.replknet import RepLKNet31B
from models.backbone.resnet import ResbackBone
from models.backbone.swin_transformer import SwinTransBackbone
from models.backbone.vit import ViTB_patch16_512

from .blocks import *
from .utils import *


class PSLKNet_ViT_p16(nn.Layer):
    #large kernel pseudo siamese network
    def __init__(self, in_channels=3, kernels=9):
        super().__init__()

        self.fa = PSBFA([64, 128, 256, 512], kernels)

        self.stage1 = STAF(in_channels, 64, kernels)
        # self.stage2 = BFIB(64, 128, kernels)
        # self.stage3 = BFIB(128, 256, kernels)
        # self.stage4 = BFIB(256, 512, kernels)

        self.backbone = ViTB_patch16_512(64,128)

        # self.cls2 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        # self.cls2 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        self.cbr1 = MF(128,64)
        self.cbr2 = MF(128+24,128)
        self.cbr3 = MF(256+96,256)
        self.cbr4 = MF(512+384,512)

        self.up1 = UpBlock(512+256, 256)
        self.up2 = UpBlock(256+128, 128)
        self.up3 = UpBlock(128+64, 64)

        self.classiier = layers.ConvBNAct(64, 2, 7, act_type="sigmoid")
    
    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        _, _, w, h = x1.shape
        a1, a2, a3, a4 = self.fa(x1, x2)
        
        f1 = self.stage1(x1, x2)
        
        m1 = self.cbr1(f1, a1)
        
        f2, f3, f4 = self.backbone(m1)
        # print(f2.shape, f3.shape, f4.shape)
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        #f2 = self.stage2(m1)
        m2 = self.cbr2(f2, a2)
        #f3 = self.stage3(m2)
        m3 = self.cbr3(f3, a3)
        #f4 = self.stage4(m3)
        m4 = self.cbr4(f4, a4)

        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        
        r1 = self.up1(m4, m3)
        r2 = self.up2(r1, m2)
        r3 = self.up3(r2, m1)

        # l1 = self.cls1(f4)
        # l1 = F.interpolate(l1, size=[w, h],mode='bilinear')

        # l2 = self.cls2(a4)
        # l2 = F.interpolate(l2, size=[w, h],mode='bilinear')

        y = F.interpolate(r3, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y 