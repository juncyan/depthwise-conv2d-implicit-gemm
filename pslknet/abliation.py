import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
# from paddleseg.models.backbones import ResNet34_vd
from models.backbone.resnet import ResbackBone, ResNet


from .blocks import *
from .utils import *

class LKC_PSNet_k13(nn.Layer):
    #large kernel siamese network
    def __init__(self, in_channels=3, kernels=13):
        super().__init__()

        # self.fa = SBFA([64, 128, 256, 512])
        self.branch1 = PSBFE(3, [64,128,256,512], kernels)
        self.branch2 = PSBFE(3, [64,128,256,512], kernels)

        # self.cls1 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        # self.cls2 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        self.cbr1 = STAF(64,64)
        self.cbr2 = STAF(128,128)
        self.cbr3 = STAF(256,256)
        self.cbr4 = STAF(512,512)

        self.up1 = UpBlock(512+256, 256)
        self.up2 = UpBlock(256+128, 128)
        self.up3 = UpBlock(128+64, 64)

        self.classiier = nn.Sequential(nn.Conv2D(64, 2, 7, 1, 3), nn.BatchNorm2D(2), nn.Sigmoid())

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        _, _, w, h = x1.shape
    
        a1, a2, a3, a4 = self.branch1(x1)
        b1, b2, b3, b4 = self.branch2(x2)

        m1 = self.cbr1(a1, b1)
        m2 = self.cbr2(a2, b2)
        m3 = self.cbr3(a3, b3)
        m4 = self.cbr4(a4, b4)
        
        r1 = self.up1(m4, m3)
        r2 = self.up2(r1, m2)
        r3 = self.up3(r2, m1)

        # l1 = self.cls1(f4)
        # l1 = F.interpolate(l1, size=[w, h],mode='bilinear')

        # l2 = self.cls2(a4)
        # l2 = F.interpolate(l2, size=[w, h],mode='bilinear')

        y = F.interpolate(r3, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y#, l1, l2
    
    # @staticmethod
    # def loss(pred, label, wdice=0.2):
    #     # label = torch.argmax(label,axis=1)
    #     prob, l1, l2 = pred

    #     # label = torch.argmax(label, 1).unsqueeze(1)
    #     label = torch.to_tensor(label, torch.float32)
        
    #     dsloss1 = nn.loss.BCELoss()(l1, label)
    #     dsloss2 = nn.loss.BCELoss()(l2, label)
    #     Dice_loss = 0.5*(dsloss1+dsloss2)

    #     label = torch.argmax(label, 1).unsqueeze(1)
    #     # label = torch.to_tensor(label, torch.float16)

    #     CT_loss = nn.loss.CrossEntropyLoss(axis=1)(prob, label)
    #     CD_loss = CT_loss + wdice * Dice_loss
    #     return CD_loss
    
    # @staticmethod
    # def predict(pred):
    #     prob, _, _ = pred
    #     return prob