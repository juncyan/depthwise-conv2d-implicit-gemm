import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import math

from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from models.utils import MLPBlock
from models.mobilesam import MobileSAM
from models.attention import RandFourierFeature, ECA


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x

class SCDSam(nn.Layer):
    def __init__(self, img_size=256,num_cls=7,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.fusion1 = Decoder(img_size)
        self.fusion2 = Decoder(img_size)

        self.conv3 = layers.ConvBNReLU(256, 128, 1)
        self.conv2 = layers.ConvBNReLU(128, 64, 1)
        self.conv1 = layers.ConvBNReLU(128+64, 64, 1)

        self.cls = layers.ConvBN(64,2,7)
        self.scls1 = layers.ConvBN(64,num_cls,7)
        self.scls2 = layers.ConvBN(64,num_cls,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
        
        b1, b, p1, p = self.extractor(x1, x2)
        tb1, tb2, tb3 = self.fusion1(b1, b)
        tp1, tp2, tp3 = self.fusion1(p1, p)
        
        t3 = paddle.concat([tb3, tp3], axis=1)
        t3 = self.conv3(t3)
        t2 = paddle.concat([tb2, tp2], axis=1)
        t2 = self.conv2(t2)
        t = paddle.concat([t3, t2], axis=1)
        t = F.interpolate(t, scale_factor=8, mode='bilinear', align_corners=True)
        t = self.conv1(t)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls2(tp1)
        return t, outa, outb
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, _, b = self.feature_extractor(x1)
        p1, p2, p3, _, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape)
        b1 = features_transfer(b1)
        # b3 = features_transfer(b3)

        p1 = features_transfer(p1)
        # p3 = features_transfer(p3)

        return b1, b, p1, p
    
    @staticmethod
    def predict(pred):
        return pred
    
    @staticmethod
    def loss(pred, label):  
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.75*l2
        return loss


class Decoder(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        self.st1conv1 = layers.ConvBNReLU(256, 128, 1)
        self.st1conv2 = EFC(128)

        self.st2conv1 = layers.ConvBNReLU(128, 64, 1)
        self.st2conv2 = EFC(64)
        
        self.conv2 = layers.ConvBNReLU(64, 64, 3)
        self.deconv6 = layers.ConvBNReLU(64,64,1)

    def forward(self, x2, x3):
        f3 = self.st1conv1(x3)
        f3 = self.st1conv2(f3)
        f3 = F.interpolate(f3, x2.shape[-2:], mode='bilinear', align_corners=True)
        
        f2 = x2 + f3
        f2 = self.st2conv1(f2)
        f2 = self.st2conv2(f2)

        f = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        
        f = self.conv2(f)
        f = self.deconv6(f)
        return f, f2, f3


class EFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.eca = ECA(kernel_size)
        self.cbr = layers.ConvBNReLU(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        y = self.eca(x)
        y = self.cbr(y)
        return y