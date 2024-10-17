import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import math

from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from models.model import BSGFM
from models.attention import RandFourierFeature, ECA
from models.segment_anything.build_sam import build_sam_vit_b, build_sam_vit_l


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x

class SamH_CD(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_h.pdparams"):
        super().__init__()
        self.sam = build_sam_vit_l(checkpoint=sam_checkpoint, img_size=img_size)
        
        self.bff1 = BSGFM(1024,64)
        self.bff3 = BSGFM(1024,64)
        self.bff4 = BSGFM(256,64)
        self.fusion = HSDecoder(64)

        self.cls1 = layers.ConvBN(64,2,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
        
        f1, f3, f4 = self.extractor(x1, x2)
        f = self.fusion(f1, f3, f4)
        f = self.cls1(f)

        return f
    
    def feature_extractor(self, x):
        f1, f2,f3 = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3
    
    def extractor(self, x1, x2):
        b1, b2, b = self.feature_extractor(x1)
        p1, p2, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b.shape)
        b1 = paddle.reshape(b1, [b1.shape[0], -1, b1.shape[-1]])
        b2 = paddle.reshape(b2, [b2.shape[0], -1, b2.shape[-1]])
        p1 = paddle.reshape(p1, [p1.shape[0], -1, p1.shape[-1]])
        p2 = paddle.reshape(p2, [p2.shape[0], -1, p2.shape[-1]])

        f1 = self.bff1(b1, p1)
        f3 = self.bff3(b2, p2)

        b4 = paddle.reshape(b, [b.shape[0], b.shape[1], -1])
        b4 = paddle.transpose(b4, perm=[0, 2, 1])
        p4 = paddle.reshape(p, [b.shape[0], b.shape[1], -1])
        p4 = paddle.transpose(p4, perm=[0, 2, 1])
        
        f4 = self.bff4(b4, p4)

        f1 = features_transfer(f1)
        f3 = features_transfer(f3)
        f4 = features_transfer(f4)
        return f1, f3, f4
    
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


class HSDecoder(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_channels=64, out_dims=64):
        super().__init__()
        dims = 64 # 2*in_channels
        self.st1conv1 = layers.ConvBNReLU(dims, in_channels, 1)
        self.st1conv2 = EFC(in_channels)
        self.rff = RandFourierFeature(in_channels, in_channels)

        self.st2conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st2conv2 = EFC(in_channels)
        
        self.st3conv1 = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.st3conv2 = EFC(in_channels)
        
        self.conv1 = layers.ConvBNReLU(3*in_channels, in_channels, 1)
        self.conv2 = layers.ConvBNReLU(in_channels, in_channels, 3)
        self.deconv6 = layers.ConvBNReLU(in_channels,out_dims,1)

    def forward(self, x1, x2, x3):
        f3 = self.st1conv1(x3)
        fr3 = self.rff(f3)
        f3 = f3 + fr3
        f3 = self.st1conv2(f3)

        f2 = x2 + f3
        f2 = self.st2conv1(f2)
        f2 = self.st2conv2(f2)

        # f3 = F.interpolate(f3, x1.shape[-2:], mode='bilinear', align_corners=True)
        # f2 = F.interpolate(f2, x1.shape[-2:], mode='bilinear', align_corners=True)
        # print(x1.shape, f2.shape, f3.shape)
        f1 = x1 + f2 + f3
        f1 = self.st3conv1(f1)
        f1 = self.st3conv2(f1)

        f = paddle.concat([f1, f2, f3], 1)
        f = self.conv1(f)
        fd = self.conv2(f)
        f = f + fd
        f = F.interpolate(f, scale_factor=16, mode='bilinear', align_corners=True)
        f = self.deconv6(f)
        return f


class EFC(nn.Layer):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.eca = ECA(kernel_size)
        self.cbr = layers.ConvBNReLU(in_channels, in_channels, kernel_size=kernel_size)
    def forward(self, x):
        y = self.eca(x)
        y = self.cbr(y)
        return y