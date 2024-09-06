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
from models.kan import KANLinear, KAN
from models.encoderfusion import BDGF, BF_PS2, DGF2D, Bit_Fusion, BSGFM, BFSGFM, BMF
from models.decoder import Fusion, Fusion2, Decoder_stage3, Decoder_SCAB, Decoder_SCAB_in2


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x


class MSamCD_FSSH(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        # self.aencoder = BMF(3,64)
        self.bff1 = BFSGFM(128,64)
        # self.bff4 = BFSGFM(320,64)
        self.cbr1 = layers.ConvBNReLU(512, 128, kernel_size=1)
        self.fusion = Decoder_SCAB_in2(64)

        # self.cbr2 = layers.ConvBNReLU(128, 64, kernel_size=1)
        self.cls1 = layers.ConvBN(64,2,7)

    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
        
        f1, b4, p4 = self.extractor(x1, x2)
        f5 = paddle.concat((b4, p4), axis=1)
        f5 = self.cbr1(f5)
        f = self.fusion(f1, f5)
        # y = self.aencoder(x1, x2)
        # f = paddle.concat([f, y],axis=1)
        # f = F.interpolate(f, scale_factor=4, mode='bilinear')
        # f = self.cbr2(f)
        f = self.cls1(f)

        return f 
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape)

        f1 = self.bff1(b1, p1)
        # f4 = self.bff4(b4, p4)
        f1 = features_transfer(f1)
        # f4 = features_transfer(f4)
        return f1, b, p
    
    @staticmethod
    def predict(pred):
        return pred
    
    @staticmethod
    def loss(pred, label):  
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.5*l2 #+ 0.5*la2
        return loss


class MSamCD_SSH(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BFSGFM(128,64)
        self.bff4 = BFSGFM(320,64)

        self.cbr = layers.ConvBNReLU(512, 128, kernel_size=1)
        self.fusion = Decoder_SCAB(64)
    
        self.cls1 = layers.ConvBN(64,2,7)

    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        f5 = paddle.concat((b4, p4), axis=1)
        f5 = self.cbr(f5)
        f = self.fusion(f1, f4, f5)
        f = self.cls1(f)

        return f #, y
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape)

        f1 = self.bff1(b1, p1)
        f4 = self.bff4(b4, p4)
        f1 = features_transfer(f1)
        f4 = features_transfer(f4)
        return f1, f4 , b, p
    
    @staticmethod
    def predict(pred):
        return pred
    
    @staticmethod
    def loss(pred, label):  
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = 0.5*l1 + l2 #+ 0.5*la2
        return loss


class MobileSamCD_osg(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/aistudio/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        self.bf1 = Bit_Fusion(256)
        self.bf2 = Bit_Fusion(640)
        
        self.conv1 = layers.ConvBNReLU(320, 256, 1)

        self.fusion = Fusion(64)

        # self.conv1 = layers.DepthwiseConvBN(6, 6, kernel_size=3)
        # self.conv2 = layers.ConvBNReLU(6, 2, kernel_size=7)
        self.cls = layers.ConvBN(64,2,7)
        
    def feature_extractor(self, x):
        (f1, f2,f3,f4),_ = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)
     
        f1 = self.bf1(b1, p1)
        f4 = self.bf2(b4, p4)

        f1 = features_transfer(f1)
        f4 = features_transfer(f4)
        return f1, f4
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f2 = self.extractor(x1, x2)

        y = self.fusion(f1, f2)

        # y = paddle.concat((fi, y), axis=1)
        y = self.cls(y)
        return y

    @staticmethod
    def loss(pred, label):
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.5*l2
        return loss