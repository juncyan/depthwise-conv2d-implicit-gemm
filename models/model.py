import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import math

from paddleseg.models import layers
from paddleseg.models.layers.attention import DualAttentionModule
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from models.utils import MLPBlock, features_transfer
from models.mobilesam import MobileSAM
from models.attention import RandFourierFeature, ECA
from models.decoder import MSIF, SemanticDv0, SemanticDv1
from models.cddecoder import CD_D0, CD_D1


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}


class SCDSam(nn.Layer):
    def __init__(self, img_size=256,num_seg=7,num_cd=2,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        # self.aecoder = AEcoder()

        self.fusion1 = SemanticDv0(256,128,64,img_size)
        self.fusion2 = SemanticDv0(256,128,64,img_size)
        self.cdfusion = CD_D0(256,128,64,img_size)

        # self.conv3 = layers.ConvBNPReLU(512,256,1)
        # self.conv2 = layers.ConvBNReLU(320, 160, 1)
        # self.conv1 = layers.ConvBNReLU(128+64, 64, 1)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        self.scls2 = layers.ConvBN(64,num_seg,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        # b1 = self.aecoder(x1)
        # p1 = self.aecoder(x2)
        b1, b, p1, p = self.extractor(x1, x2)
        # print(b1.shape, b.shape, p1.shape, p.shape)
        tb1 = self.fusion1(b1, b)
        tp1 = self.fusion1(p1, p)
        
        t= self.cdfusion(b1, b, p1, p)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls2(tp1)
        return t, outa, outb
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape)
        b1 = features_transfer(b1)
        # b3 = features_transfer(b3)

        p1 = features_transfer(p1)
        # p3 = features_transfer(p3)

        return b1, b, p1, p


class SCDSamV1(nn.Layer):
    def __init__(self, img_size=256,num_seg=7,num_cd=2,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        # self.aecoder = AEcoder()

        self.fusion1 = SemanticDv1(256,128,64,img_size)
        self.fusion2 = SemanticDv1(256,128,64,img_size)
        self.cdfusion = CD_D1(256,128,64,img_size)

        # self.conv3 = layers.ConvBNPReLU(512,256,1)
        # self.conv2 = layers.ConvBNReLU(320, 160, 1)
        # self.conv1 = layers.ConvBNReLU(128+64, 64, 1)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        self.scls2 = layers.ConvBN(64,num_seg,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        b1, b, p1, p = self.extractor(x1, x2)
        
        tb1 = self.fusion1(b1, b)
        tp1 = self.fusion1(p1, p)
        
        t= self.cdfusion(b1, b, p1, p)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls2(tp1)
        return t, outa, outb
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)

        return b1, b, p1, p
    

