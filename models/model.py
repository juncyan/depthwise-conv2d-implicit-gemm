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
from models.decoder import SemantiMambacDv0
from models.cddecoder import CD_Mamba



features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}


class SCDSam_Mamba(nn.Layer):
    def __init__(self, img_size=256,num_seg=7,num_cd=2,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        self.fusion1 = SemantiMambacDv0(320,128,64,img_size)
        self.cdfusion = CD_Mamba(320,128,64,img_size)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        # self.scls2 = layers.ConvBN(64,num_seg,7)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        b1, b, p1, p = self.extractor(x1, x2)

        tb1 = self.fusion1(b1, b)
        tp2 = self.fusion1(p1, p)
        
        t= self.cdfusion(b1, b, p1, p)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls1(tp2)
        return t, outa, outb
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape, p1.shape, p2.shape, p3.shape, p4.shape)

        return b1, b4, p1, p4
    

