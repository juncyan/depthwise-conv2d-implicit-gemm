import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import math

from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss

from models.segment_anything.build_sam import build_sam_vit_t


def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x

class Extracter(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = build_sam_vit_t(img_size=img_size, checkpoint=sam_checkpoint)

        self.cbr1 = layers.ConvBNAct(256,128,3,act_type='gelu')
        self.cbr2 = layers.ConvBNAct(640,320,3,act_type='gelu')

        # for p in self.sam():
        #     p.requires_grad = False
    
    def extract_features(self, x):
        f1, f2, f3, f4 = self.sam.image_encoder.extract_features(x)
        # print(f1.shape, f4.shape)
        f1 = features_transfer(f1)
        # f2 = features_transfer(f2)
        # f3 = features_transfer(f3)
        f4 = features_transfer(f4)
        return f1, f4
    
    def forward(self, x):
        x1 = x[:,:3, :,:]
        x2 = x[:,3: ,:,:]
        b1, b2 = self.extract_features(x1)
        p1, p2 = self.extract_features(x2)

        f1 = paddle.concat([b1, p1], 1)
        f1 = self.cbr1(f1)
        f2 = paddle.concat([b2, p2], 1)
        f2 = self.cbr2(f2)

        return f1, f2


class Fusion(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, out_dims=64):
        super(Fusion, self).__init__()

        self.deconv1 = layers.ConvBNReLU(320,256,3)
        self.deconv2 = layers.ConvBNReLU(256,256,3)
        self.deconv3 = layers.ConvBNReLU(256,128,3)
        
        self.deconv4 = layers.ConvBNReLU(256,128,3)
        self.deconv5 = layers.ConvBNReLU(128,128,3)
        self.deconv6 = layers.ConvBNReLU(128,out_dims,3)

    def forward(self, x1, x2):
        f = self.deconv1(x2)
        f = self.deconv2(f)
        f = self.deconv3(f)
        f = F.interpolate(f, x1.shape[2:], mode='bilinear', align_corners=True)

        y = paddle.concat([x1, f], 1)
        y = self.deconv4(y)
        y = self.deconv5(y)
        y = F.interpolate(y, scale_factor=8, mode='bilinear', align_corners=True)
        y = self.deconv6(y)
        return y


class SamCD(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.extracter = Extracter(img_size=img_size, sam_checkpoint=sam_checkpoint)
        self.fusion = Fusion(64)
        self.cls = layers.ConvBN(64,2,7)
        
    
    def forward(self, x):
        f1, f2 = self.extracter(x)
        y = self.fusion(f1, f2)
        y = self.cls(y)
        return y

    @staticmethod
    def loss(pred, label):
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.5*l2
        return loss
        
