import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import math

from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from models.mobilesam import MobileSAM
from models.kan import KANLinear, KAN
from models.encoderfusion import BF_PSP
from models.decoder import Fusion

features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x


class MobileSamCD_CSP(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        self.bf1 = BF_PSP(features_shape[img_size][0][0],features_shape[img_size][0][1])
        self.bf2 = BF_PSP(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        
        self.conv1 = layers.ConvBNReLU(320, 256, 1)

        self.fusion = Fusion(64)

        # self.conv1 = layers.ConvBNReLU(64+3, 2, 1)
        self.cls = layers.ConvBN(64,2,7)
        
    def feature_extractor(self, x):
        f1, f2,f3,f4 = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape)
        f1 = self.bf1(b1, p1)
        # f2 = self.bf2(b2, p2)
        # f3 = self.bf3(b3, p3)
        f4 = self.bf2(b4, p4)
        f1 = features_transfer(f1)
        f4 = features_transfer(f4)
        return f1, f4
    
    def mask_encoder(self, x):
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(None,None,None, )
        low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=x,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True, )

        original_size = [self.sam.image_encoder.img_size, self.sam.image_encoder.img_size]
        masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=original_size,
                original_size=original_size)
        return masks
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f2 = self.extractor(x1, x2)

        # fi = self.conv1(f2)
        # fi = self.mask_encoder(fi)
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


class MobileSamCD(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        self.sam.mask_decoder.train()

        self.conv1 = layers.ConvBNAct(320, 256, 1, act_type='gelu')
        self.upconv1 = layers.ConvBNAct(6, 64, 1, act_type='gelu')
        self.conv2 = layers.ConvBNAct(128, 64, 1, act_type='gelu')
        self.cls = layers.ConvBN(64,2,7)

        
    def feature_extractor(self, x):
        f1, f2,f3,f4 = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)
        b4 = features_transfer(b4)
        p4 = features_transfer(p4)
        return b4, p4
    
    def mask_encoder(self, x):
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(None,None,None, )
        low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=x,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True, )

        original_size = [self.sam.image_encoder.img_size, self.sam.image_encoder.img_size]
        masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=original_size,
                original_size=original_size)
        return masks
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f2 = self.extractor(x1, x2)

        f1 = self.conv1(f1)
        f1 = self.mask_encoder(f1)

        f2 = self.conv1(f2)
        f2 = self.mask_encoder(f2)

        y = paddle.concat((f1, f2), axis=1)
        y = self.upconv1(y)

        lmax = F.adaptive_max_pool2d(y, 1)
        lavg = F.adaptive_avg_pool2d(y, 1)
        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], axis=1)
        ya = self.conv2(ya)
        y = y + ya
       
        y = self.cls(y)
        return y

    @staticmethod
    def loss(pred, label):
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        loss = l1 + 0.5*l2
        return loss
        
