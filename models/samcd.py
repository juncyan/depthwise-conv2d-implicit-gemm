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
from models.encoderfusion import BF_PSP, BF_PS2, BF3, DGF2D, Bit_Fusion, BSGFM
from models.decoder import Fusion, Fusion2, Decoder_stage3, Decoder_SCAB


features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x


class MSamCD_S2(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BSGFM(128,64)
        self.bff4 = BSGFM(320,64)

        self.cbr = layers.ConvBNReLU(512, 128, kernel_size=1)
        self.fusion = Decoder_SCAB(64)
        
        self.conv1 = layers.ConvBNReLU(6, 2, kernel_size=1)
        # self.conv2 = layers.ConvBNReLU(4, 2, kernel_size=3)
        # self.cls2 = layers.ConvBNReLU(6, 2, kernel_size=7)

        self.cls1 = layers.ConvBN(64,2,7)
        self.cls2 = layers.ConvBN(4,2,7)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        f5 = paddle.concat((b4, p4), axis=1)
        f5 = self.cbr(f5)
        f = self.fusion(f1, f4, f5)
        f = self.cls1(f)

        b4 = self.mask_encoder(b4)
        p4 = self.mask_encoder(p4)
        y = paddle.concat((b4, p4), axis=1)
        y = self.conv1(y)

        f = paddle.concat((f, y), axis=1)
        f = self.cls2(f)

        return f, y
    
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
    
    def mask_encoder(self, x):
        # print(x.shape)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(None,None,None, )
        low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=x,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True, )
        # print(low_res_masks.shape)
        original_size = [self.sam.image_encoder.img_size, self.sam.image_encoder.img_size]
        masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=original_size,
                original_size=original_size)
        return masks
    
    @staticmethod
    def predict(preds):
        return preds[0]
    
    @staticmethod
    def loss(preds, label):
        pred = preds[0]
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        la2 = BCELoss()(preds[1], label)
        loss = l1 + 0.5*l2 + 0.5*la2
        return loss


class MSamCD_S1(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BF3(128,64)
        self.bff4 = BF3(320,64)

        self.cbr = layers.ConvBNReLU(512, 128, kernel_size=1)
        self.fusion = Decoder_stage3(64)
        
        # self.conv1 = layers.ConvBNReLU(6, 2, kernel_size=3)
        # self.conv2 = layers.ConvBNReLU(6, 6, kernel_size=3)
        # self.cls2 = layers.ConvBNReLU(6, 2, kernel_size=7)

        self.cls1 = layers.ConvBN(64,2,7)
        # self.cls2 = layers.ConvBN(64,2,7)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        f5 = paddle.concat((b4, p4), axis=1)
        f5 = self.cbr(f5)
        f = self.fusion(f1, f4, f5)

        # b4 = self.mask_encoder(b4)
        # p4 = self.mask_encoder(p4)

        # y = paddle.concat((b4, p4), axis=1)
        # y = self.conv1(y)

        # f = paddle.concat((f, y), axis=1)
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
        f4 = self.bff4(b4, p4)
        f1 = features_transfer(f1)
        f4 = features_transfer(f4)
        return f1, f4 , b, p
    
    # def mask_encoder(self, x):
    #     # print(x.shape)
    #     sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(None,None,None, )
    #     low_res_masks, _ = self.sam.mask_decoder(
    #             image_embeddings=x,
    #             image_pe=self.sam.prompt_encoder.get_dense_pe(),
    #             sparse_prompt_embeddings=sparse_embeddings,
    #             dense_prompt_embeddings=dense_embeddings,
    #             multimask_output=True, )
    #     # print(low_res_masks.shape)
    #     original_size = [self.sam.image_encoder.img_size, self.sam.image_encoder.img_size]
    #     masks = self.sam.postprocess_masks(
    #             low_res_masks,
    #             input_size=original_size,
    #             original_size=original_size)
    #     return masks
    
    @staticmethod
    def predict(preds):
        return preds
    
    @staticmethod
    def loss(preds, label):
        pred = preds
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        # la2 = BCELoss()(preds[1], label)
        loss = l1 + 0.5*l2 #+ 0.5*la2
        return loss



class MobileSamCD_S4(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BF3(128)
        self.bff4 = DGF2D(256)

        # self.cbr = layers.ConvBNReLU(512, 64, kernel_size=1)

        # self.ef1 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        # self.ef2 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        self.fusion = Fusion(64)
        
        # self.conv1 = layers.ConvBNReLU(6, 2, kernel_size=3)
        # self.conv2 = layers.ConvBNReLU(6, 6, kernel_size=3)
        # self.cls2 = layers.ConvBNReLU(6, 2, kernel_size=7)

        self.cls1 = layers.ConvBN(64,2,7)
        # self.cls2 = layers.ConvBN(64,2,7)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)

        f = self.fusion(f1, f4)

        # b4 = self.mask_encoder(b4)
        # p4 = self.mask_encoder(p4)

        # y = paddle.concat((b4, p4), axis=1)
        # y = self.conv1(y)

        # f = paddle.concat((f, y), axis=1)
        f = self.cls1(f)

        return f
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)

        f1 = self.bff1(b1, p1)
        f4 = self.bff4(b, p)
        f1 = features_transfer(f1)
        # f4 = features_transfer(f4)
        return f1, f4 , b, p
    
    # def mask_encoder(self, x):
    #     # print(x.shape)
    #     sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(None,None,None, )
    #     low_res_masks, _ = self.sam.mask_decoder(
    #             image_embeddings=x,
    #             image_pe=self.sam.prompt_encoder.get_dense_pe(),
    #             sparse_prompt_embeddings=sparse_embeddings,
    #             dense_prompt_embeddings=dense_embeddings,
    #             multimask_output=True, )
    #     # print(low_res_masks.shape)
    #     original_size = [self.sam.image_encoder.img_size, self.sam.image_encoder.img_size]
    #     masks = self.sam.postprocess_masks(
    #             low_res_masks,
    #             input_size=original_size,
    #             original_size=original_size)
    #     return masks
    
    @staticmethod
    def predict(preds):
        return preds
    
    @staticmethod
    def loss(preds, label):
        pred = preds
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        # la2 = BCELoss()(preds[1], label)
        loss = l1 + l2 #+ 0.5*la2
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