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
from models.encoderfusion import BF_PSP, BF_PS2, BF3
from models.decoder import Fusion, Fusion2, Decoder
from models.encoder import EMSF

features_shape = {256:np.array([[1024, 128],[256,160],[256,320],[256,320]]),
                  512:np.array([[4096, 128],[1024,160],[1024,320],[1024,320]]),
                  1024:np.array([[16384, 128],[4096,160],[4096,320],[4096,320]])}

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x


class MobileSamCD_S4(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BF3(features_shape[img_size][0][0],features_shape[img_size][0][1])
        self.bff4 = BF3(features_shape[img_size][-1][0],features_shape[img_size][-1][1])

        # self.ef1 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        # self.ef2 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        self.fusion = Decoder(64)
        
        self.conv1 = layers.DepthwiseConvBN(6, 6, kernel_size=3)
        self.cls2 = layers.ConvBNReLU(6, 2, kernel_size=7)

        self.cls1 = layers.ConvBN(64,2,7)
        # self.cls2 = layers.ConvBN(64,2,7)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        f = self.fusion(f1, f4)
        f = self.cls1(f)

        b4 = self.mask_encoder(b4)
        p4 = self.mask_encoder(p4)

        y = paddle.concat((b4, p4), axis=1)
        y = self.conv1(y)
        y = self.cls2(y)

        res = f + y

        return res, f, y
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)

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
        la1 = BCELoss()(preds[1], label)
        la2 = BCELoss()(preds[2], label)
        loss = l1 + l2 + 0.5* (la1+la2)
        return loss


class MobileSamCD_S3(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BF3(features_shape[img_size][0][0],features_shape[img_size][0][1])
        self.bff4 = BF3(features_shape[img_size][-1][0],features_shape[img_size][-1][1])

        # self.ef1 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        # self.ef2 = EMSF(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
        self.fusion = Fusion2(64)
        
        self.conv1 = layers.DepthwiseConvBN(6, 6, kernel_size=3)
        self.cls2 = layers.ConvBNReLU(6, 2, kernel_size=7)

        self.cls1 = layers.ConvBN(64,2,7)
        # self.cls2 = layers.ConvBN(64,2,7)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        f = self.fusion(f1, f4)
        f = self.cls1(f)

        b4 = self.mask_encoder(b4)
        p4 = self.mask_encoder(p4)

        y = paddle.concat((b4, p4), axis=1)
        y = self.conv1(y)
        y = self.cls2(y)

        res = f + y

        return res, f, y
    
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)

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
        # la1 = BCELoss()(preds[1], label)
        la2 = BCELoss()(preds[2], label)
        loss = l1 + l2 + 0.5* la2
        return loss
        

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

        self.conv1 = layers.DepthwiseConvBN(6, 6, kernel_size=3)
        self.conv2 = layers.ConvBNReLU(6, 2, kernel_size=7)
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
        
        self.bff1 = BF_PSP(features_shape[img_size][0][0],features_shape[img_size][0][1])
        self.bff4 = BF_PSP(features_shape[img_size][-1][0],features_shape[img_size][-1][1])
      
        self.conv1d1 = nn.Sequential(nn.Conv1D(features_shape[img_size][-1][0], features_shape[img_size][0][0],1),
                                    nn.BatchNorm1D(features_shape[img_size][0][0]), nn.ReLU6())
        
        self.conv1d2 = nn.Sequential(nn.Conv1D(features_shape[img_size][0][0], features_shape[img_size][0][0], 1),
                                    nn.BatchNorm1D(features_shape[img_size][0][0]), nn.ReLU6())
        
        self.mconv1 = nn.Sequential(layers.ConvBNReLU(256, 128, 1),
                                    layers.ConvBNReLU(128,128,3),
                                    layers.ConvBNReLU(128, 64, 1))
        

        self.conv1 = layers.ConvBNAct(320, 256, 1, act_type='relu')
        self.upconv1 = layers.ConvBNAct(6, 64, 1, act_type='relu')
        self.conv3 = layers.ConvBNAct(128, 64, 1, act_type='relu')
        
        self.cls1 = layers.ConvBN(64,2,7)
        self.cls2 = layers.ConvBN(64,2,3)

        
    def feature_extractor(self, x):
        f1, f2,f3,f4 = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)
        f1 = self.bff1(b1, p1)
        f4 = self.bff4(b4, p4)
        b4 = features_transfer(b4)
        p4 = features_transfer(p4)
        return f1, f4, b4, p4
    
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
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f0, f4, f1, f2 = self.extractor(x1, x2)
       
        f4 = self.conv1d1(f4)
        
        f0 = paddle.concat((f0, f4), axis=-1)
        f0 = self.conv1d2(f0)
        
        f = features_transfer(f0)
        f = self.mconv1(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear')
        f = self.cls1(f)

        f1 = self.conv1(f1)
        # f1 = F.interpolate(f1, scale_factor=4, mode='bilinear')
        f1 = self.mask_encoder(f1)

        f2 = self.conv1(f2)
        # f2 = F.interpolate(f2, scale_factor=4, mode='bilinear')
        f2 = self.mask_encoder(f2)

        y = paddle.concat((f1, f2), axis=1)
        y = self.upconv1(y)
        lmax = F.adaptive_max_pool2d(y, 1)
        lavg = F.adaptive_avg_pool2d(y, 1)
        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], axis=1)
        ya = self.conv3(ya)
        y = y + ya
       
        y = self.cls2(y)
        res = f + y
        return res, f, y
    @staticmethod
    def predict(preds):
        return preds[0]
    
    @staticmethod
    def loss(preds, label):
        pred = preds[0]
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        # la1 = BCELoss()(preds[1], label)
        la2 = BCELoss()(preds[2], label)
        loss = l1 + l2 + 0.5* la2
        return loss
        


class MobileSamCD_S2(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()
        
        self.bff1 = BF_PSP(features_shape[img_size][0][0],features_shape[img_size][0][1])
        self.bff4 = BF_PSP(features_shape[img_size][-1][0],features_shape[img_size][-1][1])

        self.fusion = Fusion(64)
        self.conv1d1 = nn.Sequential(nn.Conv1D(features_shape[img_size][-1][0], features_shape[img_size][0][0],1),
                                    nn.BatchNorm1D(features_shape[img_size][0][0]), nn.ReLU6())
        self.mlp = MLPBlock(features_shape[img_size][0][-1], features_shape[img_size][0][-1])
        
        self.conv1d2 = nn.Sequential(nn.Conv1D(features_shape[img_size][0][0], features_shape[img_size][0][0], 1),
                                    nn.BatchNorm1D(features_shape[img_size][0][0]), nn.ReLU6())
        
        self.mconv1 = nn.Sequential(layers.ConvBNReLU(256, 128, 1),
                                    layers.ConvBNReLU(128,128,3),
                                    layers.ConvBNReLU(128, 64, 1))
        

        # self.conv1 = layers.ConvBNAct(320, 256, 1, act_type='relu')
        self.upconv1 = layers.ConvBNAct(6, 32, 1, act_type='relu')
        self.conv3 = layers.ConvBNAct(64, 32, 3, act_type='relu')
        
        self.cls1 = layers.ConvBN(64,2,7)
        self.cls2 = layers.ConvBN(32,2,3)

        
    def feature_extractor(self, x):
        [f1, f2,f3,f4], f = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.feature_extractor(x1)
        p1, p2, p3, p4, p = self.feature_extractor(x2)
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
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]

        f1, f4 , b4, p4 = self.extractor(x1, x2)
        # print(f0.shape, f4.shape)
        f = self.fusion(f1, f4)
        # f4 = self.conv1d1(f4)
        # f4 = self.mlp(f4)

        # f0 = paddle.concat((f0, f4), axis=-1)
        # f0 = self.conv1d2(f0)
        
        # f = features_transfer(f0)
        # f = self.mconv1(f)
        # f = F.interpolate(f, scale_factor=8, mode='bilinear')
        f = self.cls1(f)

        b4 = self.mask_encoder(b4)

        p4 = self.mask_encoder(p4)

        y = paddle.concat((b4, p4), axis=1)
        y = self.upconv1(y)
        lmax = F.adaptive_max_pool2d(y, 1)
        lavg = F.adaptive_avg_pool2d(y, 1)
        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], axis=1)
        ya = self.conv3(ya)
        y = y + ya
       
        y = self.cls2(y)
        res = f + y
        return res, f, y
    @staticmethod
    def predict(preds):
        return preds[0]
    
    @staticmethod
    def loss(preds, label):
        pred = preds[0]
        l1 = BCELoss()(pred, label)
        label = paddle.argmax(label, axis=1)
        l2 = LovaszSoftmaxLoss()(pred, label)
        la1 = BCELoss()(preds[1], label)
        la2 = BCELoss()(preds[2], label)
        loss = l1 + l2 + 0.5*la1 + 0.5* la2
        return loss
        
