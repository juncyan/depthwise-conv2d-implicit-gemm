import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import math

import paddle.tensor
from paddleseg.models import layers
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss
from paddleseg.utils import load_entire_model

from models.mobilesam import MobileSAM
from models.kan import KANLinear, KAN


def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x

class Extracter(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size, checkpoint=sam_checkpoint)

        self.bf1 = Bit_Fusion(256,128)
        self.bf2 = Bit_Fusion(640,320)

        # for p in self.sam():
        #     p.requires_grad = False
    
    def extract_features(self, x):
        f1, f2, f3, f4 = self.sam.image_encoder.extract_features(x)
        return f1,f2,f3, f4
        # print(f1.shape, f4.shape)
        f1 = features_transfer(f1)
        # f2 = features_transfer(f2)
        # f3 = features_transfer(f3)
        f4 = features_transfer(f4)
        return f1, f4
    
    def prompt(self, x:paddle.tensor):
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
                input_size=x.shape[-2:],
                original_size=original_size)
        return masks
    
    def forward(self, x):
        x1 = x[:,:3, :,:]
        x2 = x[:,3: ,:,:]
        b1,_,_, b2 = self.extract_features(x1)
        p1,_,_, p2 = self.extract_features(x2)

        f1 = self.bf1(b1, p1)
        f2 = self.bf2(b2, p2)

        f1 = features_transfer(f1)
        f2 = features_transfer(f2)

        return f1, f2

class Bit_Fusion(nn.Layer):
    def __init__(self, in_channels=32, out_channels=64):
        super().__init__()
        self.lmax = nn.AdaptiveMaxPool1D(1)
        self.lavg = nn.AdaptiveAvgPool1D(1)
        dims = int(in_channels // 2)

        self.lc1 = nn.Linear(in_channels, dims)
        self.bn1 = nn.BatchNorm1D(dims, data_format="NLC")

        self.lc2 = nn.Linear(2*dims, dims)
        self.bn2 = nn.BatchNorm1D(dims, data_format="NLC")

        self.lc3 = nn.Linear(2*dims, dims)#nn.Linear(2*dims, dims)
        self.bn3 = nn.BatchNorm1D(dims, data_format="NLC")


    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], -1)
        y = self.lc1(x)
        y = self.bn1(y)
        y = F.relu(y)

        lavg = self.lavg(y)
        lmax = self.lmax(y)

        yavg = lavg * y
        ymax = lmax * y
        ya = paddle.concat([yavg, ymax], -1)

        ya = self.lc2(ya)
        ya = self.bn2(ya)
        ya = F.relu(ya)

        yb = paddle.concat([y, ya], -1)

        yb = yb + x

        yb = self.lc3(yb)
        yb = self.bn3(yb)
        yb = F.relu(yb)

        return yb

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


class MobileSamCD(nn.Layer):
    def __init__(self, img_size=256,sam_checkpoint=r"/home/jq/Code/weights/vit_t.pdparams"):
        super().__init__()
        self.sam = MobileSAM(img_size=img_size)
        self.sam.eval()
        if sam_checkpoint is not None:
            load_entire_model(self.sam, sam_checkpoint)
            self.sam.image_encoder.build_abs()

        self.bf1 = Bit_Fusion(256,128)
        self.bf2 = Bit_Fusion(640,320)

        self.fusion = Fusion(64)
        self.cls = layers.ConvBN(64,2,7)
        
    def feature_extractor(self, x):
        f1, f2,f3,f4 = self.sam.image_encoder.extract_features(x)
        return f1, f2, f3, f4
    
    def extractor(self, x1, x2):
        f1, f2, f3, f4 = self.feature_extractor(x1)
        p1, p2, p3, p4 = self.feature_extractor(x2)

        f1 = self.bf1(f1, p1)
        f2 = self.bf2(f4, p4)
        f1 = features_transfer(f1)
        f2 = features_transfer(f2)
        return f1, f2
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
        f1, f2 = self.extractor(x1, x2)
        
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
        
