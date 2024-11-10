import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from models.utils import MLPBlock, Transformer_block, features_transfer
from models.attention import RandFourierFeature

from .decoder import Up

class CD_D0(nn.Layer):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        self.ffa1 = FFA(in_c1)
        self.conv1 = nn.Sequential(layers.ConvBNReLU(in_c1, in_c2, 1), layers.ConvBNReLU(in_c2, in_c2, 3))


        self.ffa3 = FFA(in_c2)
        self.conv3 = nn.Sequential(layers.ConvBNReLU(in_c2, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

    
    def forward(self, x1, x2, y1, y2):

        f1 = self.ffa1(x2, y2)
        f1 = self.conv1(f1)
        f1 = F.interpolate(f1, x1.shape[-2:], mode='bilinear', align_corners=True)

        f3 = self.ffa3(x1, y1)
        f3 = f1 + f3
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)
        return f3


class CD_D1(nn.Layer):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        self.ffa1 = FFA(in_c1)
        self.conv1 = nn.Sequential(layers.ConvBNReLU(in_c1, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

        self.ln = nn.Linear(2*in_c2, in_c2)
        self.ffa3 = Transformer_block(in_c2, 8)
        self.conv3 = nn.Sequential(layers.ConvBNReLU(in_c2, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

    
    def forward(self, x1, x2, y1, y2):

        f1 = self.ffa1(x2, y2)
        f1 = self.conv1(f1)
        f1 = F.interpolate(f1, self.img_size, mode='bilinear', align_corners=True)

        f3 = paddle.concat([x1, y1], axis=-1)
        f3 = self.ln(f3)
        f3 = self.ffa3(f3,f3,f3)
    
        f3 = features_transfer(f3)
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)
        f3 = f3 + f1
        return f3

class FFA(nn.Layer):
    # feature fusing and aggregation
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
    
        self.sa = layers.ConvBNAct(2, 1, 3, act_type='sigmoid')
        self.outcbn = layers.ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)
        
        max_feature = paddle.max(y, axis=1, keepdim=True)
        mean_feature = paddle.mean(y, axis=1, keepdim=True)
        
        att_feature = paddle.concat([max_feature, mean_feature], axis=1)
        
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y