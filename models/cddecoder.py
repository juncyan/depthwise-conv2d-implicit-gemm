import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from models.utils import MLPBlock, Transformer_block, features_transfer
from models.attention import RandFourierFeature
from paddlenlp.transformers.mamba.modeling import MambaMixer, MambaConfig


class CD_Mamba(nn.Layer):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        cfg1 = MambaConfig(4096, in_c1)
        cfg2 = MambaConfig(1024, in_c2)
        self.ssm1 = nn.Sequential(MambaMixer(cfg1, 0), MambaMixer(cfg1, in_c1-1), nn.LayerNorm(in_c1))
        self.ssm2 = nn.Sequential(MambaMixer(cfg2, 0), MambaMixer(cfg2, in_c2-1), nn.LayerNorm(in_c2))
        
        self.proj1 = nn.Linear(2*in_c1, in_c1)
        self.conv1 = nn.Sequential(layers.ConvBNReLU(in_c1, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))
        self.proj2 = nn.Sequential(nn.Linear(2*in_c2, in_c2), nn.LayerNorm(in_c2))
      
        self.conv3 = nn.Sequential(layers.ConvBNReLU(in_c2, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

        self.conv4 = layers.ConvBNReLU(out_c+out_c, out_c, 1)

    
    def forward(self, x1, x2, y1, y2):
        f1 = paddle.concat([x2, y2], axis=-1)
        f1 = self.proj1(f1)
        f1 = self.ssm1(f1)

        f1 = features_transfer(f1)
        f1 = self.conv1(f1)
        f1 = F.interpolate(f1, self.img_size, mode='bilinear', align_corners=True)

        f3 = paddle.concat([x1, y1], axis=-1)
        f3 = self.proj2(f3)
        f3 = self.ssm2(f3)
        f3 = features_transfer(f3)
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)

        f4 = paddle.concat([f3,f1] , axis=1)
        f4 = self.conv4(f4)
        return f4

class FFA(nn.Layer):
    # feature fusing and aggregation
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = layers.ConvBNReLU(dims, in_channels, 1)
        self.lfc = MSIF(in_channels, dims)
    
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
    

class MSIF(nn.Layer):
    #multi-scale information fusion
    def __init__(self, in_channels, internal_channels):
        super().__init__()
        self.cbr1 = layers.ConvBNReLU(in_channels, internal_channels, 1)

        self.cond1 = nn.Conv2D(internal_channels, internal_channels, 1)
        self.cond3 = nn.Conv2D(internal_channels, internal_channels, 3, padding=3, dilation=3, groups=internal_channels)
        self.cond5 = nn.Conv2D(internal_channels, internal_channels, 3, padding=5, dilation=5, groups=internal_channels)

        self.bn1 = nn.BatchNorm2D(internal_channels)
        self.relu1 = nn.ReLU()

        self.cbr2 = layers.ConvBNReLU(internal_channels, in_channels, 1)
        
        self.lastbn = nn.BatchNorm2D(in_channels)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        y = self.cbr1(x)
        y1 = self.cond1(y)
        y2 = self.cond3(y)
        y3 = self.cond5(y)
        y = self.relu1(self.bn1(y1 + y2 + y3))
        y = self.cbr2(y)
        return self.relu(self.lastbn(x + y))