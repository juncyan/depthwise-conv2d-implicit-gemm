import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from models.attention import ECA, RandFourierFeature, FFTModel, DFFN
from models.utils import Transformer_block, features_transfer
from paddlenlp.transformers.mamba.modeling import MambaMixer, MambaConfig
# from cd_models.mamba.ppmamba import MambaBlock
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


class SemantiMambacDv0(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        
        conf1 = MambaConfig(4096, in_c2)
        self.ssm1 = nn.Sequential(MambaMixer(conf1, 0), MambaMixer(conf1, in_c2-1), nn.LayerNorm(in_c2))

        conf2 = MambaConfig(1024, in_c1)
        self.ssm2 = nn.Sequential(MambaMixer(config=conf2, layer_idx=0), MambaMixer(conf2, in_c1-1), nn.LayerNorm(in_c1))

        self.st1conv1 = layers.ConvBNReLU(in_c1, in_c2, 1)
        self.st1conv2 = DepthWiseConv2dImplicitGEMM(in_c2, 12, True) #layers.ConvBNReLU(in_c2, in_c2, 3)
        self.st1conv3 = layers.ConvBNReLU(in_c2, in_c2, 1)
        
        self.sa1 = layers.ConvBNAct(2,1,1, act_type="sigmoid")
        
        self.st2conv2 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 3)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 1)

    def forward(self, x1, x2):
        f = self.ssm2(x2)
        # f = self.fft2(f)
        f = features_transfer(f)
        
        f = self.st1conv1(f)
        max_feature1 = paddle.max(f, axis=1, keepdim=True)
        mean_feature1 = paddle.mean(f, axis=1, keepdim=True)
        att_feature1 = paddle.concat([max_feature1, mean_feature1], axis=1)
        
        y = self.sa1(att_feature1)
        f = y * f

        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True) #self.up1(f)

        f2 = self.ssm1(x1)
        # f2 = self.fft1(f2)
        f2 = features_transfer(f2)
        # f2 = f2.transpose([0, 2, 3, 1])
        # print(f1.shape, f2.shape)
        f2 = f1 + f2
        f2 = self.st2conv2(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        f2 = self.st2conv3(f2)
        f2 = self.st2conv3(f2)
        
        # f2 = f2 + f1
        return f2 
