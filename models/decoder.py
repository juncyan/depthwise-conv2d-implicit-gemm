import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from models.attention import ECA, RandFourierFeature
from models.utils import Transformer_block, features_transfer
from cd_models.mamba.mamba import Mamba, MambaConfig
# from cd_models.mamba.ppmamba import MambaBlock


class SemanticDv0(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        
        self.st1conv1 = layers.ConvBNReLU(in_c1, in_c2, 1)
        self.st1conv2 = layers.ConvBNReLU(in_c2, in_c2, 3) #MSIF(in_c2, in_c2*2)
        self.st1conv3 = layers.ConvBNReLU(in_c2, in_c2, 3)
        
        self.sa1 = layers.ConvBNAct(2,1,1, act_type="sigmoid")
        # self.up1 = UpSampling(in_c2, scale=2)

        self.st2conv1 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.st2conv2 = layers.ConvBNReLU(out_c, out_c, 3)#MSIF(out_c, out_c*2)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 3)

        self.sa2 = layers.ConvBNAct(2,1,1, act_type="sigmoid")

    def forward(self, x1, x2):
        # f = self.lk(x)
        # f = self.eca(f)
        # f = self.cbr(f)

        f = self.st1conv1(x2)
        max_feature1 = paddle.max(f, axis=1, keepdim=True)
        mean_feature1 = paddle.mean(f, axis=1, keepdim=True)
        att_feature1 = paddle.concat([max_feature1, mean_feature1], axis=1)
        
        y = self.sa1(att_feature1)
        f = y * f

        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        f1 = F.interpolate(f1, size=x1.shape[-2:], mode='bilinear', align_corners=True) #self.up1(f)

        f2 = f1 + x1
        f2 = self.st2conv1(f2)

        max_feature2 = paddle.max(f2, axis=1, keepdim=True)
        mean_feature2 = paddle.mean(f2, axis=1, keepdim=True)
        att_feature2 = paddle.concat([max_feature2, mean_feature2], axis=1)
        
        y2 = self.sa2(att_feature2)
        f2 = y2 * f2

        f2 = self.st2conv2(f2)
        f2 = self.st2conv3(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        
        # f3 = self.st3conv1(f2)
        # f3 = self.st3conv2(f3)
        # f3 = self.up3(f3)
        return f2


class SemanticDv1(nn.Layer):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        
        self.st1conv1 = layers.ConvBNReLU(in_c1, out_c, 1)
        self.st1conv2 = layers.ConvBNReLU(out_c, out_c, 3) #MSIF(in_c2, in_c2*2)
        self.st1conv3 = layers.ConvBNReLU(out_c, out_c, 1)
        
        self.sa1 = layers.ConvBNAct(2,1,1, act_type="sigmoid")
        # self.up1 = UpSampling(in_c2, scale=2)

        # self.st2conv1 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.att = Transformer_block(in_c2, 8)
        self.st2conv2 = layers.ConvBNReLU(in_c2, out_c, 3)#MSIF(out_c, out_c*2)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 13)

        self.sa2 = layers.ConvBNAct(2,1,1, act_type="sigmoid")

    def forward(self, x1, x2):
        # f = self.lk(x)
        # f = self.eca(f)
        # f = self.cbr(f)

        f = self.st1conv1(x2)
        max_feature1 = paddle.max(f, axis=1, keepdim=True)
        mean_feature1 = paddle.mean(f, axis=1, keepdim=True)
        att_feature1 = paddle.concat([max_feature1, mean_feature1], axis=1)
        
        y = self.sa1(att_feature1)
        f = y * f

        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        f1 = F.interpolate(f1, size=self.img_size, mode='bilinear', align_corners=True) #self.up1(f)

        f2 = self.att(x1,x1,x1)
        f2 = features_transfer(f2)
        # f2 = f2.transpose([0, 2, 3, 1])

        f2 = self.st2conv2(f2)
        f2 = self.st2conv3(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        
        f2 = f2 + f1
        return f2 



class DoubleConv(nn.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Layer):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.bilinear = bilinear
        if bilinear:
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2DTranspose(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        if(self.bilinear):
            x1 =nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # print("x2.size():", x2.shape)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = paddle.concat([x2, x1], axis=1)
        return self.conv(x)
 

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