import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddleseg.models import layers

from .pos_embed import get_2d_relative_pos_embed
from .vertex import DyGraphConv2d
from .nn import act_layer


class GCN_block(nn.Layer):
    def __init__(self,in_channels, kernel_size=3, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=16, drop_path=0.0, relative_pos=False):
        super().__init__()
        self.channels = in_channels
        self.n = n #节点数
        self.r = r 
        self.fc1 = layers.ConvBN(in_channels, in_channels,1) #映射层
    
        self.graph_conv = DyGraphConv2d(in_channels, 2*in_channels, kernel_size, dilation, conv,
                               act, norm, bias, stochastic, epsilon, r) #图卷积层
        
        self.fc2 = layers.ConvBN(in_channels, in_channels, 1, 0) #映射层

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity() # VIT中的DropPath
        self.relative_pos = None

        if relative_pos:  #是否使用相对连接关系
            print('using relative_pos')
            relative_pos_tensor = paddle.to_tensor(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = paddle.ParamAttr(initializer=-relative_pos_tensor.squeeze(1))
            #add_parameter(self,-relative_pos_tensor.squeeze(1))
            self.relative_pos.stop_gradient=True

    def _get_relative_pos(self, relative_pos, H, W): 
        
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        
        x = self.graph_conv(x, relative_pos)
        
        x = self.fc2(x)
        
        x = self.drop_path(x) + _tmp
        return x


class FFN(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = layers.activation(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Layer):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2D(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2D(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2D(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2D(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2D(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x



