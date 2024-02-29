import paddle
import paddle.nn as nn
import paddleseg.models.layers as layers

from .utils import SEModule

class RepLK(nn.Layer):
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dw_c = 4*in_channels
        self.bn = nn.BatchNorm2D(in_channels)
        self.c1 = nn.Conv2D(in_channels, dw_c, 1)
        self.lk = layers.DepthwiseConvBN(dw_c, dw_c, kernels)
        self.c2 = nn.Conv2D(dw_c, in_channels, 1)
    
    def forward(self, x):
        t = self.bn(x)
        t = self.c1(t)
        t = self.lk(t)
        t = self.c2(t)
        y = x + t
        return y

class Lark(nn.Layer):
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        self.lk = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.bn = nn.BatchNorm2D(in_channels)
        self.se = SEModule(in_channels)
        self.c1 = nn.Conv2D(in_channels, in_channels, 1)
        self.gelu = nn.GELU(in_channels)
        self.c2 = nn.Conv2D(in_channels, in_channels, 1)
    
    def forward(self, x):
        t = self.lk(x)
        t = self.bn(t)
        t = self.se(t)
        t = self.c1(t)
        t = self.gelu(t)
        t = self.c2(t)
        y = x + t
        return y

class ConvNeXt(nn.Layer):
    
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2D(dim,epsilon=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = paddle.create_parameter([dim], dtype='float32') if layer_scale_init_value > 0 else None
        
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
