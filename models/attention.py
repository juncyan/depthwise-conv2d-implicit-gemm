import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor
import paddleseg.models.layers as layers
from typing import Union, Optional
from models.utils import MLPBlock

class ECA(nn.Layer):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=None) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose([0,2,1])).transpose([0,2,1]).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Flatten(nn.Layer):
    def forward(self, x):
        y = x.reshape([x.shape[0], x.shape[1], -1])
        y = y.transpose([0, 2, 1])
        return y
    

class ChannelGate(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #     )
        self.flap = Flatten()
        self.mlp = MLPBlock(gate_channels, gate_channels // reduction_ratio)
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                avg_pool = self.flap(avg_pool)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                max_pool = self.flap(max_pool)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        # .unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = self.flap(scale).unsqueeze(-1)
        return x * scale

def lp_pool2d(input, norm_type,kernel_size,stride, ceil_mode = False):
    r"""
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
   
    kw, kh = kernel_size
    if stride is not None:
        out = F.avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = F.avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (paddle.sign(out) * F.relu(paddle.abs(out))).mul(kw * kh).pow(1.0 / norm_type)

def logsumexp_2d(tensor:paddle.tensor):
    tensor_flatten = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    s, _ = paddle.max(tensor_flatten, axis=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(axis=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Layer):
    def forward(self, x):
        cm = paddle.max(x,1).unsqueeze(1)
        ca = paddle.mean(x,1).unsqueeze(1)
        return paddle.concat([cm, ca], axis=1)

class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = layers.ConvBN(2,1,kernel_size)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class BAM(nn.Layer):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

class CBAM(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
