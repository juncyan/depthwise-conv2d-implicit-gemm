import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
import numpy as np
from typing import Type
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from paddle.autograd import PyLayer

class MLPBlock(nn.Layer):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Layer]=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        return self.lin2(self.act(self.lin1(x)))

class SEModule(nn.Layer):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2D(
            channels,
            reduction_channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Conv2D(
            reduction_channels,
            channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        x_se = x.reshape(
            [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).mean(-1).reshape(
                [x.shape[0], x.shape[1], 1, 1])

        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * F.sigmoid(x_se)
    

class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
    def _init_weights(self, m):
        weight_attr = paddle.ParamAttr(initializer=1.0)
        bias_attr = paddle.framework.ParamAttr(initializer=0.0)
        if isinstance(m, nn.Linear):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.LayerNorm):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.Conv2D):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            
            m.weight_attr = paddle.normal(0, np.sqrt(2.0 / fan_out))
            # m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias_attr = paddle.framework.ParamAttr(initializer=0.0)

if __name__ == "__main__":
    print('utils')
    x = paddle.rand([5,3,16,16], dtype=paddle.float32).cuda()
    y = paddle.to_tensor(0.2)
    print(x[0,0,0,0], x[0,0,0,0].item())

    # m = DepthWiseConv2D(3,1).to("gpu:0")
    # y = m(x)
    # print(x == y)