import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
import numpy as np
import math
from typing import Type
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from paddle.autograd import PyLayer


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
            mask = paddle.to_tensor(paddle.bernoulli(paddle.full(shape, keep_prob)))
            x = x / keep_prob * mask 
        return x


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


class Transformer_block(nn.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 downsample_rate: int=1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads: int):
        b, n, c = x.shape
        x = x.reshape([b, n, num_heads, c // num_heads])
        return x.transpose([0, 2, 1, 3])

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose([0, 2, 1, 3])
        return x.reshape([b, n_tokens, n_heads * c_per_head])

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @k.transpose([0, 1, 3, 2])
        attn = attn / math.sqrt(c_per_head)
        attn = F.softmax(attn, axis=-1)
        out = attn @v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

def features_transfer(x):
        x = x.transpose((0, 2, 1))
        B, C, S = x.shape
        wh = int(math.sqrt(S))
        x = x.reshape((B, C, wh, wh))
        return x



if __name__ == "__main__":
    print('utils')
    x = paddle.rand([5,3,16,16], dtype=paddle.float32).cuda()
    y = paddle.to_tensor(0.2)
    print(x[0,0,0,0], x[0,0,0,0].item())

    # m = DepthWiseConv2D(3,1).to("gpu:0")
    # y = m(x)
    # print(x == y)