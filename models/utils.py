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

    



if __name__ == "__main__":
    print('utils')
    x = paddle.rand([5,3,16,16], dtype=paddle.float32).cuda()
    y = paddle.to_tensor(0.2)
    print(x[0,0,0,0], x[0,0,0,0].item())

    # m = DepthWiseConv2D(3,1).to("gpu:0")
    # y = m(x)
    # print(x == y)