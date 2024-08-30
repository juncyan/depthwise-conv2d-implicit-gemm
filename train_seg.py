import random
import os
import numpy as np
import paddle
import paddle.nn as nn
from models.samcd import MobileSamCD

# x = paddle.rand([1, 3, 256, 256]).cuda()
# m = MobileSamCD().to("gpu")
# m.train()
# y = m(x,x)
# print(y.shape)

class T(nn.Layer):
    def __init__(self):
        super(T, self).__init__()
        print(self.training)
    
    def forward(self, x):
        return x

m = T()
m.eval()

   