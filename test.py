import paddle
import os
import cv2
import numpy as np
import pandas as pd
import glob
# from skimage import io
# from models.replk import RepLKBlock, RepLKNet

# x = paddle.randn([1, 3, 224, 224]).cuda()
# m = RepLKBlock(3,16,13,3,0.2).to("gpu")
# y = m(x)
# print(y.shape)
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


x = paddle.randn([1, 384, 13, 13]).cuda()

m1 = DepthWiseConv2dImplicitGEMM(384, 31, bias=False).to("gpu")
y = m1(x)
print(y.shape)
