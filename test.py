import os
import glob
import cv2
import shutil
import numpy as np
import random

import paddle

from models.deeplab import Block, Attention
from models.backbone.xception import XceptionL65
from models.deeplab import LKALab

device = 'gpu:1'


if __name__ == "__main__":
    print("test")
    paddle.device.set_device(device)
    x = paddle.rand([2,3,512,512]).cuda()
    # x = paddle.transpose(x, [0,2,1])
    # m = Block(16).to(device)
    m = LKALab(2, XceptionL65(), (0,1)).to(device)
    y = m(x)
    for i in y:
        print(i.shape)
    
    

    