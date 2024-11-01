import paddle
import numpy as np
import random
import shutil
import glob
import os
import cv2
from tqdm import tqdm
from models.model import SCDSam
from skimage import io

# x = paddle.rand((1, 6, 512, 512)).cuda()
# m = SCDSam(img_size=512).to('gpu')
# y = m(x)
    
x1 = [1,2,3]

for idx, (x,y,z) in enumerate(zip(x1, x1, x1)):
    print(idx, x,y,z)