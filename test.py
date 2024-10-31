import paddle
import numpy as np
import random
import shutil
import glob
import os
import cv2
from tqdm import tqdm
from models.model import SCDSam

x = paddle.rand((1, 6, 512, 512)).cuda()
m = SCDSam(img_size=512).to('gpu')
y = m(x)
    
