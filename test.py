import random
import os
import numpy as np
import paddle
import logging

from cd_models.fccdn import FCCDN
from cd_models.stanet import STANet
from cd_models.p2v import P2V
from cd_models.msfgnet import MSFGNet
from cd_models.fc_siam_conc import FCSiamConc
from cd_models.snunet import SNUNet
from cd_models.f3net import F3Net
from paddleseg.models import UNet

from datasets.cdloader import DataReader, TestReader
from work.train import train
from common import Args

from models.samcd import MobileSamCD
from models.mobilesam import MobileSAM


x = paddle.randn([4, 6, 256,256]).cuda()
# m = MobileSAM(img_size=1024).to('gpu')
# y = m(x)
# print(y.shape)
m = MobileSamCD().to('gpu')
y = m(x)