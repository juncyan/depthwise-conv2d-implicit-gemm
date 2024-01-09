# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
import time
import paddle
import paddle.nn.functional as F
import argparse
import random
import os
import numpy as np
import paddle
from paddleseg.models import UNet, DeepLabV3P, UNetPlusPlus, UPerNet, SegNeXt, ResNet50_vd, Xception65_deeplab
from paddleseg.models.losses import BCELoss
from paddleseg.transforms import Resize

from datasets.dataloader import DataReader, TestReader
from work.predict import predict
from common import Args
from models.dacdnet import ACDNet_v3
from models.dacdnet.abli import ACDNet
from models.f3net import F3Net
from models.f3net.UAlibaltion import LKAUChangeST, LKAUChange_noPPM
from models.msfgnet import MSFGNet
from paddleseg.utils import TimeAverager
from common import Metrics
from common.logger import load_logger



dataset_name = "LEVIR_d"
# dataset_name = "GVLM_CD_d"
# dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)
num_classes = 2

# model = UNet(num_classes, in_channels=6)
# model = UNetPlusPlus(num_classes, 6)
# model = DeepLabV3P(num_classes, backbone=ResNet50_vd(in_channels=6))
# model = SegNeXt(num_classes=num_classes, decoder_cfg={}, backbone=ResNet50_vd(in_channels=6))
# model = ACDNet_v3(in_channels=6, num_classes=num_classes)
# model = UChange(in_channels=6, num_classes=num_classes)
# model = UPerNet(num_classes, ResNet50_vd(in_channels=6),(0,1,2,3))
# model = ACDNet_v3(num_classes,in_channels=6)
# model = LKAUChange_noPPM()
model = MSFGNet()


datatest = TestReader(dataset_path,"test")

if __name__ == "__main__":
    print("test")
    weight_path = r"/home/jq/Code/paddle/output/levir_c/LKSWNet_2023_12_15_01/LKSWNet_best.pdparams"
    predict(model, datatest, weight_path, datatest.data_name)