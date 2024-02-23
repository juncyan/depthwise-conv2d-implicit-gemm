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
from paddleseg.models.losses import BCELoss
from paddleseg.transforms import Resize

from datasets.dataloader import DataReader, TestReader
from work.predict import predict
from common import Args
from paddleseg.utils import TimeAverager
from common import Metrics
from common.logger import load_logger
from cd_models.stanet import STANet
from pslknet.model import PSLKNet



# dataset_name = "LEVIR_c"
# dataset_name = "GVLM_CD_d"
dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)
num_classes = 2

model = STANet(3,2)


datatest = TestReader(dataset_path,"test",en_concat=False)

if __name__ == "__main__":
    print("test")
    weight_path = r"/home/jq/Code/paddle/output/clcd/STANet_2024_01_30_10/epoch_100_model.pdparams"
    predict(model, datatest, weight_path, datatest.data_name)