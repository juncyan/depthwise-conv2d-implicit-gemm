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


import paddle
from datasets.cdloader import TestReader
from work.predict import predict
from cd_models.snunet import SNUNet


dataset_name = "LEVIR_CD"
# dataset_name = "GVLM_CD"
# dataset_name = "CLCD"
# dataset_name = "SYSU_CD"
dataset_name = "WHU_BCD"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)
num_classes = 2
datatest = TestReader(dataset_path,"test",en_concat=True)

model = SNUNet(3, 2)

weight_path = r"/home/jq/Code/paddle/output/whu_bcd/SNUNet_2024_06_30_00/SNUNet_best.pdparams"
predict(model, datatest,weight_path, dataset_name)