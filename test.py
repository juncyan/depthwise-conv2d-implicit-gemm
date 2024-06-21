import random
import os
import numpy as np
import paddle

from datasets.segloader import DataReader, TestReader
# from rlklab.xception import Xception65_deeplab
from paddleseg.models import UNet, DeepLabV3P, UNetPlusPlus, UPerNet, SegNeXt, ResNet50_vd,Xception65_deeplab
from paddleseg.models import PSPNet, MobileSeg, OCRNet, hrnet, ESPNetV1
from paddleseg.models import layers

