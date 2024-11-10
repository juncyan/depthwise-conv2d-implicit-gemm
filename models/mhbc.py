import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor
import paddleseg.models.layers as layers
from typing import Union, Optional
import numpy as np
from models.utils import MLPBlock