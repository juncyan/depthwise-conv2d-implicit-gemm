import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.utils import MLPBlock

class EMSF(nn.Layer):
    # Bitemporal Fusion based on Parall Shift Pattern 
    def __init__(self, lenk_size=64, channel=32):
        super().__init__()
        self.conv1 = nn.Conv1D(2*lenk_size, lenk_size, 1)
        self.ln2 = nn.Linear(2*channel, channel)
        self.ln3 = nn.Linear(2*channel, channel)

        self.ln4 = nn.Linear(3*channel, channel)

    def forward(self, x1, x2):
        y1 = paddle.repeat_interleave(x1, 2, 1)
        y1[:, 1::2, :] = x2
        y1 = self.conv1(y1)
        
        y2 = paddle.repeat_interleave(x1, 2, 2)
        y2[:, :, 1::2] = x2
        y2 = self.ln2(y2)

        y3 = paddle.concat([x1, x2], -1)
        y3 = self.ln3(y3)

        y = paddle.concat([y1, y2, y3], -1)
        y = self.ln4(y)
        return y
