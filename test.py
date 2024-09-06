import paddle
import numpy as np


from models.samcd import MSamCD_SSH, MSamCD_FSSH
from models.dhsamcd import DHSamCD, DHSamCD_v2
from models.attention import ECA, ChannelGate, SpatialGate, FFN, get_mgrid, RandFourierFeature
from models.encoderfusion import BSGFM
from core.misc.count_params import flops
from paddleseg.utils import TimeAverager, op_flops_funs

# x = get_mgrid(256, 256,2,0.5)
# coords = np.expand_dims(x, axis=0)
# coords = paddle.to_tensor(coords).cuda()
# print(coords.shape
# x = paddle.rand([4, 16, 10]).cuda()
# m = RandFourierFeature(16, 10, 16).to("gpu")
# y = m(x)
# print(y.shape)


x = paddle.rand([2,3,512,512]).cuda()
z = paddle.rand([2,2,512,512]).cuda()
label = paddle.argmax(z, axis=1, keepdim=True)
m = DHSamCD_v2(512).to("gpu")
y = m(x, x)
p = m.predict(y)
print(p.shape)
loss = m.loss(y, label)
print(loss)

flop_p = flops(m, [1, 6, 512, 512],
custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
