import paddle
from models.decoder import VFC, SMDM
from models.encoderfusion import DGF, BF3
from models.samcd import MSamCD_S1
from models.attention import ECA, ChannelGate, SpatialGate
from models.encoderfusion import BSGFM

# x = paddle.rand([2,64,32,32]).cuda()
# m = SMDM(64,3).to("gpu")
# y = m(x,x)
# print(y.shape)

x = paddle.rand([2,32,16]).cuda()
m = BSGFM(16,32).to("gpu")
y = m(x,x)
print(y.shape)
