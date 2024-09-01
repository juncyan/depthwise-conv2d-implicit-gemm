import paddle
from models.decoder import VFC, SMDM
from models.encoderfusion import DGF, BF3

# x = paddle.rand([2,64,32,32]).cuda()
# m = SMDM(64,3).to("gpu")
# y = m(x,x)
# print(y.shape)

x = paddle.rand([2,64,320]).cuda()
m = BF3(64,320).to("gpu")
y = m(x, x)
print(y.shape)
