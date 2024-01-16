import paddle
from dacdnet.ablation import LKPSNet


x = paddle.rand([4,3,256, 256]).cuda()
m = LKPSNet().to("gpu")
y = m(x, x)
print(y.shape)