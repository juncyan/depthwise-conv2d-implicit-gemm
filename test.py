import paddle
from cd_models.dacdnet.blocks import GAM

x = paddle.rand([4,16,128,128]).cuda()
m = GAM().to("gpu")
y = m(x)
print(y.shape)