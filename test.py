import paddle
from pslknet.model import PSLKNet_k3

x = paddle.rand([4,6,256,256]).cuda()
m = PSLKNet_k3().to("gpu")
y = m(x)
print(y.shape)