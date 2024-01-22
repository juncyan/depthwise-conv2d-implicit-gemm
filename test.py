import paddle
from dacdnet.ablation import MSLKNet

x = paddle.rand([2,6,512,512]).cuda()
label = paddle.randint(0,1,[2,2,512,512]).cuda()
m = MSLKNet().to("gpu")
y = m(x)
l = m.loss(y, label)
print(l)