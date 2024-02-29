from pslknet.lkblocks import ConvNeXt
import paddle

x = paddle.rand([2,16,128,128]).cuda()
m = ConvNeXt(16).to('gpu')
y = m(x)
print(y.shape)