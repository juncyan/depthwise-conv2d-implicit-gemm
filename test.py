import paddle
import paddle.nn as nn
from paddleseg.models import layers
from pslknet.cblkb import RepConv, RepC3, RepConvBn

if __name__ == "__main__":
    x = paddle.rand([1,128,256,256]).cuda()
    m = RepConv(128).to('gpu:0')
    y1 = m(x)
    m.eval()
    paddle.save(m.state_dict(), "testrepc.pdparams")
    layer_state_dict = paddle.load("testrepc.pdparams")
    # print(layer_state_dict)

    # with paddle.no_grad():
    m.set_state_dict(layer_state_dict)
    m.eval()

    y2 = m(x)
    print(y1-y2)



