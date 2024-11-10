import paddle
from models.replk import RepConv



x = paddle.randn([1, 3, 256, 256]).cuda()
m = RepConv(3).to('gpu:0')

y1 = m(x)
print(y1)
paddle.save(m.state_dict(), 'repconv.pdparams')


m.load_dict(paddle.load('repconv.pdparams'))
m.eval()
y2 = m(x)

print(y2)