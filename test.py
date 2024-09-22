import paddle
import numpy as np


from models.samcd import SamB_CD

cp = r"/home/jq/Code/weights/vit_b.pdparams"
m = SamB_CD(256).to('gpu:0')
x = paddle.randn([1, 3, 256,256]).cuda()
y = m(x,x)
print(y.shape)
