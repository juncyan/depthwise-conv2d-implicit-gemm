import paddle
import numpy as np
import shutil
import glob
import os
# from models.samcd import SamB_CD

# cp = r"/home/jq/Code/weights/vit_b.pdparams"
# m = SamB_CD(256).to('gpu:0')
# x = paddle.randn([1, 3, 256,256]).cuda()
# y = m(x,x)
# print(y.shape)

p = r"/home/jq/Code/paddle/output/sysu_cd"

fs = os.listdir(p)
print(fs)
for f in fs:
    bp = os.path.join(p,f)
    ns = os.listdir(bp)
    for n in ns:
        if n.endswith(".csv"):
            shutil.copyfile(os.path.join(bp,n),f"./saves/{f}_{n}")
