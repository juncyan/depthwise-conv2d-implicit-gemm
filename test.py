import paddle
import paddle.nn as nn
from paddleseg.models import layers
from paddleseg.models.losses import lovasz_loss


from models.segment_anything.build_sam import build_sam_vit_t
x = paddle.randn([1, 64, 256, 256]).cuda()
pd_path = r"/home/jq/Code/weights/vit_t.pdparams"
m = build_sam_vit_t(checkpoint=pd_path).to('gpu')
m.image_encoder(x)