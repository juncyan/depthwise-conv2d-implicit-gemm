import paddle
from paddle import optimizer
from paddle.nn import functional as F
from cd_models.mamba.mamba import Mamba, MambaConfig
from cd_models.mamba.vmamba import SS2D, VSSBackbone
from models.model import SCDSam_Mamba
from paddlenlp.transformers.mamba.modeling import MambaMixer, MambaConfig

x = paddle.rand([1, 3,512, 512]).cuda()

# m = SS2D(32).to('gpu:0')
m = VSSBackbone().to('gpu:0')
y = m(x)
for i in y:
    print(i.shape)