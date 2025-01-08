__version__ = "0.0.1"

import paddle
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)
if not hasattr(paddle, "masked_fill"):
    paddle.masked_fill = masked_fill
if not hasattr(paddle.Tensor, "masked_fill"):
    paddle.Tensor.masked_fill = masked_fill

if not hasattr(paddle, "is_autocast_enabled"):
    def is_autocast_enabled():
        tracer = paddle.framework._dygraph_tracer()
        return False if tracer._amp_level == paddle.core.AmpLevel.O0 else True
    paddle.is_autocast_enabled = is_autocast_enabled

from .depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM