import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

import depthwise_conv2d_implicit_gemm_C as _extension

__all__ = ["DepthWiseConv2dImplicitGEMM"]


class _DepthWiseConv2dImplicitGEMMFP32(PyLayer):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _extension.forward_fp32(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensor()
        dx = _extension.backward_data_fp32(grad, w)
        dw = _extension.backward_filter_fp32(grad, x, w)
        return dx, dw


class _DepthWiseConv2dImplicitGEMMFP16(PyLayer):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _extension.forward_fp16(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensor()
        dx = _extension.backward_data_fp16(grad, w)
        dw = _extension.backward_filter_fp16(grad, x, w)
        return dx, dw


class DepthWiseConv2dImplicitGEMM(nn.Conv2D):
    def __init__(self, channels, kernel, bias=False):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            groups=channels,
            bias_attr=bias,
        )

    def forward(self, x):
        if x.dtype == paddle.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, self.weight)
        elif x.dtype == paddle.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, self.weight)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        if self.bias is not None:
            x = x + self.bias.reshape([1, -1, 1, 1])
        return x


if __name__ == "__main__":
    paddle.seed(0)
    if paddle.device.is_compiled_with_cuda():
        x = paddle.randn([64, 384, 32, 32], dtype=paddle.float32).cuda()
        m1 = DepthWiseConv2dImplicitGEMM(384, 31, bias=False).to("gpu")
        m2 = nn.Conv2D(
            384, 384, 31, padding=31 // 2, bias_attr=False, groups=384
        ).to("gpu")
        m2.set_state_dict(m1.state_dict())

        # 使用 PaddlePaddle 的 AMP
        amp_level = "O1"  # PaddlePaddle 的 AMP 级别
        with paddle.amp.auto_cast(level=amp_level):
            y1 = m1(x)
            y2 = m2(x)

        # 反向传播
        loss1 = (y1.mean() * 1024).backward()
        loss2 = (y2.mean() * 1024).backward()

        print("output difference:", ((y1 - y2) ** 2).mean().item())
        print(
            "gradient difference:",
            ((m1.weight.grad - m2.weight.grad) ** 2).mean().item(),
        )