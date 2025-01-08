import pytest
import paddle
import paddle.nn.functional as F

from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


def paddle_forward(x, w):
    return F.conv2d(x, w, padding=w.shape[3] // 2, groups=w.shape[0])


def test_cuda_available():
    if not paddle.device.is_compiled_with_cuda():
        pytest.exit("no cuda available")


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64, 192])
@pytest.mark.parametrize("kernel_size", [3, 7, 13, 31])
@pytest.mark.parametrize("resolution", [16, 32])
@pytest.mark.parametrize("seed", [0, 42])
def test_forward_fp32(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    paddle.seed(seed)
    with paddle.fluid.dygraph.guard():
        x = paddle.randn([batch_size, channels, resolution, resolution], dtype='float32')
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size)
        y = m(x)
        y_ref = paddle_forward(x, m.weight)
        assert y.dtype == paddle.float32
        assert paddle.allclose(y, y_ref)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_forward_fp16(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    paddle.seed(seed)
    with paddle.fluid.dygraph.guard():
        x = paddle.randn([batch_size, channels, resolution, resolution], dtype='float16')
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size)
        with paddle.amp.auto_cast():
            y = m(x)
            y_ref = paddle_forward(x, m.weight)
        assert y.dtype == paddle.float16
        assert y_ref.dtype == paddle.float16
        assert paddle.allclose(y, y_ref, rtol=1e-3, atol=1e-6), (y - y_ref).max()


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_backward_fp32(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    paddle.seed(seed)
    with paddle.fluid.dygraph.guard():
        x = paddle.randn([batch_size, channels, resolution, resolution], dtype='float32')
        x.stop_gradient = False
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size)
        y = m(x)
        y.mean().backward()
        dx = x.gradient()
        dw = m.weight.gradient()
        x.clear_grad()
        m.weight.clear_grad()
        y_ref = paddle_forward(x, m.weight)
        y_ref.mean().backward()
        dx_ref = x.gradient()
        dw_ref = m.weight.gradient()
        assert paddle.allclose(dx, dx_ref), (dx - dx_ref).max()
        assert paddle.allclose(dw, dw_ref, rtol=1e-4, atol=1e-6), (dw - dw_ref).max()


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 7, 13])
@pytest.mark.parametrize("resolution", [16])
@pytest.mark.parametrize("seed", [0, 42])
def test_backward_fp16(
    batch_size,
    channels,
    kernel_size,
    resolution,
    seed,
):
    paddle.seed(seed)
    with paddle.fluid.dygraph.guard():
        x = paddle.randn([batch_size, channels, resolution, resolution], dtype='float16')
        x.stop_gradient = False
        m = DepthWiseConv2dImplicitGEMM(channels, kernel_size)
        with paddle.amp.auto_cast():
            y = m(x)
            y.mean().backward()
        dx = x.gradient()
        dw = m.weight.gradient()
        x.clear_grad()
        m.weight.clear_grad()
        with paddle.amp.auto_cast():
            y_ref = paddle_forward(x, m.weight)
            y_ref.mean().backward()
        dx_ref = x.gradient()
        dw_ref = m.weight.gradient()
        assert dx.dtype == dx_ref.dtype
        assert dx.dtype == paddle.float16
        assert dw.dtype == dw_ref.dtype
        assert dw.dtype == paddle.float32
        assert paddle.allclose(dx, dx_ref), (dx - dx_ref).max()
        assert paddle.allclose(dw, dw_ref, rtol=1e-4, atol=1e-6), (dw - dw_ref).max()