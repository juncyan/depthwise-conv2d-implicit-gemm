# Efficient Implementation Depthwise Convolution with large kernels

## Installation
We can easily install the toolkit:
```bash
rm -rf build dist
python depth_gemm_C_setup.py install
rm -rf build dist
python setup.py bdist_wheel
pip install dist/*.whl
```
or 
```bash
rm -rf build dist
python depth_gemm_C_setup.py install
rm -rf build dist
python setup.py install
```

## Usage
Please follow ``depthwise_conv2d_implicit_gemm.py`` for the detailed usage:
```bash
import torch.nn as nn
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM

# depth-wise conv2d with kernel size 31
m1 = DepthWiseConv2dImplicitGEMM(384, 31, bias=False).cuda()

# navie pytorch implementation
m2 = nn.Conv2d(384, 384, 31, padding=31 // 2, bias=False, groups=384).cuda()
```
