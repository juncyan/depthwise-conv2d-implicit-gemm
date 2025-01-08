import paddle
import os
from pathlib import Path
# from setuptools import setup, find_packages
from site import getsitepackages
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, setup

CUTLASS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cutlass'))

def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

print(CUTLASS_ROOT)
paddle_includes = [
    os.path.join(CUTLASS_ROOT, "include"),
    os.path.join(CUTLASS_ROOT, "tools", "library", "include"),
    os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
    os.path.join(CUTLASS_ROOT, "examples", "common"),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')),
]
# for site_packages_path in getsitepackages():
#     paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
#     paddle_includes.append(
#         os.path.join(site_packages_path, "paddle", "include", "third_party")
#     )
    
prop = paddle.device.cuda.get_device_properties()
cc = prop.major * 10 + prop.minor
cc_list = [cc, ]
cc_flag = []
for arch in cc_list:
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{arch},code=sm_{arch}")

source = ["src/frontend.cpp",
        "src/forward_fp32.cu",
        "src/backward_data_fp32.cu",
        "src/backward_filter_fp32.cu",
        "src/forward_fp16.cu",
        "src/backward_data_fp16.cu",
        "src/backward_filter_fp16.cu",]

if cc >= 75:
    cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

print("sources", source)

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": append_nvcc_threads(
        [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
        ]
        + cc_flag
    ),
}

setup(
    name='depthwise_conv2d_implicit_gemm_C',
    py_modules=['depthwise_conv2d_implicit_gemm_C'],
    ext_modules=CUDAExtension(sources=source,
                              include_dirs=paddle_includes,
                              extra_compile_args=extra_compile_args,
                              verbose=True),
    cmdclass={
        'build_ext': BuildExtension
    }
    )
