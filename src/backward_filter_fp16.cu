#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutlass/convolution/device/convolution.h"
#include "cutlass/util/device_memory.h"

#include <paddle/extension.h>

#include <iostream>

// The code section below describes datatype for input, output tensors and
// computation between elements
using ElementAccumulator = float;  // Data type of accumulator
using ElementComputeEpilogue =
        float;  // Data type of epilogue computation (alpha, beta)
using ElementSrc = cutlass::half_t;   // Data type of elements in src tensor
using ElementDiff = cutlass::half_t;  // Data type of elements in diff tensor
using ElementGrad = float;            // Data type of elements in output tensor

using LayoutSrc = cutlass::layout::TensorNCHW;
using LayoutDiff = cutlass::layout::TensorNCHW;
using LayoutGrad = cutlass::layout::TensorNCHW;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ThreadblockShape =
        cutlass::gemm::GemmShape<128, 256, 32>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;  // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape =
        cutlass::gemm::GemmShape<8, 8, 4>;  // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
        cutlass::conv::threadblock::DepthwiseConvolutionWgradThreadblockSwizzle;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// This code section describes the epilogue part of the kernel, we use default
// value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementGrad,              // Data type of output matrix.
        1, ElementAccumulator,    // Data type of accumulator
        ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                  // combination
using Convolution = cutlass::conv::device::ConvolutionBackwardFilter<
        ElementSrc, LayoutSrc, ElementDiff, LayoutDiff, ElementGrad, LayoutGrad,
        ElementAccumulator, cutlass::conv::ConvType::kDepthwiseConvolution,
        MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOp, SwizzleThreadBlock, NumStages, 1, 1,
        cutlass::conv::SpecialOptimizeDesc::NONE, cutlass::arch::OpMultiplyAdd,
        cutlass::conv::ImplicitGemmMode::GEMM_NT>;

struct Options {
    bool help;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    bool reference_check;
    bool measure_performance;
    int iterations;
    bool save_workspace;
    ElementComputeEpilogue alpha;
    ElementComputeEpilogue beta;
    bool benchmark;
    std::string tag;

    Options()
            : help(false),
              input_size(1, 32, 32, 32),
              filter_size(32, 3, 3, 1),
              padding(1, 1, 1, 1),
              conv_stride(1, 1),
              dilation(1, 1),
              reference_check(false),
              measure_performance(true),
              iterations(1000),
              save_workspace(false),
              alpha(1),
              beta(0),
              benchmark(false) {}

    // Verify the problem size is compatible with the CUTLASS Convolution
    // implementation.
    bool valid() {
        int const kAlignment = 1;

        if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
            // misaligned tensors
            return false;
        }

        // Invalid padding
        if ((padding.h() != filter_size.h() / 2) ||
            (padding.w() != filter_size.w() / 2)) {
            return false;
        }

        return true;
    }

    /// Updates input and filter sizes
    void update(cutlass::Tensor4DCoord input_size,
                cutlass::Tensor4DCoord filter_size) {
        this->input_size = input_size;
        this->filter_size = filter_size;

        padding.n() = filter_size.h() / 2;
        padding.h() = filter_size.h() / 2;
        padding.w() = filter_size.w() / 2;
        padding.c() = filter_size.w() / 2;
    }

    /// Computes the output tensor size (NPQK)
    cutlass::Tensor4DCoord output_size() const {
        return cutlass::Tensor4DCoord(
                input_size.n(),
                (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                conv_stride.row() +
                        1,
                (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                conv_stride.column() +
                        1,
                filter_size.n());
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const {
        // Number of multiply-adds = NPQK * CRS / K
        int64_t fmas =
                output_size().product() *
                int64_t(filter_size.h() * filter_size.w() * filter_size.c()) /
                output_size().c();

        // Two flops per multiply-add
        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};

#define CUTLASS_CHECK(status)                                                 \
    {                                                                         \
        cutlass::Status error = status;                                       \
        if (error != cutlass::Status::kSuccess) {                             \
            std::cerr << "Got cutlass error: "                                \
                      << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess) {                                           \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

bool is_contiguous(const paddle::Tensor& tensor) {
    auto strides = tensor.strides();
    auto shape = tensor.shape();
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (strides[i] != stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

#define CHECK_CUDA(x) \
    PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    PD_CHECK(is_contiguous(x), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

paddle::Tensor backward_filter_fp16(paddle::Tensor diff, paddle::Tensor input,
                                   paddle::Tensor weight) {
    CHECK_INPUT(diff);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    // auto output = paddle::empty_like(weight, at::TensorOptions().dtype(at::kFloat));
    auto output = paddle::empty_like(weight, paddle::DataType::FLOAT32);
    // auto output = paddle::full_like(weight, 0.0f, paddle::DataType::FLOAT32);

    Options options = Options();

    options.update({input.shape()[0], input.shape()[2], input.shape()[3], input.shape()[1]},
                   {weight.shape()[0], weight.shape()[2], weight.shape()[3], 1});

    cutlass::TensorRef<ElementSrc, LayoutSrc> d_src(
            (ElementSrc*)(input.data<ElementSrc>()),
            LayoutSrc::packed(options.input_size));
    cutlass::TensorRef<ElementDiff, LayoutDiff> d_diff(
            (ElementDiff*)(diff.data<ElementDiff>()),
            LayoutDiff::packed(options.output_size()));
    cutlass::TensorRef<typename Convolution::ElementGrad,
                       typename Convolution::LayoutGrad>
            d_filter((ElementGrad*)(output.data<ElementGrad>()),
                     LayoutGrad::packed(options.filter_size));

    //
    // Define arguments for CUTLASS Convolution
    //

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    typename Convolution::Arguments arguments{
            {options.input_size, options.filter_size, options.padding,
             options.conv_stride, options.dilation, options.output_size(), mode,
             split_k_slices, options.filter_size.n()},
            d_src,     // tensor_src.device_ref(),
            d_diff,    // tensor_filter.device_ref(),
            d_filter,  // tensor_dst.device_ref(),
            {options.alpha}};

    //
    // Initialize CUTLASS Convolution
    //

    Convolution conv_op;

    size_t workspace_size = conv_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(conv_op.initialize(arguments, workspace.get()));

    //
    // Launch initialized CUTLASS kernel
    //

    CUTLASS_CHECK(conv_op());

    return output;
}
