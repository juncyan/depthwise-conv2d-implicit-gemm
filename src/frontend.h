#include <paddle/extension.h>
#include <paddle/phi/common/data_type.h>

paddle::Tensor forward_fp32(paddle::Tensor input, paddle::Tensor weight);
paddle::Tensor backward_data_fp32(paddle::Tensor input, paddle::Tensor weight);
paddle::Tensor backward_filter_fp32(paddle::Tensor diff, paddle::Tensor input,
                                   paddle::Tensor weight);
paddle::Tensor forward_fp16(paddle::Tensor input, paddle::Tensor weight);
paddle::Tensor backward_data_fp16(paddle::Tensor input, paddle::Tensor weight);
paddle::Tensor backward_filter_fp16(paddle::Tensor diff, paddle::Tensor input,
                                   paddle::Tensor weight);
