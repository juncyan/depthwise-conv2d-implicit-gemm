import paddle
from paddle import nn

from paddleseg.models import layers

##############################
#    Basic layers
##############################

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    """
    Args:
        norm (string): normalize style
        nc (int): normalize channels
    Returns:
        paddle.nn.Layer: normalize function
    """
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2D(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2D(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2D(channels[i - 1], channels[i], 1, bias_attr=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(value=0.)(m.bias)
            elif isinstance(m, nn.BatchNorm2D) or isinstance(m, nn.InstanceNorm2D):
                nn.initializer.Constant(value=0.)(m.weight)
                nn.initializer.Constant(value=0.)(m.bias)
    

def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    t = paddle.arange(0, batch_size)
    t = t.unsqueeze(-1).unsqueeze(-1)
    idx_base = t * num_vertices_reduced
    # idx_base = paddle.arange(0, batch_size).reshape(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = paddle.reshape(idx, [-1])

    x = paddle.transpose(x, [0, 2, 1, 3])
    # print(batch_size * num_vertices_reduced)
    # print(x.shape)
    # print(idx.numpy())
    feature = paddle.reshape(x, [batch_size * num_vertices_reduced, -1])[idx.numpy()]
    # print(feature.shape)
    feature = paddle.reshape(feature, [batch_size, num_vertices, k, num_dims])
    feature = paddle.transpose(feature, [0, 3, 1, 2])
    return feature
