import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg.models.layers as layers


class BFIB(nn.Layer):
    #bi-temporal feature integrating block
    def __init__(self, in_channels, out_channels, kernels = 7, stride=2):
        super().__init__()
        self.fe = LKFE(in_channels, kernels)
        
        self.ce = LKCE(in_channels, out_channels, kernels, stride)
        
    def forward(self, x):
        y = self.fe(x)
        
        y = self.ce(y)
        return y

class PSBFA(nn.Layer):
    #pseudo siamese bi-temporal feature assimilating module
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce1 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b1 = layers.ConvBNReLU(in_channels, out_channels, 3)

        self.reduce2 = layers.ConvBNReLU(in_channels, in_channels, 3, stride=2)
        self.b2 = layers.ConvBNReLU(in_channels, out_channels, 3)
    
    def forward(self, x1, x2):
        y1 = self.reduce1(x1)
        y1 = self.b1(y1)

        y2 = self.reduce2(x2)
        y2 = self.b2(y2)
        y = y1 + y2
        return y1, y2, y


class PAM(nn.Layer):
    """
    Position attention module.
    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.key_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.value_conv = nn.Conv2D(in_channels, in_channels, 1, 1)

        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = paddle.shape(x)

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = paddle.reshape(query, (0, self.mid_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = paddle.reshape(key, (0, self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = paddle.bmm(query, key)
        sim = F.softmax(sim, axis=-1)

        value = self.value_conv(x)
        value = paddle.reshape(value, (0, self.in_channels, -1))
        sim = paddle.transpose(sim, (0, 2, 1))

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = paddle.bmm(value, sim)
        feat = paddle.reshape(feat,
                              (0, self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out

class LKFE(nn.Layer):
    #large kernel feature extraction
    def __init__(self, in_channels, kernels = [7]):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2 * in_channels, 3,)
        # self.dwc = nn.Sequential(layers.DepthwiseConvBN(2 * in_channels, 2 * in_channels, kernels),
        #                          nn.GELU())
        # self.se = SEModule(2 * in_channels, 8)
        self.dwc = MLKC(2*in_channels, kernels)
        self.conv2 = layers.ConvBNReLU(2 * in_channels, in_channels, 3,)
        self.ba = nn.Sequential(nn.BatchNorm2D(in_channels), nn.ReLU())

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwc(m)
        # m = self.se(m)
        m = self.conv2(m)
        y = x + m
        return self.ba(y)
    
class LKCE(nn.Layer):
    #large kernel channel expansion
    def __init__(self, in_channels, out_channels, kernels = [7], stride=1):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(in_channels, 2*in_channels, 3, stride=stride)

        # self.dwc = nn.Sequential(layers.DepthwiseConvBN(2*in_channels, 2*in_channels, kernels, stride=stride),
        #                          nn.GELU())
        self.dwc = MLKC(2*in_channels, kernels)
        self.conv3 = layers.ConvBNReLU(2*in_channels, out_channels, 3)
       

    def forward(self, x):
        y = self.conv1(x)
        m = self.dwc(y)
        m = self.conv3(m)
        return m

class BII(nn.Layer):
    #bi-temporal images integration
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.reduce = nn.Sequential(nn.GELU(), layers.ConvBNReLU(2*in_channels, out_channels, 3, stride=2))
        self.cbr1 = layers.ConvBNReLU(out_channels, out_channels*2, 1)
        self.dws = nn.Sequential(layers.DepthwiseConvBN(out_channels*2, out_channels*2, kernels), nn.GELU())
        self.se = SEModule(out_channels*2)
        self.cbr2 = layers.ConvBNReLU(out_channels*2, out_channels, 3)
        
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        ym = paddle.concat([y1, y2], 1)

        ym = self.reduce(ym)
        # ym = self.pam(ym)
        y = self.cbr1(ym)
        y = self.dws(y)
        y = self.se(y)
        y = self.cbr2(y)
        return y

class SEModule(nn.Layer):
    def __init__(self, channels, reductions = 8):
        super(SEModule, self).__init__()
        reduction_channels = channels // reductions
        self.avg = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, reduction_channels, 1, bias_attr=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(reduction_channels, channels, 1, bias_attr=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c , _, _ = x.shape
        avg = self.avg(x)
        y = self.fc1(avg)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        y = y.expand_as(x)
        return x * y

class MLKC(nn.Layer):
    #multi-large kernel conv
    def __init__(self, in_channels, kernels=[7]):
        super().__init__()

        mid_c = (len(kernels) + 1)*in_channels
        # print(mid_c, in_channels)
        self.c1 = nn.Conv2D(in_channels, in_channels, 1)

        self.lkcs = nn.LayerList()
        for kernel in kernels:
            self.lkcs.add_sublayer(f'{kernel}',nn.Conv2D(in_channels, in_channels, kernel, padding=kernel//2, groups=in_channels))

        self.bn = nn.BatchNorm2D(in_channels)
        self.gelu = nn.GELU()

        self.cbr = layers.ConvBNReLU(in_channels, in_channels, 3)
    
    def forward(self, x):
        y = self.c1(x)
        for op in self.lkcs:
            y += op(x)
        
        # my = paddle.concat(y, 1)
        my = self.gelu(self.bn(y))
        res = self.cbr(my)
        return res

class STAF(nn.Layer):
    #Spatial and Temporal Adaptive Fusion Module
    def __init__(self, in_channels=3, out_channels=64, kernels=7):
        super().__init__()

        self.conv1 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.conv2 = layers.DepthwiseConvBN(in_channels, in_channels, kernels)
        self.cbr1 = nn.Sequential(nn.GELU(), layers.ConvBNReLU(2*in_channels, out_channels, 3, stride=2))
        # self.cbr1 = layers.ConvBNReLU(2*in_channels, out_channels, 1)
        self.dws = nn.Sequential(layers.DepthwiseConvBN(out_channels, out_channels, kernels), nn.GELU())
        self.cbr2 = layers.ConvBNReLU(out_channels, out_channels, 3)
        
        self.tdcbrs2 = layers.ConvBNReLU(in_channels, out_channels, 3, stride=2)
        self.tdc11 = layers.ConvBNReLU(out_channels, out_channels, 1)
        self.tddsc = nn.Sequential(layers.DepthwiseConvBN(out_channels, out_channels, 7), nn.GELU())
        self.tdcbr2 = layers.ConvBNReLU(out_channels, out_channels, 3)

        # self.alpha = self.create_parameter(
        #     shape=[1],
        #     dtype='float32',
        #     default_initializer=nn.initializer.Constant(0.5))

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        ym = paddle.concat([y1, y2], 1)
        ym = self.cbr1(ym)
        y = self.dws(ym)
        y = self.cbr2(y)

        Td = paddle.abs(x1 - x2)
        td = self.tdcbrs2(Td)
        td1 = self.tdc11(td)
        td2 = self.tddsc(td)
        tc = td1 + td2
        td = self.tdcbr2(tc)
        res = y +  td
        return res