import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from .encoder import LKBlock, ConvFFN, FDModule, FAModule, FAModule_v2


class ACDNet_v3(nn.Layer):
    """
    Difference Assimilation Change Detection
    with PPModule
    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
    """
    def __init__(self,num_classes,align_corners=False,use_deconv=False,in_channels=3):
        super().__init__()
        self.encode = Encoder_v2(in_channels)

        mid_channels = [64, 128, 256, 512]
        self.fd = FAModule_v2(3, mid_channels)

        self.decode = Decoder_v2(align_corners, use_deconv=use_deconv)
        # self.ppm = layers.PPModule(512,512,[1,2,3,6],True, False)

        self.cls = self.conv = nn.Conv2D(in_channels=64,out_channels=num_classes,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        _, ch, w, h = x.shape
        assert ch == 2*self.fd.channels, "input data channels is illegal"
        fdblacks = self.fd(x)
       
        x, short_cuts = self.encode(x)
        # x = self.ppm(x)
        # fuse_back_feature = []
        # for fuse, cut, ffeature in zip(self.encoder_feature_cat, short_cuts, fdblacks):
        #     ft = paddle.concat([cut, ffeature], axis=1)
        #     fuse_back_feature.append(fuse(ft))
        x = self.decode(x, short_cuts, fdblacks)
        logit = self.cls(x)
        
        return logit


class DCDNet_v1(nn.Layer):
    """
    Difference Assimilation Change Detection

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
    """

    def __init__(self,num_classes,align_corners=False,use_deconv=False,in_channels=3):
        super().__init__()
        self.encode = Encoder(in_channels)
        mid_channels = [64, 128, 256, 512]
        self.fd = FDModule(3, mid_channels)
        self.encoder_feature_cat = [layers.ConvBNReLU(ch*2, ch, 3, 1) for ch in mid_channels]
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(in_channels=64,out_channels=num_classes,kernel_size=3,stride=1,padding=1)


    def forward(self, x):
        _, ch, w, h = x.shape
        assert ch == 2*self.fd.channels, "input data channels is illegal"
        fdblacks = self.fd(x)

        logit_list = []
        x, short_cuts = self.encode(x)
        fuse_back_feature = []
        for fuse, cut, ffeature in zip(self.encoder_feature_cat, short_cuts, fdblacks):
            ft = paddle.concat([cut, ffeature], axis=1)
            fuse_back_feature.append(fuse(ft))
        x = self.decode(x, fuse_back_feature)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

class Encoder_v2(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512]]
        dpr = [x.item() for x in paddle.linspace(0.1, 0.3, 4)]
        lksizes = [19,17,17,13]

        # self.down_sample_list = nn.LayerList()
        # for i, channel in enumerate(down_channels):
        #     self.down_sample_list.append(self.down_sampling(channel[0], channel[1]))
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1], lksizes[i], 0.3)
            for i, channel in enumerate(down_channels)
        ])
        self.down_sample_list.append(layers.ASPPModule((1,2,3,6),512,512,False))

    def down_sampling(self, in_channels, out_channels, lksize, drop=0.3):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(LKBlock(out_channels, out_channels, lksize, drop))
        # modules.append(layers.ConvBNReLU(out_channels, out_channels, 3)) #origin layer
        modules.append(ConvFFN(out_channels, 4*out_channels, out_channels, drop))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts

class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        dpr = [x.item() for x in paddle.linspace(0.1, 0.3, 4)]
        lksizes = [15,13,11,9]

        # self.down_sample_list = nn.LayerList()
        # for i, channel in enumerate(down_channels):
        #     self.down_sample_list.append(self.down_sampling(channel[0], channel[1]))
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1], lksizes[i], dpr[i])
            for i, channel in enumerate(down_channels)
        ])

    def down_sampling(self, in_channels, out_channels, lksize, drop=0.3):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(LKBlock(out_channels, in_channels, lksize, drop))
        # modules.append(layers.ConvBNReLU(out_channels, out_channels, 3)) #origin layer
        # modules.append(ConvFFN(out_channels, 4*out_channels, out_channels, drop))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x

class Decoder_v2(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[256*3, 256], [128*3, 128], [64*3, 64], [32*3, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling_v2(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts, mid_features):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)], mid_features[-(i+1)])
        return x

class UpSampling_v2(nn.Layer):
    def __init__(self,in_channels,out_channels,align_corners,use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(2*(in_channels//3),out_channels // 2,kernel_size=2,stride=2,padding=1)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            layers.attention.CAM(in_channels),
            # layers.attention.PAM(in_channels),
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))
        
    def forward(self, x, short_cut, features):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(x,paddle.shape(short_cut)[2:],mode='bilinear',align_corners=self.align_corners)
        x = paddle.concat([x, short_cut, features], axis=1)
        
        x = self.double_conv(x)
        return x


class UpSampling(nn.Layer):
    def __init__(self,in_channels,out_channels,align_corners,use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(in_channels,out_channels // 2,kernel_size=2,stride=2,padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(x,paddle.shape(short_cut)[2:],mode='bilinear',align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


if __name__ == "__main__":
    print("test DACDNet")

    x = paddle.rand([1,6,512,512]).cuda()
    m = ACDNet_v3(2, in_channels=6).to("gpu")
    y = m(x)
    for i in y:
        print(i.shape)
