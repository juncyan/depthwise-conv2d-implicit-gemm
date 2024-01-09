import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import ResNet34_vd
from utils import SELayer
from backbone.dcnv2 import DCN_V2Layer
import numpy as np


# BCE-Net: Reliable Building Footprints Change Extraction based on Historical Map and Up-to-Date Images using Contrastive Learning
class BCENet(nn.Layer):
    def __init__(self, num_classes=1, num_channels=3, pretrained=False):
        super(BCENet, self).__init__()
        filters = [64, 128, 256, 512]
        self.resnet_features = ResNet34_vd(in_channels=3)

        self.conv4 = nn.Conv2D(512, 512, 3, padding=1, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(512)
        # self.rl4 = nn.ReLU()

        self.conv3 = nn.Conv2D(256, 256, 3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(256)
        # self.rl3 = nn.ReLU()

        self.conv2 = nn.Conv2D(128, 128, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(128)
        # self.rl2 = nn.ReLU()

        self.conv1 = nn.Conv2D(64, 64, 3, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        # self.rl1 = nn.ReLU()

        # self.dblock = Dblock(512)
        # self.fusin = UpFusion()
        self.fusin = New_Fusion()
        # self.pam2 = PositionAttentionModule(256)
        # self.pam3 = PositionAttentionModule(512)

        self.deconv1 = nn.Conv2D(512,256, 3, padding=1)
        self.norm1 = nn.BatchNorm2D(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2D(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2D(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2D(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2D(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2D(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2D(128)
        self.sel = SELayer(128,2)
        # self.drl4 = nn.ReLU()
        self.finalseg = nn.Conv2D(128, num_classes, 3, padding=1)

        # self.finalnew = nn.Conv2D(32, num_classes, 3, padding=1)
        self.finalmov = nn.Conv2D(128, num_classes, 3, padding=1)

        # self.conv_lab11 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        # self.bnorm21 = nn.BatchNorm2D(32)
        # self.conv_lab21 = nn.Conv2D(32, num_classes, 3, padding=1)
        self.segblock = nn.Conv2D(128, num_classes, 3, padding=1)
        self.conv_lab1 = nn.Conv2DTranspose(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bnorm2 = nn.BatchNorm2D(64)
        self.conv_lab2 = nn.Conv2D(64, num_classes, 3, padding=1)

        self.dcn = DCN_V2Layer()

    def forward(self, inputs, labelso):
        # Encoder
        # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
        e11, e12, e13, e14 = self.resnet_features(inputs)
        x_size = inputs.shape

        e1 = self.conv1(e11)
        e1 = self.bn1(e1)
        # e1 = self.rl1(e1)
        e2 = self.conv2(e12)
        e2 = self.bn2(e2)
        # e2 = self.rl2(e2)
        e3 = self.conv3(e13)
        e3 = self.bn3(e3)
        # e3 = self.pam2(e3)
        # e3 = self.rl3(e3)
        e4 = self.conv4(e14)
        e4 = self.bn4(e4)
        # e4 = self.pam3(e4)
        # e4 = self.rl4(e4)
        # e4 = self.dblock(e4)
        fu_new, featn = self.fusin(e1, e2, e3, e4, labelso)

        d1 = self.deconv1(e4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, e3.shape[2:], mode='bilinear', align_corners=False)

        d2 = self.deconv2(paddle.concat([d1, e3], axis=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2, e2.shape[2:], mode='bilinear', align_corners=False)

        d3 = self.deconv3(paddle.concat([d2, e2], axis=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, e1.shape[2:], mode='bilinear', align_corners=False)

        # d4f = self.deconv4(paddle.cat([d3, e1], axis=1))
        print("d3.shape, e1.shape",d3.shape, e1.shape)
        d4f = self.dcn(d3, e1) #self.dcn(df4)
        # d4 = self.sel(d4f)
        d4 = self.norm4(d4f)
        d4 = F.interpolate(d4, x_size[2:], mode='bilinear', align_corners=False)
        out = self.finalseg(d4)

        # new_out = self.finalnew(d4 *(1-paddle.unsqueeze(labelso,axis=1)))
        mov_out = self.finalmov((1 - F.sigmoid(d4)) * paddle.unsqueeze(labelso, axis=1))
        # # mov_out = (1 - paddle.sigmoid(out)) * paddle.unsqueeze(labelso, axis=1)
        # mov_out = self.conv_lab1(fu_mov)
        # mov_out = self.bnorm2(mov_out)
        # mov_out = self.conv_lab2(mov_out)
        print("fu_new.shape: ",fu_new.shape)
        fu_new = self.dcn(fu_new)
        fu_new = self.sel(fu_new)
        new_out = self.conv_lab1(fu_new)
        new_out = self.bnorm2(new_out)
        new_out = self.conv_lab2(new_out)
        feat_all = F.interpolate(self.segblock(d4f), (inputs.shape[2], inputs.shape[3]), mode='bilinear',
                                 align_corners=False)
        feat_mov = F.interpolate(self.segblock(featn), (inputs.size()[2], inputs.size()[3]), mode='bilinear',
                                 align_corners=False)
        # feat_all = self.segblock(d4f)
        # feat_mov = self.segblock(featn)
        # out_new = self.finalseg2(paddle.cat((d4,F.interpolate(fusion, inputs.size()[2:], mode='bilinear',align_corners=False)),axis=1))
        return out, mov_out, new_out, feat_all, feat_mov


class New_Fusion(nn.Layer):
    """ different based on Position attention module"""
    def __init__(self, **kwargs):
        super(New_Fusion, self).__init__()

        self.deconv1 = nn.Conv2D(512, 256, 3, padding=1)
        self.norm1 = nn.BatchNorm2D(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2D(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2D(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2D(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2D(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2D(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2D(128)
        self.conv_lab = nn.Conv2D(1, 1,kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()
        # self.sel = SEModule(128, reduction=2)

    def forward(self, x1,x2,x3,x4, old):

        d1 = self.deconv1(x4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, x3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.deconv2(paddle.concat([d1, x3], axis=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2, x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.deconv3(paddle.concat([d2, x2], axis=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.deconv4(paddle.concat([d3, x1], axis=1))
        # out = self.sel(outf)
        outf = self.norm4(out)
        out = F.interpolate(outf, (2*x1.shape[2],2*x1.shape[3]), mode='bilinear', align_corners=False)

        lab = self.conv_lab(paddle.unsqueeze(old,axis=1))
        backfeat = out * (1 - lab)
        return backfeat, outf


if __name__ == "__main__":
    print("bcenet")
    device = 'gpu'
    x = paddle.rand([1,3,512,512]).cuda()
    gt_old = paddle.zeros([1,512,512]).cuda()
    m = BCENet(2).to(device)
    y = m(x, gt_old)
    for i in y:
        print(i.shape)