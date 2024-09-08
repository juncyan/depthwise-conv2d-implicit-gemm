import random
import os
import numpy as np
import paddle
import logging
import argparse

from cd_models.fccdn import FCCDN
from cd_models.stanet import STANet
from cd_models.p2v import P2V
from cd_models.msfgnet import MSFGNet
from cd_models.fc_siam_conc import FCSiamConc
from cd_models.snunet import SNUNet
from cd_models.f3net import F3Net
from paddleseg.models import UNet


from models.samcd import MSamCD_SSH, MSamCD_FSSH
from models.dhsamcd import DHSamCD, DHSamCD_v2, DHSamCD_v3, DHSamCD_v4

from core.work import Work


# dataset_name = "LEVIR_CD"
# dataset_name = "LEVIR_CDP"
# dataset_name = "GVLM_CD"
# dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
dataset_name = "WHU_BCD"
# dataset_name = "S2Looking"
# dataset_name = "CLCD"

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

# res = ResNet50_vd()
# model = UNet(num_classes, in_channels=6)
# model = UNetPlusPlus(num_classes, 6)
# model = UPerNet(num_classes, ResNet50_vd(in_channels=6),(0,1,2,3))
# model = DeepLabV3P(num_classes, backbone=ResNet50_vd(in_channels=6))
# model = SegNeXt(num_classes=num_classes, decoder_cfg={}, backbone=ResNet50_vd(in_channels=6))
# model = DSAMNet(3,2)
# model = MSFGNet(3, 2)
# model = P2V(3,2)
# model = FCCDN(3,2)
# model = FCSiamConc(3,2)


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fssh',
                        help='model name (default: msfgnet)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=str, default='gpu:0',
                        choices=['gpu:0', 'gpu:1', 'cpu'],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="CLCD",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--img_ab_concat', type=bool, default=True,
                        help='img_ab_concat False')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num_workers (default: 8)')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print("main")
    args = parse_args()
    m = {"ssh":MSamCD_SSH, "fssh":MSamCD_FSSH, "v2":DHSamCD, "hv2":DHSamCD_v2,"hv3":DHSamCD_v3, 'hv4':DHSamCD_v4}
    model = m[args.model](img_size=args.img_size)
    w = Work(model, args,'./output')
    w()


