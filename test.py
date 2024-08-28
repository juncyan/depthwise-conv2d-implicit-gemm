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

from datasets.cdloader import DataReader, TestReader
from work.train import train
from common import Args

from models.samcd import MobileSamCD

from core.work import Work


# 参数、优化器及损失

# dataset_name = "LEVIR_CD"
# dataset_name = "LEVIR_CDP"
# dataset_name = "GVLM_CD"
# dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
# dataset_name = "WHU_BCD"
# dataset_name = "S2Looking"
dataset_name = "CLCD"


model = MobileSamCD(img_size=512)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='msfgnet',
                        help='model name (default: msfgnet)')
    parser.add_argument('--device', type=str, default='gpu:0',
                        choices=['gpu:0', 'gpu:1', 'cpu'],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default='LEVIR_CD',
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
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num_workers (default: 8)')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print("main")
    args = parse_args()
    w = Work(model, args,'./output')
    w(model)

