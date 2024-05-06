import random
import os
import numpy as np
import paddle
import logging

from paddleseg.models import UNet, DeepLabV3P, UNetPlusPlus, UPerNet, SegNeXt, ResNet50_vd, Xception65_deeplab
from cd_models.fccdn import FCCDN
from cd_models.stanet import STANet
from cd_models.p2v import P2V
from paddleseg.models.losses import BCELoss
from paddleseg.transforms import Resize

from datasets.cdloader import DataReader, TestReader
from work.train import train
from common import Args


# 参数、优化器及损失
batch_size = 8
iters = 100 #epochs * 445 // batch_size
base_lr = 2e-4

# dataset_name = "LEVIR_CD"
# dataset_name = "GVLM_CD"
dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
dataset_path = '/home/jq/data/{}'.format(dataset_name)

num_classes = 2
# res = ResNet50_vd()
# model = UNet(num_classes, in_channels=6)
# model = UNetPlusPlus(num_classes, 6)
# model = UPerNet(num_classes, ResNet50_vd(in_channels=6),(0,1,2,3))
# model = DeepLabV3P(num_classes, backbone=ResNet50_vd(in_channels=6))
# model = SegNeXt(num_classes=num_classes, decoder_cfg={}, backbone=ResNet50_vd(in_channels=6))
# model = DSAMNet(3,2)
model = P2V(3, 2)
# model = FCSiamConc(3,2)
# model = STANet(3,2)


model_name = model.__str__().split("(")[0]
args = Args('output/{}'.format(dataset_name.lower()), model_name)
args.batch_size = batch_size
args.device = "gpu:1"
args.num_classes = num_classes
args.pred_idx = 0
args.data_name = dataset_name
args.img_ab_concat = True
args.en_load_edge = False

paddle.device.set_device(args.device)

def seed_init(seed=32767):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    

if __name__ == "__main__":
    print("main")
    seed_init(32767)
    logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING

    train_data = DataReader(dataset_path, 'train', args.en_load_edge, args.img_ab_concat)
    val_data = DataReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)
    test_data = TestReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)

    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(),) 

    # paddle.Model().fit()
    train(model,train_data, val_data, test_data, optimizer, args, iters, 2)
