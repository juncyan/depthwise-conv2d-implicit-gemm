import random
import os
import numpy as np
import paddle

from datasets.segloader import DataReader, TestReader
# from rlklab.xception import Xception65_deeplab
from paddleseg.models import UNet, DeepLabV3P, UNetPlusPlus, UPerNet, SegNeXt, ResNet50_vd,Xception65_deeplab
from paddleseg.models import PSPNet, MobileSeg, OCRNet, hrnet, ViT_base_patch16_224


from segwork.train import train
from common import Args

# 参数、优化器及损失
batch_size = 4
iters = 200
base_lr = 2e-4

# dataset_name = "MacaoLC"
dataset_name = "Landcover"
dataset_path = '/mnt/data/Datasets//{}'.format(dataset_name)

num_classes = 5

# res = ResNet50_vd()
# model = UNet(num_classes, in_channels=3)
# model = UNetPlusPlus(num_classes, 3)
# model = UPerNet(num_classes, ResNet50_vd(in_channels=3),(0,1,2,3))
# model = DeepLabV3P(num_classes, backbone=Xception65_deeplab(in_channels=3), backbone_indices=(0,1))
# model = SegNeXt(num_classes=num_classes, decoder_cfg={}, backbone=ResNet50_vd(in_channels=3))
# model = MobileSeg(num_classes,ResNet50_vd(in_channels=3))
# model = PSPNet(num_classes, ResNet50_vd(in_channels=3))
model = OCRNet(num_classes, ResNet50_vd(in_channels=3),(1,3))

model_name = model.__str__().split("(")[0]
args = Args('output/{}'.format(dataset_name.lower()), model_name)
args.device = "gpu:1"
args.batch_size = batch_size
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
    # logging.disable(logging.INFO)

    train_data = DataReader(dataset_path, 'train', args.en_load_edge, args.img_ab_concat)
    val_data = DataReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)
    test_data = TestReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)

    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(),) 
   
    train(model,train_data, val_data, test_data, optimizer, args, iters, 2)

   