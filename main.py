import random
import os
import numpy as np
import paddle


from datasets.dataloader import DataReader, TestReader
from work.train import train
from common import Args
from pslknet.model import PSLKNet
from pslknet.abliation import PSLKNet_noBFIB

# 参数、优化器及损失
batch_size = 8
iters = 100 #epochs * 445 // batch_size
base_lr = 2e-4

# dataset_name = "LEVIR_c"
# dataset_name = "GVLM_CD_d"
# dataset_name = "CLCD"
dataset_name = "SYSCD_d"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

num_classes = 2
model = PSLKNet_noBFIB()

model_name = model.__str__().split("(")[0]
args = Args('output/{}'.format(dataset_name.lower()), model_name)
args.batch_size = batch_size
args.num_classes = num_classes
args.pred_idx = 0
args.data_name = dataset_name
args.img_ab_concat = True
args.en_load_edge = False

def seed_init(seed=32767):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    

if __name__ == "__main__":
    print("main")
    seed_init(32767)

    train_data = DataReader(dataset_path, 'train', args.en_load_edge, args.img_ab_concat)
    val_data = DataReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)
    test_data = TestReader(dataset_path, 'test', args.en_load_edge, args.img_ab_concat)

    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(),) 

    # paddle.Model().fit()
    train(model,train_data, val_data, test_data, optimizer, args, iters, 10, 2)
