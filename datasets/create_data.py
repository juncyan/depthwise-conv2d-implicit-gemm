# 构建数据集
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import paddle
from paddle.io import Dataset
from paddleseg.transforms import Compose, Resize
from .utils import one_hot_it


batch_size = 2  # 批大小


class ChangeDataset(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, dataset_path, mode, transforms=[], num_classes=2, ignore_index=255):
       
        # list_path = os.path.join(dataset_path, (mode + '_list.txt'))
        self.data_list = self.__get_list(os.path.join(dataset_path, (mode + '_list.txt')))
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])

        self.transforms = Compose(transforms, to_rgb=False, img_channels=3)  # 一定要设置to_rgb为False，否则这里有6个通道会报错
        self.is_aug = False #if len(transforms) == 0 else True
        self.num_classes = num_classes  # 分类数
        self.ignore_index = ignore_index  # 忽视的像素值
        
        assert self.data_list != None, "no data list could load"

        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(dataset_path,"A",_file))
            self.sst2_images.append(os.path.join(dataset_path,"B",_file))
            self.gt_images.append(os.path.join(dataset_path,"label",_file))

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = np.array(Image.open(A_path))
        B_img = np.array(Image.open(B_path))
        image = np.concatenate((A_img, B_img), axis=-1)  # 将两个时段的数据concat在通道层
        w, h, _ = image.shape
        #data = {"img":image, "label":label}
        # if self.is_aug:
        #     #image, label = self.transforms(img=image, label=label)
        #     #image, label = self.transforms(data)
        #     image = paddle.to_tensor(image).astype('float32')
        # else:
        #     image = paddle.to_tensor(image.transpose(2, 0, 1)).astype('float32')
        image = paddle.to_tensor(image.transpose(2, 0, 1)).astype('float32')

        label = np.array(Image.open(lab_path)) #cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        # label = label.clip(max=1)  # 这里把0-255变为0-1，否则啥也学不到，计算出来的Kappa系数还为负数
        # label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)
        #gt = np.argmax(gt,axis=2)
        else:
            label = np.array((label != 0),dtype=np.int8)
            label = self.label_color[label]
        # label = np.argmax(label,axis=-1)
        # label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        label = np.transpose(label, [2,0,1])
        label = paddle.to_tensor(label).astype('int64')
        data = {"img":image, "label":label, "trans_info":['resize', [w, h]]}
        return data
    
    def __len__(self):
        return self.data_num
    # 这个用于把list.txt读取并转为list
    def __get_list(self, list_path):
        data_list = None
        # with open(list_path, 'r') as f:
        #     data = f.readlines()
        #     for d in data:
        #         data_list.append(d.replace('\n', '').split(' '))
        with open(list_path, 'r') as f:
            data_list = f.read().split('\n')[:-1]
        return data_list

if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    # 完成三个数据的创建
    transforms = [Resize([1024, 1024])]
    train_data = ChangeDataset(dataset_path, 'train', transforms)
    test_data = ChangeDataset(dataset_path, 'test', transforms)
    val_data = ChangeDataset(dataset_path, 'val', transforms)