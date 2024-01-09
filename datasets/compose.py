import paddle
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from common.cd_dataload_512 import *

class Mydataset(paddle.io.Dataset):
    def __init__(self, path, augment=False, transform=None, target_transform=None,lab_smooth=0):
        self.aug = augment
        self.file_path = os.path.dirname(path)
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.lab_smooth=lab_smooth

    def __getitem__(self, item):
        if self.aug == False:
            imgn, labn = self.imgs[item]
            imgn = os.path.join(self.file_path, "images/" + imgn)
            label = os.path.join(self.file_path, "labels/" + labn)
            bgr_img = cv2.imread(imgn, -1)
            image = Image.fromarray(bgr_img)
            if self.transform is not None:
                image = self.transform(image)
            # gt = cv2.imread(label, 0)/255
            gt = cv2.imread(label, 0)
            gt_old = np.zeros((512,512), dtype=np.int)
            gt_new = np.zeros((512,512), dtype=np.int)
            gt_mov = np.zeros((512,512), dtype=np.int)
            gt_b = np.zeros((512, 512), dtype=np.int)


            gt_old[gt == 1] = 1
            gt_old[gt == 3] = 1
            gt_new[gt == 2] = 1
            gt_mov[gt == 3] = 1
            gt_b[gt == 1] = 1
            gt_b[gt == 2] = 1
            return image, gt_old, gt_new,gt_mov,gt_b, labn

        else:
            # 进行数据增强
            imgn, labn = self.imgs[item]
            imgn = os.path.join(self.file_path, "images/" + imgn)
            labn = os.path.join(self.file_path, "labels/" + labn)
            # if self.lab_smooth>0:
            #     gt = cv2.GaussianBlur(cv2.imread(labn, 0), ksize=(self.lab_smooth, self.lab_smooth),sigmaX=self.lab_smooth)
            # else:
            gt = cv2.imread(labn, 0)
                # gt = cv2.imread(labn, 0)/255
                # gt[gt > 2] = 0
                # gt[gt == 2] = 1
            image = cv2.imread(imgn, -1)


            # gt = cv2.imread(label, 0)/255
            # gt1 = cv2.imread(lab1, 0) / 255
            # gt2 = cv2.imread(lab2, 0) / 255
            # image1 = cv2.imread(fn1, -1)
            # image2 = cv2.imread(fn2, -1)

            # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
            # sort1 = batch[np.random.randint(0,6)]
            # sort2 = batch[np.random.randint(0, 6)]
            # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
            # image2 = cv2.merge([image2[:, :, sort2[0]], image2[:, :, sort2[1]], image2[:, :, sort2[2]]])


            # image, gt = motionblur(image, gt, blur=5, p=0.5)
            # image = random_color_jitter(image, saturation_range=0.5, brightness_range=0.5, contrast_range=0.5, u=0.5)
            # image = randomHueSaturationValue(image,hue_shift_limit=(-30, 30),sat_shift_limit=(-5, 5),val_shift_limit=(-15, 15))
            # image, gt = randomShiftScaleRotate(image, gt, shift_limit=(-0.1, 0.1), scale_limit=(-0.0, 0.0),
            #                                    aspect_limit=(-0.1, 0.1), rotate_limit=(-5, 5))
            # image, gt = randomHorizontalFlip(image, gt, u=0.5)
            # image, gt = randomVerticleFlip(image, gt, u=0.5)
            # image, gt = randomRotate90(image, gt, u=0.5)
            # image, gt = resize(image, gt, 1024, 640)
            #
            image = randomHueSaturationValue(image,
                                             hue_shift_limit=(-35, 35),
                                             sat_shift_limit=(-35, 35),
                                             val_shift_limit=(-35, 35))
            # image, gt = randomShiftScaleRotate(image, gt,
            #                                    shift_limit=(-0.15, 0.15),
            #                                    scale_limit=(-0.25, 0.5),
            #                                    aspect_limit=(-0.15, 0.15),
            #                                    rotate_limit=(-10, 10))

            # image1, image2, gt = randomHorizontalFlip(image1,image2, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)
            # image1, image2, gt = resize(image1, image2, gt, 512,512)
            # image, gt = resize(image, gt, 512, 512)

            # image = image[..., ::-1]  # bgr2rgb
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # grad = (255 * grade(gray)).astype(np.uint8)


            gt_old = np.zeros((512,512), dtype=np.uint8)
            gt_new = np.zeros((512,512), dtype=np.uint8)
            gt_mov = np.zeros((512,512), dtype=np.uint8)
            gt_b = np.zeros((512, 512), dtype=np.uint8)


            # gt_1 = np.zeros((512, 512), dtype=np.uint8)
            # gt_1[gt == 1] = 255
            # gt_agu = gt.copy()
            # builds, _ = cv2.findContours(gt_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for build in builds:
            #     if random.random() > 0.5:
            #         build_coord = np.array(build.transpose(1, 0, 2))
            #         gt_agu = cv2.fillPoly(gt_agu, build_coord, 2)
            image, gt = randomShiftScaleRotate(image, gt, u=0.75,
                                                   shift_limit=(-0.15, 0.15),
                                                   scale_limit=(-0.25, 0.5),
                                                   aspect_limit=(-0.15, 0.15),
                                                   rotate_limit=(-15, 15))
            # gt_old[gt_agu == 1] = 1
            # gt_old[gt_agu == 3] = 1
            # gt_new[gt_agu == 2] = 1
            # gt_mov[gt_agu == 3] = 1
            # gt_b[gt_agu == 1] = 1
            # gt_b[gt_agu == 2] = 1

            # cv2.imwrite(r'H:\Projects\BE_Net\Result\05-16_20-53-00\ga.png', 80 * gt_agu)
            # cv2.imwrite(r'H:\Projects\BE_Net\Result\05-16_20-53-00\gt.png', 80 * gt)
            # gt_old[gt == 1] = 1
            # gt_old[gt == 3] = 1
            # gt_new[gt == 2] = 1
            # gt_mov[gt == 3] = 1
            # gt_b[gt == 1] = 1
            # gt_b[gt == 2] = 1

            if random.random() > 0.8:
                gt_old[gt == 3] = 1
                gt_new[gt == 1] = 1
                gt_new[gt == 2] = 1
                gt_mov[gt == 3] = 1
                gt_b[gt == 1] = 1
                gt_b[gt == 2] = 1
            else:
                gt_old[gt == 1] = 1
                gt_old[gt == 3] = 1
                gt_new[gt == 2] = 1
                gt_mov[gt == 3] = 1
                gt_b[gt == 1] = 1
                gt_b[gt == 2] = 1

            if random.random() > 0.8:
                bboxs = []
                for i in range(5):
                    x0 = random.randint(0, 448)
                    y0 = random.randint(0, 384)
                    x_size = random.randint(32, 64)
                    y_size = random.randint(x_size // 2, 2 * x_size)

                    # x_size = random.randint(32, 96)
                    # y_size = random.randint(x_size // 2, 2 * x_size)
                    # x0 = random.randint(0, 511-x_size)
                    # y0 = random.randint(0, 511-y_size)
                    bboxs.append([[x0, y0], [x0, y0 + y_size], [x0 + x_size, y0 + y_size]])

                gen_remove = np.zeros((512, 512), dtype=np.uint8)
                move_area = np.array(bboxs)
                gen_remove = cv2.fillPoly(gen_remove, move_area, 255)
                angle = random.randint(0, 90) - 45
                M = cv2.getRotationMatrix2D((256, 256), angle, 1)
                gen_remove = cv2.warpAffine(gen_remove, M, (512, 512))

                ker = np.ones((5, 5), np.uint8)
                gen_remove = cv2.erode(gen_remove, ker, iterations=1) // 255
                jmov = (1 - gt_b) * gen_remove
                gt_mov[jmov == 1] = 1


            img = Image.fromarray(image)
            if self.transform is not None:
                img = self.transform(img.copy())
            return img, gt_old.copy(), gt_new.copy(),gt_mov.copy(),gt_b.copy(), labn

    def __len__(self):
        return len(self.imgs)

