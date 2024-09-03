import numpy as np
import cv2
import tifffile as tf
import os
from PIL import Image

# img1 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\after\after.tif"
# label1 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\after\after_label.tif"
# img2 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\before\before.tif"
# label2 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\before\before_label.tif"
# clab = r"D:\Datasets\Building change detection dataset\1. The two-period image data\change label\change_label.tif"



src_path = r"/mnt/data/Datasets/S2Looking/test_org/{}/{}"
dst_path = r"/mnt/data/Datasets/S2Looking/test/{}/{}"
TARGET_SIZE = 256

def crop_images(file:str, src_path, dst_path):

    im1 = cv2.imread(src_path.format("A",file))
    im2 = cv2.imread(src_path.format("B", file))
    lab1 = cv2.imread(src_path.format("label1", file))
    lab2 = cv2.imread(src_path.format("label2", file))
    label = cv2.imread(src_path.format("label", file))

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    lab1 = cv2.cvtColor(lab1, cv2.COLOR_BGR2RGB)
    lab2 = cv2.cvtColor(lab2, cv2.COLOR_BGR2RGB)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    TARGET_SIZE = 512
    width, height, _ = im1.shape
    nw = int(width / TARGET_SIZE)
    nh = int(height / TARGET_SIZE)
    width = TARGET_SIZE * nw
    height = TARGET_SIZE * nh

    ws = np.linspace(0, width, nw + 1, dtype=np.int32)
    width_idx1 = ws[0: -1]
    width_idx2 = ws[1:]
    hs = np.linspace(0, height, nh + 1, dtype=np.int32)
    height_idx1 = hs[0: -1]
    height_idx2 = hs[1:]
    f = file.split(".")[0]
    print(f, label.shape)
    for x in range(len(width_idx1)):
        for y in range(len(height_idx1)):
            file_name = "{}_{}_{}.png".format(f, x, y)
            img1 = im1[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]
            img2 = im2[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]
            la1 = lab1[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]
            la2 = lab2[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]
            la = label[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]

            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
            la1 = Image.fromarray(la1)
            la2 = Image.fromarray(la2)
            la = Image.fromarray(la)

            img1.save(dst_path.format('A', file_name))
            img2.save(dst_path.format('B', file_name))
            la1.save(dst_path.format('label1', file_name))
            la2.save(dst_path.format('label2', file_name))
            la.save(dst_path.format('label', file_name))


if __name__ == "__main__":
    print("split")
    src_dir = r"/mnt/data/Datasets/S2Looking/test_org/A"
    fils = os.listdir(src_dir)
    print(fils)
    for f in fils:
        crop_images(f, src_path, dst_path)