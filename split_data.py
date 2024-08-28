import cv2
import numpy as np
import pathlib
import os
import glob

import os
import glob
import cv2
import numpy as np

bd_dir = r"/home/jq/dataspace/datasets/ChangeNet"
img_path = r"/home/jq/dataspace/datasets/ChangeNet/096_000102/image"
lab_path = r"/home/jq/dataspace/datasets/ChangeNet/096_000102/label"

dirs = os.listdir(bd_dir)
idx = 1
for ds in dirs:
    folder = os.path.join(bd_dir, ds)
    if os.path.isfile(folder):
        continue
    img_fod = os.path.join(folder, "image")
    lab_fod = os.path.join(folder, "label")
    img_path = glob.glob(os.path.join(img_fod, "*.png"))
    if len(img_path) == 1:
        continue
    img_path1, img_path2 = img_path
    lab_path = glob.glob(os.path.join(lab_fod, "*.png"))[0]
    print(img_path1,img_path2, lab_path)
    img1 = cv2.imread(img_path1)
    print(img1.shape)
    img2 = cv2.imread(img_path2)
    print(img2.shape)
    label = cv2.imread(lab_path)
    print(label.shape)
    idx += 1

    if idx > 10:
        break
    

