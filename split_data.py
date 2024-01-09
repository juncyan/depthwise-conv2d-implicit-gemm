import cv2
import numpy as np
import pathlib
import os


# base_path = pathlib.Path('D:\Datasets\GVLM_CD')
# dst_path = 'D://Datasets//GVLM_CD_d//{}//{}'
imn1 = 'im1.png'
imn2 = 'im2.png'
labeln = 'ref.png'

# sub_folders = os.listdir(base_path)

def creat_data_list(dataset_path, mode='train'):
    with open(os.path.join(dataset_path, (mode + '_list.txt')), 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'time1')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        A_imgs_name.sort()
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            B_img = os.path.join(A_path.replace('time1', 'time2'), A_img_name)
            label_img = os.path.join(A_path.replace('time1', 'label'), A_img_name)
            f.write(A_img + ' ' + B_img + ' ' + label_img + '\n')  # 写入list.txt
    print(mode + '_data_list generated')


def crop_images():
    for sub_folder in sub_folders:
        folder_path = base_path/ sub_folder
        if os.path.isdir(folder_path):
            f1 = folder_path / imn1
            f2 = folder_path / imn2
            lab = folder_path / labeln

            im1 = cv2.imread(str(f1))
            im2 = cv2.imread(str(f2))
            label = cv2.imread(str(lab))


            width, height, _ = im1.shape
            nw = int(width/256)
            nh = int(height/256)
            width = 256 * nw
            height = 256 * nh

            ws = np.linspace(0, width, nw+1, dtype=np.int32)
            width_idx1 = ws[0: -1]
            width_idx2 = ws[1: ]
            hs = np.linspace(0, height, nh+1, dtype=np.int32)
            height_idx1 = hs[0: -1]
            height_idx2 = hs[1:]
            print(sub_folder, label.shape)
            for x in range(len(width_idx1)):
                for y in range(len(height_idx1)):
                    file_name = "{}_{}_{}.png".format(sub_folder, x, y)
                    img1 = im1[width_idx1[x] : width_idx2[x], height_idx1[y]: height_idx2[y], :]
                    img2 = im2[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]
                    lab1 = label[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]

                    cv2.imwrite(dst_path.format('A', file_name), img1)
                    cv2.imwrite(dst_path.format('B', file_name), img2)
                    cv2.imwrite(dst_path.format('label', file_name), lab1)
        # print(img.shape)


if __name__ == "__main__":
    print("datasets")
    dataset_path = '/home/data/Datasets/CLCD'  # data的文件夹
    # 分别创建三个list.txt
    creat_data_list(dataset_path, mode='train')
    creat_data_list(dataset_path, mode='test')
    creat_data_list(dataset_path, mode='val')

