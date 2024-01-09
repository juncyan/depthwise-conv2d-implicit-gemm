import cv2
from PIL import Image
import tifffile
import spectral as spy
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

def loadimg(filename):
    # 获取mat格式的数据，loadmat输出的是dict，所以需要进行定位
    # input_image = loadmat('D:/Hyper/Salinas_corrected.mat')['salinas_corrected']
    # gt = loadmat("D:/Hyper/Salinas_gt.mat")['salinas_gt']
    input_image = tifffile.tifffile.imread(filename)
    view1 = spy.imshow(data=input_image, bands=[69, 27, 11], title="img")  # 图像显示

    # view2 = spy.imshow(classes=gt, title="gt")  # 地物类别显示

    # view3 = spy.imshow(data=input_image, bands=[69, 27, 11], classes=gt)
    # view1.set_display_mode("overlay")
    # view1.class_alpha = 0.3  # 设置类别透明度为0.3
    #
    # # spy.view_cube(input_image, bands=[69, 27, 11])  # 显示后会打印相应功能及操作
    #
    # pc = spy.principal_components(input_image)  # N维特征显示 view_nd与view_cube需要ipython 命令行输入：ipython --pylab
    # xdata = pc.transform(input_image)  # 把数据转换到主成分空间
    # spy.view_nd(xdata[:, :, :15], classes=gt)

    plt.pause(10)


def reader(file_name):
    data = spy.open(file_name)

    # # 读取Matlab的.mat文件
    # # data = loadmat(file_name)['key']  # 返回的是dict,所以需要key

    # # 读取.raw的文件（BIL）
    # raw_image = np.fromfile(file_name)
    # format_image = np.zeros((lines, samples, bands))  # 参数分别对应文件行数、列数、波段数，对应的hdr文件可查询
    # for row in range(0, lines):
    #     for dim in range(0, bands):
    #         format_image[row, :, dim] = raw_image[(dim + row * bands) * samples:(dim + 1 + row * bands) * samples]

    # 最后读取结果应该都是<class 'numpy.ndarray'>了，方便进一步处理操作😄


if __name__ == "__main__":
    print("test")
    img_path = r"E:\MyCode\Datasets\GWater\GSWED_Africa2020_4326v2\8_48_-6_4_grid0_5_5_2000-2020.tif"
    img = tifffile.tifffile.imread(img_path)
    x = spy.imshow(img)
    x.class_alpha = 0.3
    print(x)
    plt.pause(1)



