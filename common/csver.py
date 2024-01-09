import cv2
import csv
import pandas as pd
import numpy as np
import math
import os
import re
import glob

#create and get label information
def writer_csv(csv_dir,operator="w",headers=None,lists=None):
    with open(csv_dir,operator,newline="") as csv_file:
        f_csv=csv.writer(csv_file)
        if headers!=None:
            f_csv.writerow(headers)
        if lists!=None:
            f_csv.writerows(lists)

def save_numpy_as_csv(scv_dir,d_numpy,fmt="%.4f"):
    assert len(d_numpy.shape) <= 2
    if len(d_numpy.shape)==1:
        d_numpy = np.expand_dims(d_numpy, 0)
    with open(scv_dir,"a") as f:
        np.savetxt(f, d_numpy, fmt=fmt,delimiter=',')

def reader_csv(csv_dir):
    ann = pd.read_csv(csv_dir)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label

def read_excel(path):
    dara_xls = pd.ExcelFile(path)
    data = {}
    for sheet in dara_xls.sheet_names:
        df = dara_xls.parse(sheet_name=sheet,header=None)
        #print(type(df.values))
        data[sheet] = df.values
    return data

def read_csv(csv_dir):
    data = pd.read_csv(csv_dir).values
    return data

def reverse_one_hot(image):
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    x = np.argmax(image, dim=-1)
    return x

def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	image = np.array(image.cpu())
	x = colour_codes[image.astype(int)]
	return x

def scale_image(input,factor):
    #效果不理想，边缘会有损失，不建议使用 2020/5/17 hjq
    #input.shape=[m,n],output.shape=[m//factor,n//factor]
    #将原tensor压缩factor

    h=input.shape[0]//factor
    w=input.shape[1]//factor

    return cv2.resize(input,(w,h),interpolation=cv2.INTER_NEAREST)

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10




def find_max_iter(logcontents, rows):
    max_iter = 1
    total_iter = 1
    find_max_iter = False
    find_total_iter = False

    for i in range(rows):
        idx = int(-i)
        if (find_max_iter == False) and ("[EVAL]" in logcontents[idx]) and ("best validation" in logcontents[idx]):
            ts = logcontents[idx].split("iter")[-1]
            max_iter = int(re.findall(r"\d+", ts)[0])
            find_max_iter = True

        if (find_total_iter == False) and ("[TRAIN]" in logcontents[idx]) and ("iter:" in logcontents[idx]):
            ts = logcontents[idx].split("iter:")[-1]
            res = re.findall(r"\d+\/\d+", ts)[0]
            total_iter = int(res.split("/")[-1])
            find_total_iter = True

        # print(find_max_iter, find_total_iter)
        # print(max_iter, total_iter)
        if find_max_iter and find_total_iter:
            return max_iter, total_iter

def extract_data(file_path):
    with open(file_path, 'r') as logfile:
        logcontents = logfile.readlines()
    rows = len(logcontents)

    max_iter, total_iter = find_max_iter(logcontents, rows)
    find_max_local = 1
    for i in range(rows):
        idx = i if total_iter / max_iter > 2 else int(-i)
        partstr = f"{max_iter}/{total_iter}"
        if partstr in logcontents[idx]:
            find_max_local = idx if idx > 0 else rows + idx
            break

    dret = {}
    for i in range(find_max_local + 2, min(rows, find_max_local + 6)):
        ts = logcontents[i]
        if "[EVAL]" in ts:
            res = ts.split("[EVAL]")[-1]
            # res = res.replace(" ", "")
            res = res.replace("\n", "")
            items = res.split(",")
            d = dict(item.split(":") for item in items)
            dret.update(d)
    return dret

def extract_data_as_array(file_path):
    ret = extract_data(file_path)
    iou = re.findall("0\.\d+", ret[" Class IoU"])

    precision = re.findall("0\.\d+", ret[" Class Precision"])
    recall = re.findall("0\.\d+", ret[" Class Recall"])

    acc = ret[" Acc"]
    miou = ret["mIoU"]
    kappa = ret["kappa"]
    f1 = ret["Macro_f1"]

    array = np.array([float(iou[0]), float(iou[1]), float(miou), float(precision[0]), float(precision[1]),
                      float(acc), float(recall[0]), float(recall[1]), float(kappa), float(f1)])
    return array

def generate_matrics_csv(base_path):
    data_names = os.listdir(base_path)
    
    for data_name in data_names:
        data = dict()
        keys = set()

        dmpath = os.path.join(base_path, data_name)
        if not os.path.isdir(dmpath):
            continue
        data_models = os.listdir(dmpath)
        
        for idx, data_model in enumerate(data_models):
            metrics_path = glob.glob(os.path.join(f"{dmpath}/{data_model}", '*.csv'))[0]
            
            metrics = pd.read_csv(metrics_path) 
            mkeys =  metrics.keys()
            for k in mkeys:
                keys.add(k)
            try:
                max_idx = metrics["miou"].idxmax()
            except:
                continue
            
            best_m = metrics.iloc[max_idx]
            model_name = data_model.split("202")[0]
            d = {f"{model_name}{idx}":dict(best_m)}
            data.update(d)
        
        keys = list(keys)
        keys.sort()
        indexs = []
        dc = {}
        for k in keys:
            dc.update({k:[]})

        for k in data.keys():
            d = data[k]
            indexs.append(k)
            for sk in keys:
                if sk in d.keys():
                    v = d[sk]
                else:
                    v = 0.
                dc[sk].append(v)
        
        dc = pd.DataFrame(dc, index=indexs)
        dc.to_csv(f"{base_path}/{data_name}_metrics.csv")

def save_log_as_csv(data_dir, save_path):
    keys = ["iou1", "iou2", "miou", "acc1", "acc2", "macc", "recall1", "recall2", "kappa", "F1"]
    res = {}
    folders = os.listdir(data_dir)
    idx = 0
    for f in folders:
        fpath = os.path.join(data_dir, f)
        mn = f.split("_")[0]
        array = None
        if os.path.isdir(fpath):
            txt_files = glob.glob(os.path.join(fpath, '*.log'))[0]
            # print(txt_files)
            array = extract_data_as_array(txt_files)
        res.update({f"{mn}{idx}": array})
        idx += 1

    d = pd.DataFrame(res)
    indexs = d.keys()
    print(keys)
    data = d.values.transpose([1, 0])
    s = pd.DataFrame(data, indexs, keys)
    s.to_csv(save_path)
