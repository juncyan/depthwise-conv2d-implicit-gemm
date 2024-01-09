import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import logging
from logging.handlers import RotatingFileHandler


#show or save figure
#mode=["plt.save","cv2.write"] = [0,1]
def load_logger(save_log_dir,save=True,print=True,config=None):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if print:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if save:
        file_handler = RotatingFileHandler(save_log_dir, maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if config != None:
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

res = [re.compile('.*Epoch\[(\d+)\] .*Train-accuracy.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Validation-accuracy.*=([.\d]+)')]


def plot_acc(log_name, color="r"):

    train_name = log_name.replace(".log", " train")
    val_name = log_name.replace(".log", " val")

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:  # i=0, match train acc
                break
            i += 1  # i=1, match validation acc
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])
        if epoch not in data:
            data[epoch] = [0] * len(res) * 2
        data[epoch][i*2] += val  # data[epoch], val:number
        data[epoch][i*2+1] += 1

    train_acc = []
    val_acc = []
    for k, v in data.items():
        if v[1]:
            train_acc.append(1.0 - v[0]/(v[1]))
        if v[2]:
            val_acc.append(1.0 - v[2]/(v[3]))

    x_train = np.arange(len(train_acc))
    x_val = np.arange(len(val_acc))
    plt.plot(x_train, train_acc, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.plot(x_val, val_acc, '-', linestyle='-', color=color, linewidth=2, label=val_name)
    plt.legend(loc="best")
    plt.xticks(np.arange(0, 131, 10))
    plt.yticks(np.arange(0.1, 0.71, 0.05))
    plt.xlim([0, 130])
    plt.ylim([0.1, 0.7])


    
if __name__ == "__main__":
    print("data_reader.utils run")
    x= ['1','3','5','7','9','0','2','4','6','8']
    with open("../snapshot/levir/metrics.csv","w",newline="") as csv_file:
        filewrite = csv.writer(csv_file)
        filewrite.writerow(["epoch","loss","PA","PA_Class","mIoU","FWIoU","Kappa",'Macro_f1'])
