import random
import os
import re
import numpy as np
import glob
import pandas as pd


# - [METRICS] Class IoU: [0.9911 0.8143]
# - [METRICS] Class Precision: [0.9958 0.8916]
# - [METRICS] Class Recall: [0.9953 0.9038]
# - [METRICS] Class Dice: [0.9955 0.8976]



flagIou = "Class IoU"
flagpre = "Class Precision"
flagrec = "Class Recall"
flagdice = "Class Dice"

def extract_floats(s):
    return re.findall(r'0?\.\d+', s)
    return re.findall(r'\.\d+|\d+', s)

def pattern_metric(str, metr):
    if metr in str:
        d2 = str.split(metr)[-1]
        d = d2.split(" ")[-1]
        floats = extract_floats(d)
        dt = [float(f) for f in floats if '.' in f or f.isdigit()]
        try:
            return dt[-1]
        except:
            return 0
    return 0

dirb = "output/clcd"
fdirs = os.listdir(dirb)

sdata = pd.DataFrame({})
for fd in fdirs:
    siou = 0
    sacc = 0
    srecall = 0
    sf1 = 0
    path = os.path.join(dirb, fd)
    name = fd[:10]
    fp = glob.glob(os.path.join(path, "*.log"))[0]
    f = open(fp, 'r')
    cts = f.readlines()
    for c in cts:
        iou = pattern_metric(c, flagIou)
        acc = pattern_metric(c, flagpre)
        recall = pattern_metric(c, flagrec)
        f1 = pattern_metric(c, flagdice)

        siou = iou if iou > siou else siou
        sacc = acc if acc > sacc else sacc
        srecall = recall if recall > srecall else srecall
        sf1 = f1 if f1 > sf1 else sf1
    
    skappa = 2 * sacc * srecall / (sacc + srecall + 1e-5)
    data = pd.DataFrame({name:np.array([skappa, siou, sacc, srecall, sf1])})
    sdata = pd.concat([sdata, data], axis=1)

s = sdata.transpose()
# print(s)
s.to_csv("clcd.csv",header=["Kappa", "IoU", "PA", "Recall", "F1"])
