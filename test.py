import os
import glob
import pandas as pd
import numpy as np


bdir = "output/syscd_d"
fs = glob.glob(os.path.join(bdir, "PSLKNet*"))
s = np.array([[]])
idx = []
for f in fs:
    csvf = glob.glob(os.path.join(f, '*.csv'))[0]
    df = pd.read_csv(csvf)
    name = csvf.split("PSLKNet_")[1]
    name = name.split("_")[0]
    idx.append(name)
    print(df.keys())
    c = df.loc[df['acc'].idxmax()]
    data = c[['recall', 'acc', 'miou', 'Kappa', 'Macro_f1']].values
    s = np.append(s, data)
s = np.reshape(s, [-1,5])
print(idx)
w = pd.DataFrame(s, columns=['Kappa', 'MIoU', "MPa", "Recall", "F1"], index=idx)
w.to_csv("clcd_kernelsize_test.csv")
print(w)



