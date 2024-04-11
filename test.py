import random
import os
import re
import numpy as np
import glob
import pandas as pd


bdir = r"/mnt/data/TrainLog_LEVIR_GVLM_CLCD/levir_cd"

fls = os.listdir(bdir)
max_values = None
idx = []
for fl in fls:
    if "PSLKNet_" in fl or "LKPSNet_" in fl:
        name = fl[:12]
        idx.append(name)
        pdp = os.path.join(bdir, fl)
        fn = glob.glob(os.path.join(pdp, "*.csv"))[0]
        df = pd.read_csv(fn)
        max_value_row = df.loc[df['miou'].idxmax()]  
        if max_values is None:
            max_values = max_value_row
        else:
            max_values = pd.concat([max_values, max_value_row], axis=1)


# max_values = pd.DataFrame(max_values)
# print(max_values)
max_values.to_csv("levirc.csv", header=idx)
