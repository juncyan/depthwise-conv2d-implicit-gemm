import pandas as pd
import numpy as np
import os

x = r"/home/jq/Code/paddle/output/s2looking/F3Net_2024_07_08_21/F3Net_metrics.csv"
df = pd.read_csv(x)
print(df.keys())
idx = df['F1_1'].idxmax()
data = df.loc[idx]
print(data)

idx = df['iou_1'].idxmax()
data = df.loc[idx]
print(data)