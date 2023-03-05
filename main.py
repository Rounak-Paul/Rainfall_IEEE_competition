import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

try:
    os.mkdir('output')
except: pass
try:
    os.remove('output/log.txt')
except: pass

def update_log(msg):
    print(msg)
    msg = f'{msg}' + '\n'
    with open('output/log.txt','a') as f:
        f.write(msg)
        f.close()

raw_data = pd.read_csv("data/train.csv")
header_row = raw_data.columns.to_numpy()
regions = header_row[2:]
update_log(f'list of regions: {regions}')

month_list = raw_data['MONTH'].to_numpy()
month_list = month_list.reshape(int(len(month_list)/12),12)
update_log(f'month list: {month_list}')

one_region = raw_data[regions[0]].to_numpy()
one_region = one_region.reshape(int(len(one_region)/12),12)

plt.imshow(one_region)
plt.show()










