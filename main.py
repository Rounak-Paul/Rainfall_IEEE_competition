import numpy as np
import pandas as pd
import os



try:
    os.mkdir('output')
except: pass
try:
    os.remove('output/log.txt')
except: pass

log_msg = ''
def update_log(msg):
    print(msg)
    global log_msg
    log_msg += msg + '\n'
    with open('output/log.txt','a') as f:
        f.write(log_msg)
        f.close()


raw_data = pd.read_csv("data/train.csv")
header_row = raw_data.columns
regions = header_row[2:].to_numpy()
update_log(f'list of regions: {regions}')



