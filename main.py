import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from csv import writer


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

file = open(f'output.csv','a',newline='')
obj = writer(file)
obj.writerow([
    'MONTH','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
])
file.close()

for region in regions:
    one_region = raw_data[region].to_numpy()
    one_region = one_region.reshape(int(len(one_region)/12),12)

    one_region = np.nan_to_num(one_region, copy=True, nan=1, posinf=None, neginf=None)
    one_region_lagged = one_region[:-1].copy()
    label = one_region[1:].copy()

    X_train = one_region_lagged
    y_train = label

    test = np.array([one_region[-1]])

    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, input_shape=(X_train.shape), activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(12)
    ])

    model.compile(optimizer='adam', loss='mse',metrics = ['accuracy'])

    model.fit(x = X_train, y = y_train, epochs=2000)

    outcome = model.predict(test)[0]
    
    final_output = np.append([region],outcome)
    
    file = open(f'output.csv','a',newline='')
    obj = writer(file)
    obj.writerow(final_output)
    file.close()

data = pd.read_csv('output.csv')
data = data.transpose()
data.to_csv('output.csv',header=None)