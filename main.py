import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from csv import writer
from scipy.fftpack import fft,ifft,fftshift
import sys


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
    arr_for_avg = np.nan_to_num(one_region, copy=True, nan=0, posinf=None, neginf=None)
    one_region = np.nan_to_num(one_region, copy=True, nan=-999, posinf=None, neginf=None)
    print(one_region.shape)
    print(len(arr_for_avg[:,0]))
    for i in range(len(one_region[0])):
        avg = np.average(arr_for_avg[:,i])
        _max = np.max(arr_for_avg[:,i])
        _min = np.min(arr_for_avg[:,i])
        for j in range(len(one_region)):
            if one_region[j,i] == -999:
                one_region[j,i] = 0
            # if one_region[j,i] > _max - _min:
            #     one_region[j,i] = _max - _min
            # if one_region[j,i] < _min + _min:
            #     one_region[j,i] = _min + _min
    
    
    # temp = []
    # for i in range(len(one_region[0])):
    #     t_fft = fft(one_region[:,i])
    #     t1 = t_fft[:int(len(t_fft)/2)]
    #     t2 = t_fft[int(len(t_fft)/2):]
    #     t = np.append(t_fft,np.zeros(len(t_fft)*2))
    #     t = np.append(t,t2)
    #     t = ifft(fftshift(t))
    #     t = t*2
    #     temp.append(t)
    # one_region = np.array(temp).T
    
    # plt.plot(np.abs(one_region[:,7]))
    # plt.show()
    
    # sys.exit()
    
    one_region_lagged = one_region[:-1].copy()
    label = one_region[1:].copy()

    X_train = one_region_lagged
    y_train = label
    print(f'x_train_shape: {X_train.shape}')
    test = np.array([one_region[-1]])

    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(1024, input_shape=(X_train[0].shape), activation='tanh'),
        keras.layers.Dense(1024, activation='tanh'),
        keras.layers.Dense(1024, activation='tanh'),
        keras.layers.Dense(1024, activation='tanh'),
        keras.layers.Dense(1024, activation='tanh'),
        keras.layers.Dense(1024, activation='tanh'),
        keras.layers.Dense(12)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error',metrics = [keras.metrics.RootMeanSquaredError()])

    
    model.fit(x = X_train, y = y_train, epochs=40)

    outcome = np.abs(model.predict(one_region)[0])
    
    final_output = np.append([region],outcome)
    
    file = open(f'output.csv','a',newline='')
    obj = writer(file)
    obj.writerow(final_output)
    file.close()

data = pd.read_csv('output.csv')
data = data.transpose()
data.to_csv('output.csv',header=None)