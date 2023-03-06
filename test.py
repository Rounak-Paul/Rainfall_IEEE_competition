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
# update_log(f'list of regions: {regions}')

month_list = raw_data['MONTH'].to_numpy()
month_list = month_list.reshape(int(len(month_list)/12),12)
# update_log(f'month list: {month_list}')

file = open(f'output.csv','a',newline='')
obj = writer(file)
obj.writerow([
    'MONTH','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
])
file.close()

raw_data = raw_data.to_numpy()

print(len(raw_data),len(raw_data[0]))

temp = []
for i in range(int(len(raw_data)/12)):
    temp.append(raw_data[i:i+12])
data = np.array(temp)
print(data.shape)

data = np.nan_to_num(data, copy=True, nan=0, posinf=None, neginf=None)

one_region_lagged = data[:-1].copy()
label = data[1:].copy()

X_train = one_region_lagged.flatten()
print(X_train)
y_train = label.flatten()
print(f'x_train_shape: {X_train.shape}')
test = np.array([data[-1]])
shape = test.shape
test = test.flatten()

model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, input_shape=(X_train.shape), activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(test))
    ])

model.compile(optimizer='adam', loss='mse',metrics = [keras.metrics.RootMeanSquaredError()])

model.fit(x = X_train, y = y_train, epochs=2000)

outcome = model.predict(test)[0]
print(np.array(outcome).reshape(shape[0],shape[1]))