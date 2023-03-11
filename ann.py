import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

one_region = np.nan_to_num(one_region, copy=True, nan=1, posinf=None, neginf=None)

label = np.random.rand(one_region.shape[0],one_region.shape[1])

print(one_region.shape,label.shape)

X_train, X_test, y_train, y_test = train_test_split(one_region, label, test_size=0.2)

# Define the neural network model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, input_shape=(X_train.shape), activation='sigmoid'),
    keras.layers.Dense(1024, activation='sigmoid'),
    keras.layers.Dense(12)
])

# Compile the model
model.compile(optimizer='adam', loss='mse',metrics = ['accuracy'])

# Train the model on the data
# model.fit(x = X_train, y = y_train, epochs=600)

plt.imshow(one_region)
plt.show()
