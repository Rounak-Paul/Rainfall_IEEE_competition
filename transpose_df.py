import pandas as pd

data = pd.read_csv('output.csv')
data = data.transpose()
data.to_csv('output_transposed.csv',header=None)