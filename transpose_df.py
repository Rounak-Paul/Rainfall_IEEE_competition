import pandas as pd

data = pd.read_csv('output.csv')
data = data*2
data.to_csv('output_transposed.csv',header=None)