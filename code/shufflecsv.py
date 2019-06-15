import pandas as pd

# Shuffle CSV file
df = pd.read_csv('NoDelta_Data.csv', delimiter = ",")

ds = df.sample(frac=0.005)
ds.to_csv('NoDelta_Data_Sample.csv',index=False)