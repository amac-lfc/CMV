import pandas as pd

# Shuffle CSV file
df = pd.read_csv('/home/shared/CMV/NoDelta_Data2.csv', delimiter = ",")

print(df.values.shape)

ds = df.sample(frac=0.005)
print(ds.values.shape)
ds.to_csv('/home/shared/CMV/NoDelta_Data_Sample2.csv',index=False)
