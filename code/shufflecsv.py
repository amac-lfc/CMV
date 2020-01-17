import pandas as pd

# Shuffle CSV file
df = pd.read_csv('/home/shared/CMV/NoDelta_Data.csv', delimiter = ",")
df2 = pd.read_csv('/home/shared/CMV/Delta_Data.csv', delimiter = ",")


print(df.values.shape)
print(df2.values.shape)


# ds = df.sample(frac=0.004)
# print(ds.values.shape)
# ds.to_csv('/home/shared/CMV/NoDelta_Data_Sample.csv',index=False)
