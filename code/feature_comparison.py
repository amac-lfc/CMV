import pandas as pd

delta_df = pd.read_csv('/home/shared/CMV/Delta_Data.csv', delimiter = ",")
nondelta_df = pd.read_csv('/home/shared/CMV/NoDelta_Data_Sample.csv', delimiter = ",")

delta_df = delta_df.drop(['author','parend_id', 'id'], axis=1)
nondelta_df = nondelta_df.drop(['author','parend_id', 'id'], axis=1)

delta_mean = delta_df.mean()
nondelta_mean = nondelta_df.mean()

delta_mean.to_csv("delta_mean.csv")
nondelta_mean.to_csv("nondelta_mean.csv")
