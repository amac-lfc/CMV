import pandas as pd
import numpy as np


print("Loading Data")
delta_data = pd.read_csv("/home/shared/CMV/SortedData/delta_comments_data.csv", dtype="object")
# print(delta_data.head(5))


# all comments start with "t1_"
comments_data = pd.read_csv("/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv", dtype="object")
# print(comments_data.head(5))
# print(comments_data.columns)

delta_data_np = delta_data.to_numpy()
# print(delta_data_np[:10])

print("Adding Reply Bodies to Body")
for row in range(len(delta_data_np)):
    row_data = delta_data_np[row]
    parent_id = delta_data_np[row, 2].split("_")[1]
    # print(row_data, parent_id)
    parent_loc = comments_data.loc[comments_data['id'] == parent_id]
    while len(parent_loc) >= 1:
        parent_row = parent_loc.to_numpy()[0]

        # print(parent_row[3], "\n\n")
        row_data[3] += " " + parent_row[3]
        row_data[2] = parent_row[2]

        parent_id = row_data[2]
        parent_loc = comments_data.loc[comments_data['id'] == parent_id]
    # print(row_data, "\n\n\n\n\n")
    delta_data_np[row] = row_data


print("Writing Data to File")
delta_data = pd.DataFrame(data=delta_data_np, columns=delta_data.columns)
delta_data.to_csv('/home/shared/CMV/SortedData/delta_data_w_replies.csv', index=False)
