import pandas as pd
import numpy as np


def get_deltas(input, output):
    print("Retreiving Deltas: Step (1/4)")
    data = pd.read_csv(input, dtype="object")

    print("Retreiving Deltas: Step (2/4)")
    delta_giving_data = data[data["body"].str.contains("Δ|!delta", na=False)]

    print("Retreiving Deltas: Step (3/4)")
    delta_winning_ids = delta_giving_data["parent_id"].to_numpy()

    print("Retreiving Deltas: Step (4/4)")
    np.savetxt(output, delta_winning_ids, fmt='%s')

    return delta_winning_ids


def create_labels(input, output_delta, output_nodelta, deltas_file):
    print("Separating Labels: Step (1/4)")
    data = pd.read_csv(input, dtype="object")
    delta_ids = np.loadtxt(deltas_file, dtype="object")
    columns = data.columns.to_numpy()


    print("Separating Labels: Step (2/4)")
    delta_data = []
    nodelta_data = []
    data_np = data.to_numpy()
    for row in data_np:
        # print(row[1])
        if "t1_" + str(row[1]) in delta_ids:
            delta_data.append(row)
        else:
            nodelta_data.append(row)


    print("Separating Labels: Step (3/4)")
    delta_data = np.array(delta_data)
    nodelta_data = np.array(nodelta_data)

    header = np.array(["author","id","parent_id","body"])

    delta_data = pd.DataFrame(data=delta_data, columns=header)
    nodelta_data = pd.DataFrame(data=nodelta_data, columns=header)


    print("Separating Labels: Step (4/4)")
    delta_data = delta_data.dropna()
    nodelta_data = nodelta_data.dropna()

    delta_data.to_csv(output_delta, index=False)
    nodelta_data.to_csv(output_nodelta, index=False)


def run():
    input = '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv'

    deltas_file = '/home/shared/CMV/SortedData/delta_winning_ids.txt'
    deltas_data_file = '/home/shared/CMV/SortedData/delta_comments_data.csv'
    nodeltas_data_file = '/home/shared/CMV/SortedData/nodelta_comments_data.csv'

    get_deltas(input, deltas_file)

    create_labels(input, deltas_data_file, nodeltas_data_file, deltas_file)
