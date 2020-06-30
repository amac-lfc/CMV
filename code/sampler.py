import pandas as pd
import numpy as np

def sample(input, output, max_rows):
    print("Sampling: Step (1/2)")
    data = pd.read_csv(input)
    sample = data.sample(max_rows)

    print("Sampling: Step (2/2)")
    sample.to_csv(output, index=False)


def run():
    delta_file = "/home/shared/CMV/SortedData/delta_comments_data.csv"
    nodelta_file = "/home/shared/CMV/SortedData/nodelta_comments_data.csv"

    delta_sample_file = "/home/shared/CMV/SampledData/delta_sample_data.csv"
    nodelta_sample_file = "/home/shared/CMV/SampledData/nodelta_sample_data.csv"

    print("Sampling Delta File")
    sample(delta_file, delta_sample_file, 2500)

    print("Sampling NoDelta File")
    sample(nodelta_file, nodelta_sample_file, 2500)
