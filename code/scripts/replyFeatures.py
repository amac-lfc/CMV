import numpy as np
import pandas as pd


# add the parent directory to path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features

# get input files
delta_input = '/home/shared/CMV/SortedData/delta_data_w_replies.csv'
word_list_input = "/home/shared/CMV/FeatureData/word_list.csv"

# make output files
output_delta = "/home/shared/CMV/FeatureData/reply_delta_feature_data.csv"

# generate features
delta_features = features.generateFeatures([delta_input], [output_delta], word_list_input, 'con')[0]

print("Writing Features to File with Pandas")
delta_features = pd.DataFrame(data=delta_features, columns=None)

delta_features.to_csv(output_delta, index=False)
