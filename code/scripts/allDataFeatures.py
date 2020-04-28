import numpy as np
import pandas as pd


# add the parent directory to path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features

# get input files
delta_input = "/home/shared/CMV/SortedData/delta_comments_data.csv"
nodelta_input = "/home/shared/CMV/SortedData/nodelta_comments_data.csv"
word_list_input = "/home/shared/CMV/FeatureData/word_list.csv"

# make output files
output_delta = "/home/shared/CMV/FeatureData/all_delta_feature_data.csv"
output_nodelta = "/home/shared/CMV/FeatureData/all_nodelta_feature_data.csv"

# generate features
delta_features, nodelta_features = features.generateFeatures([delta_input, nodelta_input], [output_delta, output_nodelta], word_list_input, 'con')

print("Writing Features to File with Pandas")
delta_features = pd.DataFrame(data=delta_features, columns=None)
nodelta_features = pd.DataFrame(data=nodelta_features, columns=None)

delta_features.to_csv(output_delta, index=False)
nodelta_features.to_csv(output_nodelta, index=False)
