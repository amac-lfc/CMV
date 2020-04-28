import pandas as pd
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import slimmer
import labeler
import sampler
import features
import models
import engineer
import accuracy


# load your data
nodelta_data = pd.read_csv("/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv")
delta_data = pd.read_csv("/home/shared/CMV/FeatureData/delta_sample_feature_data.csv")

# combine data with labels at the end
data = engineer.merge([nodelta_data, delta_data])

# separate features and labels
X, y = data[: , :-1], data[:, -1]
X = engineer.squareFeatures(X)

X_train, X_test, y_train, y_test = engineer.train_test_split(X, y, test_size=0.33)
model = models.FeedForwardNeuralNetwork(len(X[0]), outputs=1, layers=[100], epochs=200)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy.score(y_pred, y_test)
print(score)
