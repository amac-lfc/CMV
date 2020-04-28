import pandas as pd
import numpy as np

import slimmer
import labeler
import sampler
import features
import models
import engineer
import accuracy

def main():
    # slimmer.run()
    # labeler.run()
    # sampler.run()
    # features.run()

    model = models.RandomForest

    X_train, X_test, y_train, y_test = engineer.run(model = model)

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy.score(y_pred, y_test)
    print(score)

# main()

def example():
    # load your data
    nodelta_data = pd.read_csv("/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv")
    delta_data = pd.read_csv("/home/shared/CMV/FeatureData/delta_sample_feature_data.csv")

    # combine data with labels at the end
    data = engineer.merge([nodelta_data, delta_data])

    # separate features and labels
    X, y = data[: , :-1], data[:, -1]

    # creates new features where its squared
    X = engineer.squareFeatures(X)

    # split into train and test
    X_train, X_test, y_train, y_test = engineer.train_test_split(X, y, test_size=0.33)

    # pick a model
    model = models.GradientBoosting

    # train the model
    model = model.fit(X_train, y_train)

    # predict with the model
    y_pred = model.predict(X_test)

    # score the model
    score = accuracy.score(y_pred, y_test)
    print(score)

example()
