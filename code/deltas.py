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
    slimmer.run()
    labeler.run()
    sampler.run()
    features.run()

    model = models.RandomForest()

    X_train, X_test, y_train, y_test = engineer.run(model = model)

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy.score(y_pred, y_test)
    print(score)

main()

def example():

    # generate samples
    delta_file = "/home/shared/CMV/SortedData/delta_comments_data.csv"
    nodelta_file = "/home/shared/CMV/SortedData/nodelta_comments_data.csv"

    delta_sample_file = "/home/shared/CMV/SampledData/delta_sample_data.csv"
    nodelta_sample_file = "/home/shared/CMV/SampledData/nodelta_sample_data.csv"

    print("Sampling Delta File")
    sampler.sample(delta_file, delta_sample_file, 13000)

    print("Sampling NoDelta File")
    sampler.sample(nodelta_file, nodelta_sample_file, 13000)

    #generate features
    features.run()

    # load your data
    nodelta_data = pd.read_csv("/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv")
    delta_data = pd.read_csv("/home/shared/CMV/FeatureData/delta_sample_feature_data.csv")

    # combine data with labels at the end
    data = engineer.merge([nodelta_data, delta_data])

    # separate features and labels
    X, y = data[: , :-1], data[:, -1]

    X_train, X_test, y_train, y_test = engineer.train_test_split(X, y, test_size=0.33)
    model = models.LogisticRegressionClassifier(c=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy.score(y_pred, y_test)
    print(score)


    # creates new features where its squared
    X_2 = engineer.squareFeatures(X)
    X_2_train, X_2_test, y_2_train, y_2_test = engineer.train_test_split(X_2, y, test_size=0.33)


    # pick a model
    model_2 = models.LogisticRegressionClassifier(c=1)

    # train the model
    model_2 = model_2.fit(X_2_train, y_2_train)

    # predict with the model
    y_2_pred = model_2.predict(X_2_test)

    # score the model
    score = accuracy.score(y_2_pred, y_2_test)
    print(score)

# example()
