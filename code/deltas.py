import pandas as pd
import numpy as np

import slimmer
import labeler
import sampler
import features
import models
import engineer
import accuracy

def createData():

    inputs = ["/home/shared/CMV/RawData/Comments_MetaData.csv", "/home/shared/CMV/RawData/Comments_TextData.csv",
        "/home/shared/CMV/RawData/Submissions_MetaData.csv", "/home/shared/CMV/RawData/Submissions_TextData.csv"]

    outputs = ['/home/shared/CMV/SlimmedData/Slimmed_Comments_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_TextData.csv']

    columns_lst = [["name", "parent_id", "author", "link_id"], ["author", "id", "parent_id", "body"],
        ["url", "id", "author"], ["author", "id", "title", "selftext"]]

    slimmer.slim_all(inputs, outputs, columns_lst)


    input = '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv'

    deltas_file = '/home/shared/CMV/SortedData/delta_winning_ids.txt'
    deltas_data_file = '/home/shared/CMV/SortedData/delta_comments_data.csv'
    nodeltas_data_file = '/home/shared/CMV/SortedData/nodelta_comments_data.csv'

    labeler.get_deltas(input, deltas_file)

    labeler.create_labels(input, deltas_data_file, nodeltas_data_file, deltas_file)


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




def main():

    model = models.RandomForest()

    print("Prepping Data")

    nodelta_file = "/home/shared/CMV/FeatureData/all_nodelta_feature_data.csv"
    nodelta_sample_file = "/home/shared/CMV/SampledData/sampled_nodelta_feature_data.csv"

    print("Sampling NoDelta File")
    sampler.sample(nodelta_file, nodelta_sample_file, 20000)


    nodelta_data = pd.read_csv("/home/shared/CMV/SampledData/sampled_nodelta_feature_data.csv"  )
    delta_data = pd.read_csv("/home/shared/CMV/FeatureData/all_delta_feature_data.csv")

    data = engineer.merge([nodelta_data, delta_data])

    X, y = data[: , :-1], data[:, -1]

    X,y = engineer.smote(X, y)

    print("Shape of all features:", X.shape)
    X_train, X_test, y_train, y_test = engineer.train_test_split(X, y, test_size=0.33)

    print("Fitting Model")
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy.score(y_pred, y_test)
    print("Score:",score)

    cm = accuracy.confusion_matrix(y_test, y_pred, normalize='true')
    print("Confusion Matrix:", cm)

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
