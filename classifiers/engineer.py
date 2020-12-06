import pandas as pd
import numpy as np
import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def merge(dataframes):
    print("Merging Dataframes and adding labels")
    # Add label for the classifier 0 or 1
    for i in range(len(dataframes)):
        dataframes[i]['label'] = i

    data = pd.concat(dataframes).to_numpy()
    return data


def pca(X, n_features=2):
    print('Running PCA')
    pca = PCA(n_components=n_features)
    return pca.fit_transform(X)


def smote(X, y, k_neighbors = None, sampling_strategy =  None):
    print("Running SMOTE")
    sm = SMOTE(k_neighbors = k_neighbors, sampling_strategy = sampling_strategy)
    return sm.fit_resample(X, y)


def squareFeatures(X):
    print("Squaring Features")
    X_new = []
    for row in X:
        row_new = []
        for f1 in row:
            for f2 in row:
                row_new.append(f1 * f2)
        X_new.append(np.array(row_new))
    return np.array(X_new)


if __name__ == '__main__':

    model = models.RandomForest
    print("Training Model")

    nodelta_data = pd.read_csv("/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv")
    delta_data = pd.read_csv("/home/shared/CMV/FeatureData/delta_sample_feature_data.csv")

    data = merge([nodelta_data, delta_data])

    X, y = data[: , :-1], data[:, -1]
    X = squareFeatures(X)
    # X = pca(X, 10)


    print("Shape of all features:", X.shape)

    print(train_test_split(X, y, test_size=0.33))
