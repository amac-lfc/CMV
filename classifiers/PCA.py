import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import slimmer
import labeler
import sampler
import features
import models
import engineer
import lib


if __name__ == '__main__':

    print("Prepping Data")

    # Reading the delta:
    delta_file = "/mnt/h/FeatureData/all_delta_feature_data.csv"
    delta_data = pd.read_csv(delta_file)

    # Reading the no delta
    nodelta_file = "/mnt/h/FeatureData/all_nodelta_feature_data.csv"
    nodelta_data = pd.read_csv(nodelta_file)
    nodelta_data = nodelta_data.sample(n=20000)

    x = np.vstack((nodelta_data.values, delta_data.values))

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    print(principalDf)

    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1) 
    fig, ax = plt.subplots()
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['no delta', 'delta']
    ax.scatter(principalDf.loc[:20000, 'principal component 1'], principalDf.loc[:20000, 'principal component 2'], c ='r')
    ax.scatter(principalDf.loc[20000:, 'principal component 1'], principalDf.loc[20000:, 'principal component 2'], c ='b')
    ax.legend(targets)
    ax.grid()
    plt.show()


    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=nodelta_data.columns)
    print(loadings)
    x = loadings['PC1']
    y = loadings['PC2']
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(len(x)):
        ax.annotate(loadings.index[i], (x[i], y[i]))
    plt.show()
