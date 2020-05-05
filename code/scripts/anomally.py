import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features



nodelta_data = pd.read_csv("/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv")
delta_data = pd.read_csv("/home/shared/CMV/FeatureData/delta_sample_feature_data.csv")

nodelta_data['label'] = 0
delta_data['label'] = 1

data = pd.concat([nodelta_data, delta_data])

print(data.columns)
print(features.getFeaturesList("con"))

""" starting anomally detection """

Normal_len = len (nodelta_data)
Anomolous_len = len (delta_data)

start_mid = Anomolous_len // 2
start_midway = start_mid + 1

train_cv_v1  = delta_data [: start_mid]
train_test_v1 = delta_data [start_midway:Anomolous_len]

start_mid = (Normal_len * 60) // 100
start_midway = start_mid + 1

cv_mid = (Normal_len * 80) // 100
cv_midway = cv_mid + 1

train_fraud = nodelta_data [:start_mid]
train_cv    = nodelta_data [start_midway:cv_mid]
train_test  = nodelta_data [cv_midway:Normal_len]

train_cv = pd.concat([train_cv,train_cv_v1],axis=0)
train_test = pd.concat([train_test,train_test_v1],axis=0)


print(train_fraud.columns.values)
print(train_cv.columns.values)
print(train_test.columns.values)

train_cv_y = train_cv["label"]
train_test_y = train_test["label"]

train_cv.drop(labels = ["label"], axis = 1, inplace = True)
train_fraud.drop(labels = ["label"], axis = 1, inplace = True)
train_test.drop(labels = ["label"], axis = 1, inplace = True)

""" guassian stuff """


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma

def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

mu, sigma = estimateGaussian(train_fraud)
p = multivariateGaussian(train_fraud,mu,sigma)
p_cv = multivariateGaussian(train_cv,mu,sigma)
p_test = multivariateGaussian(train_test,mu,sigma)
