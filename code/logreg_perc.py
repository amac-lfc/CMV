print("Importing Modules...")

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
# import csv
import numpy as np
from confusionMatrix import *
from random import shuffle
from lib import *
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import random as rd
from sklearn.preprocessing import normalize
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
from mpl_toolkits.mplot3d import Axes3D

def grab(value,data):
    grabbed_data = []
    for i in data:
        if i[1] == value:
            grabbed_data.append(i)
    return grabbed_data

def separateData(data):
    features = []
    labels = []
    for x,y in data:
        features.append(x)
        labels.append(y)
    return features, labels

print("Loading Common Words and Creating Features List")
# common_words = open(r"delta_words.txt",mode='r',encoding="utf-8").read().split(" ")
# common_words = common_words[:NumWords]
features_list = ['author', 'parend_id', 'id', 'nested_count', 'reply_count',
            'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count', 'questions_count',
            'bold_count', 'avgSentences_count', 'enumeration_count', 'excla_count', 'high arousal', 'low arousal',
            'medium arousal', 'medium dominance', 'low dominance', 'high dominance', 'high valence', 'low valence',
            'medium valence', 'examples', 'hedges', 'self references']

print('Reading File and Creating Data')
df = pd.read_csv('/home/shared/CMV/Delta_Data2.csv', delimiter = ",")
Deltas = df.values[:,3:]
y_Deltas = np.ones(len(Deltas[:,0]),'i')
Deltas = np.column_stack((Deltas,y_Deltas))
# print(Deltas[:10])
# df = pd.read_csv('NoDelta_Data.csv', delimiter = ",")
# ds = df.sample(frac=0.005)
ds = pd.read_csv('/home/shared/CMV/NoDelta_Data_Sample2.csv', delimiter = ",")
NoDeltas = ds.values[:,3:]
y_NoDeltas = np.zeros(len(NoDeltas[:,0]),'i')
NoDeltas = np.column_stack((NoDeltas,y_NoDeltas))

print("Shuffling Data")
np.random.shuffle(Deltas)
np.random.shuffle(NoDeltas)

trainDeltas, testDeltas = Deltas[:int(len(Deltas) * .8),:], Deltas[int(len(Deltas) * .8):,:]
trainNoDeltas, testNoDeltas = NoDeltas[:int(len(NoDeltas) * .8),:], NoDeltas[int(len(NoDeltas) * .8):,:]



train_data_sm = np.concatenate((trainDeltas,trainNoDeltas), axis=0)
test_data_sm = np.concatenate((testDeltas,testNoDeltas), axis=0)

print("Duplicating Deltas")

# while trainDeltas.shape[0] < trainNoDeltas.shape[0]:
#     i = rd.randint(0,trainDeltas.shape[0]-1)
#     trainDeltas = np.concatenate((trainDeltas, trainDeltas[i,:][np.newaxis,:]), axis=0)
#
# while trainDeltas.shape[0] > trainNoDeltas.shape[0]:
#     i = rd.randint(0,trainNoDeltas.shape[0]-1)
#     trainNoDeltas = np.concatenate((trainNoDeltas, trainNoDeltas[i,:][np.newaxis,:]), axis=0)

train_data = np.concatenate((trainDeltas,trainNoDeltas), axis=0)
test_data = np.concatenate((testDeltas,testNoDeltas), axis=0)

print("# of train: ", len(train_data))
print("# of test: ", len(test_data))

print("Splitting Test Data into Deltas and No Deltas")
#train_data = fixed_data[:int(len(fixed_data) *.8)]
#test_data = fixed_data[int(len(fixed_data) * .8):]


x_train = train_data[:,:-1]
y_train = train_data[:,-1]
y_train = y_train.astype('int')

x_train_sm = train_data_sm[:,:-1]
y_train_sm = train_data_sm[:,-1]
y_train_sm = y_train_sm.astype('int')

print("X Train, Y Train: ", x_train.shape, y_train.shape)
print("X Train SM, Y Train SM: ", x_train_sm.shape, y_train_sm.shape)


x_testDeltas = testDeltas[:,:-1]
y_testDeltas = testDeltas[:,-1]
y_testDeltas = y_testDeltas.astype('int')

x_testNoDeltas = testNoDeltas[:,:-1]
y_testNoDeltas = testNoDeltas[:,-1]
y_testNoDeltas = y_testNoDeltas.astype('int')

# x_test = test_data[:,:-1]
# y_test = test_data[:,-1]
# y_test = y_test.astype('int')

#print("x_test, y_test: ", x_test.shape, y_test.shape)

print("Deltas, NoDeltas = ", len(y_testDeltas), len(y_testNoDeltas))
print("Deltas, NoDeltas = ", Deltas.shape, NoDeltas.shape)


x_train_sm = normalize(x_train_sm)
x_train = normalize(x_train)
x_testDeltas = normalize(x_testDeltas)
x_testNoDeltas = normalize(x_testNoDeltas)

#This makes it so the test data only contains data with deltas
#test_data = grab('1', test_data)



# df = pd.read_csv('Delta_Data.csv', delimiter = ",")
# Deltas = df.values[:,3:]
# y_Deltas = np.ones(len(Deltas[:,0]),'i')
# Deltas = np.column_stack((Deltas,y_Deltas))
# df2 = pd.read_csv('NoDelta_Data.csv', delimiter = ",")
# NoDeltas = df2.values[:,3:]
# y_NoDeltas = np.zeros(len(NoDeltas[:,0]),'i')
# NoDeltas = np.column_stack((NoDeltas,y_NoDeltas))
# test_data = NoDeltas#np.concatenate((Deltas,NoDeltas), axis=0)
# np.random.shuffle(test_data)
# test_data = test_data[:40000]
# x_test = test_data[:,:-1]
# y_test = test_data[:,-1]
# y_test = y_test.astype('int')

# delta_count = 0
# nodelta_count = 0
# for i in y_train:
#     if i == 1:
#         delta_count += 1
#     else:
#         nodelta_count += 1
#     print(i,end="")
# print()
# print(delta_count,nodelta_count)


print("Training Logistic Regression")
#Creating the Classifiers

# models = [["solver is liblinear", LogisticRegression(max_iter=10000, solver="liblinear", penalty="l1")],
#         ["class weight is balanced", LogisticRegression(max_iter=10000, solver="lbfgs", class_weight="balanced")],
#         ["c is 1",LogisticRegression(max_iter=10000, solver="lbfgs", C=1)],
#         ["c is 10", LogisticRegression(max_iter=10000, solver="lbfgs", C=1e2)],
#         ["c is 100", LogisticRegression(max_iter=10000, solver="lbfgs", C=1e3)],
#         ["c is 1/10", LogisticRegression(max_iter=10000, solver="lbfgs", C=1e-1)],
#         ["c is 1/100", LogisticRegression(max_iter=10000, solver="lbfgs", C=1e-2)]]

models = [["solver is liblinear, class weight is balanced, c is 1/10", LogisticRegression(max_iter=10000, solver="liblinear", penalty="l1", class_weight="balanced", C=1e-1)],
["solver is lbfgs, class weight is balanced, c is 1/10", LogisticRegression(max_iter=10000, solver="lbfgs", class_weight="balanced", C=1e-1)]]
# clf_P = Perceptron(tol=1e-6)

#Training without SMOTE
x_train = np.asarray(x_train)
sm = SMOTE(k_neighbors = 2)
x_res, y_res = sm.fit_resample(x_train, y_train)

print(y_res)

print("Delta count:", np.count_nonzero(y_res == 1))
print("Nodelta count:", np.count_nonzero(y_res == 0))


# print(x_res)

for i in range(len(models)):
    model = models[i][1].fit(x_res,y_res)

    y_predictNoDeltas = model.predict(x_testNoDeltas)
    y_predictDeltas = model.predict(x_testDeltas)

    print(f"Accuracy score for {models[i][0]} Delta is: {accuracy_score(y_testDeltas, y_predictDeltas)}")
    print(f"Accuracy score for {models[i][0]} No Delta is: {accuracy_score(y_testNoDeltas, y_predictNoDeltas)}")


#Training with SMOTE
#clf_RF = clf_RF.fit(x_res,y_res)

# clf_tree = clf_tree.fit(x_train,y_train)
# clf_MNB = clf_MNB.fit(x_train,y_train)

# getImportances(clf_RF, x_train, features_list)

# print("Checking for Accuracy")
# y_predict = clf_tree.predict(x_test)
# print(f"Accuracy score for Decision Tree is: {accuracy_score(y_test, y_predict)}")

# y_predict = clf_MNB.predict(x_test)
# print(f"Accuracy score for Naive Bayes Classifier is: {accuracy_score(y_test, y_predict)}")

# y_predict = clf_RF.predict(x_test)
# print(f"Accuracy score for Random Forest Classifier is: {accuracy_score(y_test, y_predict)}")



# y_predictNoDeltas = clf_LR.predict(x_testNoDeltas)
# y_predictDeltas = clf_LR.predict(x_testDeltas)
#
#
#
# print(f"Accuracy score for LogisticRegression Delta is: {accuracy_score(y_testDeltas, y_predictDeltas)}")
# print(f"Accuracy score for LogisticRegression No Delta is: {accuracy_score(y_testNoDeltas, y_predictNoDeltas)}")

# y_test = np.concatenate([y_testDeltas,y_testNoDeltas])
# y_pred = np.concatenate([y_predictDeltas,y_predictNoDeltas])
#
# # plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix')
# plotConfusionMatrix(y_test, y_pred, title='Logistic Regression CM', normalize=True)
#
#
# y_predictNoDeltas = clf_P.predict(x_testNoDeltas)
# y_predictDeltas = clf_P.predict(x_testDeltas)
#
# print(f"Accuracy score for Perceptron Delta is: {accuracy_score(y_testDeltas, y_predictDeltas)}")
# print(f"Accuracy score for Perceptron No Delta is: {accuracy_score(y_testNoDeltas, y_predictNoDeltas)}")


# y_predictNoDeltas = clf_RF.predict(x_testNoDeltas)
# y_predictDeltas = clf_RF.predict(x_testDeltas)

# print(f"Accuracy score for Random Forest Classifier Delta is: {accuracy_score(y_testDeltas, y_predictDeltas)}")
# print(f"Accuracy score for Random Forest Classifier No Delta is: {accuracy_score(y_testNoDeltas, y_predictNoDeltas)}")


# y_test = np.concatenate([y_testDeltas,y_testNoDeltas])
# y_pred = np.concatenate([y_predictDeltas,y_predictNoDeltas])

# plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix')
# plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix', normalize=True)


# ndeltas = np.where(y_test == 1)[0]
# ndeltas2 = np.where(y_test == 0)[0]
# print(len(ndeltas2), len(ndeltas),len(y_test),len(ndeltas)/len(y_test)*100)
