print("Importing Modules...")

import pandas as pd
# import csv
import numpy as np
from random import shuffle
from lib import *
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
NumWords = 200
common_words = open(r"delta_words.txt",mode='r',encoding="utf-8").read().split(" ")
common_words = common_words[:NumWords]
features_list = ['certainty_count', 'extremity_count', 'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count',
                'questions_count', 'bold_count', 'avgSentences_count', 'enumeration'] + common_words

print('Reading File and Creating Data')
df = pd.read_csv('Delta_Data.csv', delimiter = ",")
Deltas = df.values[:,3:]
y_Deltas = np.ones(len(Deltas[:,0]),'i')
Deltas = np.column_stack((Deltas,y_Deltas))
# print(Deltas[:10])
# df = pd.read_csv('NoDelta_Data.csv', delimiter = ",")
# ds = df.sample(frac=0.005)
ds = pd.read_csv('NoDelta_Data_Sample.csv', delimiter = ",")
NoDeltas = ds.values[:,3:]
y_NoDeltas = np.zeros(len(NoDeltas[:,0]),'i')
NoDeltas = np.column_stack((NoDeltas,y_NoDeltas))
fixed_data = np.concatenate((Deltas,NoDeltas), axis=0)

print("Randomizing and Evening Out Data")
#shuffle(fixed_data)
np.random.shuffle(fixed_data)

print("Splitting Data into Train and Test")
train_data = fixed_data[:int(len(fixed_data) *.8)]
test_data = fixed_data[int(len(fixed_data) * .8):]
x_train = train_data[:,:-1]
# x_train = x_train.astype('int')
y_train = train_data[:,-1]
y_train=y_train.astype('int')
# print(x_train.shape, y_train.shape)

#This makes it so the test data only contains data with deltas
#test_data = grab('1', test_data)

x_test = test_data[:,:-1]
# x_test = x_test.astype('int')
y_test = test_data[:,-1]
y_test = y_test.astype('int')

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


print("Training Decision Tree, Naive Bayes Classifier, and Random Forest Classifier")
#Creating the Classifiers
clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=80, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=12,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
#
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = clf_RF, param_grid = param_grid,
#                           cv = 3, n_jobs = -1, verbose = 2)
#
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# clf_RF = grid_search.best_estimator_

# clf_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')
# clf_MNB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

#Training the Classifiers
x_train = np.asarray(x_train)
clf_RF = clf_RF.fit(x_train,y_train)
# clf_tree = clf_tree.fit(x_train,y_train)
# clf_MNB = clf_MNB.fit(x_train,y_train)



# getImportances(clf_RF, x_train, features_list)

print("Checking for Accuracy")
# y_predict = clf_tree.predict(x_test)
# print(f"Accuracy score for Decision Tree is: {accuracy_score(y_test, y_predict)}")
#
# y_predict = clf_MNB.predict(x_test)
# print(f"Accuracy score for Naive Bayes Classifier is: {accuracy_score(y_test, y_predict)}")

y_predict = clf_RF.predict(x_test)
print(f"Accuracy score for Random Forest Classifier is: {accuracy_score(y_test, y_predict)}")
