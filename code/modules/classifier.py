print("Importing Modules...")

import csv
from lib import *
from random import shuffle
import numpy as np
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

def evenOutData(data, max_lines):
    with_delta = 0
    no_delta = 0
    for i in data:
        if i[1] == '0':
            no_delta += 1
        else:
            with_delta += 1

    num_of_each = min(with_delta, no_delta, max_lines)
    print(f'Using {num_of_each} Lines from Delta and {num_of_each} Lines from No Delta')

    with_delta = 0
    no_delta = 0
    new_data = []
    for i in data:
        if (i[1] == '0') and (no_delta < num_of_each):
            no_delta += 1
            new_data.append(i)
        elif (i[1] == '1') and (with_delta < num_of_each):
            with_delta += 1
            new_data.append(i)
    return new_data

def separateData(data):
    features = []
    labels = []
    for x,y in data:
        features.append(x)
        labels.append(y)
    return features, labels

def readInputFile(input_file, lines_to_read):
    data = []
    with open(input_file, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0

        for row in csv_reader:
            common_word_counts = []
            for word in common_words:
                common_word_counts.append(row[word])

            feature = [row['certainty_count'], row['extremity_count'], row['lexical_diversity_rounded'],
                       row['char_count_rounded'], row['link_count'], row['quote_count']] + common_word_counts

            data.append([feature,row['Delta_Awarded']])
            line_count += 1
            if (line_count >= lines_to_read) and (lines_to_read>0):
                  break;
    print(f'Processed {line_count} lines.')
    return data

print("Loading Common Words and Creating Features List")
common_words = open(r"../../delta_words.txt",mode='r',encoding="utf-8").read().split(" ")
features_list = ['certainty_count', 'extremity_count', 'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count'] + common_words

print('Reading File and Creating Data')
data = readInputFile("../../CSVs/true_data.csv",1000000)

print("Randomizing and Evening Out Data")
fixed_data = evenOutData(data, 5000)
shuffle(fixed_data)

print("Splitting Data into Train and Test")
train_data = fixed_data[:int(len(fixed_data) *.8)]
test_data = fixed_data[int(len(fixed_data) * .8):]
x_train, y_train = separateData(train_data)

#This makes it so the test data only contains data with deltas
#test_data = grab('1', test_data)

x_test, y_test = separateData(test_data)
x_test = [list(map(float,i)) for i in x_test]
x_train = [list(map(float,i)) for i in x_train]

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



getImportances(clf_RF, x_train, features_list)

print("Checking for Accuracy")
# y_predict = clf_tree.predict(x_test)
# print(f"Accuracy score for Decision Tree is: {accuracy_score(y_test, y_predict)}")

# y_predict = clf_MNB.predict(x_test)
# print(f"Accuracy score for Naive Bayes Classifier is: {accuracy_score(y_test, y_predict)}")

y_predict = clf_RF.predict(x_test)
print(f"Accuracy score for Random Forest Classifier is: {accuracy_score(y_test, y_predict)}")
