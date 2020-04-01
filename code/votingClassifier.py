print("Importing Modules...")

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
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
from imblearn.over_sampling import SMOTE


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

while trainDeltas.shape[0] < trainNoDeltas.shape[0]:
    i = rd.randint(0,trainDeltas.shape[0]-1)
    trainDeltas = np.concatenate((trainDeltas, trainDeltas[i,:][np.newaxis,:]), axis=0)

while trainDeltas.shape[0] > trainNoDeltas.shape[0]:
    i = rd.randint(0,trainNoDeltas.shape[0]-1)
    trainNoDeltas = np.concatenate((trainNoDeltas, trainNoDeltas[i,:][np.newaxis,:]), axis=0)

print(len(trainDeltas), len(trainNoDeltas))

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




print("Training Decision Tree, Naive Bayes Classifier, and Random Forest Classifier")
#Creating the Classifiers


clf_LG = LogisticRegression(max_iter=10000, solver="liblinear", penalty="l1",
            class_weight="balanced", C=1e-1)

clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=80, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=12,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)


x_train = np.asarray(x_train)


clf_LG = clf_LG.fit(x_train, y_train)
clf_RF = clf_RF.fit(x_train,y_train)

ensemble = VotingClassifier(estimators=[('Logistic Regression', clf_LG), ('Random Forest', clf_RF)],
                       voting='hard', weights=[1,2]).fit(x_train,y_train)


print("Checking for Accuracy")

y_predictNoDeltas = ensemble.predict(x_testNoDeltas)
y_predictDeltas = ensemble.predict(x_testDeltas)

print(f"Accuracy score for the ensemble Delta is: {accuracy_score(y_testDeltas, y_predictDeltas)}")
print(f"Accuracy score for the ensemble No Delta is: {accuracy_score(y_testNoDeltas, y_predictNoDeltas)}")

y_test = np.concatenate([y_testDeltas,y_testNoDeltas])
y_pred = np.concatenate([y_predictDeltas,y_predictNoDeltas])

# plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix')
plotConfusionMatrix(y_test, y_pred, title='Random Forest & Logistic Regression', normalize=True)
