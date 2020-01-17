print("Importing Modules...")

from imblearn.over_sampling import SMOTE
import pandas as pd
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
from xgboost import XGBClassifier


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
features_list = ['nested_count', 'reply_count','certainty_count', 'extremity_count', 'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count',
                'questions_count', 'bold_count', 'avgSentences_count', 'enumeration', 'excla'] #+ common_words

print('Reading File and Creating Data')
df = pd.read_csv('/home/shared/CMV/Delta_Data.csv', delimiter = ",")
Deltas = df.values[:,3:]
y_Deltas = np.ones(len(Deltas[:,0]),'i')
Deltas = np.column_stack((Deltas,y_Deltas))
# print(Deltas[:10])
# df = pd.read_csv('NoDelta_Data.csv', delimiter = ",")
# ds = df.sample(frac=0.005)
ds = pd.read_csv('/home/shared/CMV/NoDelta_Data_Sample.csv', delimiter = ",")
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

sm = SMOTE()
x_res, y_res = sm.fit_resample(x_train_sm, y_train_sm)


print("SM Deltas, SM NoDeltas = ", len(np.where(y_res == 1)[0]), len(np.where(y_res == 0)[0]))

print("Training Decision Tree, Naive Bayes Classifier, and Random Forest Classifier")
#Creating the Classifiers

# fit model no training data
model = XGBClassifier()

x_train = np.asarray(x_train)

model.fit(x_train, y_train)


print("Checking for Accuracy")


y_predictDeltas = model.predict(x_testNoDeltas)
y_predictNoDeltas = model.predict(x_testDeltas)


y_test = np.concatenate([y_testDeltas,y_testNoDeltas])
y_pred = np.concatenate([y_predictDeltas,y_predictNoDeltas])

# plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix')
plotConfusionMatrix(y_test, y_pred, title='Confusion Matrix', normalize=True)
