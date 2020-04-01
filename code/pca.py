print("Importing Modules...")

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
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

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# x_top_3 = x_res[:,[4, 11, 6]].copy()
#
#
# deltas = []
# nodeltas = []
# for i in range(len(y_res)):
#     if y_res[i] == 0:
#         nodeltas.append(x_top_3[i])
#     else:
#         deltas.append(x_top_3[i])
#
# deltas = np.array(deltas)
# nodeltas = np.array(nodeltas)
#
# print(len(deltas), len(nodeltas))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(rand_jitter(deltas[:, 0]), rand_jitter(deltas[:, 1]), rand_jitter(deltas[:, 2]), alpha=.2, c="red")
# ax.scatter(rand_jitter(nodeltas[:, 0]), rand_jitter(nodeltas[:, 1]), rand_jitter(nodeltas[:, 2]), alpha=.2, c="blue")
#
# ax.set_xlabel('Reply Count')
# ax.set_ylabel('Avg. Sentence Count')
# ax.set_zlabel('Character Count')
#
# plt.savefig("top_three_graphed.png")


# x_top = x_res[:,[4, 11]].copy()
#
#
# deltas = []
# nodeltas = []
# for i in range(len(y_res)):
#     if y_res[i] == 0:
#         nodeltas.append(x_top[i])
#     else:
#         deltas.append(x_top[i])
#
# deltas = np.array(deltas)
# nodeltas = np.array(nodeltas)
#
# plt.scatter(deltas[:, 0], deltas[:, 1], c="red")
# plt.scatter(nodeltas[:, 0], nodeltas[:, 1], c="blue")
# plt.title("Top Two Features Graphed")
# plt.xlabel("Reply Count")
# plt.ylabel("Avg. Sentence Count")
# plt.savefig("top_two_graphed.png")

# Pull from PCA
# pca = PCA(n_components=2)
# x_res = normalize(x_res)
# x_pca = pca.fit_transform(x_res)
#
#
# deltas = []
# nodeltas = []
# for i in range(len(y_res)):
#     if y_res[i] == 0:
#         nodeltas.append(x_pca[i])
#     else:
#         deltas.append(x_pca[i])
#
# deltas = np.array(deltas)
# nodeltas = np.array(nodeltas)
#
#
# plt.scatter(deltas[:, 0], deltas[:, 1], c="red")
# plt.scatter(nodeltas[:, 0], nodeltas[:, 1], c="blue")
# plt.title("PCA Features Graphed")
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.savefig("pca_features.png")



# plt.scatter(rand_jitter(deltas[:, 0]), rand_jitter(deltas[:, 1]), alpha=.2, c="red", label="delta")
# plt.scatter(rand_jitter(nodeltas[:, 0]), rand_jitter(nodeltas[:, 1]), alpha=.2, c="blue", label="nodelta")
# plt.title("PCA Features Graphed and Jittered")
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.savefig("pca_features_jittered.png")
