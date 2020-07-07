from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

import sys
sys.setrecursionlimit(10000)

names = np.array(["RandomForest", "AdaBoost", "GradientBoosting","LogisticRegression", "DecisionTree", 'GaussianNB', 'BernoulliNB' ,'SVM'])

def AdaBoost(): return AdaBoostClassifier(n_estimators=100, random_state=0)

def GradientBoosting(): return GradientBoostingClassifier(n_estimators=100, random_state=0)

def LogisticRegression(c=1):
    return LogisticRegressionClassifier(C=c)

def RandomForest(): return RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=80, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=12,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            random_state=0, verbose=0, warm_start=True, oob_score=True)

def DecisionTree(): return DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='best')

def SVM(C=1.0) : return SVC(C=C)
