from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
from sklearn.linear_model import Lasso as LassoClassifier
from sklearn.linear_model import Ridge as RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import numpy as np

import sys
sys.setrecursionlimit(10000)

params = {'logistic': {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'},
         'ada_boost': {'adaboostclassifier__algorithm': 'SAMME.R',
                       'adaboostclassifier__learning_rate': 0.5,
                       'adaboostclassifier__n_estimators': 1600,
                       'smote__k_neighbors': 4,
                       'smote__sampling_strategy': 1.0
                       },
         'bernoulli_nb': {'bernoullinb__alpha': 0.01,
                          'bernoullinb__class_prior': [0.88, 0.12],
                          'bernoullinb__fit_prior': True,
                          'smote__k_neighbors': 3,
                          'smote__sampling_strategy': 1.0
                          },
         'decision_tree': {'criterion': 'gini',
                           'max_depth': 10,
                           'max_features': None,
                           'max_leaf_nodes': 50,
                           'min_impurity_decrease': 0.0,
                           'min_samples_leaf': 1,
                           'min_samples_split': 5,
                           'splitter': 'best'
                           },
         'gaussian_nb': {'gaussiannb__priors': [0.020000000000000018, 0.98],
                         'gaussiannb__var_smoothing': 0.1,
                         'smote__k_neighbors': 7,
                         'smote__sampling_strategy': 1.0
                         },
         'gradient_boosting': {'gradientboostingclassifier__learning_rate': 0.1,
                               'gradientboostingclassifier__loss': 'deviance',
                               'gradientboostingclassifier__max_features': 'sqrt',
                               'gradientboostingclassifier__max_leaf_nodes': None,
                               'gradientboostingclassifier__min_samples_leaf': 1,
                               'gradientboostingclassifier__min_samples_split': 10,
                               'gradientboostingclassifier__n_estimators': 200,
                               'smote__k_neighbors': 5,
                               'smote__sampling_strategy': 1.0
                               },
         'lasso': {'lasso__alpha': 0.001,
                   'smote__k_neighbors': 8,
                   'smote__sampling_strategy': 1.0
                   },
         'random_forest': {'bootstrap': True,
                           'max_depth': 20,
                           'max_features': 'log2',
                           'n_estimators': 200
                           },
         'ridge': {'ridge__alpha': 10.0,
                   'ridge__solver': 'auto',
                   'smote__k_neighbors': 2,
                   'smote__sampling_strategy': 1.0
                   },
         'svm': {'C': 1.0,
                 'degree': 1,
                 'gamma': 0.1,
                 'kernel': 'rbf',
                 'shrinking': True
                 }}


'''
The different models are:
    1 : "RandomForest"
    2 : "AdaBoost"
    3 : "GradientBoosting"
    4 : "LogisticRegression"
    5 : "DecisionTree"
    6 : 'GaussianNB' (Gaussian naive Bayes)
    7 : 'BernoulliNB' (Bernouille naive Bayes)
    8 : 'SVM' (Support Vector Machine)
'''
names = np.array(["RandomForest", "AdaBoost", "GradientBoosting","LogisticRegression", "DecisionTree", 'GaussianNB', 'BernoulliNB' ,'SVM'])

def AdaBoost(): return AdaBoostClassifier() #**params['ada_boost']

def GradientBoosting(): return GradientBoostingClassifier() #**params['gradient_boosting']

def LogisticRegression(class_weight=None):
    return LogisticRegressionClassifier(class_weight=class_weight, **params['logistic'])

def RandomForest(class_weight=None): return RandomForestClassifier(class_weight=class_weight, **params['random_forest'])

def DecisionTree(class_weight=None): return DecisionTreeClassifier(class_weight=class_weight, **params['decision_tree'])

def SVM(class_weight=None) : return SVC(class_weight=class_weight, **params['svm'])

# add laso and ridgegression
