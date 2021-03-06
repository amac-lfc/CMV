import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

import slimmer
import labeler
import sampler
import features
import models
import engineer
import lib

from pprint import pprint

if __name__ == '__main__':

    '''
    If you need to create the data call the function below:
    createData()
    '''


    print("Prepping Data")
    # Reading the delta:
    delta_file = "/home/shared/CMV/FeatureData/all_delta_feature_data.csv"
    # delta_file = "/mnt/h/FeatureData/all_delta_feature_data.csv"
    delta_data = pd.read_csv(delta_file)

    # Reading the no delta
    nodelta_file = "/home/shared/CMV/FeatureData/all_nodelta_feature_data.csv"
    # nodelta_file = "/mnt/h/FeatureData/all_nodelta_feature_data.csv"
    nodelta_data = pd.read_csv(nodelta_file)
    print("Sampling NoDelta File")
    nodelta_data = nodelta_data.sample(n=20000)
    ## If you already saved the sample file:
    # nodelta_sample_file = "sampled_nodelta_feature_data.csv"
    # nodelta_data = pd.read_csv(nodelta_sample_file)

    # Merge the data set and add labels = 0 (No Delta) 1 (Delta)
    data = engineer.merge([nodelta_data, delta_data])

    # Split the data between features and labels
    X, y = data[: , :-1], data[:, -1]

    # Scale all the features between 0 and 1
    scaler = MinMaxScaler()
    X=scaler.fit_transform(X)

    print("Shape of all features:", X.shape)

    ## Compute class weights
    class_weight=compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
    class_weight={0:class_weight[0],1:class_weight[1]}
    print(class_weight)
    sample_weight = np.zeros(len(y))
    sample_weight[y==0]=class_weight[0]
    sample_weight[y==1]=class_weight[1]

    # This variable will keep track of the grid results by
    # classifier name.
    results = {}

    # #################################################################
    # ## Random Forest Classifier
    # #################################################################
    print("##########################")
    print("#### Random Forest...")
    
    # parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17.,18,19,20,21],
    #              'max_features': ['auto', 'sqrt', 'log2']}
    
    # new set of parameters
    parameters = {
                 'criterion': ['gini', 'entropy'],
                 'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 'max_features': ['auto', 'sqrt', 'log2', None],
                 'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [25, 100, 200],
                 'max_leaf_nodes': [5, 10, 50, 100, 200, 500, None],
                 'class_weight': [class_weight]}
    
    clf = models.RandomForestClassifier()
    
    gd = GridSearchCV(estimator=clf,
                     param_grid = parameters,
                     scoring='accuracy',
                     cv=5,
                     refit=True,
                     n_jobs=28,
                     verbose=1)
    
    gd.fit(X, y)
    
    best_parameters = gd.best_params_
    print(best_parameters)
    results["random_forest"] = best_parameters
    
       # parameters = {'criterion': ['gini', 'entropy'],
    #              'splitter': ['best', 'random'],
    #              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #              'min_samples_leaf': [1, 2, 4],
    #              'min_samples_split': [2, 5, 10],
    #              'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    #              'max_features': ['auto', 'sqrt', 'log2', None],
    #              'max_leaf_nodes': [5, 10, 50, 100, 200, 500, None],
    #              'class_weight': [class_weight]
    
    # #################################################################
    # ### Ada Boost Classifier
    # ##################################################################
    # print("##########################")
    # print("#### Ada Boost...")
    #
    # parameters = {'smote__sampling_strategy': [0.1,0.5,1.0,'auto'],
    #              'smote__k_neighbors': [i for i in range(2,9)],
    #              'adaboostclassifier__n_estimators': [25, 50, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #              'adaboostclassifier__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    #              'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R']}
    #
    # clf = models.AdaBoost()
    #
    # pipe = make_pipeline(SMOTE(),clf)
    #
    # gd = GridSearchCV(estimator=pipe,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=5,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["ada_boost"] = best_parameters


    ##################################################################
    ### Gradient Boosting Classifier
    ##################################################################
    # print("##########################")
    # print("#### Gradient Boosting...")

    # parameters = {'smote__sampling_strategy': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'auto'],
    #              'smote__k_neighbors': [1,2,3,4,5,6,7,8],

    #              'gradientboostingclassifier__learning_rate': [0.1],
    #              'gradientboostingclassifier__loss': ['deviance'],
    #              'gradientboostingclassifier__max_features': ['sqrt'],
    #              'gradientboostingclassifier__max_leaf_nodes': [None],
    #              'gradientboostingclassifier__min_samples_leaf': [1],
    #              'gradientboostingclassifier__min_samples_split': [10],
    #              'gradientboostingclassifier__n_estimators': [200],

    #              # 'gradientboostingclassifier__loss': ['deviance', 'exponential'],
    #              # 'gradientboostingclassifier__learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
    #              # 'gradientboostingclassifier__n_estimators': [25, 50, 200],
    #              # 'gradientboostingclassifier__subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    #              # 'gradientboostingclassifier__criterion': ['friedman_mse', 'mse', 'mae'],
    #              # 'gradientboostingclassifier__min_samples_leaf': [1, 2, 4],
    #              # 'gradientboostingclassifier__min_samples_split': [2, 5, 10],
    #              # 'gradientboostingclassifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #              # 'gradientboostingclassifier__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    #              # 'gradientboostingclassifier__max_features': ['auto', 'sqrt', 'log2', None],
    #              # 'gradientboostingclassifier__max_leaf_nodes': [5, 10, 50, 100, 200, 500, None],
    #              # 'gradientboostingclassifier__warm_start': [False, True]
    #              }

    # clf = models.GradientBoosting()
    # pipe = make_pipeline(SMOTE(),clf)

    # gd = GridSearchCV(estimator=pipe,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=3,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=2)

    # gd.fit(X, y)

    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["gradient_boosting"] = best_parameters

    # ##################################################################
    # ### Logistic Regression Classifier
    # ##################################################################
    # print("##########################")
    # print("#### Logistic Regression...")
    #
    # parameters = [
    #             {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    #             'solver': ['liblinear'],
    #             'penalty' : ['l1', 'l2']},
    #             {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    #             'solver': ['saga'],
    #             'penalty' : ['l1', 'l2', 'elasticnet']},
    #             {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    #             'solver': ['sag', 'lbfgs', 'newton_cg'],
    #              'penalty' : ['l2','none']},
    #             {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    #             'penalty' : ['none']}
    #              ]
    #
    # clf = models.LogisticRegressionClassifier(class_weight=class_weight)
    #
    # gd = GridSearchCV(estimator=clf,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=5,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["Logistic"] = best_parameters
    #
    #
    # ##################################################################
    # ### Decision tree
    # ##################################################################
    # print("##########################")
    # print("#### Decision Tree...")
    #
    # parameters = {'criterion': ['gini', 'entropy'],
    #              'splitter': ['best', 'random'],
    #              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #              'min_samples_leaf': [1, 2, 4],
    #              'min_samples_split': [2, 5, 10],
    #              'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    #              'max_features': ['auto', 'sqrt', 'log2', None],
    #              'max_leaf_nodes': [5, 10, 50, 100, 200, 500, None],
    #              'class_weight': [class_weight]
    #              }
    #
    # clf = models.DecisionTreeClassifier(class_weight=class_weight)
    #
    #
    # gd = GridSearchCV(estimator=clf,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=3,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["decision_tree"] = best_parameters
    #
    #
    #
    # ##################################################################
    # ### Gaussian Naive Bayes Classifier
    # ##################################################################
    #
    # print("##########################")
    # print("#### Gaussian Naive Bayes")
    #
    # # Note: This is pretty bad since it's probably making it always say
    # # class with the majority.
    # parameters = {'smote__sampling_strategy': [0.1,0.5,1.0,'auto'],
    #               'smote__k_neighbors': [i for i in range(2,9)],
    #               'gaussiannb__var_smoothing': [10**-i for i in range(1,20)],
    #               'gaussiannb__priors':[[1-(i/100), (i/100)] for i in range(1, 100)]}
    #
    # clf = models.GaussianNB()
    # pipe = make_pipeline(SMOTE(),clf)
    #
    # gd = GridSearchCV(estimator=pipe,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=2,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["gaussian_nb"] = best_parameters
    #
    #
    #
    # ##################################################################
    # ### Bernoulli Naive Bayes Classifier
    # ##################################################################
    # print("##########################")
    # print("#### Bernoulli Naive Bayes...")
    #
    # parameters = {'smote__sampling_strategy': [0.1,0.5,1.0,'auto'],
    #               'smote__k_neighbors': [i for i in range(2,9)],
    #               'bernoullinb__alpha': [10E-10, 10E-7, 10E-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    #               'bernoullinb__fit_prior': [False, True],
    #               'bernoullinb__class_prior':[[1-(i/100), (i/100)] for i in range(1, 100)]}
    #
    # clf = models.BernoulliNB()
    # pipe = make_pipeline(SMOTE(),clf)
    #
    # gd = GridSearchCV(estimator=pipe,
    #                param_grid = parameters,
    #                scoring='accuracy',
    #                cv=5,
    #                refit=True,
    #                n_jobs=28,
    #                verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["bernoulli_nb"] = best_parameters
    #
    #
    #
    # ##################################################################
    # ### Support Vector Machine
    # ##################################################################
    # print("##########################")
    # print("#### SVM...")
    #
    #
    # # parameters = {'kernel':('linear', 'rbf'), 'C':[1E-2, 10]}
    # # parameters = [{
    # #                 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    # #                 'kernel': ['linear']
    # #               },
    # #              {
    # #                 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    # #                 'gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    # #                 'kernel': ['rbf']
    # #              }]
    #
    # parameters = [{
    #                 'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    #                 'gamma': [0.001, 0.01, 0.1, 1.0, 10.0],
    #                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #                 'degree': [i for i in range(1, 3)],
    #                 'shrinking': [True]
    #              }]
    # clf = models.SVC(class_weight=class_weight)
    #
    # gd = GridSearchCV(estimator=clf,
    #                  param_grid = parameters,
    #                  scoring='accuracy',
    #                  cv=2,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["svm"] = best_parameters
    #
    #
    #
    # ##################################################################
    # ### Lasso
    # ##################################################################
    # print("##########################")
    # print("#### Lasso...")
    #
    # parameters = [{'smote__sampling_strategy': [0.1,0.5,1.0,'auto'],
    #                'smote__k_neighbors': [i for i in range(2,9)],
    #                'lasso__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}]
    #
    # # pipe = Pipeline([('smote', engineer.SMOTE()), ('clf', models.LassoClassifier())])
    # pipe = make_pipeline(SMOTE(),models.LassoClassifier())
    #
    # gd = GridSearchCV(estimator=pipe,
    #                  param_grid = parameters,
    #                  cv=5,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    #
    # fit_parameters = {'sample_weight': sample_weight}
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["lasso"] = best_parameters
    #
    #
    #
    # ##################################################################
    # ### Ridge
    # ##################################################################
    # print("##########################")
    # print("#### Ridge...")
    #
    # parameters = {'smote__sampling_strategy': [0.1,0.5,1.0,'auto'],
    #               'smote__k_neighbors': [i for i in range(2,9)],
    #               'ridge__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    #               'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] }
    #
    # pipe = make_pipeline(SMOTE(),models.RidgeClassifier())
    #
    # gd = GridSearchCV(estimator=pipe,
    #                  param_grid = parameters,
    #                  cv=5,
    #                  refit=True,
    #                  n_jobs=28,
    #                  verbose=1)
    #
    # gd.fit(X, y)
    #
    # best_parameters = gd.best_params_
    # print(best_parameters)
    # results["ridge"] = best_parameters
    #
    pprint(results)
