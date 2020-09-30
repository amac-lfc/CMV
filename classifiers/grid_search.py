import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from  sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

import slimmer
import labeler
import sampler
import features
import models
import engineer
import lib


if __name__ == '__main__':

    ''' 
    If you need to create the data call the function below:
    createData()
    '''


    print("Prepping Data")
    # Reading the delta:
    delta_file = "/mnt/h/FeatureData/all_delta_feature_data.csv"
    delta_data = pd.read_csv(delta_file)

    # Reading the no delta
    nodelta_file = "/mnt/h/FeatureData/all_nodelta_feature_data.csv"
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


    ###################################################################
    #### Support Vector Machine
    ###################################################################
   #  print("##########################")
   #  print("#### SVM...")


   #  # parameters = {'kernel':('linear', 'rbf'), 'C':[1E-2, 10]}
   #  parameters = [{
   #                  'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
   #                  'kernel': ['linear']
   #                },
   #               {
   #                  'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
   #                  'gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
   #                  'kernel': ['rbf']
   #               }]
   #  clf = models.SVC(class_weight=class_weight)

   #  gd = GridSearchCV(estimator=clf, 
   #                   param_grid = parameters,
   #                   scoring='accuracy',
   #                   cv=5,
   #                   refit=True,
   #                   n_jobs=1)
   #  gd.fit(X, y)

   #  best_parameters = gd.best_params_
   #  print(best_parameters)


    ###################################################################
    #### Random Forest Classifier
    ###################################################################
   #  print("##########################")
   #  print("#### Random Forest...")

   #  parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17.,18,19,20,21],
   #               'max_features': ['auto', 'sqrt', 'log2']}

   #  clf = models.RandomForestClassifier(class_weight=class_weight)

   #  gd = GridSearchCV(estimator=clf, 
   #                   param_grid = parameters,
   #                   scoring='accuracy',
   #                   cv=5,
   #                   refit=True,
   #                   n_jobs=1)

   #  gd.fit(X, y)

   #  best_parameters = gd.best_params_
   #  print(best_parameters)

    ###################################################################
    #### Logistic Regression Classifier
    ###################################################################
    print("##########################")
    print("#### Logistic Regression...")

    parameters = [
                {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], 
                'solver': ['liblinear'],
                'penalty' : ['l1', 'l2']},
                {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], 
                'solver': ['saga'],
                'penalty' : ['l1', 'l2', 'elasticnet']},
                {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], 
                'solver': ['sag', 'lbfgs', 'newton_cg'],
                 'penalty' : ['l2,, ''none']},
                {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], 'penalty' : ['none']}
                 ]

    clf = models.LogisticRegressionClassifier(class_weight=class_weight)

    gd = GridSearchCV(estimator=clf, 
                     param_grid = parameters,
                     scoring='accuracy',
                     cv=5,
                     refit=True,
                     n_jobs=1)

    gd.fit(X, y)

    best_parameters = gd.best_params_
    print(best_parameters)

   ###################################################################
   #### Lasso
   ###################################################################
    print("##########################")
    print("#### Lasso...")

    parameters = [{'lasso__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}]

   #  pipe = Pipeline([('smote', engineer.SMOTE()), ('clf', models.LassoClassifier())])
    pipe = make_pipeline(SMOTE(),models.LassoClassifier())

    gd = GridSearchCV(estimator=pipe, 
                     param_grid = parameters,
                     cv=5)
  

    fit_parameters = {'sample_weight': sample_weight}

    gd.fit(X, y)

    best_parameters = gd.best_params_
    print(best_parameters)


    ###################################################################
    #### Ridge
    ###################################################################
    print("##########################")
    print("#### Ridge...")

    parameters = {'ridge__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                  'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] }

    pipe = make_pipeline(SMOTE(),models.RidgeClassifier())

    gd = GridSearchCV(estimator=pipe, 
                     param_grid = parameters,
                     cv=5)

    gd.fit(X, y)

    best_parameters = gd.best_params_
    print(best_parameters)

    ###################################################################
    #### Decision tree
    ###################################################################
   #  print("##########################")
   #  print("#### Decision Tree...")

   #  parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17.,18,19,20,21],
   #               'max_features': ['auto', 'sqrt', 'log2']}

   #  clf = models.DecisionTreeClassifier(class_weight=class_weight)

   #  gd = GridSearchCV(estimator=clf, 
   #                   param_grid = parameters,
   #                   scoring='accuracy',
   #                   cv=5,
   #                   refit=True,
   #                   n_jobs=1)

   #  gd.fit(X, y)

   #  best_parameters = gd.best_params_
   #  print(best_parameters)