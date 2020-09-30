import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score

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

    '''
    Slect which models you want to use. Your options are:
    1 : "RandomForest"
    2 : "AdaBoost"
    3 : "GradientBoosting"
    4 : "LogisticRegression"
    5 : "DecisionTree"
    6 : 'GaussianNB' (Gaussian naive Bayes)
    7 : 'BernoulliNB' (Bernouille naive Bayes)
    8 : 'SVM' (Support Vector Machine)
    '''
    ModelList= [1,2,3,4,6,7,8]
    # ModelList=[4,2]

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


    scores = []
    for ModelNumber in ModelList:
        # Defined the model
        print("### Model: "+models.names[ModelNumber-1]+"...")
        if ModelNumber in [2,3,6,7]:
            model = getattr(models, models.names[ModelNumber-1])() #  this is equivalent to model = model.LogisticRegression()
        else:
            model = getattr(models, models.names[ModelNumber-1])(class_weight=class_weight)

        print("Fitting Model")
        if ModelNumber in [2,3,6,7]:
            all_accuracies = cross_val_score(estimator=model, X=X, y=y, cv=5, fit_params={'sample_weight': sample_weight} )
        else:
            all_accuracies = cross_val_score(estimator=model, X=X, y=y, cv=5)

        # all_accuracies = cross_val_score(estimator=model, X=X, y=y, cv=5)
        scores.append(all_accuracies)
        print(all_accuracies)

 
    scores=np.stack( scores, axis=0 )
    # Plot the bar chart of the different scores
    if len(ModelList) > 1:
        plt.clf()
        plt.rcdefaults()
        objects = models.names[np.array(ModelList)-1]
        y_pos = np.arange(len(objects))

        # sort the scores and object in acending order
        # indices = np.argsort(scores)
        # scores=np.array(scores)[indices]
        # objects = objects[indices]

        for k in range(5):
            plt.barh(y_pos+0.16*k, scores[:,k], align='center', alpha=0.5, height = 0.16)
        plt.yticks(y_pos+2*0.16, objects)
        plt.xlim([0,1])
        plt.title('Accuracy Scores')
        for k in range(5):
            for i in range(len(scores[:,k])):
                plt.annotate('{:.2f}'.format(scores[i,k]), xy=(scores[i,k],y_pos[i]+0.16*k))
        plt.tight_layout()
        print("Saving the accaracy scores as accuracy_scores.png")
        plt.savefig("cross_val_scores.png")