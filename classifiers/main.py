import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight


import slimmer
import labeler
import sampler
import features
import models
import engineer
import lib

def createData():

    '''
    This function create the data with all the features
    '''

    inputs = ["/home/shared/CMV/RawData/Comments_MetaData.csv", "/home/shared/CMV/RawData/Comments_TextData.csv",
        "/home/shared/CMV/RawData/Submissions_MetaData.csv", "/home/shared/CMV/RawData/Submissions_TextData.csv"]

    outputs = ['/home/shared/CMV/SlimmedData/Slimmed_Comments_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_MetaData.csv',
        '/home/shared/CMV/SlimmedData/Slimmed_Submissions_TextData.csv']

    columns_lst = [["name", "parent_id", "author", "link_id"], ["author", "id", "parent_id", "body"],
        ["url", "id", "author"], ["author", "id", "title", "selftext"]]

    slimmer.slim_all(inputs, outputs, columns_lst)


    input = '/home/shared/CMV/SlimmedData/Slimmed_Comments_TextData.csv'

    deltas_file = '/home/shared/CMV/SortedData/delta_winning_ids.txt'
    deltas_data_file = '/home/shared/CMV/SortedData/delta_comments_data.csv'
    nodeltas_data_file = '/home/shared/CMV/SortedData/nodelta_comments_data.csv'

    labeler.get_deltas(input, deltas_file)

    labeler.create_labels(input, deltas_data_file, nodeltas_data_file, deltas_file)


    # get input files
    delta_input = "/home/shared/CMV/SortedData/delta_comments_data.csv"
    nodelta_input = "/home/shared/CMV/SortedData/nodelta_comments_data.csv"
    word_list_input = "../data/word_list.csv"

    # make output files
    output_delta = "/home/shared/CMV/FeatureData/all_delta_feature_data.csv"
    output_nodelta = "/home/shared/CMV/FeatureData/all_nodelta_feature_data.csv"

    # generate features
    delta_features, nodelta_features = features.generateFeatures([delta_input, nodelta_input], [output_delta, output_nodelta], word_list_input, 'con')

    print("Writing Features to File with Pandas")
    delta_features = pd.DataFrame(data=delta_features, columns=None)
    nodelta_features = pd.DataFrame(data=nodelta_features, columns=None)

    delta_features.to_csv(output_delta, index=False)
    nodelta_features.to_csv(output_nodelta, index=False)


def create_sample_deltas(sample_size=20000):
    '''
    This function read the delta data and save a sample of it
    '''
    nodelta_file = "/mnt/h/FeatureData/all_nodelta_feature_data.csv"
    nodelta_sample_file = "../data/sampled_nodelta_feature_data.csv"
    sampler.sample(nodelta_file, nodelta_sample_file, sample_size)

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
    ModelList= [1,2,3,4,5,6,7,8]
    # ModelList=[2]

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
    X_train, X_test, y_train, y_test = engineer.train_test_split(X, y, test_size=0.33)

    ## Oversampling with SMOTE
    # X_train,y_train = engineer.smote(X_train, y_train, k_neighbors= 2, sampling_strategy=0.8)
    # class_weight = None
    # sample_weight = None

    ## Compute class weights
    class_weight=compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
    class_weight={0:class_weight[0],1:class_weight[1]}
    print(class_weight)
    sample_weight = np.zeros(len(y_train))
    sample_weight[y_train==0]=class_weight[0]
    sample_weight[y_train==1]=class_weight[1]


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
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = accuracy_score(y_pred, y_test)
        scores.append(score)
        print("Score:",score)

        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize
        print("Confusion Matrix: \n", cm)

        plot_confusion_matrix(model, X_test, y_test,
                                display_labels=['no delta', 'delta'],
                                cmap=plt.cm.Blues,
                                normalize='true')
        plt.title(models.names[ModelNumber-1])
        print("Saving the confusion matrix for {0} as confusion_matrix_for_{0}.png".format(models.names[ModelNumber-1]))
        plt.savefig("confusion_matrix_for_{0}.png".format(models.names[ModelNumber-1]))

        if ModelNumber==1:
            lib.getImportances(model, delta_data.columns[:-1],savefig="feature_importance.png")

# Plot the bar chart of the different scores
if len(ModelList) > 1:
    plt.clf()
    plt.rcdefaults()
    objects = models.names[np.array(ModelList)-1]
    y_pos = np.arange(len(objects))

    # sort the scores and object in acending order
    indices = np.argsort(scores)
    scores=np.array(scores)[indices]
    objects = objects[indices]

    plt.barh(y_pos, scores, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlim([0,1])
    plt.title('Accuracy Scores')
    for i in range(len(scores)):
        plt.annotate('{:.2f}'.format(scores[i]), xy=(scores[i],y_pos[i]))
    plt.tight_layout()
    print("Saving the accaracy scores as accuracy_scores.png")
    plt.savefig("accuracy_scores.png")
