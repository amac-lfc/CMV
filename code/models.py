from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense

names = ["AdaBoost", "GradientBoosting", "PerceptronClassifier",
    "LogisticRegressionClassifier", "RandomForest", "MLP", "DecisionTree",
    "MultinomialNB"]

def AdaBoost(): return AdaBoostClassifier(n_estimators=100, random_state=0)

def GradientBoosting(): return GradientBoostingClassifier(n_estimators=100, random_state=0)

def PerceptronClassifier(): return Perceptron(tol=1e-6)

def LogisticRegressionClassifier(c=1):
    return LogisticRegression(C=c)

def RandomForest(): return RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=80, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=12,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

def MLP(): return MLPClassifier(solver='lbfgs', alpha=1e-5,
            hidden_layer_sizes=(10, 50), random_state=1)

def DecisionTree(): return DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

def MultinomialNB(): return MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


class FeedForwardNeuralNetwork:
    def __init__(self, inputs=10, outputs=1, layers=[10,10], epochs=50):
        self.model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                self.model.add(Dense(layers[i], input_dim=inputs, activation='relu'))
            else:
                self.model.add(Dense(layers[i], activation='relu'))
        self.model.add(Dense(outputs, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.epochs = epochs

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, verbose=1)
        return self

    def predict(self, X):
        return self.model.predict_classes(X)
        return self
