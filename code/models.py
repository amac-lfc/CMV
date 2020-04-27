from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

names = ["AdaBoost", "GradientBoosting", "Perceptron",
    "LogisticRegression", "RandomForest", "MLP", "DecisionTree",
    "MultinomialNB"]

AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)

GradientBoosting = GradientBoostingClassifier(n_estimators=100, random_state=0)

Perceptron = Perceptron(tol=1e-6)

LogisticRegression = LogisticRegression(max_iter=10000, solver="liblinear", penalty="l1",
            class_weight="balanced", C=1e-1)

RandomForest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=80, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=12,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
            hidden_layer_sizes=(10, 50), random_state=1)

DecisionTree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

MultinomialNB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
