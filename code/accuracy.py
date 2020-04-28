import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score as score
from sklearn.metrics import balanced_accuracy_score as balancedScore

def run(y_pred, y_test):
    print(score(y_pred, y_test))
