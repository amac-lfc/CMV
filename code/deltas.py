import pandas as pd
import numpy as np

import slimmer
import labeler
import sampler
import features
import models
import engineer
import accuracy

def main():
    # slimmer.run()
    # labeler.run()
    # sampler.run()
    # features.run()

    model = models.RandomForest

    X_train, X_test, y_train, y_test = engineer.run(model = model)

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy.score(y_pred, y_test)
    print(score)

main()
