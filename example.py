'''
Simple example making use of the library built to provide a custom implementation of the Gradient Boosting Algorithm
'''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from gradient_boosting_engine import PytorchBasedGenericGradientBoost

# Definition of Hyper-Parameters
NUM_CLASSIFIERS = 5
MAX_DEPTH = 4
GRADIENT_BOOST_LEARNING_RATE = 0.1
MINIMIZER_LEARNING_RATE = 0.005
MINIMIZER_TRAINING_EPOCHS = 1000

# Read the training data
df = pd.read_csv("./titanic.csv", sep="\t")
X = df.loc[:, ["Age", "Fare", "Pclass"]]
y = df.loc[:, "Survived"]
X = np.nan_to_num(X, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Running the custom algorithm 
custom = PytorchBasedGenericGradientBoost("classifier", NUM_CLASSIFIERS, MAX_DEPTH, GRADIENT_BOOST_LEARNING_RATE=GRADIENT_BOOST_LEARNING_RATE, MINIMIZER_LEARNING_RATE=MINIMIZER_LEARNING_RATE, MINIMIZER_TRAINING_EPOCHS=MINIMIZER_TRAINING_EPOCHS)
custom.fit(X_train, y_train)
predictions_train = custom.predict(X_train)
predictions_test = custom.predict(X_test)
print("Custom Implementation : Accuracy score for training data : {}".format(accuracy_score(np.round(predictions_train), y_train)))
print("Custom Implementation : Accuracy score for testing data : {}".format(accuracy_score(np.round(predictions_test), y_test)))

# Running the vanilla sklearn algorithm
classifier = GradientBoostingClassifier(n_estimators=NUM_CLASSIFIERS, max_depth=MAX_DEPTH)
classifier.fit(np.array(X_train), np.array(y_train))
print("Vanilla Implementation : Accuracy score for training data : {}".format(accuracy_score(classifier.predict(X_train), y_train)))
print("Vanilla Implementation : Accuracy score for testing data : {}".format(accuracy_score(classifier.predict(X_test), y_test)))