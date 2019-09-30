
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


USE_CUDA = torch.cuda.is_available()
gpus = [0]
if USE_CUDA:
    torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

# Definition of hyper-parameters
TRAINING_EPOCHS = 3000
NUM_CLASSIFIERS = 5
MAX_DEPTH = 4
LEARNING_RATE = 0.1

# Read the training data

df = pd.read_csv("./titanic.csv", sep="\t")
X = df.loc[:, ["Age", "Fare", "Pclass"]]
y = df.loc[:, "Survived"]
X = np.nan_to_num(X, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

#######################################################################################################
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

#######################################################################################################
class LossFunctionMinimizer(nn.Module):
    def __init__(self):
        super(LossFunctionMinimizer, self).__init__()
        self.current_leaf_value = nn.Parameter(data=FloatTensor([0.0]), requires_grad=True)
    def forward(self, targets, previous_predictions):
        P = logodds = previous_predictions + self.current_leaf_value
        deviance = -(torch.sum(targets * P - torch.log(1 + torch.exp(P))))
        return deviance
    def reinitialize_variable(self):
        self.current_leaf_value.data = FloatTensor([0.0])

minimizer = LossFunctionMinimizer()
if USE_CUDA:
    minimizer.cuda()
minimizer_optimizer = torch.optim.Adam(minimizer.parameters(), lr=0.001)

def minimize_loss_function(targets, previous_predictions, TRAINING_EPOCHS=2000):
    minimizer.reinitialize_variable()
    for training_epoch in range(TRAINING_EPOCHS):
        targets_leaf_tensor = FloatTensor(targets)
        # previous_leaf_predictions_tensor = torch.zeros(targets.shape)
        loss = minimizer.forward(targets_leaf_tensor, previous_predictions)
        minimizer.zero_grad()
        loss.backward()
        minimizer_optimizer.step()
    return [el for el in minimizer.parameters()][0].cpu().detach().numpy()[0]


#######################################################################################################

class ResidualsCalculator(nn.Module):
    def __init__(self, predicted_values):
        super(ResidualsCalculator, self).__init__()
        self.predicted_values = nn.Parameter(data=torch.zeros(predicted_values.shape), requires_grad=True)
        self.predicted_values.data = predicted_values
    def forward(self, targets):
        P = logodds = self.predicted_values
        deviance = -(torch.sum(targets * P - torch.log(1 + torch.exp(P))))
        return deviance

def compute_residuals(targets, predicted_values):
    model = ResidualsCalculator(predicted_values)
    if USE_CUDA:
        model.cuda()
    loss = model.forward(targets)
    model.zero_grad()
    loss.backward()
    residuals = model.predicted_values.grad.clone()# deep copy of the input
    return residuals

#######################################################################################################

def fit_regression_tree_classifier_to_residuals(X_data, y_data): # y_data -> residuals
    tree_regressor = DecisionTreeRegressor(max_depth=MAX_DEPTH)
    tree_regressor.fit(X_data, y_data)
    leaf_buckets = []
    for i in range(X_data.shape[0]):
        leaf_buckets.append(tuple(tree_regressor.decision_path(X_data[i, :].reshape(1, -1)).todok().keys()))
    unique_paths = list(set(leaf_buckets))
    return (leaf_buckets, unique_paths, tree_regressor)

#######################################################################################################


initial_predictions = minimize_loss_function(y_train, torch.zeros(y_train.shape).cuda())
initial_probability = sigmoid(initial_predictions)
predictions_array = initial_predictions = np.ones(y_train.shape)*initial_predictions

regression_trees = []

y_values = y_train
X_values = X_train
prediction_values = predictions_array




for classifier_index in range(NUM_CLASSIFIERS):
    regression_trees.append({"tree_index": classifier_index})
    residuals = compute_residuals(FloatTensor(y_values), FloatTensor(prediction_values))
    # regression_trees[-1]["residuals"] = residuals
    leaf_buckets, unique_clusters, tree_regressor = fit_regression_tree_classifier_to_residuals(X_values, residuals.cpu())
    regression_trees[-1]["tree_regressor"] = tree_regressor

    X_values_temp = np.array([])
    y_values_temp = np.array([])
    prediction_values_temp = np.array([])

    for unique_cluster in unique_clusters:
        indices = [1 if el == unique_cluster else 0 for el in leaf_buckets]
        y_leaf = y_values[np.array(indices) == 1]
        X_leaf = X_values[np.array(indices) == 1]
        predictions_leaf = prediction_values[np.array(indices) == 1]
        prediction_for_leaf = minimize_loss_function(FloatTensor(np.array(y_leaf)), FloatTensor(predictions_leaf))
        predictions_for_leaf_array = np.ones(y_leaf.shape) * LEARNING_RATE * prediction_for_leaf + predictions_leaf
        regression_trees[-1][str(unique_cluster)] = prediction_for_leaf
        X_values_temp = X_leaf if X_values_temp.shape == (0, ) else np.append(X_values_temp, X_leaf, axis=0)
        y_values_temp = np.append(y_values_temp, y_leaf)
        prediction_values_temp = np.append(prediction_values_temp, predictions_for_leaf_array)

    y_values = y_values_temp
    X_values = X_values_temp
    prediction_values = prediction_values_temp

print("Tha accuracy of the custom implementation is : {}".format(accuracy_score(np.round(sigmoid(prediction_values)), y_values)))


# Vanilla sklearn algorithm
classifier = GradientBoostingClassifier(n_estimators=NUM_CLASSIFIERS, max_depth=MAX_DEPTH)
classifier.fit(np.array(X_train), np.array(y_train))
print("Tha accuracy of the sklearn implementation is : {}".format(accuracy_score(classifier.predict(X_train), y_train)))


# # Vanilla sklearn algorithm - This is the very same as the above (cross checking whether X_values and y_values are
# # consistent
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# classifier = GradientBoostingClassifier(n_estimators=NUM_CLASSIFIERS, max_depth=MAX_DEPTH)
# classifier.fit(np.array(X_values), np.array(y_values))
# print("Tha accuracy of the sklearn implementation is : {}".format(accuracy_score(classifier.predict(X_values), y_values)))

#######################################################################################################
# Comparisons on the test set


