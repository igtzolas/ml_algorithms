'''
This file decomposes the Gradient Boosting Algorithm
It uses pytorch to perform optimization of the loss function, 
        pytorch to compute the residuals = partial derivatives of the loss function with respect to current predictions,
        sklearn DecisionTreeRegressor to bucketize examples based on residuals

With respect to the Classification Algorithm this is the game played all along:
probability ----  log(p/(1-p)) ----> logodds
probability <----    sigmoid   <---- logodds

The loss function is the binary cross entropy function which makes use of probabilities
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeRegressor
# To be used for comparison with the custom implementation
from sklearn.ensemble import GradientBoostingClassifier

# Handling cuda availability
USE_CUDA = torch.cuda.is_available()
gpus = [0]
if USE_CUDA:
    torch.cuda.set_device(gpus[0])
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

#######################################################################################################
class LossFunctionMinimizer(nn.Module):
    def __init__(self, type):
        # type can be one of the 2 : "regressor" or "classifier"
        super(LossFunctionMinimizer, self).__init__()
        self.type = type
        self.current_leaf_value = nn.Parameter(data=FloatTensor([0.0]), requires_grad=True)
    def reinitialize_variable(self):
        self.current_leaf_value.data = FloatTensor([0.0])
    def refine_previous_predictions(self, previous_predictions):
        new_predictions = previous_predictions + self.current_leaf_value
        return new_predictions
    def loss(self, previous_predictions, targets_leaf_tensor):
        if self.type == "regressor":
            return self.loss_regressor(previous_predictions, targets_leaf_tensor)
        elif self.type == "classifier":
            return self.loss_classifier(previous_predictions, targets_leaf_tensor)
        else:
            raise Exception("Not supported")
    def loss_classifier(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        probabilities = 1.0 / (1.0 + torch.exp(-logodds))
        loss = F.binary_cross_entropy(probabilities, targets_leaf_tensor)
        return loss
    def loss_regressor(self, previous_predictions, targets_leaf_tensor):
        values = self.refine_previous_predictions(previous_predictions)
        loss = F.mse_loss(values, targets_leaf_tensor)
        return loss        

#######################################################################################################
'''
What are the residuals ???
We have a loss function : Loss(prediction_1, prediction_2, ....., prediction_m), m : number of training points

Going back to the definition of the partial derivative/gradient with respect to a parameter, it is the change incurred to 
the loss function if we change by a little bit the parameter

So, the array of partial derivatives tells us how the Loss changes by changes in the predictions

It makes sense to group predictions with the same / similar effect !!!
This is the heart of the algorithm and explains why the algorithm builds regression trees around the residuals/partial derivatives

In my opinion, the term residuals is not that correct! The term gradient should be used instead to allow for the understanding of the algorithm rational.
'''
class ResidualsCalculator(nn.Module):
    def __init__(self, predicted_values, type):
        super(ResidualsCalculator, self).__init__()
        self.type = type
        self.predicted_values = nn.Parameter(data=torch.zeros(predicted_values.shape), requires_grad=True)
        self.predicted_values.data = predicted_values
    def forward(self):
        my_parameters = self.predicted_values
        return my_parameters
    def loss(self, targets):
        if self.type == "regressor":
            return self.loss_regressor(targets)
        elif self.type == "classifier":
            return self.loss_classifier(targets)
        else:
            raise Exception("Not supported")
    def loss_classifier(self, targets):
        logodds = self.forward()
        probabilities = 1.0 / (1.0 + torch.exp(-logodds))
        loss = F.binary_cross_entropy(probabilities, targets)
        return loss
    def loss_regressor(self, previous_predictions, targets):
        values = self.forward()
        loss = F.mse_loss(values, targets)
        return loss  

#######################################################################################################
def fit_regression_tree_classifier_to_residuals(X_data, y_data, max_depth): # y_data -> residuals
    tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
    tree_regressor.fit(X_data, y_data)
    leaf_buckets = []
    for i in range(X_data.shape[0]):
        leaf_buckets.append(tuple(tree_regressor.decision_path(X_data[i, :].reshape(1, -1)).todok().keys()))
    unique_paths = list(set(leaf_buckets))
    return (leaf_buckets, unique_paths, tree_regressor)

#######################################################################################################
# Combining all the pieces together
class PytorchBasedGenericGradientBoost():
    def __init__(self, type, n_trees, max_depth, GRADIENT_BOOST_LEARNING_RATE = 0.1, MINIMIZER_LEARNING_RATE = 0.001, MINIMIZER_TRAINING_EPOCHS = 5000):
        '''
        type : "regressor" or "classifier"
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.type = type
        self.gradient_boost_learning_rate = GRADIENT_BOOST_LEARNING_RATE
        self.minimizer_learning_rate = MINIMIZER_LEARNING_RATE
        self.minimizer_training_epochs = MINIMIZER_TRAINING_EPOCHS
        # Variables to hold output of algorithm
        self.initial_prediction = None
        self.regression_trees = []
        # Get an instance of a minimizer
        self.minimizer = LossFunctionMinimizer(self.type)
        if USE_CUDA:
            self.minimizer.cuda()
        self.minimizer_optimizer = torch.optim.Adam(self.minimizer.parameters(), lr=self.minimizer_learning_rate)
    def minimize_loss_function(self, targets, previous_predictions):
        self.minimizer.reinitialize_variable()
        for training_epoch in range(self.minimizer_training_epochs):
            targets_leaf_tensor = FloatTensor(targets)
            loss = self.minimizer.loss_classifier(previous_predictions, targets_leaf_tensor)
            self.minimizer.zero_grad()
            loss.backward()
            self.minimizer_optimizer.step()
        return [el for el in self.minimizer.parameters()][0].cpu().detach().numpy()[0]
    def compute_residuals(self, targets, predicted_values):
        model = ResidualsCalculator(predicted_values, self.type)
        if USE_CUDA:
            model.cuda()
        loss = model.loss(targets)
        model.zero_grad()
        loss.backward()
        residuals = model.predicted_values.grad.clone() # deep copy of the input/gradients
        return residuals
    def fit(self, X, y):
        X_values = X.copy()
        y_values = y.copy()
        # Initialization phase
        if USE_CUDA:
            initial_values = torch.zeros(y_values.shape,1).cuda()
        else:
            initial_values = torch.zeros(y_values.shape)
        self.initial_prediction = self.minimize_loss_function(y_values, initial_values)
        prediction_values = np.ones(y_values.shape) * self.initial_prediction

        for classifier_index in range(self.n_trees):
            self.regression_trees.append({"tree_index": classifier_index})
            residuals = self.compute_residuals(FloatTensor(y_values), FloatTensor(prediction_values))
            leaf_buckets, unique_clusters, tree_regressor = fit_regression_tree_classifier_to_residuals(X_values, residuals.cpu(), self.max_depth)
            self.regression_trees[-1]["tree_regressor"] = tree_regressor

            X_values_temp = np.array([])
            y_values_temp = np.array([])
            prediction_values_temp = np.array([])

            for unique_cluster in unique_clusters:
                indices = [1 if el == unique_cluster else 0 for el in leaf_buckets]
                y_leaf = y_values[np.array(indices) == 1]
                X_leaf = X_values[np.array(indices) == 1]
                predictions_leaf = prediction_values[np.array(indices) == 1]
                prediction_for_leaf = self.minimize_loss_function(FloatTensor(np.array(y_leaf)), FloatTensor(predictions_leaf))
                predictions_for_leaf_array = np.ones(y_leaf.shape) * self.gradient_boost_learning_rate * prediction_for_leaf + predictions_leaf
                self.regression_trees[-1][str(unique_cluster)] = prediction_for_leaf
                X_values_temp = X_leaf if X_values_temp.shape == (0, ) else np.append(X_values_temp, X_leaf, axis=0)
                y_values_temp = np.append(y_values_temp, y_leaf)
                prediction_values_temp = np.append(prediction_values_temp, predictions_for_leaf_array)
            y_values = y_values_temp
            X_values = X_values_temp
            prediction_values = prediction_values_temp    
    def predict(self, X):
        predictions = []
        for index in range(X.shape[0]):
            prediction = self.initial_prediction
            for tree_index in range(self.n_trees):
                tree = self.regression_trees[tree_index]
                prediction += self.gradient_boost_learning_rate * tree[str(tuple(tree["tree_regressor"].decision_path(X[index, :].reshape(1,-1)).todok().keys()))]
            predictions.append(prediction)
        if self.type == "regressor":
            return predictions
        elif self.type == "classifier":
            return torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            raise Exception("Not supported")