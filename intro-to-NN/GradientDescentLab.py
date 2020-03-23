# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:16:19 2020

@author: andre
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Some helper functions for plotting and drawing lines

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)
    
# Reading and plotting data
data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plot_points(X,y)
plt.show()

# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    expX = np.exp(-x)
    return 1/(1+expX)

# Output (prediction) formula
def output_formula(features, weights, bias):
    score = sum(features[i]*weights[i] for i in len(features))
    score = score + bias
    return sigmoid(score)

# or it can be also
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    logYhat = np.log(output)
    logYhat2 = np.log(1-output)
    return -y*logYhat-(1-y)*(1-logYhat2)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    weights = weights + learnrate*(y - output)*x
    bias = bias + learnrate*(y - output)
    return weights, bias

### TRAINING FUNCTION
    
np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):                                    # For each feature and target in the dataset     
            output = output_formula(x, weights, bias)                          # Calculate scores (predictions): 1st created formula 
            error = error_formula(y, output)                                   # Calculate log-loss error: 2nd created formula  
            weights, bias = update_weights(x, y, weights, bias, learnrate)     # Updates W,b: 3rd created formula 
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))                            # Calculates the error by averaging errors for all observation in dataset 
        errors.append(loss)                                                    # Errors is a vector containing the log-loss error at each epoch     
        if e % (epochs / 10) == 0:                                             # Print info every 10 epochs 
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()
    
    return weights, bias

### TRAINING
weights, bias = train(X, y, epochs, learnrate, True)

# Thies returns the coefficients of the line describing our model and our data
# This coefficients have been found with the gradient descend algorithm