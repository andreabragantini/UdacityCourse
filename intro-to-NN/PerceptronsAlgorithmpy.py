# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:54:21 2020

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines, W, b


data = pd.read_csv('data.csv', sep=',', header=None, names =['X0','X1','label'])
X = np.array([data['X0'], data['X1']]).T
y = np.array(data['label'])

boundary_lines, W, b = trainPerceptronAlgorithm(X,y)

coef0 = boundary_lines[-1][0][0]
coef1 = boundary_lines[-1][1][0]

# Plotting
red = data[data['label']==1]
blue = data[data['label']==0]

plt.plot(red['X0'], red['X1'], 'ro')
plt.plot(blue['X0'], blue['X1'], 'o')
plt.plot(np.linspace(0,1,11), coef0*np.linspace(0,1,11)+coef1)
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

