# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:40:23 2020

@author: andre
"""

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    prob = []
    for i in L:
        prob.append(np.exp(i)/sum(np.exp(j) for j in L))
        
    return prob

L = [4, 6, 8, -4]
results = softmax(L)
print(results)

# Other solution

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())