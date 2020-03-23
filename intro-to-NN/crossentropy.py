# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:30:47 2020

@author: andre
"""

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ce = 0
    for i in range(len(Y)):
        a = Y[i]*np.log(P[i])
        b = (1-Y[i])*np.log(1-P[i])
        ce -= a+b    
    
    return ce

Y = [0,1]
P = [.9,.1]
results = cross_entropy(Y,P)
print(results)

# Other solution

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))