# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:25:57 2018

@author: FOX0L48
x: meshgrid
A: co-efficient matrix
b: weight function
n_x: number of grid points in the discretization
"""
import numpy as np
from scipy.sparse import diags
import tensorflow as tf
import matplotlib.pyplot as plt


#jacobi iterations
nn = 100

scale_factor = nn * nn

xx = np.linspace(0, 1, nn)
x = xx[1:nn-1]

load_function = 2. * np.exp(x) + x * np.exp(x) 
#incorporating the boundary
boundary_x0 = 0.
boundary_x1 = scale_factor * np.exp(1)
load_function[0] = load_function[0] - boundary_x0
load_function[nn-3] = load_function[nn-3] - boundary_x1

exact_solution = x * np.exp(x)

D = scale_factor * -2. * np.ones((nn-2, 1), float).flatten()
E = scale_factor * np.ones((nn-3, 1), float).flatten()
A = diags([E, np.zeros((nn-2, 1), float).flatten(), E], [-1, 0, 1]).toarray()

sol2 = np.zeros(np.size(x))

n_iter = 6000
#array for training data:
training_data = np.zeros((n_iter, nn-2), float)
training_data[0, :] = sol2
for i in range(n_iter):
  #dataTrain[i, :, 0] = sol2 #input to the network
  sol2 = (load_function - np.dot(A, sol2)) / D
  training_data[i, :] = sol2 #iteration data for training the network
  
#plotting the results
plt.plot(x, exact_solution, 'r', x, sol2, 'b')
plt.show()










