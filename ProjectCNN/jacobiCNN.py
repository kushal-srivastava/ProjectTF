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
import matplotlib.pyplot as plt


#jacobi iterations
nn = 10

scale_factor = - nn * nn

xx = np.linspace(0, 1, nn)
x = xx[1:nn-1]

load_function = 2 * np.exp(x) + x * np.exp(x) 
#incorporating the boundary
boundary_x0 = 0.
boundary_x1 = 2 * np.exp(1) + np.exp(1)
load_function[0] = load_function[0] - boundary_x0
load_function[nn-3] = load_function[nn-3] - boundary_x1

exact_solution = 4 * x - 8 * np.exp(x) + 5 * x * np.exp(x) - np.exp(x) * x * x

D = scale_factor * -2 * np.ones((nn-2, 1), float).flatten()
E = scale_factor * np.ones((nn-3, 1), float).flatten()
A = diags([E, np.zeros((nn-2, 1), float).flatten(), E], [-1, 0, 1]).toarray()

sol2 = np.zeros(np.size(x))

n_iter = 200
#array for training data:
inputData = np.zeros((n_iter, np.size(sol2)), float)
for i in range(n_iter):
  sol2 = (load_function - np.dot(A, sol2)) / D

print(np.max(exact_solution - sol2))

#plotting the results
plt.plot(x, exact_solution, 'r')
plt.show()





