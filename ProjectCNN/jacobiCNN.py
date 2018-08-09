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

load_function = np.multiply(-x,np.exp(x), (x+3))
#incorporating the boundary
load_function[0] = load_function[0] - (-8) / scale_factor
load_function[nn-3] = load_function[nn-3] - (4 + 4*np.exp(1)) / scale_factor

exact_solution = 4 * x - 8 * np.exp(x) + 5 * np.multiply(x, np.exp(x)) - np.multiply(np.exp(x), x, x)

D = scale_factor * -2 * np.ones((nn-2, 1), float).flatten()
E = scale_factor * np.ones((nn-3, 1), float).flatten()
A = diags([E, np.zeros((nn-2, 1), float).flatten(), E], [-1, 0, 1]).toarray()

sol2 = np.zeros(np.size(x))

for i in range(1000):
  sol2 = (load_function - np.dot(A, sol2)) / D
 
print(exact_solution - sol2)



