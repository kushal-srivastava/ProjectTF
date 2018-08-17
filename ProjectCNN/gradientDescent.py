# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 20:43:47 2018

@author: FOX0L48
"""

import numpy as np
import matplotlib.pyplot as plt

nn = 1000
alpha = 0.01
x = np.linspace(-6,6, nn)
y =x * x + 2*x
dy = 2 * x + 2

x_min = -4
min_x = np.zeros(nn)
for i in range(nn):
  x_min = x_min - alpha * (2 * x_min + 2)
  min_x[i] = x_min

print('minimum of y is at x = ' + str(x_min))
plt.plot(x, y, 'r', x, dy, 'b', x, min_x, 'g')
plt.show()
