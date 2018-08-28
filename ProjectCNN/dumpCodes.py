FOX0L48@SR-71:~/Documents/workspace/ProjectTF/ProjectCNN$ python
Python 2.7.12 (default, Dec  4 2017, 14:50:18) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
import numpy as np
import scipy as sp
nn = 25000
h = 1/25000
scale_factor = nn * nn

np.identity(3,3)
scale_factor = - nn * nn
D = scale_factor * 2 * np.ones((nn, 1), float).flatten()
E = scale_factor * np.ones((nn, 1), float).flatten()
x = np.linspace(0, 1, nn)
load_function = np.multiply(-x,np.exp(x), (x+3)) 
sol = (load_function - np.multiply(E, x[0:nn-2]) - np.multiply(E, x[1:nn-1]))
E = scale_factor * np.ones((nn-1, 1), float).flatten()
np.multiply(E, x[0:nn-1])
np.multiply(E, x[1:nn])



##
nx = 25000

x = np.linspace(0, 1, nx)
h = 1/nx

shape_vector = [1, -2, 1]


main_diag = np.ones((nx, 1), float).flatten()
off_diag = np.ones((nx-1, 1), float).flatten()

A = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
# to get the main array: A.toarray()

E1 = np.insert(E, nn-3, 0)
E2 = np.insert(E, 0, 0)
sol = np.zeros(np.size(x))
#for i in range(1000):
#  sol1 = 1/D * (load_function - np.multiply(E1, np.insert(sol[1:nn-2], nn-3, 0)) - np.multiply(E2, np.insert(sol[0:nn-3], 0, 0) ) )
#  sol = sol1
#%%
import tensorflow as tf
import numpy as np
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
  print(sess.run(scatter))
