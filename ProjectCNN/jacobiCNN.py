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
training_data = np.zeros((n_iter, nn, 2), float)

training_data[:, 0, 0] = xx[0]
training_data[:, nn-1, 0] = xx[nn-1]

training_data[:, 0, 1] = boundary_x0
training_data[:, nn-1, 1] = boundary_x1

for i in range(n_iter):
  training_data[i, 1:nn-1, 0] = sol2 #input to the network
  sol2 = (load_function - np.dot(A, sol2)) / D
  training_data[i, 1:nn-1, 1] = sol2 #iteration data for training the network
  
#plotting the results
plt.plot(x, exact_solution, 'r', x, sol2, 'b')
plt.show()

#-------training network--------------
featureSize = nn
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(n_iter / batch_size))
batch_index = 0
n_epochs = 100

x_batch = tf.placeholder(tf.float32, shape=(None, nn), name="batchX")
y_batch = tf.placeholder(tf.float32, shape=(None, nn), name="batchY")

jac_outputIter = tf.Variable(tf.zeros(nn), dtype=tf.float32, name="jacobiOutput")
net_inputIter = tf.Variable(tf.zeros(nn), dtype=tf.float32, name="iterationInput")

net_outputIter = tf.Variable(tf.zeros(nn), dtype=tf.float32, name="iterationOutput")
weights = tf.Variable( tf.random_uniform([nn, nn], -1.0, 1.0), name="weights")
iter_error = tf.Variable(tf.zeros(nn), dtype=tf.float32, name="iter_error")

net_outputIter = tf.matmul(weights, tf.transpose(net_inputIter)

#
##the execution graph
iter_error = tf.subtract(jac_outputIter, net_outputIter)
mse = tf.reduce_mean(tf.square(iter_error), name="mse")

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


with tf.Session() as sess:
  sess.run(init)
    
  for epoch in range(n_epochs):
    x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
    if epoch % 100 == 0:
      print("Epoch", epoch, "MSE =", mse.eval())
    sess.run(training_op)
    
best_weight = weights.eval()
print(best_weight)

def fetch_batch(epoch, batch_index, batch_size):
  x_batch = training_data[batch_index * batch_size:batch_index:batch_size + batch_size -1, :, 0]
  y_batch = training_data[batch_index * batch_size:batch_index:batch_size + batch_size -1, :, 1]
  return x_batch, y_batch
  













