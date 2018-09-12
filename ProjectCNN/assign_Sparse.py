import numpy as np
import tensorflow as tf
from jacobiSolver import jacobiSolver as jS
from sparseSolverNetwork import fetch_batch

global sess

def matrix_reshape(weights, w_reshaped):
    tmp = sess.run(weights)
    return_matrix = sess.run( tf.assign(w_reshaped, tf.diag(tmp)) )

problem_size = 5
learning_rate = 0.01
batch_size = 10
n_epochs = 1

x = tf.placeholder(tf.float32, shape=(None, problem_size), name="x")
w = tf.Variable(np.random.rand(5), tf.float32, name="lin_weights")
y = tf.placeholder(tf.float32, shape=(None, problem_size), name="y")
w_reshaped = tf.Variable(np.diag(np.random.rand(5)), dtype=tf.float32, name="w_reshaped")

output = tf.matmul(w_reshaped, tf.transpose(x), name="Mul_output")
mse = tf.reduce_mean(tf.square(y - tf.transpose(output)))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(mse, var_list = [w] )

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        w_temp = sess.run(w)
        print(w_temp)
        #sess.run(tf.assign(w_reshaped, w_temp))



