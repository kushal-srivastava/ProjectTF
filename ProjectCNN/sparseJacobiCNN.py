import numpy as np
import tensorflow as tf
from scipy.sparse import diags
"""
todo
* function to fetch data for batch gradient
* batch_size: 
* epoch
* batch_index: batch number used as seed for random input.
"""
def fetch_batch(data, batch_size,epoch):
    global n_iter
    np.random.seed(epoch)
    shuffle_index = np.arange(n_iter)
    np.random.shuffle( shuffle_index )
    x_batch = data[shuffle_index[:batch_size], :, 0]
    y_batch = data[shuffle_index[:batch_size], :, 1]
    return x_batch, y_batch

"""
* nn: number of discretization points
* scale_factor: (1/ h2) scaling of the elements 
  in co-efficient matrix
* learning_rate:
* batch_index: batch num. used as seed to generate random vector
  while reshuffling the input data
* n_batches: 
* n_epochs: number of training iterations
"""
nn = 7
n_iter = 100
x = np.linspace(0, 1, nn)
scale_factor = nn * nn

learning_rate = 0.3
batch_size = 10
batch_index = 0
n_batches = int(np.ceil(n_iter / batch_size))
n_epochs =100

load_function = 2. * np.exp(x) + x * np.exp(x)

"""
todo
* moving the boundary points to the right hand side
* Creating tridiagonal matrix using:
    D: diagonal vector
    E: off-diagonal vectors
* initialize a random tri-diagonal matrix to store be used
  as initial weights for training the network
"""
boundary_x0 = 0.
boundary_x1 = scale_factor * np.exp(1)
load_function[1] = load_function[1] - boundary_x0
load_function[nn-2] = load_function[nn-2] - boundary_x1

D = scale_factor * -2. * np.ones((nn-2, 1), float).flatten()
E = scale_factor * np.ones((nn-3, 1), float).flatten()
A = diags([E, np.zeros((nn-2, 1), float).flatten(), E], [-1, 0, 1]).toarray()


init_weights = np.diag(np.random.rand(nn), 0) + np.diag(np.random.rand(nn-1), -1) \
                                            + np.diag(np.random.rand(nn-1), 1)

D_1 =  np.ones((nn, 1), float).flatten()
E_1 =  np.ones((nn-1, 1), float).flatten()
A_1 = diags([E_1, D_1, E_1], [-1, 0, 1]).toarray()
"""
todo
* jacobi iterations to generate the data for training
* n_iter: number of iterations for jacobi
* fix the boundary points for both input and output 
* generate random input for the solver
* store the input and output in training_data 
"""

training_data = np.zeros((n_iter, nn, 2), float)

training_data[:, 0, 0] = x[0]
training_data[:, nn-1, 0] = x[nn-1]

training_data[:, 0, 1] = boundary_x0
training_data[:, nn-1, 1] = boundary_x1

for i in range(n_iter):
    sol2 = np.random.rand(nn-2)
    training_data[i, 1:nn-1, 0] = sol2 #input to the network
    sol2 = (load_function[1:nn-1] - np.dot(A, sol2)) / D
    training_data[i, 1:nn-1, 1] = sol2 #iteration data for training the network

"""
tensorflow variables
* assign random_weights to weights using convert_to_tensor
* initialize global variables
* X:
* y: 
* trainable_weights: index of weights to train
* output: 
* error: 
* mse:
"""
const_matrix = tf.constant(A_1,dtype=tf.float32)
#input iteration value: training_data[:, :, 0]
X = tf.placeholder(tf.float32, shape=(None, nn), name="batchX")
#y output iteration value: training_data[:, :, 1]
y = tf.placeholder(tf.float32, shape=(None, nn), name="batchY")
weights = tf.Variable(np.random.rand(nn, nn), dtype = tf.float32)

#gen_tridiag = weights * const_matrix
#weight_assign = tf.assign(weights, gen_tridiag)

#weights = tf.convert_to_tensor(init_weights, dtype = tf.float32)
#trainable_weights = tf.where(weights > tf.constant(0.0))

output = tf.multiply(weights, tf.transpose(X), name="Mul_output")
#error = tf.subtract(y, output, name="error")
mse = tf.reduce_mean(tf.square(y - output), name="mse")

optimizer = tf.train.GradientDescentOptimizer(0.5)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
#print(error)
#epoch = 11
#x_batch, y_batch = fetch_batch(training_data, batch_size, epoch)

#%%
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        x_batch, y_batch = fetch_batch(training_data, batch_size, epoch)
        try:
          sess.run(training_op, feed_dict={X:x_batch, y:y_batch})
        except:
          print("Training operation failed.")
#        sess.run(weight_assign)
        if epoch % 10 == 0:
            print(mse.eval(feed_dict={X:x_batch, y:y_batch}))
        #batch_index += 1

