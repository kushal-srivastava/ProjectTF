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
def fetch_batch(data, batch_size,epoch, n_iter):
    np.random.seed(epoch)
    shuffle_index = np.arange(n_iter)
    np.random.shuffle( shuffle_index )
    x_batch = data[shuffle_index[:batch_size], :, 0]
    y_batch = data[shuffle_index[:batch_size], :, 1]
    return x_batch, y_batch


"""
* nn: number of discretization points
* n_iter: number of data samples
* learning_rate:
* n_epochs: number of training iterations
* D_1: vector of ones with len: nn
* E_1: vector of ones with len:nn-1
* A_1: tridiagonal matrix(ones) used to reshape the weights
"""
def solveNetwork(data, given_learning_rate, n_epochs, batch_size):
	n_iter, nn, _ = np.shape(data)

	#n_iter = 10000
	#batch_size = 500
	#n_epochs =50000
	#learning_rate_array = 0.6*np.exp(-20000*np.linspace(0.000001,0.001,5))

	D_1 =  np.ones((nn, 1), float).flatten()
	E_1 =  np.ones((nn-1, 1), float).flatten()
	A_1 = diags([E_1, D_1, E_1], [-1, 0, 1]).toarray()

	const_matrix = tf.constant(A_1,dtype=tf.float32)

	#input iteration value: training_data[:, :, 0]
	X = tf.placeholder(tf.float32, shape=(None, nn), name="batchX")

	#y output iteration value: training_data[:, :, 1]
	y = tf.placeholder(tf.float32, shape=(None, nn), name="batchY")

	learning_rate = tf.placeholder_with_default(0.001, shape=(), name="learning_rate")
	weights = tf.Variable(np.random.rand(nn, nn), dtype = tf.float32)

	#maintaining the tridiagonal nature of the weight matrix
	gen_tridiag = weights * const_matrix
	weight_assign = tf.assign(weights, gen_tridiag)

	output = tf.matmul(weights, tf.transpose(X), name="Mul_output")
	mse = tf.reduce_mean(tf.square(y - tf.transpose(output)), name="mse")

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

	training_op = optimizer.minimize(mse)
	init = tf.global_variables_initializer()

	learning_rate_epoch = 0.6
	count = 0
	#print("here")
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(n_epochs):
			if epoch >= 30000:
				learning_rate_epoch = 0.01
				count += 1
			x_batch, y_batch = fetch_batch(data, batch_size, epoch, n_iter)
			try:
				sess.run(training_op, feed_dict={X:x_batch, y:y_batch, learning_rate:learning_rate_epoch})
			except:
				print("Training operation failed.")				
			sess.run(weight_assign)
			if epoch % 1000 == 0:
				print(mse.eval(feed_dict={X:x_batch, y:y_batch}))
		return_weights = sess.run(weights)
	return return_weights


