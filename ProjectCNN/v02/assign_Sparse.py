import numpy as np
import tensorflow as tf
from jacobiSolver import jacobiSolver as jS

"""
* define fetch_batch
"""
def fetch_batch(data, batch_size, epoch):
    n_iter, _, _ = data.shape
    np.random.seed(epoch)
    shuffle_index = np.arange(n_iter)
    np.random.shuffle( shuffle_index )
    x_batch = data[shuffle_index[:batch_size],:,0]
    y_batch = data[shuffle_index[:batch_size],:,1]
    return x_batch, y_batch
"""
Function to train the network to perform one jacobi number_of_jacobiIterations
Return: trained weights for a specific problem size.
"""
def sparseTrainer(data, learning_rate, n_epochs, batch_size):
      
	_, problem_size, _ = data.shape
	
    """
    Algorithm tuning variables
    """
    count_LRswitch = 30000
    msePrint_frequency = 1000

    """
    X: placeholder for input vector
    y: placeholder for output vector
    w_0: variable holding the diagonal entries of the Load matrix
   	w_u: variable holding the off-diagonal entries (upper) of the Load matrix
   	w_l: variable holding the off-diagonal entries (lower) of the Load matrix
    """
	X = tf.placeholder(tf.float64, shape=(None, problem_size), name="x")

	w_0 = tf.Variable(np.random.rand(problem_size), tf.float64, name="vector_diag")

	w_u = tf.Variable(np.random.rand(problem_size-1), tf.float64, name="weights_u")

	w_l = tf.Variable(np.random.rand(problem_size-1), tf.float64, name="weights_l")

	y = tf.placeholder(tf.float64, shape=(None, problem_size), name="y")

	tf_learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

	# w = tf.diag(w_0) + tf.pad(tf.diag(w_u), [[0,1], [1,0]], "CONSTANT") +\
	                  # tf.pad(tf.diag(w_l), [[1,0], [0,1]], "CONSTANT")
	"""
	We represent the Linear Operation (W . x = (L + D + U) . x) where in every individual
	matrix (L, D, U) is recreated using a tensorflow variable vector (trainable)
	"""
	output = tf.matmul( (tf.diag(w_0) + tf.pad(tf.diag(w_u), [[0,1], [1,0]], "CONSTANT") + tf.pad(tf.diag(w_l), [[1,0], [0,1]], "CONSTANT") ), 
		tf.transpose(X) )

	mse = tf.reduce_mean(tf.square(y - tf.transpose(output)))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=tf_learning_rate)
	training_op = optimizer.minimize(mse)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(n_epochs):
			if epoch > count_LRswitch:
				learning_rate = 0.01
			x_batch, y_batch = fetch_batch(data, batch_size, epoch)
			try:
				sess.run(training_op, feed_dict={X:x_batch, y:y_batch, tf_learning_rate:learning_rate})
			except:
				print("Training operation failed.")
			if epoch % msePrint_frequency == 0:
				print(mse.eval(feed_dict={X:x_batch, y:y_batch, learning_rate:learning_rate}))
		temp = sess.run(w)
		print(temp)

    


