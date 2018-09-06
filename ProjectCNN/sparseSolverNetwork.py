import numpy as np
import tensorflow as tf

"""
* define fetch_batch
"""
def fetch_batch(data, batch_size, epoch):
    n_iter, _, _ = np.shape()
    np.random.seed(epoch)
    shuffle_index = np.arrange(n_iter)
    np.random.shuffle( shuffle_index )
    x_batch = data[shuffle_index[:batch_size],:,0]
    y_batch = data[shuffle_index[:batch_size],:,1]
    return x_batch, y_batch
   
def sparseNetworkSolver(data, given_learning_rate, n_epochs, batch_size):
    n_iter, nn, _ = np.shape(data)

    """
    * indices: generate a list to store the non-zero indices of the weights
    * values: generate a list to store the values at those indices
    * dense_shape: shape of the co-efficient matrix
    """
    indices=[]
    indices.append([0,0])
    indices.append([0,1])
    for i in range(1,nn-1):
        indices.append([i,i-1])
        indices.append([i,i])
        indices.append([i,i+1])
    indices.append([nn-1,nn-2])
    indices.append([nn-1,nn-1])
    
    nn_nonZero_entries = 2 + 3*(nn-2) + 2
    values = np.random.rand(nn_nonZero_entries)

    weights = tf.SparseTensor(indices=indices, values=values, dense_shape=[nn,nn])
    X = tf.placeholder(tf.float32, shape=(None, nn), name="batchX")
    y = tf.placeholder(tf.float32, shape=(None, nn), name="batchY")
    learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
    
    return_weights = tf.sparse_tensor_to_dense(weights)

    output = tf.sparse_matmul(weights, tf.transpose(X), a_is_sparse=True)
    mse = tf.reduce_mean(tf.square(y-tf.transpose(output)), name="mse")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()

    count = 0
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch >= 30000:
                given_learning_rate = 0.01
            x_batch, y_batch = fetch_batch(data, batch_size, epoch, n_iter)
            try:
                sess.run(training_op, feed_dict={X:x_batch, y:y_batch, learning_rate: given_learning_rate})
            except:
                sess.run("Training operation failed")
            if epoch % 100 == 0:
                print(mse.eval(feed_dict={X:x_batch, y:y_batch}))
        learned_weights = sess.run(return_weights)
    return learned_weights


