import numpy as np
import tensorflow as tf
from scipy.sparse import diags

nn=5
init_weights = np.diag(np.random.rand(nn), 0) + np.diag(np.random.rand(nn-1), -1) \
                                            + np.diag(np.random.rand(nn-1), 1)
print(init_weights)
#input iteration value: training_data[:, :, 0]
X = tf.placeholder(tf.float32, shape=(None, nn), name="batchX")
#y output iteration value: training_data[:, :, 1]
y = tf.placeholder(tf.float32, shape=(None, nn), name="batchY")
#weights = tf.Variable(init_weights, dtype = tf.float32)
weights = tf.Variable(np.ones(5), dtype = tf.float32,name="weight")
#print(weights)
trainable_weights = tf.where(weights > tf.constant(0.0))
output = tf.multiply(weights, X)
error = tf.subtract(y, output)
mse = tf.reduce_mean(tf.square(error), name="mse")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    v_temp = tf.get_variable("weight:0")
    v_temp[:3] = 0
    print(sess.run(weights))


