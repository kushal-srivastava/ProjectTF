# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
###########################################
#using sklearn
#from sklearn.datasets import fetch_mldata
#mnist = fetch_mldata('MNIST original')
###########################################

###########################################
#using tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
###########################################


#init = tf.global_variables_initializer()
#tensorflow variables

mnistData_train = tf.Variable(55000, 784)
mnistClass_train = tf.Variable(55000, 10)
mnistData_test = tf.Variable(10000, 784)
mnistClass_test = tf.Variable(10000, 10)
mnistData_cv = tf.Variable(5000, 784)
mnistClass_cv = tf.Variable(5000, 10)

#X y_ are placeholders a value that we will input when we
# run the computation.
#None: will depend on the batch size for computation, hence 
#reserved for runtime
x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))

W = tf.Variable(np.random.rand(784, 10).astype(np.float32))
b = tf.Variable(tf.zeros([10]))
###########################################
#setting up the data
# mnist_train: training set
# mnist_test: test set
# mnist_cv: cross validation data
# mnist dataset contains 70000 images along with labels
#tensorflow dataset divides it into 3 subsets
# {55k, 10k, 5k} {train, test, validate}
###########################################
#shuffle step not needed as data has already been divided into 
# respective classes
#shuffle_index = np.random.permutation(70000)
mnistData_train = mnist.train.images[:,:]
mnistData_test = mnist.test.images[:,:]
mnistData_cv = mnist.validation.images[:,:]
#
mnistClass_train = mnist.train.labels[:,:]
mnistClass_test = mnist.test.labels[:,:]
mnistClass_cv = mnist.validation.labels[:,:]


y = tf.nn.softmax(tf.matmul(x, W) + b)
#defining the loss function
cross_entropy = tf.reduce_mean( \
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
###########################################
sess = tf.InteractiveSession()

sess.run(init)
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()



