# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:12:18 2018

@author: kushal
"""

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
print(sess.run(linear_model, {x:[1,2, 3, 4]}))
#print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

n_epochs = 1000

for epochs in range(n_epochs):
    sess.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

print(sess.run([W, b]))
sess.close()

#%%
"""
mnist dataset
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
#every image is 28x28
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable( tf.zeros([10]) )

sess.run( tf.global_variables_initializer() )

y_ = tf.matmul(x,w) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range (1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0], y: batch[1]})

correct_prediction = tf.equal( tf.argmax(y_, 1), tf.argmax(y,1) )
accuracy = tf.reduce_mean(tf.cast( correct_prediction, tf.float32) )
print(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))


