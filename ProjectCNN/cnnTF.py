import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib as plt

from sklearn.datasets import load_sample_images

#load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)

#variable for storing training data
x = tf.placeholder(tf.float32, [None, 784])
# here "None" means that the dimension can be of any length

#variable for storing the weigtts:
W = tf.Variable(tf.zeros([784, 10]))
#variable for storing bias
b = tf.Variable(tf.zeros([10])

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


